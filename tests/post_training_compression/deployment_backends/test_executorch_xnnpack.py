"""Tests for versatil.post_training_compression.deployment_backends.executorch_xnnpack module."""

import re
from collections.abc import Callable
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from torchao.quantization import Int8DynamicActivationIntxWeightConfig, quantize_
from torchao.quantization.granularity import PerGroup

from versatil.inference.policy_runtime.executorch_adapter import ExecuTorchModuleAdapter
from versatil.post_training_compression.constants import (
    ArtifactFormat,
    CompressionFilename,
    DeploymentBackendName,
)
from versatil.post_training_compression.deployment_backends.executorch_xnnpack import (
    ExecutorchXNNPACKBackend,
    _lower_exported_program,
)
from versatil.post_training_compression.export import _export_with_dynamic_batch

XNNPACK_MODULE = (
    "versatil.post_training_compression.deployment_backends.executorch_xnnpack"
)


@pytest.fixture
def eager_xnnpack_model_factory(
    rng: np.random.Generator,
) -> Callable[[], nn.Module]:
    def factory() -> nn.Module:
        model = nn.Sequential(
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
        )
        with torch.no_grad():
            for parameter in model.parameters():
                data = rng.standard_normal(parameter.shape).astype(np.float32)
                parameter.copy_(torch.from_numpy(data))
        model.eval()
        quantize_(
            model=model,
            config=Int8DynamicActivationIntxWeightConfig(
                weight_dtype=torch.int4,
                weight_granularity=PerGroup(32),
            ),
        )
        return model

    return factory


@pytest.fixture
def xnnpack_example_inputs_factory(
    rng: np.random.Generator,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    def factory(batch_size: int = 2) -> tuple[torch.Tensor, ...]:
        features = torch.from_numpy(
            rng.standard_normal((batch_size, 64)).astype(np.float32)
        )
        return (features,)

    return factory


@pytest.fixture(scope="session")
def executorch_backend_factory() -> Callable[..., ExecutorchXNNPACKBackend]:
    def factory(max_batch_size: int) -> ExecutorchXNNPACKBackend:
        return ExecutorchXNNPACKBackend(max_batch_size=max_batch_size)

    return factory


@pytest.fixture
def executorch_runtime_factory(
    tmp_path: Path,
) -> Callable[..., ExecuTorchModuleAdapter]:
    def factory(model_bytes: bytes, filename: str) -> ExecuTorchModuleAdapter:
        model_path = tmp_path / filename
        model_path.write_bytes(model_bytes)
        return ExecuTorchModuleAdapter(model_path=str(model_path))

    return factory


@pytest.mark.unit
class TestExecutorchXNNPACKBackend:
    @pytest.mark.parametrize("max_batch_size", [1, 8, 16])
    def test_stores_configuration(self, max_batch_size: int) -> None:
        backend = ExecutorchXNNPACKBackend(max_batch_size=max_batch_size)

        assert backend.max_batch_size == max_batch_size

    @pytest.mark.parametrize("max_batch_size", [0, -1])
    def test_rejects_invalid_max_batch_size(self, max_batch_size: int) -> None:
        with pytest.raises(
            ValueError,
            match=re.escape("max_batch_size must be >= 1."),
        ):
            ExecutorchXNNPACKBackend(max_batch_size=max_batch_size)

    def test_export_lowers_model_to_pte_bytes(self) -> None:
        backend = ExecutorchXNNPACKBackend(max_batch_size=8)
        model = MagicMock(spec=nn.Module)
        example_inputs = (torch.zeros(2, 4),)
        exported_program = MagicMock()

        with (
            patch(
                f"{XNNPACK_MODULE}._export_with_dynamic_batch",
                return_value=exported_program,
            ) as mock_export,
            patch.object(
                backend,
                "_lower_to_pte_buffer",
                return_value=b"pte-bytes",
            ) as mock_lower,
        ):
            artifact = backend.export(model=model, example_inputs=example_inputs)

        mock_export.assert_called_once_with(
            model=model,
            example_inputs=example_inputs,
            max_batch_size=8,
        )
        mock_lower.assert_called_once_with(exported_program=exported_program)
        assert artifact.converted_model is None
        assert artifact.example_inputs is example_inputs
        assert artifact.model_bytes == b"pte-bytes"
        assert artifact.model_filename == CompressionFilename.EXECUTORCH_MODEL.value
        assert artifact.artifact_format == ArtifactFormat.EXECUTORCH_PTE
        assert artifact.backend_name == DeploymentBackendName.EXECUTORCH_XNNPACK.value


@pytest.mark.integration
@pytest.mark.requires_executorch
class TestExecutorchXNNPACKBackendIntegration:
    @pytest.mark.parametrize(
        "batch_size, expectation",
        [
            (1, does_not_raise()),
            (2, does_not_raise()),
            (
                8,
                pytest.raises(
                    RuntimeError,
                    match=re.escape("Failed to execute method forward, error: 0x10"),
                ),
            ),
        ],
    )
    def test_unbounded_dynamic_batch_uses_example_size_as_runtime_limit(
        self,
        eager_xnnpack_model_factory: Callable[[], nn.Module],
        xnnpack_example_inputs_factory: Callable[..., tuple[torch.Tensor, ...]],
        executorch_backend_factory: Callable[..., ExecutorchXNNPACKBackend],
        executorch_runtime_factory: Callable[..., ExecuTorchModuleAdapter],
        batch_size: int,
        expectation: AbstractContextManager[None],
    ) -> None:
        model = eager_xnnpack_model_factory()
        example_inputs = xnnpack_example_inputs_factory(batch_size=2)
        exported_program = _export_with_dynamic_batch(
            model=model,
            example_inputs=example_inputs,
        )

        model_bytes = ExecutorchXNNPACKBackend._lower_to_pte_buffer(
            exported_program=exported_program,
        )
        runtime = executorch_runtime_factory(
            model_bytes=model_bytes, filename=CompressionFilename.EXECUTORCH_MODEL.value
        )
        bounded_artifact = executorch_backend_factory(max_batch_size=8).export(
            model=model, example_inputs=example_inputs
        )
        bounded_runtime = executorch_runtime_factory(
            model_bytes=bounded_artifact.model_bytes, filename="bounded.pte"
        )

        # Use matching XNNPACK kernels to isolate dynamic-shape handling.
        inputs = xnnpack_example_inputs_factory(batch_size=batch_size)
        with torch.no_grad():
            expected = bounded_runtime(observation_tensors=inputs)[
                0
            ]  # (batch_size, 64) -> (batch_size, 16)
            assert expected.shape == (batch_size, 16)
            with expectation:
                actual = runtime(observation_tensors=inputs)[
                    0
                ]  # (batch_size, 64) -> (batch_size, 16)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_export_lowers_eager_quantized_model_with_bounded_dynamic_batch(
        self,
        eager_xnnpack_model_factory: Callable[[], nn.Module],
        xnnpack_example_inputs_factory: Callable[..., tuple[torch.Tensor, ...]],
    ) -> None:
        backend = ExecutorchXNNPACKBackend(max_batch_size=8)
        model = eager_xnnpack_model_factory()
        example_inputs = xnnpack_example_inputs_factory(batch_size=2)

        artifact = backend.export(model=model, example_inputs=example_inputs)

        assert len(artifact.model_bytes) > 0
        assert artifact.model_filename == CompressionFilename.EXECUTORCH_MODEL.value
        assert artifact.artifact_format == ArtifactFormat.EXECUTORCH_PTE
        assert artifact.backend_name == DeploymentBackendName.EXECUTORCH_XNNPACK.value


@pytest.mark.unit
class TestLowerExportedProgram:
    def test_delegates_to_executorch_xnnpack_partitioner(self) -> None:
        exported_program = MagicMock()
        edge_program = MagicMock()
        executorch_program = MagicMock()
        executorch_program.buffer = b"pte"
        edge_program.to_executorch.return_value = executorch_program
        executorch_exir = MagicMock()
        executorch_exir.to_edge_transform_and_lower.return_value = edge_program
        partitioner = MagicMock()
        xnnpack_partitioner = MagicMock()
        xnnpack_partitioner.XnnpackPartitioner.return_value = partitioner

        result = _lower_exported_program(
            exported_program=exported_program,
            executorch_exir=executorch_exir,
            xnnpack_partitioner=xnnpack_partitioner,
        )

        executorch_exir.to_edge_transform_and_lower.assert_called_once_with(
            exported_program,
            partitioner=[partitioner],
        )
        edge_program.to_executorch.assert_called_once_with()
        assert result == b"pte"
