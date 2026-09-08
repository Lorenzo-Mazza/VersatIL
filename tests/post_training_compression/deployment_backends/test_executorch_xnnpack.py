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
from torchao.core.config import AOBaseConfig
from torchao.quantization import (
    Int4WeightOnlyConfig,
    Int8DynamicActivationIntxWeightConfig,
    Int8WeightOnlyConfig,
    IntxWeightOnlyConfig,
    quantize_,
)
from torchao.quantization.granularity import PerGroup
from torchao.quantization.quant_primitives import MappingType
from torchao.quantization.quantize_.workflows.intx.intx_packing_format import (
    IntxPackingFormat,
)

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
from versatil.quantization.constants import PT2EBackendName, QuantizationMode

XNNPACK_MODULE = (
    "versatil.post_training_compression.deployment_backends.executorch_xnnpack"
)


@pytest.fixture
def xnnpack_quantization_config_factory() -> Callable[..., MagicMock]:
    def factory(
        packing: IntxPackingFormat = IntxPackingFormat.UNPACKED_TO_INT8,
        mapping: MappingType = MappingType.SYMMETRIC,
        weight_dtype: torch.dtype = torch.int4,
    ) -> MagicMock:
        config = MagicMock(spec=Int8DynamicActivationIntxWeightConfig)
        config.intx_packing_format = packing
        config.weight_mapping_type = mapping
        config.weight_dtype = weight_dtype
        return config

    return factory


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


@pytest.mark.unit
class TestExecutorchXNNPACKQuantizationValidation:
    @pytest.mark.parametrize("mode", [*list(QuantizationMode), "unsupported"])
    def test_declares_supported_quantization_workflows(self, mode: str) -> None:
        backend = ExecutorchXNNPACKBackend(max_batch_size=8)
        expectation = (
            pytest.raises(
                ValueError,
                match=re.escape(
                    "Deployment backend executorch_xnnpack supports quantization modes "
                    "['none', 'pt2e', 'eager'], got 'unsupported'."
                ),
            )
            if mode == "unsupported"
            else does_not_raise()
        )
        with expectation:
            backend.validate_quantization(mode=mode)

    @pytest.mark.parametrize("pt2e_backend", list(PT2EBackendName))
    def test_pt2e_export_requires_the_xnnpack_quantizer(
        self, pt2e_backend: PT2EBackendName
    ) -> None:
        backend = ExecutorchXNNPACKBackend(max_batch_size=8)
        expectation = (
            does_not_raise()
            if pt2e_backend == PT2EBackendName.XNNPACK
            else pytest.raises(
                ValueError,
                match=re.escape(
                    "Deployment backend executorch_xnnpack supports PT2E backends "
                    "['xnnpack'], got ['x86_inductor']."
                ),
            )
        )
        with expectation:
            backend.validate_quantization(
                mode=QuantizationMode.PT2E.value,
                pt2e_backend_names=(pt2e_backend.value,),
            )

    def test_smoothquant_requires_inductor_lowering(
        self,
        deployment_target_factory: Callable[..., MagicMock],
        deployment_model_factory: Callable[..., MagicMock],
    ) -> None:
        target = deployment_target_factory(
            config=MagicMock(spec=AOBaseConfig), smoothquant=True
        )
        model = deployment_model_factory(device="cpu")

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Target '(root)': SmoothQuantSchema requires the torch_inductor "
                "deployment backend. XNNPACK lowering requires separate validation."
            ),
        ):
            ExecutorchXNNPACKBackend(max_batch_size=8).validate_eager_target(
                model=model, target=target, module_names={"projection"}
            )

        model.get_submodule.assert_not_called()

    @pytest.mark.parametrize(
        "config_type",
        [Int8WeightOnlyConfig, Int4WeightOnlyConfig, IntxWeightOnlyConfig],
    )
    def test_linear_lowering_requires_dynamic_int8_activations(
        self,
        deployment_target_factory: Callable[..., MagicMock],
        deployment_model_factory: Callable[..., MagicMock],
        config_type: type[AOBaseConfig],
    ) -> None:
        target = deployment_target_factory(config=MagicMock(spec=config_type))
        model = deployment_model_factory(device="cpu")

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Target '(root)': XNNPACK linear lowering requires "
                "Int8DynamicActivationIntxWeightConfig with symmetric INT4 weights "
                "and unpacked_to_int8 packing."
            ),
        ):
            ExecutorchXNNPACKBackend(max_batch_size=8).validate_eager_target(
                model=model, target=target, module_names={"projection"}
            )

        model.get_submodule.assert_not_called()

    @pytest.mark.parametrize(
        "packing, mapping, group_size, expectation",
        [
            (
                IntxPackingFormat.UNPACKED_TO_INT8,
                MappingType.SYMMETRIC,
                32,
                does_not_raise(),
            ),
            (
                IntxPackingFormat.UNPACKED_TO_INT8,
                MappingType.SYMMETRIC_NO_CLIPPING_ERR,
                None,
                does_not_raise(),
            ),
            (
                IntxPackingFormat.OPAQUE_TORCHAO_AUTO,
                MappingType.SYMMETRIC,
                32,
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "Target '(root)': XNNPACK requires unpacked_to_int8 weight packing."
                    ),
                ),
            ),
            (
                IntxPackingFormat.UNPACKED_TO_INT8,
                MappingType.ASYMMETRIC,
                32,
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "Target '(root)': XNNPACK requires symmetric weights without affine offsets."
                    ),
                ),
            ),
            (
                IntxPackingFormat.UNPACKED_TO_INT8,
                MappingType.SYMMETRIC,
                16,
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "Target '(root)': XNNPACK grouped INT4 requires a group size divisible by 32."
                    ),
                ),
            ),
        ],
        ids=["group32", "per_channel", "opaque", "asymmetric", "group16"],
    )
    def test_validates_int4_weight_representation(
        self,
        deployment_target_factory: Callable[..., MagicMock],
        deployment_model_factory: Callable[..., MagicMock],
        xnnpack_quantization_config_factory: Callable[..., MagicMock],
        packing: IntxPackingFormat,
        mapping: MappingType,
        group_size: int | None,
        expectation: AbstractContextManager[None],
    ) -> None:
        config = xnnpack_quantization_config_factory(
            packing=packing, mapping=mapping, weight_dtype=torch.int4
        )
        target = deployment_target_factory(config=config, group_size=group_size)
        model = deployment_model_factory(device="cpu")

        with expectation:
            ExecutorchXNNPACKBackend(max_batch_size=8).validate_eager_target(
                model=model, target=target, module_names={"projection"}
            )
            model.get_submodule.assert_called_once_with("projection")

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    @pytest.mark.parametrize("for_conversion", [True, False])
    def test_conversion_checks_selected_weight_devices(
        self,
        deployment_target_factory: Callable[..., MagicMock],
        deployment_model_factory: Callable[..., MagicMock],
        xnnpack_quantization_config_factory: Callable[..., MagicMock],
        device: str,
        for_conversion: bool,
    ) -> None:
        config = xnnpack_quantization_config_factory(
            packing=IntxPackingFormat.UNPACKED_TO_INT8,
            mapping=MappingType.SYMMETRIC,
            weight_dtype=torch.int4,
        )
        target = deployment_target_factory(config=config, group_size=32)
        model = deployment_model_factory(device=device)
        expectation = (
            pytest.raises(
                ValueError,
                match=re.escape(
                    "Target '(root)': this XNNPACK export workflow requires CPU weights."
                ),
            )
            if for_conversion and device == "cuda"
            else does_not_raise()
        )

        with expectation:
            ExecutorchXNNPACKBackend(max_batch_size=8).validate_eager_target(
                model=model,
                target=target,
                module_names={"projection"},
                for_conversion=for_conversion,
            )

        if for_conversion:
            model.get_submodule.assert_called_once_with("projection")
        else:
            model.get_submodule.assert_not_called()

    @pytest.mark.parametrize("known_config", [True, False])
    def test_logs_representations_that_require_lowering_validation(
        self,
        deployment_target_factory: Callable[..., MagicMock],
        deployment_model_factory: Callable[..., MagicMock],
        xnnpack_quantization_config_factory: Callable[..., MagicMock],
        caplog: pytest.LogCaptureFixture,
        known_config: bool,
    ) -> None:
        config = (
            xnnpack_quantization_config_factory(
                packing=IntxPackingFormat.UNPACKED_TO_INT8,
                mapping=MappingType.SYMMETRIC,
                weight_dtype=torch.int8,
            )
            if known_config
            else MagicMock(spec=AOBaseConfig)
        )
        target = deployment_target_factory(config=config, group_size=32)
        model = deployment_model_factory(device="cpu")

        ExecutorchXNNPACKBackend(max_batch_size=8).validate_eager_target(
            model=model, target=target, module_names={"projection"}
        )

        expected = (
            "Target '(root)': XNNPACK eager linear weights other than INT4 "
            "are unverified by these rules."
            if known_config
            else "Target '(root)': XNNPACK lowering for "
            f"{type(config).__module__}.{type(config).__qualname__} is unverified."
        )
        assert caplog.messages == [expected]
        model.get_submodule.assert_called_once_with("projection")


@pytest.mark.integration
@pytest.mark.requires_executorch
class TestExecutorchXNNPACKBackendIntegration:
    @pytest.mark.parametrize("batch_size", [1, 2, 8])
    def test_unbounded_dynamic_batch_uses_example_size_as_runtime_limit(
        self,
        eager_xnnpack_model_factory: Callable[[], nn.Module],
        xnnpack_example_inputs_factory: Callable[..., tuple[torch.Tensor, ...]],
        tmp_path: Path,
        batch_size: int,
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
        model_path = tmp_path / CompressionFilename.EXECUTORCH_MODEL.value
        model_path.write_bytes(model_bytes)
        runtime = ExecuTorchModuleAdapter(model_path=str(model_path))
        bounded_artifact = ExecutorchXNNPACKBackend(max_batch_size=8).export(
            model=model, example_inputs=example_inputs
        )
        bounded_path = tmp_path / "bounded.pte"
        bounded_path.write_bytes(bounded_artifact.model_bytes)
        bounded_runtime = ExecuTorchModuleAdapter(model_path=str(bounded_path))

        # Use matching XNNPACK kernels to isolate dynamic-shape handling.
        inputs = xnnpack_example_inputs_factory(batch_size=batch_size)
        with torch.no_grad():
            expected = bounded_runtime(observation_tensors=inputs)[
                0
            ]  # (batch_size, 64) -> (batch_size, 16)
            assert expected.shape == (batch_size, 16)
            if batch_size <= example_inputs[0].shape[0]:
                actual = runtime(observation_tensors=inputs)[
                    0
                ]  # (batch_size, 64) -> (batch_size, 16)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            else:
                with pytest.raises(
                    RuntimeError,
                    match=re.escape("Failed to execute method forward, error: 0x10"),
                ):
                    runtime(
                        observation_tensors=inputs
                    )  # (batch_size, 64) -> runtime capacity error

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
