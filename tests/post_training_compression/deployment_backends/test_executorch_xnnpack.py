"""Tests for versatil.post_training_compression.deployment_backends.executorch_xnnpack module."""

import re
from collections.abc import Callable
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from types import ModuleType
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
from versatil.quantization.constants import (
    FXNodeOp,
    PT2EBackendName,
    QuantizationMode,
    QuantizationModuleType,
)

XNNPACK_MODULE = (
    "versatil.post_training_compression.deployment_backends.executorch_xnnpack"
)


@pytest.fixture
def exported_embedding_graph_factory() -> Callable[..., MagicMock]:
    def factory(gradient_args: tuple[int | bool, ...]) -> MagicMock:
        exported_program = MagicMock(spec=torch.export.ExportedProgram)
        weights = MagicMock(spec=torch.fx.Node)
        indices = MagicMock(spec=torch.fx.Node)
        embedding = MagicMock(spec=torch.fx.Node)
        embedding.op = FXNodeOp.CALL_FUNCTION.value
        embedding.target = torch.ops.aten.embedding.default
        embedding.args = (weights, indices, *gradient_args)
        linear = MagicMock(spec=torch.fx.Node)
        linear.op = FXNodeOp.CALL_FUNCTION.value
        linear.target = torch.ops.aten.linear.default
        linear.args = (indices, weights, None)
        exported_program.graph.nodes = [embedding, linear]
        return exported_program

    return factory


@pytest.fixture
def lowering_dependencies_factory() -> Callable[..., tuple[MagicMock, MagicMock]]:
    def factory(model_bytes: bytes) -> tuple[MagicMock, MagicMock]:
        executorch_exir = MagicMock(spec=ModuleType)
        executorch_exir.to_edge_transform_and_lower = MagicMock()
        executorch_exir.ExecutorchBackendConfig = MagicMock()
        edge_program = executorch_exir.to_edge_transform_and_lower.return_value
        edge_program.to_executorch.return_value.buffer = model_bytes
        xnnpack_partitioner = MagicMock(spec=ModuleType)
        xnnpack_partitioner.XnnpackPartitioner = MagicMock()
        return executorch_exir, xnnpack_partitioner

    return factory


@pytest.fixture
def xnnpack_quantization_config_factory() -> Callable[..., MagicMock]:
    def factory(
        packing: IntxPackingFormat = IntxPackingFormat.UNPACKED_TO_INT8,
        mapping: MappingType = MappingType.SYMMETRIC,
        weight_dtype: torch.dtype = torch.int4,
        embedding: bool = False,
        version: int = 2,
    ) -> MagicMock:
        config = MagicMock(
            spec=IntxWeightOnlyConfig
            if embedding
            else Int8DynamicActivationIntxWeightConfig
        )
        config.intx_packing_format = packing
        config.weight_mapping_type = mapping
        config.mapping_type = mapping
        config.weight_dtype = weight_dtype
        config.version = version
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


@pytest.mark.unit
class TestExecutorchXNNPACKQuantizationValidation:
    @pytest.mark.parametrize(
        "weight_dtype, mapping, packing, version, valid",
        [
            (
                torch.int2,
                MappingType.SYMMETRIC,
                IntxPackingFormat.UNPACKED_TO_INT8,
                2,
                True,
            ),
            (
                torch.int4,
                MappingType.SYMMETRIC,
                IntxPackingFormat.UNPACKED_TO_INT8,
                2,
                True,
            ),
            (
                torch.int8,
                MappingType.SYMMETRIC,
                IntxPackingFormat.UNPACKED_TO_INT8,
                2,
                True,
            ),
            (
                torch.int3,
                MappingType.SYMMETRIC,
                IntxPackingFormat.UNPACKED_TO_INT8,
                2,
                False,
            ),
            (
                torch.int4,
                MappingType.ASYMMETRIC,
                IntxPackingFormat.UNPACKED_TO_INT8,
                2,
                False,
            ),
            (torch.int4, MappingType.SYMMETRIC, "packed", 2, False),
            (
                torch.int4,
                MappingType.SYMMETRIC,
                IntxPackingFormat.UNPACKED_TO_INT8,
                1,
                False,
            ),
        ],
    )
    def test_embedding_lowering_requires_supported_weight_only_format(
        self,
        deployment_target_factory: Callable[..., MagicMock],
        deployment_model_factory: Callable[..., MagicMock],
        xnnpack_quantization_config_factory: Callable[..., MagicMock],
        weight_dtype: torch.dtype,
        mapping: MappingType,
        packing: str,
        version: int,
        valid: bool,
    ) -> None:
        config = xnnpack_quantization_config_factory(
            embedding=True,
            weight_dtype=weight_dtype,
            mapping=mapping,
            packing=packing,
            version=version,
        )
        target = deployment_target_factory(
            config=config, group_size=32, module_type=QuantizationModuleType.EMBEDDING
        )
        model = deployment_model_factory(device="cpu")
        expectation = (
            does_not_raise()
            if valid
            else pytest.raises(
                ValueError,
                match=re.escape(
                    "Target '(root)': XNNPACK embeddings require version=2, symmetric "
                    "INT2, INT4 or INT8 weights and unpacked_to_int8 packing."
                ),
            )
        )
        with expectation:
            ExecutorchXNNPACKBackend(max_batch_size=4).validate_eager_target(
                model=model,
                target=target,
                module_names={"embedding"},
                for_conversion=True,
            )
        if valid:
            model.get_submodule.assert_called_once_with("embedding")
        else:
            model.get_submodule.assert_not_called()

    def test_embedding_lowering_rejects_activation_quantization(
        self,
        deployment_target_factory: Callable[..., MagicMock],
        deployment_model_factory: Callable[..., MagicMock],
        xnnpack_quantization_config_factory: Callable[..., MagicMock],
    ) -> None:
        target = deployment_target_factory(
            config=xnnpack_quantization_config_factory(embedding=False),
            module_type=QuantizationModuleType.EMBEDDING,
        )
        model = deployment_model_factory(device="cpu")
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Target '(root)': XNNPACK embeddings require IntxWeightOnlyConfig."
            ),
        ):
            ExecutorchXNNPACKBackend(max_batch_size=4).validate_eager_target(
                model=model,
                target=target,
                module_names={"embedding"},
            )
        model.get_submodule.assert_not_called()

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
    @pytest.mark.parametrize("gradient_args", [(), (0,), (0, True, True)])
    def test_export_canonicalizes_embedding_gradient_arguments(
        self,
        exported_embedding_graph_factory: Callable[..., MagicMock],
        lowering_dependencies_factory: Callable[..., tuple[MagicMock, MagicMock]],
        gradient_args: tuple[int | bool, ...],
    ) -> None:
        exported_program = exported_embedding_graph_factory(gradient_args=gradient_args)
        embedding, linear = exported_program.graph.nodes
        weights, indices = embedding.args[:2]
        executorch_exir, xnnpack_partitioner = lowering_dependencies_factory(
            model_bytes=b"pte"
        )

        _lower_exported_program(
            exported_program=exported_program,
            executorch_exir=executorch_exir,
            xnnpack_partitioner=xnnpack_partitioner,
        )

        assert embedding.args == (weights, indices)
        assert linear.args == (indices, weights, None)
        exported_program.graph_module.recompile.assert_called_once_with()

    def test_delegates_to_executorch_xnnpack_partitioner(
        self,
        exported_embedding_graph_factory: Callable[..., MagicMock],
        lowering_dependencies_factory: Callable[..., tuple[MagicMock, MagicMock]],
    ) -> None:
        exported_program = exported_embedding_graph_factory(gradient_args=())
        executorch_exir, xnnpack_partitioner = lowering_dependencies_factory(
            model_bytes=b"pte"
        )
        partitioner = xnnpack_partitioner.XnnpackPartitioner.return_value
        edge_program = executorch_exir.to_edge_transform_and_lower.return_value

        result = _lower_exported_program(
            exported_program=exported_program,
            executorch_exir=executorch_exir,
            xnnpack_partitioner=xnnpack_partitioner,
        )

        executorch_exir.to_edge_transform_and_lower.assert_called_once_with(
            exported_program,
            partitioner=[partitioner],
        )
        executorch_exir.ExecutorchBackendConfig.assert_called_once_with(
            do_quant_fusion_and_const_prop=True
        )
        edge_program.to_executorch.assert_called_once_with(
            config=executorch_exir.ExecutorchBackendConfig.return_value
        )
        assert result == b"pte"
