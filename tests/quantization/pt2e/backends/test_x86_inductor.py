"""Tests for versatil.quantization.pt2e.backends.x86_inductor module."""

import os
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
    get_default_x86_inductor_quantization_config,
)

from versatil.quantization.pt2e.backends.x86_inductor import (
    X86InductorBackend,
    _nodes_match_module_path,
    _PerCallX86InductorQuantizer,
)


@pytest.fixture
def linear_activations_factory() -> Callable[..., torch.Tensor]:
    def factory(batch_size: int, input_dimension: int) -> torch.Tensor:
        return torch.ones(batch_size, input_dimension)  # (batch_size, input_dimension)

    return factory


@pytest.mark.unit
@pytest.mark.parametrize("serialized_prefix", [False, True])
@pytest.mark.parametrize(
    "module_paths,expected",
    [
        (["decoder.vlm_backbone"], True),
        (["decoder.vlm_backbone.language_model"], True),
        (["decoder.vlm_backbone_other.language_model"], False),
        (["decoder.vlm_backbone.linear", "decoder.other.linear"], False),
    ],
)
def test_module_filter_requires_every_pattern_node_inside_the_target(
    module_scope_nodes_factory: Callable[..., list[MagicMock]],
    module_paths: list[str],
    expected: bool,
    serialized_prefix: bool,
) -> None:
    nodes = module_scope_nodes_factory(
        module_paths=module_paths, serialized_prefix=serialized_prefix
    )

    assert (
        _nodes_match_module_path(nodes=nodes, module_path="decoder.vlm_backbone")
        == expected
    )


@pytest.mark.integration
class TestX86InductorHelperMethodScopes:
    @pytest.mark.parametrize("is_dynamic", [False, True])
    @pytest.mark.parametrize(
        "module_path,quantized_branches",
        [
            ("", (True, True)),
            ("decoder", (True, True)),
            ("decoder.vlm_backbone", (True, False)),
            ("decoder.vlm_backbone.language_model", (True, False)),
        ],
    )
    def test_quantizes_requested_helper_scope_and_preserves_sibling_boundary(
        self,
        helper_method_policy_factory: Callable[[], nn.Module],
        linear_activations_factory: Callable[..., torch.Tensor],
        module_path: str,
        quantized_branches: tuple[bool, bool],
        is_dynamic: bool,
    ) -> None:
        policy = helper_method_policy_factory()
        inputs = linear_activations_factory(batch_size=2, input_dimension=8)
        exported = torch.export.export(policy, (inputs,), strict=False).module()
        linear_nodes = [
            node
            for node in exported.graph.nodes
            if node.target == torch.ops.aten.linear.default
        ]
        original_scopes = [node.meta["nn_module_stack"].copy() for node in linear_nodes]
        assert all(
            name != "decoder.vlm_backbone"
            for scope in original_scopes
            for name, _ in scope.values()
        )
        quantizer = X86InductorBackend(is_dynamic=is_dynamic).create_quantizer(
            module_path=module_path
        )

        prepared = prepare_pt2e(exported, quantizer)
        with torch.no_grad():
            prepared(inputs)  # (batch_size, output_dimension)
        converted = convert_pt2e(prepared)

        converted_linears = [
            node
            for node in converted.graph.nodes
            if node.target == torch.ops.aten.linear.default
        ]
        assert len(converted_linears) == 2
        for node, expected_quantized, scope in zip(
            converted_linears, quantized_branches, original_scopes, strict=True
        ):
            assert node.meta["nn_module_stack"] == scope
            weight_node = node.args[1]
            assert (
                weight_node.target
                == torch.ops.quantized_decomposed.dequantize_per_channel.default
            ) == expected_quantized
        with torch.no_grad():
            torch.testing.assert_close(
                converted(inputs), policy(inputs), rtol=0.05, atol=0.01
            )  # (batch_size, output_dimension)

    @pytest.mark.parametrize("configuration_source", ["global", "operator"])
    def test_module_exclusion_precedes_operator_and_global_configuration(
        self,
        helper_method_policy_factory: Callable[[], nn.Module],
        linear_activations_factory: Callable[..., torch.Tensor],
        configuration_source: str,
    ) -> None:
        policy = helper_method_policy_factory()
        inputs = linear_activations_factory(batch_size=2, input_dimension=8)
        exported = torch.export.export(policy, (inputs,), strict=False).module()
        quantizer = _PerCallX86InductorQuantizer()
        configuration = get_default_x86_inductor_quantization_config(is_dynamic=True)
        if configuration_source == "global":
            quantizer.set_global(quantization_config=configuration)
        else:
            quantizer.set_module_type_qconfig(nn.Linear, configuration)
        quantizer.set_module_name_qconfig("decoder.vlm_backbone", None)

        prepared = prepare_pt2e(exported, quantizer)
        with torch.no_grad():
            prepared(inputs)  # (batch_size, output_dimension)
        converted = convert_pt2e(prepared)

        linear_nodes = [
            node
            for node in converted.graph.nodes
            if node.target == torch.ops.aten.linear.default
        ]
        assert linear_nodes[0].args[1].op == "get_attr"
        assert (
            linear_nodes[1].args[1].target
            == torch.ops.quantized_decomposed.dequantize_per_channel.default
        )


class TestX86InductorSourceMetadata:
    @pytest.mark.unit
    @pytest.mark.parametrize("existing_source", [True, False])
    @pytest.mark.parametrize("serialized_type", [True, False])
    def test_records_call_identifiers_and_preserves_module_scope(
        self,
        source_metadata_graph_factory: Callable[..., MagicMock],
        source_metadata_quantizer_factory: Callable[[], _PerCallX86InductorQuantizer],
        existing_source: bool,
        serialized_type: bool,
    ) -> None:
        graph = source_metadata_graph_factory(
            existing_source=existing_source,
            serialized_type=serialized_type,
        )
        node = graph.graph.nodes[0]
        module_stack = node.meta["nn_module_stack"].copy()
        quantizer = source_metadata_quantizer_factory()
        quantizer.transform_for_annotation(model=graph)
        assert node.meta["source_fn_stack"] == (
            [("already_recorded_call", nn.Linear)]
            if existing_source
            else [("linear@1", nn.Linear)]
        )
        assert node.meta["nn_module_stack"] == module_stack

    @pytest.mark.integration
    @pytest.mark.parametrize("iterations", [1, 3])
    def test_shared_layer_calls_have_separate_partitions_and_shared_weights(
        self,
        repeated_linear_model_factory: Callable[..., nn.Module],
        linear_activations_factory: Callable[..., torch.Tensor],
        iterations: int,
    ) -> None:
        model = repeated_linear_model_factory(iterations=iterations)
        inputs = linear_activations_factory(batch_size=2, input_dimension=8)
        exported = torch.export.export(model, (inputs,), strict=False).module()
        quantizer = X86InductorBackend(is_dynamic=True).create_quantizer(
            module_path="block.0"
        )
        expected = exported(inputs)  # (batch, output_dim)
        parameters = dict(exported.named_parameters())
        quantizer.transform_for_annotation(model=exported)
        partitions = get_source_partitions(exported.graph, [nn.Linear, nn.GELU])
        assert len(partitions[nn.Linear]) == iterations + 1
        assert len(partitions[nn.GELU]) == iterations
        assert all(
            len(partition.output_nodes) == 1
            for source_partitions in partitions.values()
            for partition in source_partitions
        )
        quantizer.annotate(model=exported)
        linear_nodes = [
            node
            for node in exported.graph.nodes
            if node.target == torch.ops.aten.linear.default
        ]
        for node in linear_nodes[:-1]:
            assert node.args[1] is linear_nodes[0].args[1]
            assert node.meta["quantization_annotation"]._annotated
        assert "quantization_annotation" not in linear_nodes[-1].meta
        assert dict(exported.named_parameters()) == parameters
        torch.testing.assert_close(exported(inputs), expected)  # (batch, output_dim)


@pytest.mark.unit
class TestX86InductorBackendStorage:
    @pytest.mark.parametrize("is_dynamic", [True, False])
    @pytest.mark.parametrize("is_qat", [True, False])
    def test_stores_configuration(
        self,
        x86_inductor_backend_factory,
        is_dynamic,
        is_qat,
    ):
        backend = x86_inductor_backend_factory(
            is_dynamic=is_dynamic,
            is_qat=is_qat,
        )

        assert backend.is_dynamic == is_dynamic
        assert backend.is_qat == is_qat
        assert backend.supported_device_types == ("cpu",)


@pytest.mark.integration
class TestX86InductorBackendCreateQuantizer:
    @pytest.mark.parametrize(
        "module_path",
        ["", "encoder.backbone"],
        ids=["global", "per_module"],
    )
    def test_targets_correct_scope(self, x86_inductor_backend_factory, module_path):
        backend = x86_inductor_backend_factory()

        quantizer = backend.create_quantizer(module_path=module_path)

        if module_path == "":
            assert quantizer.global_config.weight.dtype == torch.int8
        else:
            assert module_path in quantizer.module_name_qconfig
            assert quantizer.module_name_qconfig[module_path].weight.dtype == torch.int8

    @pytest.mark.parametrize("is_dynamic", [True, False])
    def test_dynamic_flag_propagates(
        self,
        x86_inductor_backend_factory,
        is_dynamic,
    ):
        backend = x86_inductor_backend_factory(is_dynamic=is_dynamic)

        quantizer = backend.create_quantizer(module_path="")

        assert quantizer.global_config.input_activation.is_dynamic == is_dynamic

    @pytest.mark.parametrize("is_qat", [True, False])
    def test_qat_flag_propagates(self, x86_inductor_backend_factory, is_qat):
        backend = x86_inductor_backend_factory(is_qat=is_qat)

        quantizer = backend.create_quantizer(module_path="")

        assert quantizer.global_config.is_qat == is_qat

    @pytest.mark.parametrize("reduce_range", [True, False])
    def test_reduce_range_flag_propagates(
        self,
        x86_inductor_backend_factory,
        reduce_range,
    ):
        backend = x86_inductor_backend_factory(reduce_range=reduce_range)

        quantizer = backend.create_quantizer(module_path="")

        expected_quant_max = 127 if reduce_range else 255
        assert quantizer.global_config.input_activation.quant_max == expected_quant_max


@pytest.mark.unit
class TestX86InductorBackendEnvironmentContext:
    @pytest.mark.parametrize("original_setting", [True, False])
    @patch("versatil.quantization.pt2e.backends.x86_inductor.inductor_config")
    def test_sets_and_restores_env_vars(
        self,
        mock_inductor_config: MagicMock,
        x86_inductor_backend_factory: Callable[..., X86InductorBackend],
        original_setting: bool,
    ) -> None:
        backend = x86_inductor_backend_factory()
        original_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        original_freezing = os.environ.get("TORCHINDUCTOR_FREEZING")
        mock_inductor_config.cpp_wrapper = original_setting
        mock_inductor_config.freezing = original_setting

        with backend.environment_context():
            assert os.environ.get("TORCHINDUCTOR_FREEZING") == "1"
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            assert mock_inductor_config.cpp_wrapper is True
            assert mock_inductor_config.freezing is True

        assert os.environ.get("CUDA_VISIBLE_DEVICES") == original_cuda
        assert os.environ.get("TORCHINDUCTOR_FREEZING") == original_freezing
        assert mock_inductor_config.cpp_wrapper is original_setting
        assert mock_inductor_config.freezing is original_setting

    @pytest.mark.parametrize("original_setting", [True, False])
    @patch("versatil.quantization.pt2e.backends.x86_inductor.inductor_config")
    def test_restores_environment_on_exception(
        self,
        mock_inductor_config: MagicMock,
        x86_inductor_backend_factory: Callable[..., X86InductorBackend],
        original_setting: bool,
    ) -> None:
        backend = x86_inductor_backend_factory()
        original_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        original_freezing = os.environ.get("TORCHINDUCTOR_FREEZING")
        mock_inductor_config.cpp_wrapper = original_setting
        mock_inductor_config.freezing = original_setting

        with (
            pytest.raises(RuntimeError, match="test error"),
            backend.environment_context(),
        ):
            raise RuntimeError("test error")

        assert os.environ.get("CUDA_VISIBLE_DEVICES") == original_cuda
        assert os.environ.get("TORCHINDUCTOR_FREEZING") == original_freezing
        assert mock_inductor_config.cpp_wrapper is original_setting
        assert mock_inductor_config.freezing is original_setting

    @patch("versatil.quantization.pt2e.backends.x86_inductor.inductor_config")
    def test_removes_env_var_when_originally_unset(
        self,
        mock_inductor_config,
        x86_inductor_backend_factory,
    ):
        backend = x86_inductor_backend_factory()
        os.environ.pop("TORCHINDUCTOR_FREEZING", None)

        with backend.environment_context():
            assert os.environ["TORCHINDUCTOR_FREEZING"] == "1"

        assert "TORCHINDUCTOR_FREEZING" not in os.environ


@pytest.mark.unit
class TestX86InductorBackendActivateEnvironment:
    @patch("versatil.quantization.pt2e.backends.x86_inductor.inductor_config")
    @patch.dict(os.environ, {}, clear=True)
    def test_sets_env_vars_permanently(
        self,
        mock_inductor_config: MagicMock,
        x86_inductor_backend_factory: Callable[..., X86InductorBackend],
    ) -> None:
        backend = x86_inductor_backend_factory()
        mock_inductor_config.cpp_wrapper = False
        mock_inductor_config.freezing = False

        backend.activate_environment()

        assert os.environ.get("TORCHINDUCTOR_FREEZING") == "1"
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        assert mock_inductor_config.cpp_wrapper is True
        assert mock_inductor_config.freezing is True
