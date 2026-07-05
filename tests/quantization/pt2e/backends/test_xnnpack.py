"""Tests for versatil.quantization.pt2e.backends.xnnpack module."""

import os
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from versatil.quantization.constants import PT2EBackendName
from versatil.quantization.pt2e.backends.xnnpack import (
    _XNNPACK_QUANTIZER_MODULE,
    XNNPACKPT2EBackend,
)

XNNPACK_OPERATOR_TARGETS = {
    torch.ops.aten.linear.default,
    torch.ops.aten.conv2d.default,
    torch.ops.aten.convolution.default,
}


@pytest.fixture
def xnnpack_backend_factory() -> Callable[..., XNNPACKPT2EBackend]:
    def factory(
        is_dynamic: bool = False,
        is_qat: bool = False,
        is_per_channel: bool = True,
    ) -> XNNPACKPT2EBackend:
        return XNNPACKPT2EBackend(
            is_dynamic=is_dynamic,
            is_qat=is_qat,
            is_per_channel=is_per_channel,
        )

    return factory


@pytest.mark.unit
class TestXNNPACKPT2EBackendStorage:
    @pytest.mark.parametrize("is_dynamic", [True, False])
    @pytest.mark.parametrize("is_qat", [True, False])
    @pytest.mark.parametrize("is_per_channel", [True, False])
    def test_stores_configuration(
        self,
        xnnpack_backend_factory: Callable[..., XNNPACKPT2EBackend],
        is_dynamic: bool,
        is_qat: bool,
        is_per_channel: bool,
    ) -> None:
        backend = xnnpack_backend_factory(
            is_dynamic=is_dynamic,
            is_qat=is_qat,
            is_per_channel=is_per_channel,
        )

        assert backend.name == PT2EBackendName.XNNPACK.value
        assert backend.is_dynamic == is_dynamic
        assert backend.is_qat == is_qat
        assert backend.is_per_channel == is_per_channel
        assert backend.supported_device_types == ("cpu",)


@pytest.mark.unit
def test_create_quantizer_imports_xnnpack_on_demand(
    xnnpack_backend_factory: Callable[..., XNNPACKPT2EBackend],
) -> None:
    backend = xnnpack_backend_factory(
        is_dynamic=True,
        is_qat=True,
        is_per_channel=False,
    )
    quantizer = MagicMock()
    config = MagicMock()
    quantizer_module = MagicMock()
    quantizer_module.XNNPACKQuantizer.return_value = quantizer
    quantizer_module.get_symmetric_quantization_config.return_value = config

    with patch(
        "versatil.quantization.pt2e.backends.xnnpack.importlib.import_module",
        return_value=quantizer_module,
    ) as import_module:
        result = backend.create_quantizer(module_path="encoder")

    assert result is quantizer
    import_module.assert_called_once_with(_XNNPACK_QUANTIZER_MODULE)
    quantizer_module.XNNPACKQuantizer.assert_called_once_with()
    quantizer_module.get_symmetric_quantization_config.assert_called_once_with(
        is_per_channel=False,
        is_qat=True,
        is_dynamic=True,
    )
    operator_targets = {
        call_args.args[0] for call_args in quantizer.set_operator_type.call_args_list
    }
    assert operator_targets == XNNPACK_OPERATOR_TARGETS
    for call_args in quantizer.set_operator_type.call_args_list:
        assert call_args.args[1] is config
    quantizer.set_filter_function.assert_called_once()
    node_filter = quantizer.set_filter_function.call_args.args[0]
    matching_node = MagicMock()
    matching_node.meta = {
        "val": torch.zeros(2, 4),
        "nn_module_stack": {
            "encoder.backbone": ("L['self'].encoder.backbone", nn.Linear),
        },
    }
    assert node_filter(matching_node)


@pytest.mark.integration
@pytest.mark.requires_executorch
class TestXNNPACKPT2EBackendCreateQuantizer:
    @pytest.mark.parametrize(
        "module_path",
        ["", "encoder", "encoder.backbone"],
        ids=["global", "parent_module", "exact_module"],
    )
    def test_targets_correct_scope(
        self,
        xnnpack_backend_factory: Callable[..., XNNPACKPT2EBackend],
        module_path: str,
    ) -> None:
        backend = xnnpack_backend_factory(is_per_channel=True)

        quantizer = backend.create_quantizer(module_path=module_path)

        config = quantizer.operator_type_config[torch.ops.aten.linear.default]
        assert quantizer.global_config is None
        assert quantizer.module_name_config == {}
        assert set(quantizer.operator_type_config) == XNNPACK_OPERATOR_TARGETS
        assert config.input_activation.dtype == torch.int8
        assert config.weight.dtype == torch.int8
        assert config.weight.qscheme == torch.per_channel_symmetric
        matching_node = MagicMock()
        matching_node.meta = {
            "val": torch.zeros(2, 4),
            "nn_module_stack": {
                "encoder.backbone": ("L['self'].encoder.backbone", nn.Linear),
            },
        }
        if module_path == "":
            assert quantizer.filter_fn(matching_node)
        else:
            assert quantizer.filter_fn(matching_node)
            unmatched_node = MagicMock()
            unmatched_node.meta = {
                "val": torch.zeros(2, 4),
                "nn_module_stack": {
                    "decoder": ("L['self'].decoder", nn.Linear),
                },
            }
            assert not quantizer.filter_fn(unmatched_node)

    @pytest.mark.parametrize("is_dynamic", [True, False])
    def test_dynamic_flag_propagates(
        self,
        xnnpack_backend_factory: Callable[..., XNNPACKPT2EBackend],
        is_dynamic: bool,
    ) -> None:
        backend = xnnpack_backend_factory(is_dynamic=is_dynamic)

        quantizer = backend.create_quantizer(module_path="")
        config = quantizer.operator_type_config[torch.ops.aten.linear.default]

        assert config.input_activation.is_dynamic == is_dynamic
        if is_dynamic:
            assert config.output_activation is None
        else:
            assert config.output_activation.dtype == torch.int8

    @pytest.mark.parametrize(
        "is_per_channel, expected_qscheme",
        [
            (True, torch.per_channel_symmetric),
            (False, torch.per_tensor_symmetric),
        ],
    )
    def test_per_channel_flag_propagates(
        self,
        xnnpack_backend_factory: Callable[..., XNNPACKPT2EBackend],
        is_per_channel: bool,
        expected_qscheme: torch.qscheme,
    ) -> None:
        backend = xnnpack_backend_factory(is_per_channel=is_per_channel)

        quantizer = backend.create_quantizer(module_path="")
        config = quantizer.operator_type_config[torch.ops.aten.linear.default]

        assert config.weight.qscheme == expected_qscheme

    @pytest.mark.parametrize("is_qat", [True, False])
    def test_qat_flag_propagates(
        self,
        xnnpack_backend_factory: Callable[..., XNNPACKPT2EBackend],
        is_qat: bool,
    ) -> None:
        backend = xnnpack_backend_factory(is_qat=is_qat)

        quantizer = backend.create_quantizer(module_path="")
        config = quantizer.operator_type_config[torch.ops.aten.linear.default]

        assert config.is_qat == is_qat

    def test_filter_rejects_bool_tensor_nodes(
        self,
        xnnpack_backend_factory: Callable[..., XNNPACKPT2EBackend],
    ) -> None:
        backend = xnnpack_backend_factory()
        quantizer = backend.create_quantizer(module_path="")
        float_node = MagicMock()
        float_node.meta = {"val": torch.zeros(2, 4)}
        bool_node = MagicMock()
        bool_node.meta = {"val": torch.zeros(2, 4, dtype=torch.bool)}

        assert quantizer.filter_fn(float_node)
        assert not quantizer.filter_fn(bool_node)


@pytest.mark.unit
class TestXNNPACKPT2EBackendRuntimeHooks:
    def test_environment_context_does_not_change_environment(
        self,
        xnnpack_backend_factory: Callable[..., XNNPACKPT2EBackend],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("TORCHINDUCTOR_FREEZING", "before")
        backend = xnnpack_backend_factory()

        with backend.environment_context():
            assert os.environ["TORCHINDUCTOR_FREEZING"] == "before"

        assert os.environ["TORCHINDUCTOR_FREEZING"] == "before"
