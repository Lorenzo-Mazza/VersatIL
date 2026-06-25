"""Tests for versatil.quantization.pt2e.backends.x86_inductor module."""

import os
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn


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
    @patch("versatil.quantization.pt2e.backends.x86_inductor.inductor_config")
    def test_sets_and_restores_env_vars(
        self,
        mock_inductor_config,
        x86_inductor_backend_factory,
    ):
        backend = x86_inductor_backend_factory()
        original_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        original_freezing = os.environ.get("TORCHINDUCTOR_FREEZING")

        with backend.environment_context():
            assert os.environ.get("TORCHINDUCTOR_FREEZING") == "1"
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            assert mock_inductor_config.cpp_wrapper is True

        assert os.environ.get("CUDA_VISIBLE_DEVICES") == original_cuda
        assert os.environ.get("TORCHINDUCTOR_FREEZING") == original_freezing

    @patch("versatil.quantization.pt2e.backends.x86_inductor.inductor_config")
    def test_restores_environment_on_exception(
        self,
        mock_inductor_config,
        x86_inductor_backend_factory,
    ):
        backend = x86_inductor_backend_factory()
        original_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        original_freezing = os.environ.get("TORCHINDUCTOR_FREEZING")
        mock_inductor_config.cpp_wrapper = False

        with (
            pytest.raises(RuntimeError, match="test error"),
            backend.environment_context(),
        ):
            raise RuntimeError("test error")

        assert os.environ.get("CUDA_VISIBLE_DEVICES") == original_cuda
        assert os.environ.get("TORCHINDUCTOR_FREEZING") == original_freezing
        assert mock_inductor_config.cpp_wrapper is False

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
    def test_sets_env_vars_permanently(
        self,
        mock_inductor_config,
        x86_inductor_backend_factory,
    ):
        backend = x86_inductor_backend_factory()
        os.environ.pop("TORCHINDUCTOR_FREEZING", None)
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        backend.activate_environment()

        assert os.environ.get("TORCHINDUCTOR_FREEZING") == "1"
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        assert mock_inductor_config.cpp_wrapper is True

        os.environ.pop("TORCHINDUCTOR_FREEZING", None)
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)


@pytest.mark.unit
class TestX86InductorBackendLower:
    @patch(
        "versatil.quantization.pt2e.backends.x86_inductor.lower_pt2e_quantized_to_x86"
    )
    def test_delegates_to_torchao_lowering(
        self,
        mock_lower,
        x86_inductor_backend_factory,
    ):
        mock_converted = MagicMock(spec=nn.Module)
        mock_lowered = MagicMock(spec=nn.Module)
        mock_lower.return_value = mock_lowered
        example_inputs = (MagicMock(),)

        result = x86_inductor_backend_factory().lower(
            converted_model=mock_converted,
            example_inputs=example_inputs,
        )

        mock_lower.assert_called_once_with(mock_converted, example_inputs)
        assert result is mock_lowered
