"""Tests for versatil.quantization.backends.x86_inductor module."""

import os
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
import torch.nn as nn

from versatil.quantization.backends.x86_inductor import (
    X86InductorBackend,
)


@pytest.fixture
def backend_factory() -> Callable[..., X86InductorBackend]:
    """Factory for X86InductorBackend with configurable flags."""

    def factory(
        is_dynamic: bool = False,
        is_qat: bool = False,
        reduce_range: bool = False,
    ) -> X86InductorBackend:
        return X86InductorBackend(
            is_dynamic=is_dynamic,
            is_qat=is_qat,
            reduce_range=reduce_range,
        )

    return factory


@pytest.mark.unit
class TestX86InductorBackendStorage:
    @pytest.mark.parametrize("is_dynamic", [True, False])
    @pytest.mark.parametrize("is_qat", [True, False])
    @pytest.mark.parametrize("reduce_range", [True, False])
    def test_stores_configuration(
        self, backend_factory, is_dynamic, is_qat, reduce_range
    ):
        backend = backend_factory(
            is_dynamic=is_dynamic,
            is_qat=is_qat,
            reduce_range=reduce_range,
        )

        assert backend.is_dynamic == is_dynamic


@pytest.mark.integration
class TestX86InductorBackendCreateQuantizer:
    @pytest.mark.parametrize(
        "module_path, config_attr",
        [
            ("", "global_config"),
            ("encoder.backbone", "module_name_qconfig"),
        ],
        ids=["global", "per_module"],
    )
    def test_targets_correct_scope(self, backend_factory, module_path, config_attr):
        backend = backend_factory()

        quantizer = backend.create_quantizer(module_path=module_path)

        config_value = getattr(quantizer, config_attr)
        assert config_value is not None
        if isinstance(config_value, dict):
            assert len(config_value) > 0

    @pytest.mark.parametrize("is_dynamic", [True, False])
    def test_dynamic_flag_propagates(self, backend_factory, is_dynamic):
        backend = backend_factory(is_dynamic=is_dynamic)

        quantizer = backend.create_quantizer(module_path="")

        assert quantizer.global_config.input_activation.is_dynamic == is_dynamic

    @pytest.mark.parametrize("is_qat", [True, False])
    def test_qat_flag_propagates(self, backend_factory, is_qat):
        backend = backend_factory(is_qat=is_qat)

        quantizer = backend.create_quantizer(module_path="")

        assert quantizer.global_config.is_qat == is_qat


@pytest.mark.unit
class TestX86InductorBackendEnvironmentContext:
    @patch("versatil.quantization.backends.x86_inductor.inductor_config")
    def test_sets_and_restores_env_vars(self, mock_inductor_config):
        backend = X86InductorBackend()
        original_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        original_freezing = os.environ.get("TORCHINDUCTOR_FREEZING")

        with backend.environment_context():
            assert os.environ.get("TORCHINDUCTOR_FREEZING") == "1"
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            assert mock_inductor_config.cpp_wrapper is True

        assert os.environ.get("CUDA_VISIBLE_DEVICES") == original_cuda
        assert os.environ.get("TORCHINDUCTOR_FREEZING") == original_freezing

    @patch("versatil.quantization.backends.x86_inductor.inductor_config")
    def test_restores_environment_on_exception(self, mock_inductor_config):
        backend = X86InductorBackend()
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

    @patch("versatil.quantization.backends.x86_inductor.inductor_config")
    def test_removes_env_var_when_originally_unset(self, mock_inductor_config):
        backend = X86InductorBackend()
        os.environ.pop("TORCHINDUCTOR_FREEZING", None)

        with backend.environment_context():
            assert os.environ["TORCHINDUCTOR_FREEZING"] == "1"

        assert "TORCHINDUCTOR_FREEZING" not in os.environ


@pytest.mark.unit
class TestX86InductorBackendLower:
    @patch("versatil.quantization.backends.x86_inductor.lower_pt2e_quantized_to_x86")
    def test_delegates_to_torchao_lowering(self, mock_lower):
        mock_converted = MagicMock(spec=nn.Module)
        mock_lowered = MagicMock(spec=nn.Module)
        mock_lower.return_value = mock_lowered
        example_inputs = (MagicMock(),)

        result = X86InductorBackend().lower(
            converted_model=mock_converted,
            example_inputs=example_inputs,
        )

        mock_lower.assert_called_once_with(mock_converted, example_inputs)
        assert result is mock_lowered
