"""Tests for versatil.quantization.strategies module."""

import pytest
from torchao.quantization import Int8DynamicActivationInt8WeightConfig

from versatil.quantization.backends.x86_inductor import X86InductorBackend
from versatil.quantization.strategies import PT2EStrategy, QuantizeApiStrategy


@pytest.mark.unit
class TestPT2EStrategy:
    @pytest.mark.parametrize("is_dynamic", [True, False])
    def test_needs_calibration_reflects_dynamic_flag(self, is_dynamic):
        backend = X86InductorBackend(is_dynamic=is_dynamic)
        strategy = PT2EStrategy(pt2e_backend=backend)

        assert strategy.needs_calibration == (not is_dynamic)

    def test_backend_accessible_via_property(self):
        backend = X86InductorBackend(is_dynamic=True)
        strategy = PT2EStrategy(pt2e_backend=backend)

        assert strategy.pt2e_backend.is_dynamic is True


@pytest.mark.unit
class TestQuantizeApiStrategy:
    def test_config_accessible_via_attribute(self):
        config = Int8DynamicActivationInt8WeightConfig()
        strategy = QuantizeApiStrategy(quantize_config=config)

        assert isinstance(
            strategy.quantize_config, Int8DynamicActivationInt8WeightConfig
        )
