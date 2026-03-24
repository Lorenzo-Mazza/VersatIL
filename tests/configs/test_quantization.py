"""Tests for versatil.configs.quantization module."""

import hydra
import pytest
from omegaconf import OmegaConf

from versatil.configs.quantization import (
    Int4WeightOnlyQuantizeConfig,
    PT2EStrategyConfig,
    QuantizeApiStrategyConfig,
    X86InductorBackendConfig,
)
from versatil.quantization.backends.x86_inductor import X86InductorBackend
from versatil.quantization.strategies import PT2EStrategy, QuantizeApiStrategy


@pytest.mark.unit
class TestX86InductorBackendConfig:
    @pytest.mark.parametrize("is_dynamic", [True, False])
    @pytest.mark.parametrize("is_qat", [True, False])
    def test_hydra_instantiates_backend(self, is_dynamic, is_qat):
        config = OmegaConf.structured(
            X86InductorBackendConfig(
                is_dynamic=is_dynamic,
                is_qat=is_qat,
            )
        )

        backend = hydra.utils.instantiate(config)

        assert isinstance(backend, X86InductorBackend)
        assert backend.is_dynamic == is_dynamic


@pytest.mark.unit
class TestPT2EStrategyConfig:
    def test_hydra_instantiates_with_default_backend(self):
        config = OmegaConf.structured(PT2EStrategyConfig())

        result = hydra.utils.instantiate(config)

        assert isinstance(result, PT2EStrategy)
        assert isinstance(result.pt2e_backend, X86InductorBackend)

    @pytest.mark.parametrize("is_dynamic", [True, False])
    def test_propagates_backend_config(self, is_dynamic):
        config = OmegaConf.structured(
            PT2EStrategyConfig(
                pt2e_backend=X86InductorBackendConfig(
                    is_dynamic=is_dynamic,
                ),
            )
        )

        result = hydra.utils.instantiate(config)

        assert result.pt2e_backend.is_dynamic == is_dynamic


@pytest.mark.unit
class TestQuantizeApiStrategyConfig:
    def test_hydra_instantiates_with_int4_config(self):
        config = OmegaConf.structured(
            QuantizeApiStrategyConfig(
                quantize_config=Int4WeightOnlyQuantizeConfig(
                    group_size=64,
                ),
            )
        )

        result = hydra.utils.instantiate(config)

        assert isinstance(result, QuantizeApiStrategy)
        assert result.quantize_config.group_size == 64
