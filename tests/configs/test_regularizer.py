"""Tests for versatil.configs.regularizer module."""

import pytest
from omegaconf import MISSING

from versatil.configs.regularizer import (
    BaseRegularizerConfig,
    FiniteDifferenceLipschitzRegularizerConfig,
    JacobianFrobeniusLipschitzRegularizerConfig,
    SpectralJacobianLipschitzRegularizerConfig,
)
from versatil.metrics.regularization_context import PolicyGraphInputDomain


@pytest.mark.unit
def test_base_regularizer_config_target_defaults_to_missing():
    config = BaseRegularizerConfig()
    assert config._target_ == MISSING


@pytest.mark.unit
class TestFiniteDifferenceLipschitzRegularizerConfig:
    def test_target_points_to_regularizer(self):
        config = FiniteDifferenceLipschitzRegularizerConfig(input_keys=["feature"])
        assert (
            config._target_
            == "versatil.metrics.regularizers.FiniteDifferenceLipschitzRegularizer"
        )

    def test_stores_defaults(self):
        config = FiniteDifferenceLipschitzRegularizerConfig(input_keys=["feature"])
        assert config.input_domain == PolicyGraphInputDomain.ENCODED_FEATURES.value
        assert config.output_keys is None
        assert config.weight == 1e-3
        assert config.target == 1.0
        assert config.noise_scale == 1e-2
        assert config.symmetric is True
        assert config.detach_inputs is True
        assert config.max_batch_size is None
        assert config.apply_during_eval is False
        assert config.disable_decoder_stochastic is True
        assert config.scale_by_dimension_ratio is False


@pytest.mark.unit
class TestJacobianFrobeniusLipschitzRegularizerConfig:
    def test_target_points_to_regularizer(self):
        config = JacobianFrobeniusLipschitzRegularizerConfig(input_keys=["feature"])
        assert (
            config._target_
            == "versatil.metrics.regularizers.JacobianFrobeniusLipschitzRegularizer"
        )

    def test_stores_defaults(self):
        config = JacobianFrobeniusLipschitzRegularizerConfig(input_keys=["feature"])
        assert config.input_domain == PolicyGraphInputDomain.ENCODED_FEATURES.value
        assert config.output_keys is None
        assert config.weight == 1e-4
        assert config.number_of_probes == 1
        assert config.detach_inputs is True
        assert config.max_batch_size is None
        assert config.apply_during_eval is False
        assert config.disable_decoder_stochastic is True
        assert config.scale_by_dimension_ratio is False


@pytest.mark.unit
class TestSpectralJacobianLipschitzRegularizerConfig:
    def test_target_points_to_regularizer(self):
        config = SpectralJacobianLipschitzRegularizerConfig(input_keys=["feature"])
        assert (
            config._target_
            == "versatil.metrics.regularizers.SpectralJacobianLipschitzRegularizer"
        )

    def test_stores_defaults(self):
        config = SpectralJacobianLipschitzRegularizerConfig(input_keys=["feature"])
        assert config.input_domain == PolicyGraphInputDomain.ENCODED_FEATURES.value
        assert config.output_keys is None
        assert config.weight == 1e-4
        assert config.target == 1.0
        assert config.number_of_power_iterations == 1
        assert config.detach_inputs is True
        assert config.max_batch_size is None
        assert config.apply_during_eval is False
        assert config.disable_decoder_stochastic is True
        assert config.scale_by_dimension_ratio is False
