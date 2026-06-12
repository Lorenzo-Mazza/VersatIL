"""Tests for versatil.configs.regularizer module."""

import pytest
from omegaconf import MISSING

from versatil.configs.regularizer import (
    BaseRegularizerConfig,
    FiniteDifferenceLipschitzRegularizerConfig,
    ImageAugmentationConsistencyRegularizerConfig,
    JacobianFrobeniusLipschitzRegularizerConfig,
    SpectralJacobianLipschitzRegularizerConfig,
)
from versatil.metrics.constants import (
    FiniteDifferencePerturbationMode,
    ImageAugmentationConsistencyLossMode,
)
from versatil.metrics.regularization_context import PolicyGraphInputDomain


@pytest.mark.unit
def test_base_regularizer_config_requires_explicit_target():
    config = BaseRegularizerConfig()
    assert config._target_ == MISSING


@pytest.mark.unit
@pytest.mark.parametrize(
    "input_domain",
    [
        PolicyGraphInputDomain.ENCODED_FEATURES.value,
        PolicyGraphInputDomain.OBSERVATION.value,
    ],
)
@pytest.mark.parametrize(
    "perturbation_mode",
    [
        FiniteDifferencePerturbationMode.GAUSSIAN_DENSE.value,
        FiniteDifferencePerturbationMode.GAUSSIAN_CHANNEL_BROADCAST.value,
    ],
)
def test_finite_difference_config_stores_configuration(
    input_domain: str,
    perturbation_mode: str,
):
    config = FiniteDifferenceLipschitzRegularizerConfig(
        input_keys=["left_rgb"],
        input_domain=input_domain,
        output_keys=["action"],
        weight=0.5,
        target=0.25,
        noise_scale=0.02,
        detach_inputs=False,
        max_batch_size=4,
        apply_during_eval=True,
        eps=1e-8,
        disable_decoder_stochastic=False,
        scale_by_dimension_ratio=True,
        perturbation_mode=perturbation_mode,
    )

    assert (
        config._target_
        == "versatil.metrics.regularizers.FiniteDifferenceLipschitzRegularizer"
    )
    assert config.input_keys == ["left_rgb"]
    assert config.input_domain == input_domain
    assert config.output_keys == ["action"]
    assert config.weight == 0.5
    assert config.target == 0.25
    assert config.noise_scale == 0.02
    assert config.detach_inputs is False
    assert config.max_batch_size == 4
    assert config.apply_during_eval is True
    assert config.eps == 1e-8
    assert config.disable_decoder_stochastic is False
    assert config.scale_by_dimension_ratio is True
    assert config.perturbation_mode == perturbation_mode


@pytest.mark.unit
@pytest.mark.parametrize(
    "loss_mode",
    [
        ImageAugmentationConsistencyLossMode.FLAT_OUTPUT_MSE.value,
        ImageAugmentationConsistencyLossMode.POSITION_TRAJECTORY_L2.value,
    ],
)
@pytest.mark.parametrize("detach_targets", [True, False])
def test_image_augmentation_config_stores_configuration(
    loss_mode: str,
    detach_targets: bool,
):
    config = ImageAugmentationConsistencyRegularizerConfig(
        input_keys=["agentview_rgb"],
        output_keys=["ee_pos_action"],
        weight=0.5,
        color_augmentation=None,
        spatial_augmentation=None,
        loss_mode=loss_mode,
        detach_inputs=False,
        detach_targets=detach_targets,
        max_batch_size=4,
        apply_during_eval=True,
        input_min=0.0,
        input_max=2.0,
        max_pixel_value=127.0,
    )

    assert (
        config._target_
        == "versatil.metrics.regularizers.ImageAugmentationConsistencyRegularizer"
    )
    assert config.input_keys == ["agentview_rgb"]
    assert config.output_keys == ["ee_pos_action"]
    assert config.weight == 0.5
    assert config.color_augmentation is None
    assert config.spatial_augmentation is None
    assert config.loss_mode == loss_mode
    assert config.detach_inputs is False
    assert config.detach_targets is detach_targets
    assert config.max_batch_size == 4
    assert config.apply_during_eval is True
    assert config.input_min == 0.0
    assert config.input_max == 2.0
    assert config.max_pixel_value == 127.0


@pytest.mark.unit
@pytest.mark.parametrize("number_of_probes", [1, 4])
@pytest.mark.parametrize("scale_by_dimension_ratio", [True, False])
def test_jacobian_frobenius_config_stores_configuration(
    number_of_probes: int,
    scale_by_dimension_ratio: bool,
):
    config = JacobianFrobeniusLipschitzRegularizerConfig(
        input_keys=["left_rgb"],
        input_domain=PolicyGraphInputDomain.DECODER_FEATURES.value,
        output_keys=["action"],
        weight=0.5,
        number_of_probes=number_of_probes,
        detach_inputs=False,
        max_batch_size=4,
        apply_during_eval=True,
        disable_decoder_stochastic=False,
        scale_by_dimension_ratio=scale_by_dimension_ratio,
    )

    assert (
        config._target_
        == "versatil.metrics.regularizers.JacobianFrobeniusLipschitzRegularizer"
    )
    assert config.input_keys == ["left_rgb"]
    assert config.input_domain == PolicyGraphInputDomain.DECODER_FEATURES.value
    assert config.output_keys == ["action"]
    assert config.weight == 0.5
    assert config.number_of_probes == number_of_probes
    assert config.detach_inputs is False
    assert config.max_batch_size == 4
    assert config.apply_during_eval is True
    assert config.disable_decoder_stochastic is False
    assert config.scale_by_dimension_ratio is scale_by_dimension_ratio


@pytest.mark.unit
@pytest.mark.parametrize("number_of_power_iterations", [1, 3])
@pytest.mark.parametrize("scale_by_dimension_ratio", [True, False])
def test_spectral_jacobian_config_stores_configuration(
    number_of_power_iterations: int,
    scale_by_dimension_ratio: bool,
):
    config = SpectralJacobianLipschitzRegularizerConfig(
        input_keys=["left_rgb"],
        input_domain=PolicyGraphInputDomain.OBSERVATION.value,
        output_keys=["action"],
        weight=0.5,
        target=0.25,
        number_of_power_iterations=number_of_power_iterations,
        detach_inputs=False,
        max_batch_size=4,
        apply_during_eval=True,
        eps=1e-8,
        disable_decoder_stochastic=False,
        scale_by_dimension_ratio=scale_by_dimension_ratio,
    )

    assert (
        config._target_
        == "versatil.metrics.regularizers.SpectralJacobianLipschitzRegularizer"
    )
    assert config.input_keys == ["left_rgb"]
    assert config.input_domain == PolicyGraphInputDomain.OBSERVATION.value
    assert config.output_keys == ["action"]
    assert config.weight == 0.5
    assert config.target == 0.25
    assert config.number_of_power_iterations == number_of_power_iterations
    assert config.detach_inputs is False
    assert config.max_batch_size == 4
    assert config.apply_during_eval is True
    assert config.eps == 1e-8
    assert config.disable_decoder_stochastic is False
    assert config.scale_by_dimension_ratio is scale_by_dimension_ratio
