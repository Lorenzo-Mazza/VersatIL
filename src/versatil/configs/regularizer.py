"""Configuration objects for policy-level regularizers."""

from dataclasses import dataclass

from omegaconf import MISSING

from versatil.configs.data.augmentations import AugmentationPipelineConfig
from versatil.metrics.constants import (
    FiniteDifferencePerturbationMode,
    ImageAugmentationConsistencyLossMode,
)
from versatil.metrics.regularization_context import PolicyGraphInputDomain


@dataclass
class BaseRegularizerConfig:
    """Base configuration for policy regularizers."""

    _target_: str = MISSING


@dataclass
class FiniteDifferenceLipschitzRegularizerConfig(BaseRegularizerConfig):
    """Configuration for finite-difference local sensitivity regularization."""

    _target_: str = "versatil.metrics.regularizers.FiniteDifferenceLipschitzRegularizer"
    input_keys: list[str] = MISSING
    input_domain: str = PolicyGraphInputDomain.ENCODED_FEATURES.value
    output_keys: list[str] | None = None
    weight: float = 1e-3
    target: float = 1.0
    noise_scale: float = 1e-2
    detach_inputs: bool = True
    max_batch_size: int | None = None
    apply_during_eval: bool = False
    eps: float = 1e-12
    disable_decoder_stochastic: bool = True
    scale_by_dimension_ratio: bool = False
    perturbation_mode: str = FiniteDifferencePerturbationMode.GAUSSIAN_DENSE.value


@dataclass
class ImageAugmentationConsistencyRegularizerConfig(BaseRegularizerConfig):
    """Configuration for image augmentation consistency regularization."""

    _target_: str = (
        "versatil.metrics.regularizers.ImageAugmentationConsistencyRegularizer"
    )
    input_keys: list[str] = MISSING
    output_keys: list[str] | None = None
    weight: float = 1e-3
    color_augmentation: AugmentationPipelineConfig | None = None
    spatial_augmentation: AugmentationPipelineConfig | None = None
    loss_mode: str = ImageAugmentationConsistencyLossMode.POSITION_TRAJECTORY_L2.value
    detach_inputs: bool = True
    detach_targets: bool = True
    max_batch_size: int | None = None
    apply_during_eval: bool = False
    input_min: float = -1.0
    input_max: float = 1.0
    max_pixel_value: float = 255.0


@dataclass
class JacobianFrobeniusLipschitzRegularizerConfig(BaseRegularizerConfig):
    """Configuration for Hutchinson Jacobian Frobenius regularization."""

    _target_: str = (
        "versatil.metrics.regularizers.JacobianFrobeniusLipschitzRegularizer"
    )
    input_keys: list[str] = MISSING
    input_domain: str = PolicyGraphInputDomain.ENCODED_FEATURES.value
    output_keys: list[str] | None = None
    weight: float = 1e-4
    number_of_probes: int = 1
    detach_inputs: bool = True
    max_batch_size: int | None = None
    apply_during_eval: bool = False
    disable_decoder_stochastic: bool = True
    scale_by_dimension_ratio: bool = False


@dataclass
class SpectralJacobianLipschitzRegularizerConfig(BaseRegularizerConfig):
    """Configuration for local spectral-Jacobian Lipschitz regularization."""

    _target_: str = "versatil.metrics.regularizers.SpectralJacobianLipschitzRegularizer"
    input_keys: list[str] = MISSING
    input_domain: str = PolicyGraphInputDomain.ENCODED_FEATURES.value
    output_keys: list[str] | None = None
    weight: float = 1e-4
    target: float = 1.0
    number_of_power_iterations: int = 1
    detach_inputs: bool = True
    max_batch_size: int | None = None
    apply_during_eval: bool = False
    eps: float = 1e-12
    disable_decoder_stochastic: bool = True
    scale_by_dimension_ratio: bool = False
