"""Loss configuration for policy training."""

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

from versatil.metrics.kernels import KernelType


@dataclass
class BaseLossConfig:
    """Base configuration for loss modules."""

    _target_: str = MISSING


@dataclass
class RegressionLossConfig(BaseLossConfig):
    """Configuration for regression loss (position, orientation)."""

    _target_: str = "versatil.metrics.losses.regression.RegressionLoss"
    action_keys: list[str] = MISSING
    mse_weight: float = 1.0
    l1_weight: float = 0.0
    huber_weight: float = 0.0
    huber_delta: float = 1.0
    per_key_weights: dict[str, float] | None = None


@dataclass
class GripperLossConfig(BaseLossConfig):
    """Configuration for gripper loss."""

    _target_: str = "versatil.metrics.losses.gripper.GripperLoss"
    key: str = MISSING
    actions_metadata: Any = "${task.action_space.actions_metadata}"
    bce_weight: float = 1.0
    mse_weight: float = 1.0
    pos_weight: float | None = None


@dataclass
class KLDivergenceLossConfig(BaseLossConfig):
    """Configuration for KL divergence loss."""

    _target_: str = "versatil.metrics.losses.divergence.KLDivergenceLoss"
    weight: float = 0.0001
    prior_regularization_weight: float = 0.0


@dataclass
class GaussianEntropyLossConfig(BaseLossConfig):
    """Configuration for entropy loss."""

    _target_: str = "versatil.metrics.losses.divergence.GaussianEntropyLoss"
    key: str = MISSING
    weight: float = 0.0


@dataclass
class BinaryKLDivergenceLossConfig(BaseLossConfig):
    """Configuration for binary KL divergence loss."""

    _target_: str = "versatil.metrics.losses.divergence.BinaryKLDivergenceLoss"
    weight: float = 0.0001
    free_bits: float = 0.0
    latent_bits: int = MISSING
    entropy_weight: float = 0.005


@dataclass
class MaximumMeanDiscrepancyLossConfig(BaseLossConfig):
    """Configuration for Maximum Mean Discrepancy (MMD) loss."""

    _target_: str = (
        "versatil.metrics.losses.maximum_mean_discrepancy.MaximumMeanDiscrepancyLoss"
    )
    weight: float = 1.0
    prior_regularization_weight: float = 0.0
    prior_target_key: str = "${latent_key:POSTERIOR_LATENT}"
    kernel_type: str = KernelType.RBF.value
    bandwidth_multipliers: list[float] | None = field(
        default_factory=lambda: [0.2, 0.5, 1.0, 2.0, 5.0]
    )
    use_median_heuristic: bool = True
    use_fixed_gaussian_as_prior: bool = False


@dataclass
class ConditionalMaximumMeanDiscrepancyLossConfig(BaseLossConfig):
    """Configuration for conditional state-latent MMD loss."""

    _target_: str = "versatil.metrics.losses.maximum_mean_discrepancy.ConditionalMaximumMeanDiscrepancyLoss"
    weight: float = 1.0
    state_weight: float = 1.0
    prior_target_key: str = "${latent_key:POSTERIOR_LATENT}"
    condition_key: str = "${latent_key:PRIOR_CONDITION}"
    kernel_type: str = KernelType.RBF.value
    bandwidth_multipliers: list[float] | None = field(
        default_factory=lambda: [0.2, 0.5, 1.0, 2.0, 5.0]
    )
    use_median_heuristic: bool = True
    condition_kernel_type: str = KernelType.RBF.value
    condition_bandwidth_multipliers: list[float] | None = field(
        default_factory=lambda: [0.2, 0.5, 1.0, 2.0, 5.0]
    )
    condition_use_median_heuristic: bool = True
    normalize_condition: bool = True


@dataclass
class VQCommitmentLossConfig(BaseLossConfig):
    """Configuration for VQ commitment loss."""

    _target_: str = "versatil.metrics.losses.vector_quantization.VQCommitmentLoss"
    num_codes: int = MISSING
    num_residual_layers: int = MISSING
    weight: float = 1.0


@dataclass
class VQPriorCrossEntropyLossConfig(BaseLossConfig):
    """Configuration for VQ prior cross-entropy loss."""

    _target_: str = (
        "versatil.metrics.losses.vector_quantization.VQPriorCrossEntropyLoss"
    )
    weight: float = 1.0


@dataclass
class BinaryMaximumMeanDiscrepancyLossConfig(BaseLossConfig):
    """Configuration for Binary Maximum Mean Discrepancy (MMD) loss."""

    _target_: str = "versatil.metrics.losses.maximum_mean_discrepancy.BinaryMaximumMeanDiscrepancyLoss"
    weight: float = 1.0


@dataclass
class TrajectoryLengthLossConfig(BaseLossConfig):
    """Configuration for trajectory length loss."""

    _target_: str = "versatil.metrics.losses.trajectory.TrajectoryLengthLoss"
    weight: float = 0.1
    action_key: str = MISSING


@dataclass
class TrajectorySmoothnessConfig(BaseLossConfig):
    """Configuration for trajectory smoothness loss."""

    _target_: str = "versatil.metrics.losses.trajectory.TrajectorySmoothness"
    weight: float = 0.01
    action_key: str = MISSING


@dataclass
class ActionTokenLossConfig(BaseLossConfig):
    """Configuration for action token cross-entropy loss."""

    _target_: str = "versatil.metrics.losses.classification.ActionTokenLoss"
    weight: float = 1.0
    label_smoothing: float = 0.2


@dataclass
class PhaseClassificationLossConfig(BaseLossConfig):
    """Configuration for phase classification loss."""

    _target_: str = "versatil.metrics.losses.classification.PhaseClassificationLoss"
    key: str = MISSING
    cross_entropy_weight: float = 1.0
    entropy_weight: float = 0.0
    label_smoothing: float = 0.0


@dataclass
class GripperMixtureNLLossConfig(BaseLossConfig):
    """Configuration for gripper Mixture Negative Log-Likelihood loss."""

    _target_: str = "versatil.metrics.losses.mixture.GripperMixtureNLLoss"
    key: str = MISSING
    actions_metadata: Any = "${task.action_space.actions_metadata}"
    weight: float = 1.0
    learned_variance: bool = False
    sigma: float = 0.5
    min_variance: float = 1e-4


@dataclass
class CompositeLossConfig(BaseLossConfig):
    """Configuration for composite loss with custom modules."""

    _target_: str = "versatil.metrics.losses.composite.CompositeLoss"
    loss_modules: dict[str, Any] = field(default_factory=dict)
    weights: dict[str, float] | None = None


@dataclass
class PriorDenoisingLossConfig(BaseLossConfig):
    """Configuration for diffusion prior denoising loss."""

    _target_: str = "versatil.metrics.losses.prior_denoising.PriorDenoisingLoss"
    weight: float = 1.0


@dataclass
class MoELossConfig:
    """Configuration for Mixture of Experts (MoE) loss."""

    _target_: str = "versatil.metrics.losses.mixture_of_experts.MoELoss"
    base_loss: BaseLossConfig = MISSING
    entropy_weight: float = 0.0
    load_balance_weight: float = 0.0


@dataclass
class GaussianMixtureNLLossConfig(BaseLossConfig):
    """Configuration for Gaussian Mixture Negative Log-Likelihood loss."""

    _target_: str = "versatil.metrics.losses.mixture.GaussianMixtureNLLoss"
    action_keys: list[str] = MISSING
    weight: float = 1.0
    per_key_weights: dict[str, float] | None = None
    learned_variance: bool = True
    sigmas: dict[str, float] | None = None
    min_variance: float = 1e-4


@dataclass
class VICLatentLossConfig(BaseLossConfig):
    """Configuration for VICReg-style covariance + variance loss."""

    _target_: str = "versatil.metrics.losses.latent_geometry.VICLatentLoss"
    key: str = "${latent_key:POSTERIOR_MU}"
    covariance_weight: float = 3.0
    variance_weight: float = 10.0
    gamma: float = 0.3


@dataclass
class PosteriorGeometryLossConfig(BaseLossConfig):
    """Configuration for posterior latent moment regularization."""

    _target_: str = "versatil.metrics.losses.latent_geometry.PosteriorGeometryLoss"
    key: str = "${latent_key:POSTERIOR_MU}"
    mean_weight: float = 0.0
    std_weight: float = 0.0
    target_std: float = 1.0
    max_std_weight: float = 0.0
    max_std: float = 2.0
    covariance_weight: float = 0.0
    epsilon: float = 1e-6


@dataclass
class OptimalTransportLossConfig(BaseLossConfig):
    """Configuration for Optimal Transport loss using Sinkhorn divergence."""

    _target_: str = "versatil.metrics.losses.optimal_transport.OptimalTransportLoss"
    action_keys: list[str] = MISSING
    weight: float = 1.0
    p: int = 2
    blur_fraction: float = 0.1
    reach_multiplier: float | None = None
    expected_std: float = 1.0
    time_scale: float = 1.0


@dataclass
class LatentOptimalTransportLossConfig(BaseLossConfig):
    """Configuration for latent Sinkhorn divergence between posterior and prior."""

    _target_: str = (
        "versatil.metrics.losses.optimal_transport.LatentOptimalTransportLoss"
    )
    weight: float = 1.0
    prior_target_key: str = "${latent_key:POSTERIOR_LATENT}"
    p: int = 2
    blur_fraction: float = 0.1
    reach_multiplier: float | None = None


@dataclass
class RelaxedConditionalLatentOptimalTransportLossConfig(BaseLossConfig):
    """Configuration for relaxed conditional latent Sinkhorn divergence."""

    _target_: str = "versatil.metrics.losses.optimal_transport.RelaxedConditionalLatentOptimalTransportLoss"
    weight: float = 1.0
    prior_target_key: str = "${latent_key:POSTERIOR_LATENT}"
    condition_key: str = "${latent_key:PRIOR_CONDITION}"
    p: int = 2
    blur_fraction: float = 0.1
    reach_multiplier: float | None = None
    state_weight: float = 1.0
    normalize_condition: bool = True
