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

    _target_: str = "versatil.metrics.components.RegressionLoss"
    action_keys: list[str] = MISSING
    mse_weight: float = 1.0
    l1_weight: float = 0.0
    huber_weight: float = 0.0
    huber_delta: float = 1.0
    per_key_weights: dict[str, float] | None = None


@dataclass
class GripperLossConfig(BaseLossConfig):
    """Configuration for gripper loss."""

    _target_: str = "versatil.metrics.components.GripperLoss"
    key: str = MISSING
    actions_metadata: Any = "${task.action_space.actions_metadata}"
    bce_weight: float = 1.0
    mse_weight: float = 1.0
    pos_weight: float | None = None


@dataclass
class KLDivergenceLossConfig(BaseLossConfig):
    """Configuration for KL divergence loss."""

    _target_: str = "versatil.metrics.components.KLDivergenceLoss"
    weight: float = 0.0001
    prior_regularization_weight: float = 0.0


@dataclass
class GaussianEntropyLossConfig(BaseLossConfig):
    """Configuration for entropy loss."""

    _target_: str = "versatil.metrics.components.GaussianEntropyLoss"
    key: str = MISSING
    weight: float = 0.0


@dataclass
class BinaryKLDivergenceLossConfig(BaseLossConfig):
    """Configuration for binary KL divergence loss."""

    _target_: str = "versatil.metrics.components.BinaryKLDivergenceLoss"
    weight: float = 0.0001
    free_bits: float = 0.0
    latent_bits: int = MISSING
    entropy_weight: float = 0.005


@dataclass
class MaximumMeanDiscrepancyLossConfig(BaseLossConfig):
    """Configuration for Maximum Mean Discrepancy (MMD) loss."""

    _target_: str = "versatil.metrics.components.MaximumMeanDiscrepancyLoss"
    weight: float = 1.0
    prior_regularization_weight: float = 0.0
    kernel_type: str = KernelType.RBF.value
    bandwidth_multipliers: list[float] | None = field(
        default_factory=lambda: [0.2, 0.5, 1.0, 2.0, 5.0]
    )
    use_median_heuristic: bool = True
    use_fixed_gaussian_as_prior: bool = False


@dataclass
class VQCommitmentLossConfig(BaseLossConfig):
    """Configuration for VQ commitment loss."""

    _target_: str = "versatil.metrics.components.VQCommitmentLoss"
    num_codes: int = MISSING
    num_residual_layers: int = MISSING
    weight: float = 1.0


@dataclass
class VQPriorCrossEntropyLossConfig(BaseLossConfig):
    """Configuration for VQ prior cross-entropy loss."""

    _target_: str = "versatil.metrics.components.VQPriorCrossEntropyLoss"
    weight: float = 1.0


@dataclass
class BinaryMaximumMeanDiscrepancyLossConfig(BaseLossConfig):
    """Configuration for Binary Maximum Mean Discrepancy (MMD) loss."""

    _target_: str = "versatil.metrics.components.BinaryMaximumMeanDiscrepancyLoss"
    weight: float = 1.0


@dataclass
class TrajectoryLengthLossConfig(BaseLossConfig):
    """Configuration for trajectory length loss."""

    _target_: str = "versatil.metrics.components.TrajectoryLengthLoss"
    weight: float = 0.1
    action_key: str = MISSING


@dataclass
class TrajectorySmoothnessConfig(BaseLossConfig):
    """Configuration for trajectory smoothness loss."""

    _target_: str = "versatil.metrics.components.TrajectorySmoothness"
    weight: float = 0.01
    action_key: str = MISSING


@dataclass
class ActionTokenLossConfig(BaseLossConfig):
    """Configuration for action token cross-entropy loss."""

    _target_: str = "versatil.metrics.components.ActionTokenLoss"
    weight: float = 1.0
    label_smoothing: float = 0.2


@dataclass
class PhaseClassificationLossConfig(BaseLossConfig):
    """Configuration for phase classification loss."""

    _target_: str = "versatil.metrics.components.PhaseClassificationLoss"
    key: str = MISSING
    cross_entropy_weight: float = 1.0
    entropy_weight: float = 0.0
    label_smoothing: float = 0.0


@dataclass
class GripperMixtureNLLossConfig(BaseLossConfig):
    """Configuration for gripper Mixture Negative Log-Likelihood loss."""

    _target_: str = "versatil.metrics.components.GripperMixtureNLLoss"
    key: str = MISSING
    actions_metadata: Any = "${task.action_space.actions_metadata}"
    weight: float = 1.0
    learned_variance: bool = False
    sigma: float = 0.5
    min_variance: float = 1e-4


@dataclass
class CompositeLossConfig(BaseLossConfig):
    """Configuration for composite loss with custom modules."""

    _target_: str = "versatil.metrics.composite.CompositeLoss"
    loss_modules: dict[str, Any] = field(default_factory=dict)
    weights: dict[str, float] | None = None


@dataclass
class PriorDenoisingLossConfig(BaseLossConfig):
    """Configuration for diffusion prior denoising loss."""

    _target_: str = "versatil.metrics.components.PriorDenoisingLoss"
    weight: float = 1.0


@dataclass
class MoELossConfig:
    """Configuration for Mixture of Experts (MoE) loss."""

    _target_: str = "versatil.metrics.components.MoELoss"
    base_loss: BaseLossConfig = MISSING
    entropy_weight: float = 0.0
    load_balance_weight: float = 0.0


@dataclass
class GaussianMixtureNLLossConfig(BaseLossConfig):
    """Configuration for Gaussian Mixture Negative Log-Likelihood loss."""

    _target_: str = "versatil.metrics.components.GaussianMixtureNLLoss"
    action_keys: list[str] = MISSING
    weight: float = 1.0
    per_key_weights: dict[str, float] | None = None
    learned_variance: bool = True
    sigmas: dict[str, float] | None = None
    min_variance: float = 1e-4


@dataclass
class VICLatentLossConfig(BaseLossConfig):
    """Configuration for VICReg-style covariance + variance loss."""

    _target_: str = "versatil.metrics.components.VICLatentLoss"
    key: str = "${latent_key:POSTERIOR_MU}"
    covariance_weight: float = 3.0
    variance_weight: float = 10.0
    gamma: float = 0.3


@dataclass
class PosteriorGeometryLossConfig(BaseLossConfig):
    """Configuration for posterior latent moment regularization."""

    _target_: str = "versatil.metrics.components.PosteriorGeometryLoss"
    key: str = "${latent_key:POSTERIOR_MU}"
    mean_weight: float = 0.0
    std_weight: float = 0.0
    target_std: float = 1.0
    max_std_weight: float = 0.0
    max_std: float = 2.0
    covariance_weight: float = 0.0
    eps: float = 1e-6


@dataclass
class OptimalTransportLossConfig(BaseLossConfig):
    """Configuration for Optimal Transport loss using Sinkhorn divergence."""

    _target_: str = "versatil.metrics.ot_loss.OptimalTransportLoss"
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

    _target_: str = "versatil.metrics.ot_loss.LatentOptimalTransportLoss"
    weight: float = 1.0
    p: int = 2
    blur_fraction: float = 0.1
    reach_multiplier: float | None = None
