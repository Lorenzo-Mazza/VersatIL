"""Loss configuration for policy training."""

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING


@dataclass
class BaseLossConfig:
    """Base configuration for loss modules."""

    _target_: str = MISSING


@dataclass
class RegressionLossConfig(BaseLossConfig):
    """Configuration for regression loss (position, orientation)."""

    _target_: str = "versatil.metrics.RegressionLoss"
    action_keys: list[str] = MISSING
    mse_weight: float = 1.0
    l1_weight: float = 0.0
    huber_weight: float = 0.0
    huber_delta: float = 1.0
    per_key_weights: dict[str, float] | None = None


@dataclass
class GripperLossConfig(BaseLossConfig):
    """Configuration for gripper loss."""

    _target_: str = "versatil.metrics.GripperLoss"
    key: str = MISSING
    actions_metadata: Any = "${task.action_space.actions_metadata}"
    bce_weight: float = 1.0
    mse_weight: float = 1.0
    pos_weight: float | None = None


@dataclass
class KLDivergenceLossConfig(BaseLossConfig):
    """Configuration for KL divergence loss."""

    _target_: str = "versatil.metrics.KLDivergenceLoss"
    weight: float = 0.0001
    prior_regularization_weight: float = 0.0


@dataclass
class GaussianEntropyLossConfig(BaseLossConfig):
    """Configuration for entropy loss."""

    _target_: str = "versatil.metrics.GaussianEntropyLoss"
    key: str = MISSING
    weight: float = 0.0


@dataclass
class BinaryKLDivergenceLossConfig(BaseLossConfig):
    """Configuration for binary KL divergence loss."""

    _target_: str = "versatil.metrics.BinaryKLDivergenceLoss"
    weight: float = 0.0001
    free_bits: float = 0.0
    latent_bits: int = MISSING
    entropy_weight: float = 0.005


@dataclass
class MaximumMeanDiscrepancyLossConfig(BaseLossConfig):
    """Configuration for Maximum Mean Discrepancy (MMD) loss."""

    _target_: str = "versatil.metrics.MaximumMeanDiscrepancyLoss"
    weight: float = 1.0
    prior_regularization_weight: float = 0.0
    kernel_bandwidths: list[float] | None = field(
        default_factory=lambda: [0.2, 0.5, 1.0, 2.0, 5.0]
    )
    use_fixed_gaussian_as_prior: bool = False


@dataclass
class BinaryMaximumMeanDiscrepancyLossConfig(BaseLossConfig):
    """Configuration for Binary Maximum Mean Discrepancy (MMD) loss."""

    _target_: str = "versatil.metrics.BinaryMaximumMeanDiscrepancyLoss"
    weight: float = 1.0


@dataclass
class TrajectoryLengthLossConfig(BaseLossConfig):
    """Configuration for trajectory length loss."""

    _target_: str = "versatil.metrics.TrajectoryLengthLoss"
    weight: float = 0.1
    action_key: str = MISSING


@dataclass
class TrajectorySmoothnessConfig(BaseLossConfig):
    """Configuration for trajectory smoothness loss."""

    _target_: str = "versatil.metrics.TrajectorySmoothness"
    weight: float = 0.01
    action_key: str = MISSING


@dataclass
class PhaseClassificationLossConfig(BaseLossConfig):
    """Configuration for phase classification loss."""

    _target_: str = "versatil.metrics.PhaseClassificationLoss"
    key: str = MISSING
    cross_entropy_weight: float = 1.0
    entropy_weight: float = 0.0
    label_smoothing: float = 0.0


@dataclass
class GripperMixtureNLLossConfig(BaseLossConfig):
    """Configuration for gripper Mixture Negative Log-Likelihood loss."""

    _target_: str = "versatil.metrics.GripperMixtureNLLoss"
    key: str = MISSING
    actions_metadata: Any = "${task.action_space.actions_metadata}"
    weight: float = 1.0
    learned_variance: bool = False
    sigma: float = 0.5
    min_variance: float = 1e-4


@dataclass
class CompositeLossConfig(BaseLossConfig):
    """Configuration for composite loss with custom modules."""

    _target_: str = "versatil.metrics.CompositeLoss"
    loss_modules: dict[str, Any] = field(default_factory=dict)
    weights: dict[str, float] | None = None


@dataclass
class PriorDenoisingLossConfig(BaseLossConfig):
    """Configuration for diffusion prior denoising loss."""

    _target_: str = "versatil.metrics.PriorDenoisingLoss"
    weight: float = 1.0


@dataclass
class MoELossConfig:
    """Configuration for Mixture of Experts (MoE) loss."""

    _target_: str = "versatil.metrics.MoELoss"
    base_loss: BaseLossConfig = MISSING


@dataclass
class GaussianMixtureNLLossConfig(BaseLossConfig):
    """Configuration for Gaussian Mixture Negative Log-Likelihood loss."""

    _target_: str = "versatil.metrics.GaussianMixtureNLLoss"
    action_keys: list[str] = MISSING
    weight: float = 1.0
    per_key_weights: dict[str, float] | None = None
    learned_variance: bool = True
    sigmas: dict[str, float] | None = None
    min_variance: float = 1e-4


@dataclass
class OptimalTransportLossConfig(BaseLossConfig):
    """Configuration for Optimal Transport loss using Sinkhorn divergence."""

    _target_: str = "versatil.metrics.ot_loss.OptimalTransportLoss"
    action_keys: list[str] = MISSING
    weight: float = 1.0
    p: int = 1
    epsilon: float = 0.01
    time_scale: float = 1.0


