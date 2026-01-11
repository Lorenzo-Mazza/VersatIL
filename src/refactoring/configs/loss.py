"""Loss configuration for policy training."""

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

from refactoring.data.constants import (
    GripperType,
)


@dataclass
class BaseLossConfig:
    """Base configuration for loss modules."""

    _target_: str = MISSING


@dataclass
class RegressionLossConfig(BaseLossConfig):
    """Configuration for regression loss (position, orientation)."""

    _target_: str = "refactoring.metrics.RegressionLoss"
    action_keys: list[str] = MISSING
    mse_weight: float = 1.0
    l1_weight: float = 0.0
    huber_weight: float = 0.0
    huber_delta: float = 1.0
    per_key_weights: dict[str, float] | None = None


@dataclass
class GripperLossConfig(BaseLossConfig):
    """Configuration for gripper loss."""

    _target_: str = "refactoring.metrics.GripperLoss"
    key: str = MISSING
    actions_metadata: Any = "${task.action_space.actions_metadata}"
    bce_weight: float = 1.0
    mse_weight: float = 1.0
    pos_weight: float | None = None


@dataclass
class KLDivergenceLossConfig(BaseLossConfig):
    """Configuration for KL divergence loss (VAE)."""

    _target_: str = "refactoring.metrics.KLDivergenceLoss"
    weight: float = 0.0001
    prior_regularization_weight: float = 0.0


@dataclass
class GaussianEntropyLossConfig(BaseLossConfig):
    """Configuration for entropy loss."""

    _target_: str = "refactoring.metrics.GaussianEntropyLoss"
    key: str = MISSING
    weight: float = 0.0


@dataclass
class BinaryKLDivergenceLossConfig(BaseLossConfig):
    """Configuration for binary KL divergence loss (Free Transformer)."""

    _target_: str = "refactoring.metrics.BinaryKLDivergenceLoss"
    weight: float = 0.0001
    free_bits: float = 0.0
    latent_bits: int = MISSING
    entropy_weight: float = 0.005


@dataclass
class MaximumMeanDiscrepancyLossConfig(BaseLossConfig):
    """Configuration for Maximum Mean Discrepancy (MMD) loss."""

    _target_: str = "refactoring.metrics.MaximumMeanDiscrepancyLoss"
    weight: float = 1.0
    prior_regularization_weight: float = 0.0
    kernel_bandwidths: list[float] | None = field(default_factory=lambda: [0.2, 0.5, 1.0, 2.0, 5.0])


@dataclass
class BinaryMaximumMeanDiscrepancyLossConfig(BaseLossConfig):
    """Configuration for Binary Maximum Mean Discrepancy (MMD) loss."""

    _target_: str = "refactoring.metrics.BinaryMaximumMeanDiscrepancyLoss"
    weight: float = 1.0


@dataclass
class TrajectoryLengthLossConfig(BaseLossConfig):
    """Configuration for trajectory length loss."""

    _target_: str = "refactoring.metrics.TrajectoryLengthLoss"
    weight: float = 0.1
    action_key: str = MISSING


@dataclass
class TrajectorySmoothnessConfig(BaseLossConfig):
    """Configuration for trajectory smoothness loss."""

    _target_: str = "refactoring.metrics.TrajectorySmoothness"
    weight: float = 0.01
    action_key: str = MISSING


@dataclass
class PhaseClassificationLossConfig(BaseLossConfig):
    """Configuration for phase classification loss."""

    _target_: str = "refactoring.metrics.PhaseClassificationLoss"
    key: str = MISSING
    cross_entropy_weight: float = 1.0
    entropy_weight: float = 0.0
    label_smoothing: float = 0.0


@dataclass
class ActionTokenLossConfig(BaseLossConfig):
    """Configuration for action to token loss (TokenACT-style models)."""

    _target_: str = "refactoring.metrics.ActionTokenLoss"
    label_smoothing: float = 0.0


@dataclass
class ActionReconstructionLossConfig(BaseLossConfig):
    """Configuration for action reconstruction loss (ACT-style models)."""

    _target_: str = "refactoring.metrics.ActionReconstructionLoss"
    action_keys: list[str] | None = None
    mse_weight: float = 1.0
    l1_weight: float = 0.0
    gripper_bce_weight: float = 1.0
    kl_weight: float = 0.0001
    length_weight: float = 0.0
    smoothness_weight: float = 0.0
    gripper_type: str = GripperType.BINARY.value
    use_vae: bool = False


@dataclass
class PhaseActionLossConfig(BaseLossConfig):
    """Configuration for phase-conditioned action loss (PhaseACT models)."""

    _target_: str = "refactoring.metrics.PhaseActionLoss"
    action_keys: list[str] | None = None
    mse_weight: float = 1.0
    l1_weight: float = 0.0
    gripper_bce_weight: float = 1.0
    kl_weight: float = 0.0001
    length_weight: float = 0.0
    smoothness_weight: float = 0.0
    phase_ce_weight: float = 1.0
    phase_entropy_weight: float = 0.0
    label_smoothing: float = 0.0
    gripper_type: str = GripperType.BINARY.value
    use_vae: bool = False


@dataclass
class FixedVarianceGaussianNLLossConfig(BaseLossConfig):
    """Configuration for fixed variance Gaussian Negative Log-Likelihood loss."""

    _target_: str = "refactoring.metrics.FixedVarianceGaussianNLLoss"
    action_keys: list[str] = MISSING
    sigmas: dict[str, float] | None = None
    per_key_weights: dict[str, float] | None = None
    weight: float = 1.0


@dataclass
class FixedVarianceGripperMixtureNLLossConfig(BaseLossConfig):
    """Configuration for gripper Mixture Negative Log-Likelihood loss."""

    _target_: str = "refactoring.metrics.FixedVarianceGripperMixtureNLLoss"
    key: str = MISSING
    actions_metadata: Any = "${task.action_space.actions_metadata}"
    sigma: float = 0.5
    weight: float = 1.0


@dataclass
class CompositeLossConfig(BaseLossConfig):
    """Configuration for composite loss with custom modules."""

    _target_: str = "refactoring.metrics.CompositeLoss"
    loss_modules: dict[str, Any] = field(default_factory=dict)
    weights: dict[str, float] | None = None


@dataclass
class PriorDenoisingLossConfig(BaseLossConfig):
    """Configuration for diffusion prior denoising loss."""

    _target_: str = "refactoring.metrics.PriorDenoisingLoss"
    weight: float = 1.0


@dataclass
class MoELossConfig:
    """Configuration for Mixture of Experts (MoE) loss."""

    _target_: str = "refactoring.metrics.MoELoss"
    base_loss: BaseLossConfig = MISSING
