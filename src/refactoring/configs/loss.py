"""Loss configuration for policy training."""

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

from refactoring.data.constants import (
    POSITION_ACTION_KEY,
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
    action_keys: list[str] = field(default_factory=lambda: [POSITION_ACTION_KEY])
    mse_weight: float = 1.0
    l1_weight: float = 0.0
    huber_weight: float = 0.0
    huber_delta: float = 1.0
    per_key_weights: dict[str, float] | None = None


@dataclass
class GripperLossConfig(BaseLossConfig):
    """Configuration for gripper loss."""

    _target_: str = "refactoring.metrics.GripperLoss"
    gripper_type: str = GripperType.BINARY.value
    bce_weight: float = 1.0
    mse_weight: float = 1.0
    pos_weight: float | None = None


@dataclass
class KLDivergenceLossConfig(BaseLossConfig):
    """Configuration for KL divergence loss (VAE)."""

    _target_: str = "refactoring.metrics.KLDivergenceLoss"
    weight: float = 0.0001


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
    action_key: str = POSITION_ACTION_KEY


@dataclass
class TrajectorySmoothnessConfig(BaseLossConfig):
    """Configuration for trajectory smoothness loss."""

    _target_: str = "refactoring.metrics.TrajectorySmoothness"
    weight: float = 0.01
    action_key: str = POSITION_ACTION_KEY


@dataclass
class PhaseClassificationLossConfig(BaseLossConfig):
    """Configuration for phase classification loss."""

    _target_: str = "refactoring.metrics.PhaseClassificationLoss"
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
class CompositeLossConfig(BaseLossConfig):
    """Configuration for composite loss with custom modules."""

    _target_: str = "refactoring.metrics.CompositeLoss"
    loss_modules: dict[str, Any] = field(default_factory=dict)
    weights: dict[str, float] | None = None


@dataclass
class MoELossConfig:
    """Configuration for Mixture of Experts (MoE) loss."""
    _target_: str = "refactoring.metrics.MoELoss"
    base_loss: BaseLossConfig = MISSING
