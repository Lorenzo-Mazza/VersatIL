"""Metrics package for loss computation and performance tracking.

This package provides a modular system for computing losses and tracking metrics:

"""

from refactoring.metrics.accumulators import (
    MetricsAccumulator,
    to_scalar,
)
from refactoring.metrics.base import (
    BaseLoss,
    LossOutput,
    reduce_loss_with_padding,
)
from refactoring.metrics.components import (
    BinaryKLDivergenceLoss,
    GripperLoss,
    KLDivergenceLoss,
    PhaseClassificationLoss,
    RegressionLoss,
    TrajectoryLengthLoss,
    TrajectorySmoothness,
)
from refactoring.metrics.composite import (
    ActionReconstructionLoss,
    CompositeLoss,
    PhaseActionLoss,
)
from refactoring.metrics.constants import (
    LossModuleName,
    MetadataKey,
    MetricKey,
    PredictionKey,
    TargetKey,
)

__all__ = [
    "BaseLoss",
    "LossOutput",
    "reduce_loss_with_padding",
    "RegressionLoss",
    "GripperLoss",
    "KLDivergenceLoss",
    "BinaryKLDivergenceLoss",
    "TrajectoryLengthLoss",
    "TrajectorySmoothness",
    "PhaseClassificationLoss",
    "CompositeLoss",
    "ActionReconstructionLoss",
    "PhaseActionLoss",
    "MetricsAccumulator",
    "to_scalar",
    "MetricKey",
    "MetadataKey",
    "LossModuleName",
    "PredictionKey",
    "TargetKey",
]
