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
    MaximumMeanDiscrepancyLoss,
    BinaryMaximumMeanDiscrepancyLoss,
    PhaseClassificationLoss,
    RegressionLoss,
    TrajectoryLengthLoss,
    TrajectorySmoothness,
    MoELoss,
    FixedVarianceGaussianNLLoss,
    FixedVarianceGripperMixtureNLLoss

)
from refactoring.metrics.composite import (
    CompositeLoss,
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
    "MaximumMeanDiscrepancyLoss",
    "BinaryMaximumMeanDiscrepancyLoss",
    "BinaryKLDivergenceLoss",
    "TrajectoryLengthLoss",
    "TrajectorySmoothness",
    "PhaseClassificationLoss",
    "CompositeLoss",
    "FixedVarianceGaussianNLLoss",
    "FixedVarianceGripperMixtureNLLoss",
    "MetricsAccumulator",
    "to_scalar",
    "MetricKey",
    "MetadataKey",
    "LossModuleName",
    "PredictionKey",
    "TargetKey",
    "MoELoss",
]
