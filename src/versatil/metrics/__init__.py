"""Metrics package for loss computation and performance tracking.

This package provides a modular system for computing losses and tracking metrics:

"""

from versatil.metrics.accumulators import (
    MetricsAccumulator,
    to_scalar,
)
from versatil.metrics.base import (
    BaseLoss,
    LossOutput,
    reduce_loss_with_padding,
)
from versatil.metrics.components import (
    ActionTokenLoss,
    BinaryKLDivergenceLoss,
    FixedVarianceGaussianNLLoss,
    FixedVarianceGripperMixtureNLLoss,
    GaussianEntropyLoss,
    GripperLoss,
    KLDivergenceLoss,
    MaximumMeanDiscrepancyLoss,
    BinaryMaximumMeanDiscrepancyLoss,
    MetadataPassthrough,
    MoELoss,
    PhaseClassificationLoss,
    PriorDenoisingLoss,
    RegressionLoss,
    TrajectoryLengthLoss,
    TrajectorySmoothness,
)
from versatil.metrics.composite import (
    CompositeLoss,
)
from versatil.metrics.constants import (
    LossModuleName,
    MetadataKey,
    MetricKey,
    PredictionKey,
    TargetKey,
)

__all__ = [
    "ActionTokenLoss",
    "BaseLoss",
    "BinaryKLDivergenceLoss",
    "BinaryMaximumMeanDiscrepancyLoss",
    "CompositeLoss",
    "FixedVarianceGaussianNLLoss",
    "FixedVarianceGripperMixtureNLLoss",
    "GaussianEntropyLoss",
    "GripperLoss",
    "KLDivergenceLoss",
    "LossModuleName",
    "LossOutput",
    "MaximumMeanDiscrepancyLoss",
    "MetadataKey",
    "MetadataPassthrough",
    "MetricKey",
    "MetricsAccumulator",
    "MoELoss",
    "PhaseClassificationLoss",
    "PredictionKey",
    "PriorDenoisingLoss",
    "reduce_loss_with_padding",
    "RegressionLoss",
    "TargetKey",
    "to_scalar",
    "TrajectoryLengthLoss",
    "TrajectorySmoothness",
]
