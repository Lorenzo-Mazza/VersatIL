"""Metrics package for loss computation and performance tracking.

This package provides a modular system for computing losses and tracking metrics:

- **Base classes**: BaseLoss, LossOutput for defining loss modules
- **Loss components**: Individual loss functions (regression, classification, etc.)
- **Composite losses**: Pre-configured loss combinations for common use cases
- **Metrics accumulators**: Classes for tracking and aggregating metrics across batches
- **Constants**: Enums and constants for loss types and metric keys
- **Add-ons**: Optional loss functions with heavy dependencies (geomloss+pykeops)
  - OptimalTransportLoss: Uses lazy imports to avoid slow compilation overhead
  - Access via: refactoring.metrics.add-ons.OptimalTransportLoss or Hydra _target_

Example usage:
    ```python
    from refactoring.metrics import ActionReconstructionLoss, MetricsAccumulator

    loss_fn = ActionReconstructionLoss(
        action_keys=["position_action"],
        mse_weight=1.0,
        use_vae=True,
    )

    metrics = MetricsAccumulator()

    for batch in dataloader:
        predictions = model(batch)
        loss_output = loss_fn(predictions, batch, is_pad=batch["is_pad"])
        metrics.add_loss_output(loss_output)

    print(metrics.to_dict())
    ```

Note on OptimalTransportLoss:
    OptimalTransportLoss is NOT imported by default to avoid triggering slow
    PyKeOps JIT compilation. Use it via Hydra config with _target_:
    ```yaml
    loss:
        _target_: refactoring.metrics.add-ons.OptimalTransportLoss
        action_keys: [position_action]
        weight: 0.1
    ```
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
    MEAN_KEY,
    PREDICTION_TYPE_KEY,
    VARIANCE_KEY,
    LossModuleName,
    MetadataKey,
    MetricKey,
    PredictionKey,
    ReductionMode,
    TargetKey,
)

__all__ = [
    "BaseLoss",
    "LossOutput",
    "reduce_loss_with_padding",
    "RegressionLoss",
    "GripperLoss",
    "KLDivergenceLoss",
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
    "ReductionMode",
    "LossModuleName",
    "PredictionKey",
    "TargetKey",
    "VARIANCE_KEY",
    "MEAN_KEY",
    "PREDICTION_TYPE_KEY",
]
