"""Regression losses for continuous action predictions."""

import torch
import torch.nn.functional as F

from versatil.metrics.base import (
    BaseLoss,
    LossOutput,
    WeightsDictionary,
    reduce_loss_with_padding,
)
from versatil.metrics.constants import MetricKey


class RegressionLoss(BaseLoss):
    """Regression loss for continuous action predictions (position, orientation).

    Supports MSE, L1, and Huber loss functions with optional per-modality weighting.
    """

    def __init__(
        self,
        action_keys: list[str],
        mse_weight: float = 1.0,
        l1_weight: float = 0.0,
        huber_weight: float = 0.0,
        huber_delta: float = 1.0,
        per_key_weights: dict[str, float] | None = None,
    ):
        """Initialize regression loss.

        Args:
            action_keys: List of action keys to compute loss for (e.g., ['position', 'orientation'])
            mse_weight: Weight for MSE loss
            l1_weight: Weight for L1 loss
            huber_weight: Weight for Huber loss
            huber_delta: Delta parameter for Huber loss
            per_key_weights: Optional dictionary of per-key weights
        """
        super().__init__()
        self.action_keys = action_keys
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.huber_weight = huber_weight
        self.huber_delta = huber_delta
        self.per_key_weights = per_key_weights or {}

    @property
    def weights(self) -> WeightsDictionary:
        """Getter that returns dictionary with weight keys and scalar coefficients."""
        return {
            "mse_weight": self.mse_weight,
            "l1_weight": self.l1_weight,
            "huber_weight": self.huber_weight,
        }

    def set_weights(self, new_weights: WeightsDictionary) -> None:
        """Setter that updates the weight scalar coefficients."""
        self._validate_weights(new_weights)
        self.mse_weight = new_weights["mse_weight"]
        self.l1_weight = new_weights["l1_weight"]
        self.huber_weight = new_weights["huber_weight"]

    def get_required_keys(self) -> set[str]:
        """Get required target keys for regression loss.

        Returns:
            Set of action keys this loss operates on
        """
        return set(self.action_keys)

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute regression loss.

        Args:
            predictions: Dictionary with predicted actions
            targets: Dictionary with ground truth actions
            is_pad: Optional padding mask (B, horizon)

        Returns:
            LossOutput with regression loss components
        """
        component_losses = {}
        total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)

        for action_key in self.action_keys:
            if action_key not in predictions or action_key not in targets:
                raise ValueError(
                    f"Predictions and targets must contain key '{action_key}' for RegressionLoss."
                )

            pred = predictions[action_key].float()
            target = targets[action_key].float()
            key_weight = self.per_key_weights.get(action_key, 1.0)

            if self.mse_weight > 0:
                mse = F.mse_loss(pred, target, reduction="none")
                mse_reduced = reduce_loss_with_padding(mse, is_pad, reduction="mean")
                loss_key = f"{action_key}_{MetricKey.MSE_LOSS.value}"
                component_losses[loss_key] = mse_reduced
                total_loss = total_loss + self.mse_weight * key_weight * mse_reduced

            if self.l1_weight > 0:
                l1 = F.l1_loss(pred, target, reduction="none")
                l1_reduced = reduce_loss_with_padding(l1, is_pad, reduction="mean")
                loss_key = f"{action_key}_{MetricKey.L1_LOSS.value}"
                component_losses[loss_key] = l1_reduced
                total_loss = total_loss + self.l1_weight * key_weight * l1_reduced

            if self.huber_weight > 0:
                huber = F.huber_loss(
                    pred, target, delta=self.huber_delta, reduction="none"
                )
                huber_reduced = reduce_loss_with_padding(
                    huber, is_pad, reduction="mean"
                )
                loss_key = f"{action_key}_{MetricKey.HUBER_LOSS.value}"
                component_losses[loss_key] = huber_reduced
                total_loss = total_loss + self.huber_weight * key_weight * huber_reduced

        return LossOutput(total_loss=total_loss, component_losses=component_losses)
