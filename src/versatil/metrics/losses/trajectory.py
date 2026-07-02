"""Trajectory shape regularization losses."""

import torch

from versatil.metrics.base import (
    LossOutput,
    ScalarWeightedLoss,
    reduce_loss_with_padding,
)
from versatil.metrics.constants import MetricKey


class TrajectoryLengthLoss(ScalarWeightedLoss):
    """Loss for trajectory length consistency.

    Penalizes differences between predicted and ground truth trajectory lengths.
    """

    def __init__(self, action_key: str, weight: float = 0.001):
        """Initialize trajectory length loss.

        Args:
            weight: Weight for length loss
            action_key: Action key to compute length for
        """
        super().__init__()
        self.weight = weight
        self.action_key = action_key

    def get_required_keys(self) -> set[str]:
        """Get required target keys for trajectory length loss.

        Returns:
            Set containing the action key this loss operates on
        """
        return {self.action_key}

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute trajectory length loss.

        Args:
            predictions: Dictionary with predicted actions
            targets: Dictionary with ground truth actions
            is_pad: Optional padding mask

        Returns:
            LossOutput with length loss
        """
        if self.action_key not in predictions or self.action_key not in targets:
            raise ValueError(
                f"Predictions and targets must contain key '{self.action_key}' for TrajectoryLengthLoss."
            )
        pred = predictions[self.action_key]
        target = targets[self.action_key]

        pred_steps = torch.norm(pred[:, 1:] - pred[:, :-1], dim=-1)  # (B, H-1)
        target_steps = torch.norm(target[:, 1:] - target[:, :-1], dim=-1)  # (B, H-1)

        if is_pad is not None:
            # A step between t-1 and t is valid only if both timesteps are valid
            valid_steps = (~is_pad[:, 1:]) & (~is_pad[:, :-1])  # (B, H-1)
            pred_length = reduce_loss_with_padding(
                pred_steps, is_pad=~valid_steps, reduction="mean"
            )
            target_length = reduce_loss_with_padding(
                target_steps, is_pad=~valid_steps, reduction="mean"
            )
        else:
            pred_length = pred_steps.mean()
            target_length = target_steps.mean()

        length_loss = (pred_length - target_length) ** 2

        return LossOutput(
            total_loss=self.weight * length_loss,
            component_losses={MetricKey.LENGTH_LOSS.value: length_loss},
        )


class TrajectorySmoothness(ScalarWeightedLoss):
    """Loss for trajectory smoothness (acceleration regularization)."""

    def __init__(self, action_key: str, weight: float = 0.001):
        """Initialize smoothness loss.

        Args:
            weight: Weight for smoothness loss
            action_key: Action key to compute smoothness for
        """
        super().__init__()
        self.weight = weight
        self.action_key = action_key

    def get_required_keys(self) -> set[str]:
        """Get required target keys for trajectory smoothness loss.

        Note: This loss only uses predictions, not targets, but we return
        the action key for consistency with other trajectory losses.

        Returns:
            Empty set since this loss doesn't use targets
        """
        return set()

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute trajectory smoothness loss.

        Args:
            predictions: Dictionary with predicted actions
            targets: Not used for smoothness
            is_pad: Optional padding mask

        Returns:
            LossOutput with smoothness loss
        """
        if self.action_key not in predictions:
            raise ValueError(
                f"Predictions must contain key '{self.action_key}' for TrajectorySmoothness loss."
            )
        pred = predictions[self.action_key]
        if (
            pred.shape[1] < 3
        ):  # If trajectory too short, no acceleration can be computed
            return LossOutput(
                total_loss=torch.tensor(0.0, device=pred.device),
                component_losses={MetricKey.SMOOTHNESS_LOSS.value: torch.tensor(0.0)},
            )
        velocities = pred[:, 1:] - pred[:, :-1]
        accelerations = velocities[:, 1:] - velocities[:, :-1]
        smoothness = torch.norm(accelerations, dim=-1)
        if is_pad is not None:
            # Acceleration at position t uses timesteps t, t+1, t+2 — invalid if any is padded
            pad_mask_accel = is_pad[:, :-2] | is_pad[:, 1:-1] | is_pad[:, 2:]
            smoothness = reduce_loss_with_padding(
                smoothness, pad_mask_accel, reduction="mean"
            )
        else:
            smoothness = smoothness.mean()

        return LossOutput(
            total_loss=self.weight * smoothness,
            component_losses={MetricKey.SMOOTHNESS_LOSS.value: smoothness},
        )
