"""Gripper action loss and shared gripper metadata resolution."""

import torch
import torch.nn.functional as F

from versatil.common.omegaconf_ops import resolve_dict_keys
from versatil.data.constants import BinaryGripperRange, GripperType
from versatil.data.metadata import (
    ActionMetadata,
    GripperActionMetadata,
    GripperObservationMetadata,
    OnTheFlyActionMetadata,
)
from versatil.metrics.base import (
    BaseLoss,
    LossOutput,
    WeightsDictionary,
    reduce_loss_with_padding,
)
from versatil.metrics.constants import MetricKey


def resolve_gripper_metadata(
    key: str,
    actions_metadata: dict[str, ActionMetadata],
) -> tuple[str, str]:
    """Resolve gripper type and binary range from action-space metadata.

    Args:
        key: Gripper action key to look up.
        actions_metadata: Metadata of the action space keyed by action name.

    Returns:
        Tuple of (gripper_type, binary_gripper_range) values.

    Raises:
        ValueError: If the key is missing from the action space or its
            metadata is not gripper metadata.
    """
    resolved_metadata = resolve_dict_keys(dict(actions_metadata))
    if key not in resolved_metadata:
        raise ValueError(
            f"{key} is not available to the action space. Can't compute gripper loss. "
            f"Available keys: {list(resolved_metadata)}"
        )
    meta = resolved_metadata[key]
    if isinstance(meta, GripperActionMetadata):
        return meta.gripper_type, meta.binary_gripper_range
    if isinstance(meta, OnTheFlyActionMetadata):
        source = meta.source_metadata
        if isinstance(source, GripperObservationMetadata):
            return source.gripper_type, source.binary_gripper_range
        raise ValueError(
            f"Expected GripperObservationMetadata for key '{key}', got {type(source).__name__}"
        )
    raise ValueError(
        f"Expected gripper metadata for key '{key}', got {type(meta).__name__}"
    )


class GripperLoss(BaseLoss):
    """Loss for gripper action prediction (binary or continuous)."""

    def __init__(
        self,
        key: str,
        actions_metadata: dict[str, ActionMetadata],
        bce_weight: float = 0.005,
        mse_weight: float = 0.0,
        pos_weight: torch.Tensor | float | None = None,
    ):
        """Initialize gripper loss.

        Args:
            key: Action key for gripper
            actions_metadata: Dict of metadata of the action space
            bce_weight: Weight for binary cross entropy (binary gripper)
            mse_weight: Weight for MSE loss (continuous gripper)
            pos_weight: Optional positive class weight for BCE
        """
        super().__init__()
        self.key = key
        self.bce_weight = bce_weight
        self.mse_weight = mse_weight
        if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
            pos_weight = torch.tensor(float(pos_weight))
        self.register_buffer("pos_weight", pos_weight)
        self.gripper_type, self.binary_gripper_range = resolve_gripper_metadata(
            key=key, actions_metadata=actions_metadata
        )

    @property
    def requires_action_space_targets(self) -> bool:
        """Whether the loss needs action-space metadata for BCE targets."""
        return self.bce_weight > 0

    @property
    def weights(self) -> WeightsDictionary:
        """Getter that returns dictionary with weight keys and scalar coefficients."""
        return {"bce_weight": self.bce_weight, "mse_weight": self.mse_weight}

    def set_weights(self, new_weights: WeightsDictionary) -> None:
        """Setter that updates the weight scalar coefficients."""
        self._validate_weights(new_weights)
        self.bce_weight = new_weights["bce_weight"]
        self.mse_weight = new_weights["mse_weight"]

    def get_required_keys(self) -> set[str]:
        """Get required target keys for gripper loss.

        Returns:
            Set containing the gripper action key
        """
        return {self.key}

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute gripper loss.

        Args:
            predictions: Dictionary with 'gripper_action' key
            targets: Dictionary with ground truth gripper actions
            is_pad: Optional padding mask

        Returns:
            LossOutput with gripper loss
        """
        if self.key not in predictions or self.key not in targets:
            raise ValueError(
                f"Predictions and targets must contain key '{self.key}' for GripperLoss."
            )
        pred_gripper = predictions[self.key]
        target_gripper = targets[self.key]

        if self.gripper_type == GripperType.BINARY.value:
            if self.binary_gripper_range == BinaryGripperRange.MINUS_ONE_ONE.value:
                target_gripper = (target_gripper.float() + 1.0) / 2.0
            bce = F.binary_cross_entropy_with_logits(
                pred_gripper,
                target_gripper.float(),
                pos_weight=self.pos_weight,
                reduction="none",
            )
            bce_reduced = reduce_loss_with_padding(bce, is_pad, reduction="mean")
            return LossOutput(
                total_loss=self.bce_weight * bce_reduced,
                component_losses={MetricKey.GRIPPER_BCE.value: bce_reduced},
            )
        else:
            mse = F.mse_loss(pred_gripper, target_gripper, reduction="none")
            mse_reduced = reduce_loss_with_padding(mse, is_pad, reduction="mean")
            return LossOutput(
                total_loss=self.mse_weight * mse_reduced,
                component_losses={MetricKey.GRIPPER_MSE.value: mse_reduced},
            )
