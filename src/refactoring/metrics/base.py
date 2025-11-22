"""Base classes for loss computation and metrics tracking."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from refactoring.metrics.constants import MetricKey


@dataclass
class LossOutput:
    """Output from loss computation containing total loss and component losses.

    Attributes:
        total_loss: Scalar tensor representing the total weighted loss
        component_losses: Dictionary mapping loss component names to their values
        metadata: Optional dictionary for additional information (e.g., predictions for metrics)
    """

    total_loss: torch.Tensor
    component_losses: dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        """Convert loss output to dictionary of scalar values.

        Returns:
            Dictionary with MetricKey.TOTAL_LOSS and all component losses as floats
        """
        result = {MetricKey.TOTAL_LOSS.value: self.total_loss.item()}
        for key, value in self.component_losses.items():
            result[key] = value.item() if isinstance(value, torch.Tensor) else value
        return result

    def __add__(self, other: "LossOutput") -> "LossOutput":
        """Add two LossOutput objects component-wise.

        Args:
            other: Another LossOutput instance

        Returns:
            New LossOutput with summed losses
        """
        if not isinstance(other, LossOutput):
            raise TypeError(f"Cannot add LossOutput with {type(other)}")

        new_total = self.total_loss + other.total_loss
        new_components = {}

        all_keys = set(self.component_losses.keys()) | set(
            other.component_losses.keys()
        )
        device = self.total_loss.device
        zero = torch.tensor(0.0, device=device)
        for key in all_keys:
            val1 = self.component_losses.get(key, zero)
            val2 = other.component_losses.get(key, zero)
            new_components[key] = val1 + val2

        return LossOutput(
            total_loss=new_total,
            component_losses=new_components,
            metadata={**self.metadata, **other.metadata},
        )

    def scale(self, factor: float) -> "LossOutput":
        """Scale all losses by a constant factor.

        Args:
            factor: Scaling factor

        Returns:
            New LossOutput with scaled losses
        """
        return LossOutput(
            total_loss=self.total_loss * factor,
            component_losses={k: v * factor for k, v in self.component_losses.items()},
            metadata=self.metadata,
        )


class BaseLoss(nn.Module, ABC):
    """Abstract base class for loss computation modules.

    All loss modules should inherit from this class and implement the forward method.
    Loss modules can be composed together to create complex loss functions.
    """

    @abstractmethod
    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute loss given predictions and targets.

        Args:
            predictions: Dictionary of model predictions
            targets: Dictionary of ground truth values
            is_pad: Optional boolean tensor indicating padded positions (B, horizon)

        Returns:
            LossOutput containing total loss and component losses
        """
        raise NotImplementedError

    @abstractmethod
    def get_required_keys(self) -> set[str]:
        """Get the set of keys this loss expects from the targets dictionary.

        This is used for validation to ensure the action space contains all
        necessary keys for the configured loss. For composite losses, this should
        recursively collect keys from all sub-losses.

        Returns:
            Set of target dictionary keys required by this loss
        """
        raise NotImplementedError


def reduce_loss_with_padding(
    loss_tensor: torch.Tensor,
    is_pad: torch.Tensor | None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Apply padding-aware reduction to a loss tensor.

    Args:
        loss_tensor: Loss values of shape (B, horizon, ...)
        is_pad: Boolean mask of shape (B, horizon) where True indicates padding
        reduction: Reduction mode ('mean', 'sum', or 'none')

    Returns:
        Reduced loss tensor
    """
    if is_pad is None:
        if reduction == "mean":
            return loss_tensor.mean()
        elif reduction == "sum":
            return loss_tensor.sum()
        else:
            return loss_tensor
    pad_mask = (~is_pad).float()
    while pad_mask.dim() < loss_tensor.dim():
        pad_mask = pad_mask.unsqueeze(-1)
    masked_loss = loss_tensor * pad_mask
    if reduction == "mean":
        return masked_loss.sum() / (pad_mask.sum() + 1e-8)
    elif reduction == "sum":
        return masked_loss.sum()
    else:
        return masked_loss

