"""Composite loss class that combine multiple loss components."""

import warnings
from typing import Any

import torch
import torch.nn as nn

from versatil.data.tokenization.tokenizer import Tokenizer
from versatil.metrics.base import BaseLoss, LossOutput, WeightsDictionary


class CompositeLoss(BaseLoss):
    """Composite loss that sums multiple sub-loss modules."""

    def __init__(
        self,
        loss_modules: dict[str, BaseLoss],
        weights: dict[str, float] | None = None,
    ):
        """Initialize composite loss.

        Args:
            loss_modules: Dictionary of loss module names to loss instances.
            weights: Deprecated legacy composite weights. Kept only for config
                compatibility and ignored at runtime.
        """
        super().__init__()
        self.loss_modules = nn.ModuleDict(loss_modules)
        if weights is not None and any(weight != 1.0 for weight in weights.values()):
            warnings.warn(
                "CompositeLoss.weights is deprecated and ignored at runtime. "
                "Move weights into the child loss configurations instead.",
                DeprecationWarning,
                stacklevel=2,
            )

    @property
    def weights(self) -> WeightsDictionary:
        """Getter that returns dictionary with weight keys and scalar coefficients."""
        return {name: child.weights for name, child in self.loss_modules.items()}

    def set_weights(self, new_weights: WeightsDictionary) -> None:
        """Setter that updates the weight scalar coefficients."""
        self._validate_weights(new_weights)
        for name, child in self.loss_modules.items():
            child.set_weights(new_weights[name])

    def set_tokenizer(self, tokenizer: Tokenizer | None) -> None:
        """Pass tokenizer metadata to every child loss."""
        for child in self.loss_modules.values():
            child.set_tokenizer(tokenizer=tokenizer)

    def get_required_keys(self) -> set[str]:
        """Get required target keys by recursively collecting from all sub-modules.

        Returns:
            Union of all required keys from all sub-modules
        """
        required_keys = set()
        for loss_module in self.loss_modules.values():
            required_keys.update(loss_module.get_required_keys())
        return required_keys

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Sum all sub-loss outputs (each sub-loss applies its own scalar weight).

        Args:
            predictions: Model output dictionary
            targets: Ground truth dictionary
            is_pad: Optional padding mask

        Returns:
            LossOutput with the summed total loss and all component losses
        """
        device = next(iter(predictions.values())).device
        total_loss = torch.tensor(0.0, device=device)
        all_component_losses: dict[str, torch.Tensor] = {}
        all_metadata: dict[str, Any] = {}

        for name, loss_module in self.loss_modules.items():
            loss_output: LossOutput = loss_module(predictions, targets, is_pad)
            total_loss = total_loss + loss_output.total_loss
            for component_name, component_value in loss_output.component_losses.items():
                prefixed_name = f"{name}/{component_name}"
                all_component_losses[prefixed_name] = component_value
            all_metadata.update(loss_output.metadata)

        return LossOutput(
            total_loss=total_loss,
            component_losses=all_component_losses,
            metadata=all_metadata,
        )
