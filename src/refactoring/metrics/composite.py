"""Composite loss class that combine multiple loss components."""


import torch
import torch.nn as nn

from refactoring.metrics.base import BaseLoss, LossOutput


class CompositeLoss(BaseLoss):
    """Composite loss that combines multiple loss modules with weights.

    This loss module orchestrates multiple sub-losses and combines them
    with configurable weights. It's useful for complex training objectives
    that involve multiple loss terms.
    """

    def __init__(
        self,
        loss_modules: dict[str, BaseLoss],
        weights: dict[str, float] | None = None,
    ):
        """Initialize composite loss.

        Args:
            loss_modules: Dictionary of loss module names to loss instances
            weights: Optional dictionary of weights for each loss module
        """
        super().__init__()
        self.loss_modules = nn.ModuleDict(loss_modules)
        self.weights = weights or dict.fromkeys(loss_modules.keys(), 1.0)

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
        """Compute weighted sum of all loss modules.

        Args:
            predictions: Model output dictionary
            targets: Ground truth dictionary
            is_pad: Optional padding mask

        Returns:
            LossOutput with total weighted loss and all component losses
        """
        device = next(iter(predictions.values())).device
        total_loss = torch.tensor(0.0, device=device)
        all_component_losses = {}
        all_metadata = {}

        for name, loss_module in self.loss_modules.items():
            loss_output: LossOutput = loss_module(predictions, targets, is_pad)
            weight = self.weights.get(name, 1.0)
            total_loss = total_loss + weight * loss_output.total_loss
            for comp_name, comp_value in loss_output.component_losses.items():
                prefixed_name = f"{name}/{comp_name}"
                all_component_losses[prefixed_name] = comp_value

            all_metadata.update(loss_output.metadata)

        return LossOutput(
            total_loss=total_loss,
            component_losses=all_component_losses,
            metadata=all_metadata,
        )
