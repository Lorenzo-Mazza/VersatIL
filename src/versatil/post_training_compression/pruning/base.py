"""Abstract base class for weight pruning strategies."""

import abc

import torch
from torch import nn


class BasePruner(abc.ABC):
    """Base interface for all pruning strategies."""

    @abc.abstractmethod
    def prune(self, module: nn.Module) -> tuple[int, int]:
        """Apply pruning to module.

        Args:
            module: Neural network module to prune.

        Returns:
            Tuple of (total_parameters, zero_parameters).
        """

    @staticmethod
    def compute_sparsity(module: nn.Module) -> tuple[int, int]:
        """Count total and zero parameters in module.

        Args:
            module: Neural network module to inspect.

        Returns:
            Tuple of (total_parameters, zero_parameters).
        """
        total_parameters = 0
        zero_parameters = 0
        for parameter in module.parameters():
            total_parameters += parameter.numel()
            zero_parameters += int(torch.sum(parameter == 0).item())
        return total_parameters, zero_parameters
