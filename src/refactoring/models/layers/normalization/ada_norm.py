import torch
from torch import nn

from refactoring.models.layers import ConditionalModulation
from refactoring.models.layers.activation import ActivationFunction


class AdaNorm(nn.Module):
    """Adaptive normalization layer"""
    def __init__(
            self,
            base_norm: nn.Module,
            condition_dim: int,
            feature_dim: int,
            activation: str = ActivationFunction.SILU.value
    ):
        super().__init__()
        self.norm = base_norm
        self.condition_dim = condition_dim
        self.feature_dim = feature_dim
        self.activation = activation
        self.modulation = ConditionalModulation(
            condition_dim=condition_dim,
            feature_dim=feature_dim,
            use_shift=True,
            activation=ActivationFunction.SILU.value,
            init_strategy="identity",
        )


    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional conditioning"""
        x = self.norm(x)
        return self.modulation(x, condition)
