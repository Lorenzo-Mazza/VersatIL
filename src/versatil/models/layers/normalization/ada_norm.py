import torch
from torch import nn

from versatil.models.layers import ConditionalModulation
from versatil.models.layers.activation import ActivationFunction


class AdaNorm(nn.Module):
    """Adaptive normalization layer"""

    def __init__(
        self,
        base_norm: nn.Module,
        condition_dim: int,
        feature_dim: int,
        use_gate: bool = False,
        activation: str = ActivationFunction.SILU.value,
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
            use_gate=use_gate,
            activation=ActivationFunction.SILU.value,
            init_strategy="zero" if use_gate else "identity",
        )

    def forward(
        self, x: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional conditioning

        Returns:
            The normalized and modulated input x or a tuple with x and the gate value (alpha) to be used by the residual connection, if use_gate is True.
        """
        x = self.norm(x)
        return self.modulation(x, condition)
