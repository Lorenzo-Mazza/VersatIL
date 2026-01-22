"""Final prediction layer for DiT with adaptive layer normalization modulation."""

import torch
import torch.nn as nn

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.normalization.ada_norm import AdaNorm


class FinalPredictionLayer(nn.Module):
    """Final layer that predicts noise (epsilon) with adaptive LN modulation.

    Uses the standard DiT modulation: norm(x) * (1 + scale) + shift
    """

    def __init__(
            self,
            hidden_dimension: int,
            output_dimension: int,
            activation: str = ActivationFunction.SILU.value,
    ) -> None:
        """Initialize the final prediction layer of the transformer.

        Args:
            hidden_dimension: Input hidden dimension.
            output_dimension: Output dimension.
            activation: Activation function for the AdaNorm modulation network.
        """
        super().__init__()
        base_norm = nn.LayerNorm(
            hidden_dimension, elementwise_affine=False, eps=1e-6  # Layer norm without learnable affine parameters
        )
        self.ada_norm = AdaNorm(
            base_norm=base_norm,
            condition_dim=hidden_dimension,
            feature_dim=hidden_dimension,
            use_gate=False, # no gate needed at the end
            activation=activation,
        )
        self.output_linear = nn.Linear(hidden_dimension, output_dimension, bias=True)
        self.reset_parameters()

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the output with adaptive modulation.

        Args:
            hidden_states: Input tensor (batch_size (B), sequence_length (S), hidden_dim (D)).
            conditioning_embedding: Combined timestep + encoder conditioning (batch_size, hidden_dim).

        Returns:
            Predicted tensor (batch_size, sequence_length, output_dim).
        """
        modulated_states = self.ada_norm(hidden_states, conditioning_embedding) # (B, S, D)
        return self.output_linear(modulated_states) # (B, S, output_dim)


    def reset_parameters(self) -> None:
        """Reset parameters to zeros (DiT initialization)."""
        nn.init.zeros_(self.output_linear.weight)
        if self.output_linear.bias is not None:
            nn.init.zeros_(self.output_linear.bias)
