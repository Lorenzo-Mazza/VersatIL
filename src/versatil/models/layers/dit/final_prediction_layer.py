"""Final prediction layer for DiT with adaptive layer normalization modulation.

This layer normalizes, modulates with condition, and projects to output dimension.
"""

import torch
import torch.nn as nn


class FinalPredictionLayer(nn.Module):
    """Final layer that predicts noise (epsilon) with adaptive LN modulation.

    This layer normalizes, modulates with condition, and projects to output dim.
    Uses adaptive layer normalization (adaLN) pattern similar to DiT.
    """

    def __init__(self, hidden_dim: int, output_dim: int) -> None:
        """Initialize the FinalPredictionLayer.

        Args:
            hidden_dim: Input hidden dimensionality.
            output_dim: Output dimensionality (action_dim).
        """
        super().__init__()

        # Layer norm without learnable affine parameters
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)

        # Output projection
        self.output_linear = nn.Linear(hidden_dim, output_dim, bias=True)

        # Adaptive LN modulation: produces shift and scale from condition
        self.adaptive_ln_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
        timestep_embedding: torch.Tensor,
        condition_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the output.

        Args:
            input_tensor: Input tensor (sequence_length, batch_size, hidden_dim).
            timestep_embedding: Timestep embedding (batch_size, hidden_dim).
            condition_tensor: Condition tensor (sequence_length, batch_size, hidden_dim).

        Returns:
            Predicted tensor (batch_size, sequence_length, output_dim).
        """
        # Combine condition by mean over sequence and add to timestep
        condition_mean = torch.mean(condition_tensor, dim=0)  # (B, D)
        combined_condition = condition_mean + timestep_embedding  # (B, D)

        # Compute shift and scale for modulation
        modulation_params = self.adaptive_ln_modulation(
            combined_condition
        )  # (B, 2*D)
        shift, scale = modulation_params.chunk(2, dim=1)  # Each: (B, D)

        # Apply modulation after norm
        # input_tensor: (S, B, D), shift/scale: (B, D) -> broadcast to (1, B, D)
        normalized = self.final_norm(input_tensor)  # (S, B, D)
        modulated = normalized * scale[None] + shift[None]  # (S, B, D)

        # Project to output dimension
        output = self.output_linear(modulated)  # (S, B, output_dim)

        # Transpose to batch-first: (B, S, output_dim)
        return output.transpose(0, 1)

    def reset_parameters(self) -> None:
        """Reset parameters to zeros (DiT initialization)."""
        for param in self.parameters():
            nn.init.zeros_(param)

