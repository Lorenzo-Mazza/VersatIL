"""DiT transformer decoder with modulation-based conditioning.

This decoder processes action tokens and conditions each layer on
corresponding encoder layer outputs using modulation instead of cross-attention.
"""

import copy
from typing import List

import torch
import torch.nn as nn

from refactoring.models.layers.dit.dit_decoder_layer import DiTDecoderLayer
from refactoring.models.layers.dit.final_prediction_layer import FinalPredictionLayer


class DiTDecoder(nn.Module):
    """Stack of DiT decoder blocks with final prediction layer.

    Each decoder layer is conditioned on corresponding encoder layer output
    via modulation. The final layer predicts noise (epsilon) for diffusion.
    """

    def __init__(
        self,
        base_block: DiTDecoderLayer,
        num_layers: int,
        action_dim: int,
        hidden_dim: int,
    ) -> None:
        """Initialize the DiTDecoder.

        Args:
            base_block: The base DiTDecoderLayer to copy.
            num_layers: Number of decoder layers.
            action_dim: Action dimensionality for final output.
            hidden_dim: Hidden dimensionality.
        """
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(base_block) for _ in range(num_layers)])

        # Reset parameters for each layer
        for layer in self.layers:
            layer.reset_parameters()

        # Final prediction layer
        self.final_prediction_layer = FinalPredictionLayer(hidden_dim, action_dim)

    def forward(
        self,
        source_tensor: torch.Tensor,
        timestep_embedding: torch.Tensor,
        all_condition_tensors: List[torch.Tensor],
    ) -> torch.Tensor:
        """Apply the decoder layers.

        Args:
            source_tensor: Input tensor (sequence_length, batch_size, embedding_dim).
            timestep_embedding: Timestep embedding (batch_size, embedding_dim).
            all_condition_tensors: List of condition tensors from encoder layers,
                each (sequence_length, batch_size, embedding_dim).

        Returns:
            Predicted noise (batch_size, sequence_length, action_dim).
        """
        current_tensor = source_tensor
        for layer, condition in zip(self.layers, all_condition_tensors):
            current_tensor = layer(current_tensor, timestep_embedding, condition)

        # Final prediction layer
        return self.final_prediction_layer(
            current_tensor, timestep_embedding, all_condition_tensors[-1]
        )

