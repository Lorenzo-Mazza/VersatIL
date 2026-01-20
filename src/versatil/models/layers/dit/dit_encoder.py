"""DiT transformer encoder that returns intermediate layer outputs.

This encoder processes observation tokens and returns outputs from each layer
for hierarchical conditioning of the decoder.
"""

import copy
from typing import List

import torch
import torch.nn as nn

from versatil.models.layers.dit.dit_encoder_layer import DiTEncoderLayer

class DiTEncoder(nn.Module):
    """Stack of DiT encoder layers."""

    def __init__(self, base_block: DiTEncoderLayer, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(base_block) for _ in range(num_layers)])

        # Reset parameters for each layer
        for layer in self.layers:
            layer.reset_parameters()

    def forward(
        self, source_tensor: torch.Tensor, positional_embedding: torch.Tensor
    ) -> List[torch.Tensor]:
        """Apply the encoder layers.

        Args:
            source_tensor: Input tensor (sequence_length, batch_size, embedding_dim).
            positional_embedding: Positional embedding (sequence_length, batch_size, embedding_dim).

        Returns:
            List of outputs from each layer, each (sequence_length, batch_size, embedding_dim).
        """
        current_tensor = source_tensor
        layer_outputs = []
        for layer in self.layers:
            current_tensor = layer(current_tensor, positional_embedding)
            layer_outputs.append(current_tensor)
        return layer_outputs

