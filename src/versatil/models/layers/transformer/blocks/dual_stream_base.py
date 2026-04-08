"""Base class for dual-stream attention blocks."""

import torch
import torch.nn as nn

from versatil.models.layers.normalization.typedefs import BlockNormalization
from versatil.models.layers.transformer.blocks.feedforward import FeedforwardBlock


class DualStreamBlock(nn.Module):
    """Shared components for dual-stream blocks.

    Both full and precomputed variants have secondary-stream normalization,
    feedforward, and dropout. This base holds those modules.
    """

    def __init__(
        self,
        attention_normalization_secondary: BlockNormalization,
        feedforward_block_secondary: FeedforwardBlock,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention_normalization_secondary = attention_normalization_secondary
        self.feedforward_block_secondary = feedforward_block_secondary
        self.attention_dropout_secondary = nn.Dropout(dropout)

    def _apply_secondary_attention_residual(
        self,
        residual: torch.Tensor,
        attention_output: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        """Apply gated residual connection for secondary attention output."""
        return residual + gate * self.attention_dropout_secondary(attention_output)
