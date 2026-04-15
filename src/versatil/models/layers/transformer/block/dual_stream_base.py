"""Base class for dual-stream attention blocks."""

import torch
import torch.nn as nn

from versatil.models.layers.normalization.typedefs import BlockNormalization


class DualStreamBlock(nn.Module):
    """Shared components for dual-stream attention blocks.

    Both full and secondary-stream-precomputed variants have primary-stream normalization
    and dropout. This base holds those modules.
    """

    def __init__(
        self, attention_normalization_primary: BlockNormalization, dropout: float = 0.1
    ):
        super().__init__()
        self.attention_normalization_primary = attention_normalization_primary
        self.attention_dropout_primary = nn.Dropout(dropout)

    def _apply_primary_attention_residual(
        self,
        residual: torch.Tensor,
        attention_output: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        """Apply gated residual connection for primary attention output."""
        return residual + gate * self.attention_dropout_primary(attention_output)
