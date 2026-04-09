"""Base transformer block: normalization -> operation -> gated residual."""

import abc

import torch
import torch.nn as nn

from versatil.models.layers.normalization.typedefs import BlockNormalization


class TransformerBlock(abc.ABC, nn.Module):
    """Composable base building block for transformer layers.

    Subclasses implement a specific operation (attention, feedforward) wrapped
    in normalization and a gated residual connection. The normalization module
    determines the conditioning behavior:
    - UnconditionedNorm: ignores condition, gate = ones(1)
    - AdaNorm: conditioned via (x, condition) -> (normed, gate)
    """

    def __init__(
        self,
        normalization: BlockNormalization,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.normalization = normalization
        self.residual_dropout = nn.Dropout(dropout)

    def apply_residual(
        self,
        residual: torch.Tensor,
        output: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        """Gated residual: residual + gate * dropout(output)."""
        return residual + gate * self.residual_dropout(output)
