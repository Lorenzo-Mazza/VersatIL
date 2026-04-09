"""Dual-stream attention block: per-stream norm + joint attention."""

import torch
import torch.nn as nn

from versatil.models.layers.normalization.typedefs import BlockNormalization
from versatil.models.layers.positional_encoding.rotary import RotaryPositionalEncoding
from versatil.models.layers.transformer.attention.joint_attention import JointAttention
from versatil.models.layers.transformer.blocks.dual_stream_base import (
    DualStreamBlock,
)


class DualStreamAttentionBlock(DualStreamBlock):
    """Dual-stream joint attention block.

    Both streams have independent normalization and share attention through
    joint K/V concatenation. Supports optional conditioning via adaptive
    normalization.
    """

    def __init__(
        self,
        joint_attention: JointAttention,
        attention_normalization_primary: BlockNormalization,
        attention_normalization_secondary: BlockNormalization,
        dropout: float = 0.1,
    ):
        """Initialize DualStreamAttentionBlock.

        Args:
            joint_attention: JointAttention module for dual-stream attention.
            attention_normalization_primary: Normalization for primary stream.
            attention_normalization_secondary: Normalization for secondary stream.
            dropout: Dropout rate for attention residual connections.
        """
        super().__init__(
            attention_normalization_secondary=attention_normalization_secondary,
            dropout=dropout,
        )
        self.joint_attention = joint_attention
        self.attention_normalization_primary = attention_normalization_primary
        self.attention_dropout_primary = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states_primary: torch.Tensor,
        hidden_states_secondary: torch.Tensor,
        conditioning: torch.Tensor | None = None,
        attention_mask_primary: torch.Tensor | None = None,
        attention_mask_secondary: torch.Tensor | None = None,
        joint_attention_mask: torch.Tensor | None = None,
        positional_encoding_primary: RotaryPositionalEncoding | None = None,
        positional_encoding_secondary: RotaryPositionalEncoding | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through dual-stream attention block.

        Args:
            hidden_states_primary: Primary stream tokens (B, S, D_p).
            hidden_states_secondary: Secondary stream tokens (B, T, D_s).
            conditioning: Conditioning vector for adaptive normalization (B, C).
            attention_mask_primary: Padding mask (B, S), True = masked.
            attention_mask_secondary: Padding mask (B, T), True = masked.
            joint_attention_mask: Pre-built joint mask (B, 1, S+T, S+T).
            positional_encoding_primary: Optional RoPE for primary stream.
            positional_encoding_secondary: Optional RoPE for secondary stream.

        Returns:
            Tuple of (primary_output (B, S, D_p), secondary_output (B, T, D_s)).
        """
        residual_primary = hidden_states_primary
        residual_secondary = hidden_states_secondary
        normed_primary, gate_primary = self.attention_normalization_primary(
            x=hidden_states_primary, condition=conditioning
        )
        normed_secondary, gate_secondary = self.attention_normalization_secondary(
            x=hidden_states_secondary, condition=conditioning
        )
        attention_output_primary, attention_output_secondary = self.joint_attention(
            hidden_states_primary=normed_primary,
            hidden_states_secondary=normed_secondary,
            attention_mask_primary=attention_mask_primary,
            attention_mask_secondary=attention_mask_secondary,
            joint_attention_mask=joint_attention_mask,
            positional_encoding_primary=positional_encoding_primary,
            positional_encoding_secondary=positional_encoding_secondary,
        )
        hidden_states_primary = (
            residual_primary
            + gate_primary * self.attention_dropout_primary(attention_output_primary)
        )
        hidden_states_secondary = self._apply_secondary_attention_residual(
            residual=residual_secondary,
            attention_output=attention_output_secondary,
            gate=gate_secondary,
        )
        return hidden_states_primary, hidden_states_secondary
