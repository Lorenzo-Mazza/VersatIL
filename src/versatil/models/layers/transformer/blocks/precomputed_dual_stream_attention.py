"""Dual-stream attention block where primary Q/K/V come from an external backbone."""

import torch

from versatil.models.layers.normalization.typedefs import BlockNormalization
from versatil.models.layers.positional_encoding.rotary import RotaryPositionalEncoding
from versatil.models.layers.transformer.attention.precomputed_primary_joint_attention import (
    PrecomputedPrimaryJointAttention,
)
from versatil.models.layers.transformer.blocks.dual_stream_base import (
    DualStreamBlock,
)
from versatil.models.layers.transformer.blocks.feedforward import FeedforwardBlock


class PrecomputedDualStreamAttentionBlock(DualStreamBlock):
    """Dual-stream attention block where primary Q/K/V are precomputed externally.

    Only the secondary stream has normalization and feedforward. The primary
    attention output is returned raw for external post-processing by the backbone.
    """

    def __init__(
        self,
        joint_attention: PrecomputedPrimaryJointAttention,
        attention_normalization_secondary: BlockNormalization,
        feedforward_block_secondary: FeedforwardBlock,
        dropout: float = 0.1,
    ):
        """Initialize PrecomputedDualStreamAttentionBlock.

        Args:
            joint_attention: PrecomputedPrimaryJointAttention module.
            attention_normalization_secondary: Normalization for secondary stream before attention.
            feedforward_block_secondary: Feedforward block for secondary stream.
            dropout: Dropout rate for attention residual connections.
        """
        super().__init__(
            attention_normalization_secondary=attention_normalization_secondary,
            feedforward_block_secondary=feedforward_block_secondary,
            dropout=dropout,
        )
        self.joint_attention = joint_attention

    def forward(
        self,
        precomputed_primary: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        hidden_states_secondary: torch.Tensor,
        conditioning: torch.Tensor | None = None,
        attention_mask_primary: torch.Tensor | None = None,
        attention_mask_secondary: torch.Tensor | None = None,
        joint_attention_mask: torch.Tensor | None = None,
        positional_encoding_secondary: RotaryPositionalEncoding | None = None,
        precomputed_secondary_rope: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with precomputed primary Q/K/V.

        Args:
            precomputed_primary: Pre-projected primary (Q, K, V) tuple,
                each shaped (B, H/KV_H, S, D_head).
            hidden_states_secondary: Secondary stream tokens (B, T, D_s).
            conditioning: Conditioning vector for adaptive normalization (B, C).
            attention_mask_primary: Padding mask (B, S), True = masked.
            attention_mask_secondary: Padding mask (B, T), True = masked.
            joint_attention_mask: Pre-built joint mask (B, 1, S+T, S+T).
            positional_encoding_secondary: Optional RoPE for secondary stream.
            precomputed_secondary_rope: Pre-computed (cos, sin) for secondary positions.

        Returns:
            Tuple of (raw_primary_output (B, S, H*D_head),
            processed_secondary_output (B, T, D_s)).
        """
        residual_secondary = hidden_states_secondary
        normed_secondary, gate_secondary = self.attention_normalization_secondary(
            x=hidden_states_secondary, condition=conditioning
        )
        attention_output_primary, attention_output_secondary = self.joint_attention(
            precomputed_primary=precomputed_primary,
            hidden_states_secondary=normed_secondary,
            attention_mask_primary=attention_mask_primary,
            attention_mask_secondary=attention_mask_secondary,
            joint_attention_mask=joint_attention_mask,
            positional_encoding_secondary=positional_encoding_secondary,
            precomputed_secondary_rope=precomputed_secondary_rope,
        )
        hidden_states_secondary = self._apply_secondary_attention_residual(
            residual=residual_secondary,
            attention_output=attention_output_secondary,
            gate=gate_secondary,
        )
        hidden_states_secondary = self.feedforward_block_secondary(
            hidden_states=hidden_states_secondary, conditioning=conditioning
        )
        return attention_output_primary, hidden_states_secondary
