"""Dual-stream attention block where secondary Q/K/V are precomputed externally."""

import torch

from versatil.models.layers.normalization.typedefs import BlockNormalization
from versatil.models.layers.positional_encoding.rotary import RotaryPositionalEncoding
from versatil.models.layers.transformer.attention.precomputed_joint_attention import (
    PrecomputedPrimaryJointAttention,
)
from versatil.models.layers.transformer.block.dual_stream_base import (
    DualStreamBlock,
)
from versatil.models.layers.transformer.cache.conditioning import (
    ConditioningLayerCache,
)


class PrecomputedDualStreamAttentionBlock(DualStreamBlock):
    """Dual-stream attention block where secondary Q/K/V are precomputed externally.

    Only the primary stream has normalization. The secondary attention output
    is returned raw for external post-processing.
    """

    def __init__(
        self,
        joint_attention: PrecomputedPrimaryJointAttention,
        attention_normalization_primary: BlockNormalization,
        dropout: float = 0.1,
    ):
        """Initialize PrecomputedDualStreamAttentionBlock.

        Args:
            joint_attention: PrecomputedPrimaryJointAttention module.
            attention_normalization_primary: Normalization for primary stream.
            dropout: Dropout rate for attention residual connections.
        """
        super().__init__(
            attention_normalization_primary=attention_normalization_primary,
            dropout=dropout,
        )
        self.joint_attention = joint_attention

    def forward(
        self,
        hidden_states_primary: torch.Tensor,
        conditioning_cache: ConditioningLayerCache,
        conditioning: torch.Tensor | None = None,
        attention_mask_primary: torch.Tensor | None = None,
        attention_mask_secondary: torch.Tensor | None = None,
        joint_attention_mask: torch.Tensor | None = None,
        positional_encoding_primary: RotaryPositionalEncoding | None = None,
        precomputed_primary_rope: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with precomputed secondary Q/K/V from conditioning cache.

        Args:
            hidden_states_primary: Primary stream tokens (B, T, D_p).
            conditioning_cache: Precomputed secondary Q/K/V. queries, keys, values
                each shaped (B, H/KV_H, S, D_head).
            conditioning: Conditioning vector for adaptive normalization (B, C).
            attention_mask_primary: Padding mask (B, T), True = masked.
            attention_mask_secondary: Padding mask (B, S), True = masked.
            joint_attention_mask: Pre-built joint mask (B, 1, T+S, T+S).
            positional_encoding_primary: Optional RoPE for primary stream.
            precomputed_primary_rope: Pre-computed (cos, sin) rotary positional
              encodings for primary stream positions.

        Returns:
            Tuple of (projected_primary_output (B, T, D_p),
            raw_secondary_output (B, S, H*D_head)).
        """
        residual_primary = hidden_states_primary
        normed_primary, gate_primary = self.attention_normalization_primary(
            x=hidden_states_primary, condition=conditioning
        )
        attention_output_primary, attention_output_secondary = self.joint_attention(
            hidden_states_primary=normed_primary,
            conditioning_cache=conditioning_cache,
            attention_mask_primary=attention_mask_primary,
            attention_mask_secondary=attention_mask_secondary,
            joint_attention_mask=joint_attention_mask,
            positional_encoding_primary=positional_encoding_primary,
            precomputed_primary_rope=precomputed_primary_rope,
        )
        hidden_states_primary = self._apply_primary_attention_residual(
            residual=residual_primary,
            attention_output=attention_output_primary,
            gate=gate_primary,
        )
        return hidden_states_primary, attention_output_secondary
