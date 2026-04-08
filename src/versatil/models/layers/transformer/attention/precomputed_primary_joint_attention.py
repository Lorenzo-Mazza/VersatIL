"""Joint attention where primary Q/K/V are provided externally."""

import torch
import torch.nn as nn

from versatil.models.layers.positional_encoding.rotary import RotaryPositionalEncoding
from versatil.models.layers.transformer.attention.joint_attention_base import (
    JointAttentionBase,
)
from versatil.models.layers.transformer.attention.query_key_norm import QueryKeyNorm
from versatil.models.layers.transformer.positional_encoding import (
    apply_rope_positional_encoding,
)


class PrecomputedPrimaryJointAttention(JointAttentionBase):
    """Joint attention where the primary stream's Q/K/V come from an external backbone.

    Only the secondary stream has learned projections and output
    projection. The primary attention output is returned raw (no O-projection)
    for external post-processing by the backbone.

    Used by VLA decoders (Pi0, SmolVLA) where the primary stream is a
    frozen VLM whose Q/K/V are extracted by the backbone.
    """

    def __init__(
        self,
        primary_embedding_dimension: int,
        number_of_heads: int,
        secondary_embedding_dimension: int,
        number_of_key_value_heads: int | None = None,
        head_dimension: int | None = None,
        dropout: float = 0.0,
        use_query_key_norm: bool = True,
        normalization_epsilon: float = 1e-6,
        bias: bool = True,
    ):
        """Initialize PrecomputedPrimaryJointAttention.

        Args:
            primary_embedding_dimension: Hidden dimension for the primary stream.
                Used to derive head_dimension if not provided.
            number_of_heads: Number of query attention heads for both streams.
            secondary_embedding_dimension: Hidden dimension for the secondary stream.
            number_of_key_value_heads: Number of key/value heads for GQA.
                Defaults to ``number_of_heads``.
            head_dimension: Per-head dimension. Defaults to
                ``primary_embedding_dimension // number_of_heads``.
            dropout: Dropout rate for attention weights.
            use_query_key_norm: Whether to apply QK-normalization to the secondary stream.
            normalization_epsilon: Epsilon for normalization layers.
            bias: Whether to use bias in projections.
        """
        number_of_key_value_heads = number_of_key_value_heads or number_of_heads
        head_dimension = (
            head_dimension or primary_embedding_dimension // number_of_heads
        )
        super().__init__(
            number_of_heads=number_of_heads,
            number_of_key_value_heads=number_of_key_value_heads,
            head_dimension=head_dimension,
            dropout=dropout,
        )
        self.primary_embedding_dimension = primary_embedding_dimension
        self.secondary_embedding_dimension = secondary_embedding_dimension
        self.use_query_key_norm = use_query_key_norm
        query_dimension = number_of_heads * head_dimension
        key_value_dimension = number_of_key_value_heads * head_dimension

        self.query_projection_secondary = nn.Linear(
            secondary_embedding_dimension, query_dimension, bias=bias
        )
        self.key_projection_secondary = nn.Linear(
            secondary_embedding_dimension, key_value_dimension, bias=bias
        )
        self.value_projection_secondary = nn.Linear(
            secondary_embedding_dimension, key_value_dimension, bias=bias
        )
        self.output_projection_secondary = nn.Linear(
            query_dimension, secondary_embedding_dimension, bias=bias
        )
        self.output_projection_secondary.SQUARE_ROOT_WEIGHT = True

        if use_query_key_norm:
            self.query_key_norm_secondary = QueryKeyNorm(
                head_dimension, epsilon=normalization_epsilon
            )

    def forward(
        self,
        precomputed_primary: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        hidden_states_secondary: torch.Tensor,
        attention_mask_primary: torch.Tensor | None = None,
        attention_mask_secondary: torch.Tensor | None = None,
        joint_attention_mask: torch.Tensor | None = None,
        positional_encoding_secondary: RotaryPositionalEncoding | None = None,
        precomputed_secondary_rope: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute joint attention with precomputed primary Q/K/V.

        Args:
            precomputed_primary: Pre-projected primary (Q, K, V) tuple,
                each shaped (B, H/KV_H, S, D_head).
            hidden_states_secondary: Secondary stream tokens (B, T, D_s).
            attention_mask_primary: Padding mask (B, S), True = masked.
            attention_mask_secondary: Padding mask (B, T), True = masked.
            joint_attention_mask: Pre-built joint mask (B, 1, S+T, S+T).
            positional_encoding_secondary: Optional RoPE for secondary stream.
            precomputed_secondary_rope: Pre-computed (cos, sin) for the secondary
                stream. Applied via half-rotation instead of RoPE module.

        Returns:
            Tuple of (raw_primary_output (B, S, H*D_head),
            projected_secondary_output (B, T, D_s)).
        """
        query_primary, key_primary, value_primary = precomputed_primary
        query_secondary = self._reshape_for_query(
            self.query_projection_secondary(hidden_states_secondary)
        )  # (B, H, T, D_head)
        key_secondary = self._reshape_for_key_value(
            self.key_projection_secondary(hidden_states_secondary)
        )  # (B, KV_H, T, D_head)
        value_secondary = self._reshape_for_key_value(
            self.value_projection_secondary(hidden_states_secondary)
        )  # (B, KV_H, T, D_head)

        if precomputed_secondary_rope is not None:
            cos_secondary, sin_secondary = precomputed_secondary_rope
            query_secondary = RotaryPositionalEncoding.apply_rotation_half(
                query_secondary, sin_secondary, cos_secondary
            )
            key_secondary = RotaryPositionalEncoding.apply_rotation_half(
                key_secondary, sin_secondary, cos_secondary
            )
        elif positional_encoding_secondary is not None:
            query_secondary, key_secondary = apply_rope_positional_encoding(
                queries=query_secondary,
                keys=key_secondary,
                positional_encoding=positional_encoding_secondary,
                cache_position=0,
            )

        if self.use_query_key_norm:
            query_secondary, key_secondary = self.query_key_norm_secondary(
                query_secondary, key_secondary
            )

        sequence_length_primary = query_primary.shape[2]
        sequence_length_secondary = hidden_states_secondary.shape[1]
        attention_output_primary, attention_output_secondary = self._joint_sdpa(
            query_primary=query_primary,
            key_primary=key_primary,
            value_primary=value_primary,
            query_secondary=query_secondary,
            key_secondary=key_secondary,
            value_secondary=value_secondary,
            sequence_length_primary=sequence_length_primary,
            sequence_length_secondary=sequence_length_secondary,
            attention_mask_primary=attention_mask_primary,
            attention_mask_secondary=attention_mask_secondary,
            joint_attention_mask=joint_attention_mask,
        )
        # Primary output returned raw — no O-projection (handled by backbone)
        output_secondary = self.output_projection_secondary(attention_output_secondary)
        return attention_output_primary, output_secondary
