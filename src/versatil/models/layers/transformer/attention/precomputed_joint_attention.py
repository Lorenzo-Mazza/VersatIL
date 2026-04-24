"""Joint attention where secondary Q/K/V are provided externally."""

import torch
import torch.nn as nn

from versatil.models.layers.positional_encoding.rotary import RotaryPositionalEncoding
from versatil.models.layers.transformer.attention.joint_attention_base import (
    JointAttentionBase,
)
from versatil.models.layers.transformer.attention.query_key_norm import QueryKeyNorm
from versatil.models.layers.transformer.cache.conditioning import (
    ConditioningLayerCache,
)
from versatil.models.layers.transformer.positional_encoding import (
    apply_rope_positional_encoding,
)


class PrecomputedPrimaryJointAttention(JointAttentionBase):
    """Joint attention where the secondary stream's Q/K/V is precomputed externally.

    Only the primary stream has learned projections and output
    projection. The secondary attention output is returned raw for external post-processing.

    Note:
        This computation method is used by VLA decoders (Pi0, SmolVLA) whose secondary observation stream
        is extracted by a VLM backbone.
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
            number_of_heads: Number of query attention heads for both streams.
            secondary_embedding_dimension: Hidden dimension for the secondary stream.
                Used to derive head_dimension if not provided.
            number_of_key_value_heads: Number of key/value heads for GQA.
                Defaults to ``number_of_heads``.
            head_dimension: Per-head dimension. Defaults to
                ``secondary_embedding_dimension // number_of_heads``.
            dropout: Dropout rate for attention weights.
            use_query_key_norm: Whether to apply QK-normalization to the primary stream.
            normalization_epsilon: Epsilon for normalization layers.
            bias: Whether to use bias in projections.
        """
        number_of_key_value_heads = number_of_key_value_heads or number_of_heads
        if head_dimension is None:
            if secondary_embedding_dimension % number_of_heads != 0:
                raise ValueError(
                    f"secondary_embedding_dimension ({secondary_embedding_dimension}) "
                    "must be divisible by number_of_heads "
                    f"({number_of_heads}) when head_dimension is not provided."
                )
            head_dimension = secondary_embedding_dimension // number_of_heads
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

        self.query_projection_primary = nn.Linear(
            primary_embedding_dimension, query_dimension, bias=bias
        )
        self.key_projection_primary = nn.Linear(
            primary_embedding_dimension, key_value_dimension, bias=bias
        )
        self.value_projection_primary = nn.Linear(
            primary_embedding_dimension, key_value_dimension, bias=bias
        )
        self.output_projection_primary = nn.Linear(
            query_dimension, primary_embedding_dimension, bias=bias
        )
        self.output_projection_primary.SQUARE_ROOT_WEIGHT = True

        if use_query_key_norm:
            self.query_key_norm_primary = QueryKeyNorm(
                head_dimension, epsilon=normalization_epsilon
            )

    def forward(
        self,
        hidden_states_primary: torch.Tensor,
        conditioning_cache: ConditioningLayerCache,
        attention_mask_primary: torch.Tensor | None = None,
        attention_mask_secondary: torch.Tensor | None = None,
        joint_attention_mask: torch.Tensor | None = None,
        positional_encoding_primary: RotaryPositionalEncoding | None = None,
        precomputed_primary_rope: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute joint attention with precomputed secondary Q/K/V.

        Args:
            hidden_states_primary: Primary stream tokens (B, T, D_p).
            conditioning_cache: Precomputed secondary Q/K/V. queries, keys, values
                each shaped (B, H/KV_H, S, D_head).
            attention_mask_primary: Padding mask (B, T), True = masked.
            attention_mask_secondary: Padding mask (B, S), True = masked.
            joint_attention_mask: Pre-built joint mask (B, 1, T+S, T+S).
            positional_encoding_primary: Optional RoPE layer for primary stream.
            precomputed_primary_rope: Pre-computed (cos, sin) for the primary
                stream. Applied via half-rotation instead of RoPE module.

        Returns:
            Tuple of (projected_primary_output (B, T, D_p),
            raw_secondary_output (B, S, H*D_head)).
        """
        query_primary = self._reshape_for_query(
            self.query_projection_primary(hidden_states_primary)
        )  # (B, H, T, D_head)
        key_primary = self._reshape_for_key_value(
            self.key_projection_primary(hidden_states_primary)
        )  # (B, KV_H, T, D_head)
        value_primary = self._reshape_for_key_value(
            self.value_projection_primary(hidden_states_primary)
        )  # (B, KV_H, T, D_head)
        if conditioning_cache.queries is None:
            raise ValueError(
                "conditioning_cache.queries must be provided for precomputed "
                "joint attention."
            )
        query_secondary = conditioning_cache.queries
        key_secondary = conditioning_cache.keys
        value_secondary = conditioning_cache.values
        if precomputed_primary_rope is not None:
            cos_primary, sin_primary = precomputed_primary_rope
            query_primary = RotaryPositionalEncoding.apply_rotation_half(
                tensor=query_primary, sine=sin_primary, cosine=cos_primary
            )
            key_primary = RotaryPositionalEncoding.apply_rotation_half(
                tensor=key_primary, sine=sin_primary, cosine=cos_primary
            )
        elif positional_encoding_primary is not None:
            query_primary, key_primary = apply_rope_positional_encoding(
                queries=query_primary,
                keys=key_primary,
                positional_encoding=positional_encoding_primary,
                cache_position=0,
            )

        if self.use_query_key_norm:
            query_primary, key_primary = self.query_key_norm_primary(
                query_primary, key_primary
            )
        sequence_length_primary = hidden_states_primary.shape[1]
        sequence_length_secondary = query_secondary.shape[2]
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
        output_primary = self.output_projection_primary(attention_output_primary)
        return output_primary, attention_output_secondary
