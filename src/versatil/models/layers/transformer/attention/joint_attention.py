"""Joint attention with full Q/K/V projections for both streams."""

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


class JointAttention(JointAttentionBase):
    """Dual-stream joint attention where both streams have Q/K/V projections.

    Keys and values from both streams are concatenated, allowing each
    stream to attend to both itself and the other stream.
    """

    def __init__(
        self,
        primary_embedding_dimension: int,
        number_of_heads: int,
        secondary_embedding_dimension: int | None = None,
        number_of_key_value_heads: int | None = None,
        head_dimension: int | None = None,
        dropout: float = 0.0,
        use_query_key_norm: bool = True,
        normalization_epsilon: float = 1e-6,
        bias: bool = True,
    ):
        """Initialize JointAttention.

        Args:
            primary_embedding_dimension: Hidden dimension for the primary stream.
            number_of_heads: Number of query attention heads for both streams.
            secondary_embedding_dimension: Hidden dimension for the secondary stream.
                Defaults to ``primary_embedding_dimension``.
            number_of_key_value_heads: Number of key/value heads for GQA.
                Defaults to ``number_of_heads``.
            head_dimension: Per-head dimension. Defaults to
                ``primary_embedding_dimension // number_of_heads``.
            dropout: Dropout rate for attention weights.
            use_query_key_norm: Whether to apply QK-normalization.
            normalization_epsilon: Epsilon for normalization layers.
            bias: Whether to use bias in projections.
        """
        secondary_embedding_dimension = (
            secondary_embedding_dimension or primary_embedding_dimension
        )
        number_of_key_value_heads = number_of_key_value_heads or number_of_heads
        if head_dimension is None:
            if primary_embedding_dimension % number_of_heads != 0:
                raise ValueError(
                    f"primary_embedding_dimension ({primary_embedding_dimension}) "
                    "must be divisible by number_of_heads "
                    f"({number_of_heads}) when head_dimension is not provided."
                )
            head_dimension = primary_embedding_dimension // number_of_heads
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
        self.output_projection_primary.SQUARE_ROOT_WEIGHT = True
        self.output_projection_secondary.SQUARE_ROOT_WEIGHT = True

        if use_query_key_norm:
            self.query_key_norm_primary = QueryKeyNorm(
                head_dimension, epsilon=normalization_epsilon
            )
            self.query_key_norm_secondary = QueryKeyNorm(
                head_dimension, epsilon=normalization_epsilon
            )

    def forward(
        self,
        hidden_states_primary: torch.Tensor,
        hidden_states_secondary: torch.Tensor,
        attention_mask_primary: torch.Tensor | None = None,
        attention_mask_secondary: torch.Tensor | None = None,
        joint_attention_mask: torch.Tensor | None = None,
        positional_encoding_primary: RotaryPositionalEncoding | None = None,
        positional_encoding_secondary: RotaryPositionalEncoding | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute joint attention for both streams.

        Args:
            hidden_states_primary: Primary stream tokens (B, S, D_p).
            hidden_states_secondary: Secondary stream tokens (B, T, D_s).
            attention_mask_primary: Padding mask (B, S), True = masked.
            attention_mask_secondary: Padding mask (B, T), True = masked.
            joint_attention_mask: Pre-built joint mask (B, 1, S+T, S+T).
            positional_encoding_primary: Optional RoPE for primary stream.
            positional_encoding_secondary: Optional RoPE for secondary stream.

        Returns:
            Tuple of (primary_output (B, S, D_p), secondary_output (B, T, D_s)).
        """
        query_primary = self._reshape_for_query(
            self.query_projection_primary(hidden_states_primary)
        )  # (B, H, S, D_head)
        key_primary = self._reshape_for_key_value(
            self.key_projection_primary(hidden_states_primary)
        )  # (B, KV_H, S, D_head)
        value_primary = self._reshape_for_key_value(
            self.value_projection_primary(hidden_states_primary)
        )  # (B, KV_H, S, D_head)
        query_secondary = self._reshape_for_query(
            self.query_projection_secondary(hidden_states_secondary)
        )  # (B, H, T, D_head)
        key_secondary = self._reshape_for_key_value(
            self.key_projection_secondary(hidden_states_secondary)
        )  # (B, KV_H, T, D_head)
        value_secondary = self._reshape_for_key_value(
            self.value_projection_secondary(hidden_states_secondary)
        )  # (B, KV_H, T, D_head)

        sequence_length_primary = hidden_states_primary.shape[1]
        sequence_length_secondary = hidden_states_secondary.shape[1]
        if self.use_query_key_norm:
            query_primary, key_primary = self.query_key_norm_primary(
                query_primary, key_primary
            )
            query_secondary, key_secondary = self.query_key_norm_secondary(
                query_secondary, key_secondary
            )

        if positional_encoding_primary is not None:
            query_primary, key_primary = apply_rope_positional_encoding(
                queries=query_primary,
                keys=key_primary,
                positional_encoding=positional_encoding_primary,
                cache_position=0,
            )
        if positional_encoding_secondary is not None:
            # Both streams share one joint softmax, so they must live in one
            # position space: secondary tokens continue after the primary
            # sequence. Restarting at 0 would give cross-stream logits
            # fictional relative distances (primary token i and secondary
            # token i would collide at the same position).
            query_secondary, key_secondary = apply_rope_positional_encoding(
                queries=query_secondary,
                keys=key_secondary,
                positional_encoding=positional_encoding_secondary,
                cache_position=sequence_length_primary,
            )
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
        output_secondary = self.output_projection_secondary(attention_output_secondary)
        return output_primary, output_secondary
