"""Dual-stream layer where the secondary stream has precomputed Q/K/V: joint attention + primary FFN."""

import torch
import torch.nn as nn

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.normalization.factory import create_block_normalization
from versatil.models.layers.transformer.attention.precomputed_joint_attention import (
    PrecomputedPrimaryJointAttention,
)
from versatil.models.layers.transformer.block.feedforward import (
    FeedforwardBlock,
    build_feedforward,
)
from versatil.models.layers.transformer.block.precomputed_dual_stream_attention import (
    PrecomputedDualStreamAttentionBlock,
)
from versatil.models.layers.transformer.cache.conditioning import (
    ConditioningLayerCache,
)


class PrecomputedDualStreamLayer(nn.Module):
    """Joint attention with precomputed secondary Q/K/V, plus primary feedforward.

    The secondary stream provides pre-projected Q/K/V from an external source.
    Only the primary stream has learnable normalization and feedforward.
    """

    def __init__(
        self,
        primary_embedding_dimension: int,
        secondary_embedding_dimension: int,
        number_of_heads: int,
        number_of_key_value_heads: int,
        head_dimension: int,
        primary_feedforward_dimension: int,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        conditioning_dimension: int | None = None,
        use_gating: bool = False,
        dropout: float = 0.1,
        activation: str = ActivationFunction.SILU.value,
        bias: bool = False,
        use_query_key_norm: bool = False,
    ):
        """Initialize PrecomputedDualStreamLayer.

        Args:
            primary_embedding_dimension: Primary stream embedding dimension.
            secondary_embedding_dimension: Secondary stream embedding dimension.
            number_of_heads: Number of attention heads.
            number_of_key_value_heads: Number of K/V heads.
            head_dimension: Dimension per attention head.
            primary_feedforward_dimension: FFN hidden dimension for primary stream.
            normalization_type: Normalization type for primary stream.
            conditioning_dimension: Conditioning dimension for adaptive normalization.
            use_gating: Whether to use gating in adaptive normalization.
            dropout: Dropout rate for residual connections.
            activation: Activation function for FFN.
            bias: Whether to use bias in linear layers.
            use_query_key_norm: Whether to apply QK-normalization.
        """
        super().__init__()
        self.attention_block = PrecomputedDualStreamAttentionBlock(
            joint_attention=PrecomputedPrimaryJointAttention(
                primary_embedding_dimension=primary_embedding_dimension,
                number_of_heads=number_of_heads,
                secondary_embedding_dimension=secondary_embedding_dimension,
                number_of_key_value_heads=number_of_key_value_heads,
                head_dimension=head_dimension,
                dropout=dropout,
                use_query_key_norm=use_query_key_norm,
                bias=bias,
            ),
            attention_normalization_primary=create_block_normalization(
                normalization_type=normalization_type,
                dimension=primary_embedding_dimension,
                condition_dim=conditioning_dimension,
                use_gating=use_gating,
            ),
            dropout=dropout,
        )
        self.feedforward_block_primary = FeedforwardBlock(
            feedforward=build_feedforward(
                embedding_dimension=primary_embedding_dimension,
                feedforward_dimension=primary_feedforward_dimension,
                activation=activation,
                dropout=dropout,
                bias=bias,
            ),
            normalization=create_block_normalization(
                normalization_type=normalization_type,
                dimension=primary_embedding_dimension,
                condition_dim=conditioning_dimension,
                use_gating=use_gating,
            ),
            dropout=dropout,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning_cache: ConditioningLayerCache,
        conditioning: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        precomputed_rope: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Forward pass with precomputed secondary Q/K/V from conditioning cache.

        Args:
            hidden_states: Primary stream tokens (B, T, D).
            conditioning_cache: Precomputed secondary Q/K/V. queries, keys, values
                each shaped (B, H/KV_H, S, D_head).
            conditioning: Conditioning vector for adaptive normalization (B, C).
            attention_mask: Pre-built joint mask (B, 1, S+T, S+T).
            precomputed_rope: Pre-computed (cos, sin) rotary positional encodings
                for primary stream positions.

        Returns:
            Processed primary stream output (B, T, D).
        """
        hidden_states, _ = self.attention_block(
            conditioning_cache=conditioning_cache,
            hidden_states_primary=hidden_states,
            conditioning=conditioning,
            joint_attention_mask=attention_mask,
            precomputed_primary_rope=precomputed_rope,
        )
        hidden_states = self.feedforward_block_primary(
            hidden_states=hidden_states, conditioning=conditioning
        )
        return hidden_states

    def forward_with_secondary(
        self,
        hidden_states_primary: torch.Tensor,
        conditioning_cache: ConditioningLayerCache,
        conditioning: torch.Tensor | None = None,
        joint_attention_mask: torch.Tensor | None = None,
        precomputed_primary_rope: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both primary hidden states and secondary attention output.

        Args:
            hidden_states_primary: Primary stream tokens (B, T, D).
            conditioning_cache: Precomputed secondary Q/K/V.
            conditioning: Conditioning vector for adaptive normalization (B, C).
            joint_attention_mask: Pre-built joint mask (B, 1, S+T, S+T).
            precomputed_primary_rope: Pre-computed (cos, sin) rotary positional encodings
                for primary stream positions.

        Returns:
            Tuple of (`processed_primary_output` (B, T, D_s),
            `raw_secondary_output` (B, S, H*D_head)).
        """
        hidden_states_primary, attention_output_secondary = self.attention_block(
            conditioning_cache=conditioning_cache,
            hidden_states_primary=hidden_states_primary,
            conditioning=conditioning,
            joint_attention_mask=joint_attention_mask,
            precomputed_primary_rope=precomputed_primary_rope,
        )
        hidden_states_primary = self.feedforward_block_primary(
            hidden_states=hidden_states_primary, conditioning=conditioning
        )
        return hidden_states_primary, attention_output_secondary
