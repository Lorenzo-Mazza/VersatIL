"""Cross-attention layer with K/V dimension projection for bridging two embedding spaces."""

import torch
import torch.nn as nn

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.normalization.factory import create_block_normalization
from versatil.models.layers.transformer.attention.cached_attention import (
    CachedAttention,
)
from versatil.models.layers.transformer.block.feedforward import (
    FeedforwardBlock,
    build_feedforward,
)
from versatil.models.layers.transformer.block.precomputed_cross_attention import (
    PrecomputedCrossAttentionBlock,
)
from versatil.models.layers.transformer.cache.conditioning import (
    ConditioningLayerCache,
)


class PrecomputedKVCrossAttentionLayer(nn.Module):
    """Projects precomputed conditioning K/V to local dimension, cross-attends, then feedforward.

    Bridges two embedding spaces via learned K/V projections. The conditioning
    cache provides precomputed K/V from an external source
    which may have a different hidden dimension. Projections map them into
    the local attention space before cross-attention.
    """

    def __init__(
        self,
        embedding_dimension: int,
        conditioning_key_value_dimension: int,
        number_of_heads: int,
        number_of_key_value_heads: int,
        head_dimension: int,
        feedforward_dimension: int,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        conditioning_dimension: int | None = None,
        use_gating: bool = False,
        dropout: float = 0.1,
        activation: str = ActivationFunction.SILU.value,
    ):
        """Initialize PrecomputedKVCrossAttentionLayer.

        Args:
            embedding_dimension: Hidden dimension of the main input stream.
            conditioning_key_value_dimension: K/V dimension from the conditioning source.
            number_of_heads: Number of attention heads.
            number_of_key_value_heads: Number of K/V heads.
            head_dimension: Dimension per attention head.
            feedforward_dimension: FFN hidden dimension.
            normalization_type: Normalization type for attention and FFN blocks.
            conditioning_dimension: Dimension of conditioning vector for adaptive norm.
            use_gating: Whether to use gating in adaptive normalization.
            dropout: Dropout rate for residual connections.
            activation: Activation function for FFN.
        """
        super().__init__()
        local_key_value_dimension = number_of_key_value_heads * head_dimension
        self.key_projection = nn.Linear(
            conditioning_key_value_dimension, local_key_value_dimension, bias=False
        )
        self.value_projection = nn.Linear(
            conditioning_key_value_dimension, local_key_value_dimension, bias=False
        )
        if number_of_key_value_heads == number_of_heads:
            attention_type = AttentionType.MULTI_HEAD.value
        else:
            attention_type = AttentionType.GROUPED_QUERY.value
        self.cross_attention_block = PrecomputedCrossAttentionBlock(
            attention=CachedAttention(
                embedding_dimension=embedding_dimension,
                number_of_heads=number_of_heads,
                number_of_key_value_heads=number_of_key_value_heads,
                head_dimension=head_dimension,
                dropout=dropout,
                bias=False,
                attention_type=attention_type,
            ),
            normalization=create_block_normalization(
                normalization_type=normalization_type,
                dimension=embedding_dimension,
                conditioning_dimension=conditioning_dimension,
                use_gating=use_gating,
            ),
            dropout=dropout,
        )
        self.feedforward_block = FeedforwardBlock(
            feedforward=build_feedforward(
                embedding_dimension=embedding_dimension,
                feedforward_dimension=feedforward_dimension,
                activation=activation,
                dropout=dropout,
                bias=False,
            ),
            normalization=create_block_normalization(
                normalization_type=normalization_type,
                dimension=embedding_dimension,
                conditioning_dimension=conditioning_dimension,
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
        """Project conditioning K/V, cross-attend with optional RoPE, then FFN.

        Args:
            hidden_states: Local stream tokens (B, S, D).
            conditioning_cache: Precomputed K/V from conditioning source.
                Keys and values have shape (B, P, conditioning_kv_dim).
            conditioning: Conditioning vector for adaptive normalization (B, C).
                Ignored when normalization is unconditioned.
            attention_mask: Optional mask (B, 1, S, P).
            precomputed_rope: Optional precomputed (cos, sin) rotary positional encodings for query.

        Returns:
            Updated hidden states (B, S, D).
        """
        projected_keys = self.key_projection(conditioning_cache.keys)
        projected_values = self.value_projection(conditioning_cache.values)
        hidden_states = self.cross_attention_block(
            hidden_states=hidden_states,
            keys=projected_keys,
            values=projected_values,
            conditioning=conditioning,
            attention_mask=attention_mask,
            precomputed_query_rope=precomputed_rope,
        )
        hidden_states = self.feedforward_block(
            hidden_states=hidden_states,
            conditioning=conditioning,
        )
        return hidden_states
