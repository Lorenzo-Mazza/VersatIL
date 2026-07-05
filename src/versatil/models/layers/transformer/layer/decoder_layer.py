"""General transformer decoder layer with KV cache support."""

import torch
import torch.nn as nn

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.normalization.factory import create_block_normalization
from versatil.models.layers.positional_encoding.rotary import (
    RotaryPositionalEncoding,
)
from versatil.models.layers.transformer.attention.cached_attention import (
    CachedAttention,
)
from versatil.models.layers.transformer.block.cross_attention import (
    CrossAttentionBlock,
)
from versatil.models.layers.transformer.block.feedforward import (
    FeedforwardBlock,
    build_feedforward,
)
from versatil.models.layers.transformer.block.self_attention import (
    SelfAttentionBlock,
)
from versatil.models.layers.transformer.cache.conditioning import ConditioningLayerCache
from versatil.models.layers.transformer.cache.generation import GenerationLayerCache


class TransformerDecoderLayer(nn.Module):
    """Self-attention + optional cross-attention + feedforward blocks.

    Note:
        Supports generation caching for autoregressive decoding and conditioning
        caching for static context reuse.
        Optionally supports conditioning via adaptive normalization and cross-attention.
    """

    def __init__(
        self,
        embedding_dimension: int,
        number_of_heads: int,
        number_of_key_value_heads: int | None = None,
        feedforward_dimension: int | None = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        activation: str = ActivationFunction.SWIGLU.value,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        attention_type: str = AttentionType.GROUPED_QUERY.value,
        use_cross_attention: bool = True,
        bias: bool = True,
        normalization_epsilon: float = 1e-6,
        conditioning_dimension: int | None = None,
        use_gating: bool = False,
        cross_attention_normalization_type: str | None = None,
        cross_attention_conditioning_dimension: int | None = None,
    ):
        """Initialize Transformer decoder layer.

        Args:
            embedding_dimension: Model embedding dimension.
            number_of_heads: Number of attention heads.
            number_of_key_value_heads: Number of K/V heads (for GQA).
            feedforward_dimension: FFN hidden dimension (defaults to 4 * embedding_dimension).
            dropout: Dropout probability for residual connections.
            attention_dropout: Dropout probability for attention weights.
            activation: Activation function (use ActivationFunction enum values).
            normalization_type: Type of normalization (use NormalizationType enum values).
            attention_type: Type of attention (use AttentionType enum values).
            use_cross_attention: Whether to include cross-attention block.
            bias: Whether to use bias in linear layers.
            normalization_epsilon: Epsilon for normalization layers.
            conditioning_dimension: Conditioning dimension for adaptive normalization.
                When set, wraps normalization in AdaNorm.
            use_gating: Whether to use gating in adaptive normalization (AdaLN-Zero).
            cross_attention_normalization_type: Normalization type for the cross-attention
                block. Defaults to ``normalization_type`` when None.
            cross_attention_conditioning_dimension: Conditioning dimension for
                cross-attention normalization. None means no conditioning.
        """
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        self.use_cross_attention = use_cross_attention
        if feedforward_dimension is None:
            feedforward_dimension = 4 * embedding_dimension
        self.self_attention_block = SelfAttentionBlock(
            attention=CachedAttention(
                embedding_dimension=embedding_dimension,
                number_of_heads=number_of_heads,
                number_of_key_value_heads=number_of_key_value_heads,
                dropout=attention_dropout,
                bias=bias,
                attention_type=attention_type,
            ),
            normalization=create_block_normalization(
                normalization_type=normalization_type,
                dimension=embedding_dimension,
                epsilon=normalization_epsilon,
                conditioning_dimension=conditioning_dimension,
                use_gating=use_gating,
            ),
            dropout=dropout,
        )
        if use_cross_attention:
            self.cross_attention_block = CrossAttentionBlock(
                attention=CachedAttention(
                    embedding_dimension=embedding_dimension,
                    number_of_heads=number_of_heads,
                    number_of_key_value_heads=number_of_key_value_heads,
                    dropout=attention_dropout,
                    bias=bias,
                    attention_type=attention_type,
                ),
                normalization=create_block_normalization(
                    normalization_type=cross_attention_normalization_type
                    or normalization_type,
                    dimension=embedding_dimension,
                    epsilon=normalization_epsilon,
                    conditioning_dimension=cross_attention_conditioning_dimension,
                    use_gating=use_gating
                    if cross_attention_conditioning_dimension is not None
                    else False,
                ),
                dropout=dropout,
            )
        else:
            self.cross_attention_block = None
        self.feedforward_block = FeedforwardBlock(
            feedforward=build_feedforward(
                embedding_dimension=embedding_dimension,
                feedforward_dimension=feedforward_dimension,
                activation=activation,
                dropout=dropout,
                bias=bias,
            ),
            normalization=create_block_normalization(
                normalization_type=normalization_type,
                dimension=embedding_dimension,
                epsilon=normalization_epsilon,
                conditioning_dimension=conditioning_dimension,
                use_gating=use_gating,
            ),
            dropout=dropout,
        )

    def precompute_conditioning_kv(
        self, encoded_features: torch.Tensor
    ) -> ConditioningLayerCache | None:
        """Precompute conditioning K/V for this layer's cross-attention block.

        Args:
            encoded_features: Encoder features (B, memory_length, D).

        Returns:
            ConditioningLayerCache if this layer has cross-attention, None otherwise.
        """
        if self.cross_attention_block is None:
            return None
        else:
            return self.cross_attention_block.precompute_kv(encoded_features)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoded_features: torch.Tensor | None = None,
        self_attention_mask: torch.Tensor | None = None,
        cross_attention_mask: torch.Tensor | None = None,
        generation_cache: GenerationLayerCache | None = None,
        conditioning_cache: ConditioningLayerCache | None = None,
        positional_encoding: RotaryPositionalEncoding | None = None,
        conditioning: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, GenerationLayerCache | None]:
        """Forward pass through decoder layer.

        Args:
            hidden_states: Input embeddings (B, T, D).
            encoded_features: Encoder output for cross-attention (B, S, D).
                Required when use_cross_attention=True and no conditioning_cache.
            self_attention_mask: Causal mask (B, 1, T, T), True = masked.
            cross_attention_mask: Cross-attention mask (B, 1, T, S), True = masked.
            generation_cache: Cached K/V from the main sequence. When provided,
                an updated cache is returned.
            conditioning_cache: Precomputed K/V for static conditioning.
            positional_encoding: Optional rotary positional encoding module.
            conditioning: Conditioning vector for adaptive normalization (B, C).

        Returns:
            Tuple of (output hidden states (B, T, D), updated GenerationLayerCache or None).

        Raises:
            ValueError: if cross-attention is enabled without encoded_features or conditioning_cache.
        """
        if self.use_cross_attention and (
            encoded_features is None and conditioning_cache is None
        ):
            raise ValueError(
                "Either encoded_features or conditioning_cache must be provided when using cross-attention"
            )
        hidden_states, new_cache = self.self_attention_block(
            hidden_states=hidden_states,
            conditioning=conditioning,
            attention_mask=self_attention_mask,
            positional_encoding=positional_encoding,
            generation_cache=generation_cache,
        )
        if self.use_cross_attention:
            hidden_states = self.cross_attention_block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoded_features,
                conditioning=conditioning,
                attention_mask=cross_attention_mask,
                conditioning_cache=conditioning_cache,
            )
        hidden_states = self.feedforward_block(
            hidden_states=hidden_states,
            conditioning=conditioning,
        )
        return hidden_states, new_cache
