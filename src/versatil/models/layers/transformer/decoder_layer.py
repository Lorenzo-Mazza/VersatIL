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
from versatil.models.layers.transformer.blocks.cross_attention import (
    CrossAttentionBlock,
)
from versatil.models.layers.transformer.blocks.feedforward import (
    FeedforwardBlock,
    build_feedforward,
)
from versatil.models.layers.transformer.blocks.self_attention import (
    SelfAttentionBlock,
)
from versatil.models.layers.transformer.kv_cache import LayerKVCache


class TransformerDecoderLayer(nn.Module):
    """Self-attention + optional cross-attention + feedforward blocks.

    Supports KV caching for autoregressive generation and optional
    conditioning via adaptive normalization types.
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
        autoregressive: bool = True,
        condition_dim: int | None = None,
        use_gating: bool = False,
        cross_attention_normalization_type: str | None = None,
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
            autoregressive: Whether the model is autoregressive (affects caching).
            condition_dim: Conditioning dimension for adaptive normalization.
                Required when normalization_type is adaptive.
            use_gating: Whether to use gating in adaptive normalization (AdaLN-Zero).
            cross_attention_normalization_type: Normalization type for the cross-attention
                block. Defaults to ``normalization_type`` when None.
        """
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        self.use_cross_attention = use_cross_attention
        self.autoregressive = autoregressive
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
                condition_dim=condition_dim,
                use_gating=use_gating,
            ),
            dropout=dropout,
        )
        cross_norm_type = cross_attention_normalization_type or normalization_type
        cross_condition_dim = condition_dim
        cross_use_gating = use_gating
        if cross_attention_normalization_type is not None:
            cross_is_adaptive = NormalizationType(cross_norm_type).is_adaptive
            if not cross_is_adaptive:
                cross_condition_dim = None
                cross_use_gating = False
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
                    normalization_type=cross_norm_type,
                    dimension=embedding_dimension,
                    epsilon=normalization_epsilon,
                    condition_dim=cross_condition_dim,
                    use_gating=cross_use_gating,
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
                condition_dim=condition_dim,
                use_gating=use_gating,
            ),
            dropout=dropout,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoded_features: torch.Tensor | None = None,
        self_attention_mask: torch.Tensor | None = None,
        cross_attention_mask: torch.Tensor | None = None,
        layer_cache: LayerKVCache | None = None,
        use_cache: bool = False,
        positional_encoding: RotaryPositionalEncoding | None = None,
        conditioning: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, LayerKVCache | None]:
        """Forward pass through decoder layer.

        Args:
            hidden_states: Input embeddings (B, T, D).
            encoded_features: Encoder output for cross-attention (B, S, D).
                Required when use_cross_attention=True and no cached cross K/V.
            self_attention_mask: Causal mask (B, 1, T, T), True = masked.
            cross_attention_mask: Cross-attention mask (B, 1, T, S), True = masked.
            layer_cache: Cached K/V from previous autoregressive steps.
            use_cache: Whether to return updated cache. Only valid if autoregressive=True.
            positional_encoding: Optional rotary positional encoding module.
            conditioning: Conditioning vector for adaptive normalization (B, C).
                Ignored when constructed with plain normalization.

        Returns:
            Tuple of (output hidden states (B, T, D), updated cache or None).

        Raises:
            ValueError: If use_cache=True for non-autoregressive model, or if
                encoded_features is missing when cross-attention is enabled
                without cached K/V.
        """
        if use_cache and not self.autoregressive:
            raise ValueError(
                "use_self_attention_cache=True only valid for autoregressive models"
            )
        hidden_states, new_cache = self.self_attention_block(
            hidden_states=hidden_states,
            conditioning=conditioning,
            attention_mask=self_attention_mask,
            positional_encoding=positional_encoding,
            layer_cache=layer_cache,
            use_cache=use_cache,
        )
        if self.use_cross_attention:
            if encoded_features is None and (
                layer_cache is None or layer_cache.cross_attention_keys is None
            ):
                raise ValueError(
                    "encoded_features required when use_cross_attention=True "
                    "and no cached cross KV"
                )
            hidden_states, _ = self.cross_attention_block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoded_features,
                conditioning=conditioning,
                attention_mask=cross_attention_mask,
                layer_cache=layer_cache,
            )
        hidden_states = self.feedforward_block(
            hidden_states=hidden_states,
            conditioning=conditioning,
        )
        return hidden_states, new_cache
