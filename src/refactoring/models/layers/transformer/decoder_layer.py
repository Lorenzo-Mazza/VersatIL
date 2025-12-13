"""General transformer decoder layer with KV cache support."""

import torch
import torch.nn as nn

from refactoring.models.layers.activation import ActivationFunction
from refactoring.models.layers.constants import AttentionType
from refactoring.models.layers.normalization.constants import NormalizationType
from refactoring.models.layers.transformer.attention import CachedAttention
from refactoring.models.layers.transformer.kv_cache import LayerKVCache
from refactoring.models.layers.normalization.factory import create_normalization_layer
from refactoring.models.layers.positional_encoding.rotary import RotaryPositionalEncoding
from refactoring.models.layers.swiglu import SwiGLU


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer with self-attention, cross-attention, and FFN.

    Architecture (Pre-Norm variant):
        x = x + SelfAttn(Norm(x))
        x = x + CrossAttn(Norm(x), encoded_features)
        x = x + FFN(Norm(x))

    Supports KV caching for efficient autoregressive generation, if autoregressive.
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
    ):
        """Initialize Transformer decoder layer.

        Args:
            embedding_dimension: Model embedding dimension
            number_of_heads: Number of attention heads
            number_of_key_value_heads: Number of K/V heads (for GQA)
            feedforward_dimension: FFN hidden dimension (defaults to 4 * embedding_dimension)
            dropout: Dropout probability for residual connections
            attention_dropout: Dropout probability for attention weights
            activation: Activation function (use ActivationFunction enum values)
            normalization_type: Type of normalization (use NormalizationType enum values)
            attention_type: Type of attention (use AttentionType enum values)
            use_cross_attention: Whether to use cross-attention (False for decoder-only models)
            bias: Whether to use bias in linear layers
            normalization_epsilon: Epsilon for normalization layers
            autoregressive: Whether the model is autoregressive (affects caching behavior)
        """
        super().__init__()

        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        self.use_cross_attention = use_cross_attention
        self.autoregressive = autoregressive
        if feedforward_dimension is None:
            feedforward_dimension = 4 * embedding_dimension
        self.self_attention = CachedAttention(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_key_value_heads=number_of_key_value_heads,
            dropout=attention_dropout,
            bias=bias,
            attention_type=attention_type,
        )
        self.self_attention_normalization = create_normalization_layer(
            normalization_type=normalization_type,
            dimension=embedding_dimension,
            epsilon=normalization_epsilon,
        )
        if use_cross_attention:
            self.cross_attention = CachedAttention(
                embedding_dimension=embedding_dimension,
                number_of_heads=number_of_heads,
                number_of_key_value_heads=number_of_key_value_heads,
                dropout=attention_dropout,
                bias=bias,
                attention_type=attention_type,
            )
            self.cross_attention_normalization = create_normalization_layer(
                normalization_type=normalization_type,
                dimension=embedding_dimension,
                epsilon=normalization_epsilon,
            )
        else:
            self.cross_attention = None
            self.cross_attention_normalization = None

        if activation == ActivationFunction.SWIGLU.value:
            self.feedforward_network = nn.Sequential(
                SwiGLU(input_dim=embedding_dimension, hidden_dim=feedforward_dimension, bias=bias),
                nn.Dropout(dropout),
                nn.Linear(feedforward_dimension, embedding_dimension, bias=bias),
            )
        else:
            activation_class = ActivationFunction(activation).to_torch_activation()
            self.feedforward_network = nn.Sequential(
                nn.Linear(embedding_dimension, feedforward_dimension, bias=bias),
                activation_class(),
                nn.Dropout(dropout),
                nn.Linear(feedforward_dimension, embedding_dimension, bias=bias),
            )
        self.feedforward_normalization = create_normalization_layer(
            normalization_type=normalization_type,
            dimension=embedding_dimension,
            epsilon=normalization_epsilon,
        )
        self.feedforward_network[-1].SQUARE_ROOT_WEIGHT = True # Flag for initialization (GPT2 style)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoded_features: torch.Tensor | None = None,
        self_attention_mask: torch.Tensor | None = None,
        cross_attention_mask: torch.Tensor | None = None,
        layer_cache: LayerKVCache | None = None,
        use_cache: bool = False,
        positional_encoding: RotaryPositionalEncoding | None = None,
    ) -> tuple[torch.Tensor, LayerKVCache | None]:
        """Forward pass through decoder layer.

        Args:
            hidden_states: Input embeddings (B, seq_len, D)
            encoded_features: Encoded visual features (B, num_features, D). Required if use_cross_attention=True
            self_attention_mask: Optional causal mask for self-attention with shape (B,1, query length, key length)
             where True=masked. If None, no causal masking is applied.
            cross_attention_mask: Optional mask for cross-attention, with shape (B,1, query length, key length)
             where True=masked. If None, no cross-attention masking is applied.
            layer_cache: Optional cached K/V from previous steps
            use_cache: Whether to return updated cache. Only valid if autoregressive=True
            positional_encoding: Optional rotary positional encoding module

        Returns:
            Tuple of (hidden_states, new_cache), where hidden_states has shape (B, query_len, D) and
            new_cache is a LayerKVCache or None.

        Raises:
            ValueError: If use_self_attention_cache=True for non-autoregressive model
        """
        if use_cache and not self.autoregressive:
            raise ValueError("use_self_attention_cache=True only valid for autoregressive models")
        residual = hidden_states
        hidden_states = self.self_attention_normalization(hidden_states)
        self_attention_output, new_cache = self.self_attention(
            query_input=hidden_states,
            key_input=hidden_states,
            value_input=hidden_states,
            attention_mask=self_attention_mask,
            layer_cache=layer_cache,
            use_self_attention_cache=use_cache,
            use_cross_attention_cache=False,
            positional_encoding=positional_encoding,
        )
        hidden_states = residual + self.dropout(self_attention_output)

        if self.use_cross_attention:
            if encoded_features is None and (layer_cache is None or layer_cache.cross_attention_keys is None):
                raise ValueError("encoded_features required when use_cross_attention=True and no cached cross KV")
            residual = hidden_states
            hidden_states = self.cross_attention_normalization(hidden_states)
            use_cross_cache = layer_cache is not None and layer_cache.cross_attention_keys is not None
            cross_attention_output, _ = self.cross_attention(
                query_input=hidden_states,
                key_input=encoded_features if not use_cross_cache else None,
                value_input=encoded_features if not use_cross_cache else None,
                attention_mask=cross_attention_mask,
                layer_cache=layer_cache,
                use_self_attention_cache=False,
                use_cross_attention_cache=use_cross_cache,
            )
            hidden_states = residual + self.dropout(cross_attention_output)

        residual = hidden_states
        hidden_states = self.feedforward_normalization(hidden_states)
        feedforward_output = self.feedforward_network(hidden_states)
        hidden_states = residual + self.dropout(feedforward_output)

        return hidden_states, new_cache