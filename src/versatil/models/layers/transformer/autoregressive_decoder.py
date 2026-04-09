"""GPT-style transformer decoder with KV cache support."""

import torch
import torch.nn as nn

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.normalization.factory import create_normalization_layer
from versatil.models.layers.positional_encoding.learned import (
    LearnedPositionalEncoding1D,
)
from versatil.models.layers.positional_encoding.rotary import (
    RotaryPositionalEncoding,
)
from versatil.models.layers.positional_encoding.sinusoidal import (
    SinusoidalPositionalEncoding1D,
)
from versatil.models.layers.transformer.kv_cache import (
    DecoderKVCache,
    LayerKVCache,
    initialize_decoder_cache,
)
from versatil.models.layers.transformer.layer.decoder_layer import (
    TransformerDecoderLayer,
)
from versatil.models.layers.transformer.masking import create_full_padding_mask
from versatil.models.layers.transformer.transformer_mixin import TransformerMixin


class GPTDecoder(TransformerMixin, nn.Module):
    """GPT-style autoregressive decoder, with KV caching, extended to support cross-attention.

    Stacks multiple TransformerDecoderLayer modules and manages KV cache across layers.
    Applies causal masking for autoregressive generation.
    """

    def __init__(
        self,
        number_of_layers: int,
        embedding_dimension: int,
        number_of_heads: int,
        number_of_key_value_heads: int | None = None,
        feedforward_dimension: int | None = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        activation: str = ActivationFunction.SWIGLU.value,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        attention_type: str = AttentionType.GROUPED_QUERY.value,
        use_cross_attention: bool = False,
        positional_encoding_type: str | None = None,
        maximum_sequence_length: int = 2048,
        bias: bool = True,
        normalization_epsilon: float = 1e-6,
        initializer_range: float = 0.02,
    ):
        """Initialize GPT decoder.

        Args:
            number_of_layers: Number of decoder layers
            embedding_dimension: Model embedding dimension
            number_of_heads: Number of attention heads
            number_of_key_value_heads: Number of K/V heads (for GQA)
            feedforward_dimension: FFN hidden dimension
            dropout: Dropout probability for residual connections
            attention_dropout: Dropout probability for attention weights
            activation: Activation function (use ActivationFunction enum values)
            normalization_type: Type of normalization (use NormalizationType enum values)
            attention_type: Type of attention (use AttentionType enum values)
            use_cross_attention: Whether to use cross-attention (False for decoder-only models)
            positional_encoding_type: Type of positional encoding (use PositionalEncodingType enum values, or None)
            maximum_sequence_length: Maximum sequence length for positional encoding
            bias: Whether to use bias in linear layers
            normalization_epsilon: Epsilon for normalization layers
            initializer_range: Standard deviation for weight initialization
        """
        super().__init__()

        self.number_of_layers = number_of_layers
        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        self.maximum_sequence_length = maximum_sequence_length
        self.use_cross_attention = use_cross_attention
        self.initializer_range = initializer_range
        self.number_of_residual_blocks = (
            3 if use_cross_attention else 2
        )  # Self-Attention + Feedforward
        if attention_type == AttentionType.GROUPED_QUERY.value:
            if number_of_key_value_heads is None:
                raise ValueError("number_of_key_value_heads required for GQA")
            self.number_of_key_value_heads = number_of_key_value_heads
        else:
            self.number_of_key_value_heads = number_of_heads
        self.head_dimension = embedding_dimension // number_of_heads
        self._setup_positional_encoding(
            positional_encoding_type=positional_encoding_type,
            embedding_dimension=embedding_dimension,
            maximum_sequence_length=maximum_sequence_length,
            number_of_heads=number_of_heads,
        )

        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    embedding_dimension=embedding_dimension,
                    number_of_heads=number_of_heads,
                    number_of_key_value_heads=number_of_key_value_heads,
                    feedforward_dimension=feedforward_dimension,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation=activation,
                    normalization_type=normalization_type,
                    attention_type=attention_type,
                    use_cross_attention=use_cross_attention,
                    bias=bias,
                    normalization_epsilon=normalization_epsilon,
                    autoregressive=True,
                )
                for _ in range(number_of_layers)
            ]
        )

        self.final_normalization = create_normalization_layer(
            normalization_type=normalization_type,
            dimension=embedding_dimension,
            epsilon=normalization_epsilon,
        )
        self.apply(self._init_weights)

    def precompute_cross_attention_kv(
        self,
        encoded_features: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Precompute cross-attention K/V for all layers.

        Projects encoded features through each layer's cross-attention K/V projections
        to avoid redundant computation during generation.

        Args:
            encoded_features: Encoded visual features (B, num_features, D)

        Returns:
            List of (keys, values) tuples, one per layer. Each has shape (B, kv_heads, num_features, head_dim)
        """
        cross_kv_per_layer = []
        batch_size = encoded_features.shape[0]
        num_features = encoded_features.shape[1]

        for layer in self.layers:
            projected_key = layer.cross_attention_block.attention.key_projection(
                encoded_features
            )  # (B, len, embed_dim)
            projected_value = layer.cross_attention_block.attention.value_projection(
                encoded_features
            )  # (B, len, embed_dim)
            # Reshape to (B, len, kv_heads, head_dim)
            projected_key = projected_key.view(
                batch_size,
                num_features,
                self.number_of_key_value_heads,
                self.head_dimension,
            )
            projected_value = projected_value.view(
                batch_size,
                num_features,
                self.number_of_key_value_heads,
                self.head_dimension,
            )
            # Transpose to (B, kv_heads, len, head_dim)
            projected_key = projected_key.transpose(1, 2)
            projected_value = projected_value.transpose(1, 2)
            cross_kv_per_layer.append((projected_key, projected_value))

        return cross_kv_per_layer

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoded_features: torch.Tensor | None = None,
        self_attention_mask: torch.Tensor | None = None,
        cross_attention_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        decoder_cache: DecoderKVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, DecoderKVCache | None]:
        """Forward pass through decoder.

        Args:
            hidden_states: Input token embeddings (B, query_length, D)
            encoded_features: Encoder visual features (B, num_features, D). Required if self.use_cross_attention=True
            self_attention_mask: Optional custom self-attention mask (B, 1, query_length, query_length) where True means masked.
                If None, generates standard triangular causal mask.
            cross_attention_mask: Optional mask for cross-attention, where True means masked position with shape (B,1, query length, key length).
            key_padding_mask: Optional current key padding mask for padded observation tokens (B, query_length) where True means masked.
            decoder_cache: Optional cached K/V from previous steps
            use_cache: Whether to return updated cache

        Returns:
            Tuple of (final_hidden_states, updated_decoder_cache)
        """
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        query_length = hidden_states.shape[1]
        cache_length = decoder_cache.get_length() if decoder_cache else 0
        cached_key_padding_mask = (
            decoder_cache.key_padding_mask if decoder_cache else None
        )
        total_mask, full_key_padding_mask = create_full_padding_mask(
            key_padding_mask=key_padding_mask,
            cached_key_padding_mask=cached_key_padding_mask,
            self_attention_mask=self_attention_mask,
            batch_size=batch_size,
            query_length=query_length,
            cache_length=cache_length,
            device=device,
        )  # (B, 1, query_length, key_length), (B, key_length), where key_length = cache_length + query_length

        if isinstance(
            self.positional_encoding,
            (SinusoidalPositionalEncoding1D, LearnedPositionalEncoding1D),
        ):
            hidden_states = hidden_states + self.positional_encoding(
                hidden_states, offset=cache_length
            )
        rope_pe = (
            self.positional_encoding
            if isinstance(self.positional_encoding, RotaryPositionalEncoding)
            else None
        )

        cross_kv_per_layer = None
        if self.use_cross_attention:
            if (
                decoder_cache is not None
                and decoder_cache.layers[0].cross_attention_keys is not None
            ):
                cross_kv_per_layer = [
                    (cache.cross_attention_keys, cache.cross_attention_values)
                    for cache in decoder_cache.layers
                ]
            else:
                if encoded_features is None:
                    raise ValueError(
                        "encoded_features required when use_cross_attention=True and no cached cross KV"
                    )
                cross_kv_per_layer = self.precompute_cross_attention_kv(
                    encoded_features
                )

        layer_caches = None
        if decoder_cache is not None:
            layer_caches = decoder_cache.layers
        elif use_cache:
            layer_caches = initialize_decoder_cache(
                batch_size=batch_size,
                num_layers=self.number_of_layers,
                num_heads=self.number_of_key_value_heads,
                head_dimension=self.head_dimension,
                device=device,
                dtype=hidden_states.dtype,
            )

        new_layer_caches = []

        for layer_index, layer in enumerate(self.layers):
            original_layer_cache = (
                layer_caches[layer_index] if layer_caches is not None else None
            )
            # Build layer cache with self KV and optional cross KV
            if original_layer_cache is None:
                empty_shape = (
                    batch_size,
                    self.number_of_key_value_heads,
                    0,
                    self.head_dimension,
                )
                self_keys = torch.empty(
                    empty_shape, device=device, dtype=hidden_states.dtype
                )
                self_values = torch.empty(
                    empty_shape, device=device, dtype=hidden_states.dtype
                )
            else:
                self_keys = original_layer_cache.self_attention_keys
                self_values = original_layer_cache.self_attention_values

            if self.use_cross_attention and cross_kv_per_layer is not None:
                cross_keys, cross_values = cross_kv_per_layer[layer_index]
            else:
                cross_keys, cross_values = None, None

            layer_cache = LayerKVCache(
                self_attention_keys=self_keys,
                self_attention_values=self_values,
                cross_attention_keys=cross_keys,
                cross_attention_values=cross_values,
            )
            hidden_states, new_layer_cache = layer(
                hidden_states=hidden_states,
                encoded_features=encoded_features,
                self_attention_mask=total_mask,
                cross_attention_mask=cross_attention_mask,
                layer_cache=layer_cache,
                use_cache=use_cache,
                positional_encoding=rope_pe,
            )

            if use_cache:
                new_layer_caches.append(new_layer_cache)

        hidden_states = self.final_normalization(hidden_states)
        new_decoder_cache = None
        if use_cache:
            new_decoder_cache = DecoderKVCache(
                layers=new_layer_caches,
                key_padding_mask=full_key_padding_mask,
            )

        return hidden_states, new_decoder_cache
