"""GPT-style autoregressive decoder."""

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
from versatil.models.layers.transformer.cache.conditioning import (
    ConditioningCache,
    ConditioningLayerCache,
)
from versatil.models.layers.transformer.cache.generation import (
    GenerationCache,
    initialize_generation_cache,
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

    def create_empty_generation_cache(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> GenerationCache:
        """Create an initial empty GenerationCache for autoregressive generation.

        Args:
            batch_size: Batch size.
            device: Device for cache tensors.
            dtype: Data type for cache tensors.

        Returns:
            GenerationCache with empty layers ready for the first generation step.
        """
        return GenerationCache(
            layers=initialize_generation_cache(
                batch_size=batch_size,
                num_layers=self.number_of_layers,
                num_heads=self.number_of_key_value_heads,
                head_dimension=self.head_dimension,
                device=device,
                dtype=dtype,
            ),
        )

    def precompute_conditioning_kv(
        self,
        encoded_features: torch.Tensor,
    ) -> ConditioningCache:
        """Precompute conditioning K/V for all layers.

        Projects encoded features through each layer's cross-attention K/V projections
        once, so they can be reused across all generation steps.

        Args:
            encoded_features: Encoded features (B, num_features, D).

        Returns:
            ConditioningCache with one ConditioningLayerCache per layer.
        """
        batch_size = encoded_features.shape[0]
        num_features = encoded_features.shape[1]
        layer_caches = []

        for layer in self.layers:
            cross_attention = layer.cross_attention_block.attention
            projected_key = cross_attention.key_projection(encoded_features)
            projected_value = cross_attention.value_projection(encoded_features)
            # (B, S, kv_heads * head_dim) → (B, kv_heads, S, head_dim)
            projected_key = projected_key.view(
                batch_size,
                num_features,
                self.number_of_key_value_heads,
                self.head_dimension,
            ).transpose(1, 2)
            projected_value = projected_value.view(
                batch_size,
                num_features,
                self.number_of_key_value_heads,
                self.head_dimension,
            ).transpose(1, 2)
            layer_caches.append(
                ConditioningLayerCache(keys=projected_key, values=projected_value)
            )

        return ConditioningCache(layers=layer_caches)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoded_features: torch.Tensor | None = None,
        self_attention_mask: torch.Tensor | None = None,
        cross_attention_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        generation_cache: GenerationCache | None = None,
        conditioning_cache: ConditioningCache | None = None,
    ) -> tuple[torch.Tensor, GenerationCache | None]:
        """Forward pass through decoder.

        Args:
            hidden_states: Input token embeddings (B, query_length, D).
            encoded_features: Encoder features (B, num_features, D). Required when
                use_cross_attention=True and no conditioning_cache.
            self_attention_mask: Custom causal mask (B, 1, query_length, query_length),
                True = masked. If None, generates standard triangular causal mask.
            cross_attention_mask: Mask for cross-attention (B, 1, query_length, key_length),
                True = masked.
            key_padding_mask: Padding mask for observation tokens (B, query_length),
                True = masked.
            generation_cache: Cached K/V from previous generation steps. When provided,
                an updated cache is returned.
            conditioning_cache: Precomputed K/V for static conditioning. When provided,
                encoded_features is not needed for cross-attention.

        Returns:
            Tuple of (output (B, query_length, D), updated GenerationCache or None).
        """
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        query_length = hidden_states.shape[1]
        cache_length = generation_cache.get_length() if generation_cache else 0
        cached_key_padding_mask = (
            generation_cache.key_padding_mask if generation_cache else None
        )
        total_mask, full_key_padding_mask = create_full_padding_mask(
            key_padding_mask=key_padding_mask,
            cached_key_padding_mask=cached_key_padding_mask,
            self_attention_mask=self_attention_mask,
            batch_size=batch_size,
            query_length=query_length,
            cache_length=cache_length,
            device=device,
        )

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
        use_cache = generation_cache is not None
        new_layer_caches = []
        for layer_index, layer in enumerate(self.layers):
            layer_generation_cache = (
                generation_cache.layers[layer_index]
                if generation_cache is not None
                else None
            )
            layer_conditioning_cache = (
                conditioning_cache.layers[layer_index]
                if conditioning_cache is not None
                else None
            )
            hidden_states, new_layer_cache = layer(
                hidden_states=hidden_states,
                encoded_features=encoded_features,
                self_attention_mask=total_mask,
                cross_attention_mask=cross_attention_mask,
                generation_cache=layer_generation_cache,
                conditioning_cache=layer_conditioning_cache,
                positional_encoding=rope_pe,
            )
            if use_cache:
                new_layer_caches.append(new_layer_cache)

        hidden_states = self.final_normalization(hidden_states)
        new_generation_cache = None
        if use_cache:
            new_generation_cache = GenerationCache(
                layers=new_layer_caches,
                key_padding_mask=full_key_padding_mask,
            )

        return hidden_states, new_generation_cache
