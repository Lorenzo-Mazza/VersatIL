"""GPT-style transformer decoder with KV cache support."""

import torch
import torch.nn as nn
import math

from refactoring.models.layers.activation import ActivationFunction
from refactoring.models.layers.constants import AttentionType, NormalizationType
from refactoring.models.layers.gpt_transformer.gpt_decoder_layer import GPTDecoderLayer
from refactoring.models.layers.gpt_transformer.kv_cache import DecoderKVCache, LayerKVCache, initialize_decoder_cache
from refactoring.models.layers.gpt_transformer.normalization import create_normalization_layer
from refactoring.models.layers.gpt_transformer.positional_encoding import create_positional_encoding
from refactoring.models.layers.positional_encoding.sinusoidal import SinusoidalPositionalEncoding1D
from refactoring.models.layers.rms_norm import RMSNorm


class GPTDecoder(nn.Module):
    """GPT-style autoregressive decoder with KV caching.

    Stacks multiple GPTDecoderLayer modules and manages KV cache across layers.
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
        use_cross_attention: bool = True,
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
        # Determine number of K/V heads for cache initialization
        if attention_type == AttentionType.GROUPED_QUERY.value:
            if number_of_key_value_heads is None:
                raise ValueError("number_of_key_value_heads required for GQA")
            self.number_of_key_value_heads = number_of_key_value_heads
        else:
            self.number_of_key_value_heads = number_of_heads

        self.head_dimension = embedding_dimension // number_of_heads

        # Create positional encoding if specified
        self.positional_encoding = None
        if positional_encoding_type is not None:
            self.positional_encoding = create_positional_encoding(
                encoding_type=positional_encoding_type,
                embedding_dimension=embedding_dimension,
                maximum_length=maximum_sequence_length,
                num_heads=number_of_heads,
            )

        # Stack of decoder layers
        self.layers = nn.ModuleList([
            GPTDecoderLayer(
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
            )
            for _ in range(number_of_layers)
        ])

        # Final layer normalization
        self.final_normalization = create_normalization_layer(
            normalization_type=normalization_type,
            dimension=embedding_dimension,
            epsilon=normalization_epsilon,
        )
        self.apply(self._init_weights)


    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):  # No Conv1D in your arch, so removed
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
            if hasattr(module, 'weight') and module.weight is not None:
                module.weight.data.fill_(1.0)
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        # > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        # > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        # > -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if 'output_projection.weight' in name or 'feedforward_network' in name:
                num_norm_layers = 3 if self.use_cross_attention else 2
                p.data.normal_(mean=0.0, std=(self.initializer_range / math.sqrt(num_norm_layers * self.number_of_layers)))


    def generate_causal_mask(
        self,
        sequence_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate causal attention mask.

        Args:
            sequence_length: Sequence length
            device: Device to create mask on

        Returns:
            Causal mask (1, 1, seq_len, seq_len) as boolean tensor where True means masked position
        """
        # Create causal mask: True for positions that should be masked (future positions)
        mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=device, dtype=torch.bool),
            diagonal=1
        )
        # Add batch and head dimensions: (seq_len, seq_len) -> (1, 1, seq_len, seq_len)
        mask = mask.unsqueeze(0).unsqueeze(0)
        return mask

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
            # Project encoded features to K and V using this layer's cross-attention projections
            projected_key = layer.cross_attention.key_projection(encoded_features)
            projected_value = layer.cross_attention.value_projection(encoded_features)

            # Reshape to (B, num_features, kv_heads, head_dim)
            projected_key = projected_key.view(
                batch_size, num_features, self.number_of_key_value_heads, self.head_dimension
            )
            projected_value = projected_value.view(
                batch_size, num_features, self.number_of_key_value_heads, self.head_dimension
            )

            # Transpose to (B, kv_heads, num_features, head_dim)
            projected_key = projected_key.transpose(1, 2)
            projected_value = projected_value.transpose(1, 2)

            cross_kv_per_layer.append((projected_key, projected_value))

        return cross_kv_per_layer

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoded_features: torch.Tensor | None = None,
        cross_attention_mask: torch.Tensor | None = None,
        decoder_cache: DecoderKVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, DecoderKVCache | None]:
        """Forward pass through decoder.

        Args:
            hidden_states: Input token embeddings (B, seq_len, D)
            encoded_features: Encoded visual features (B, num_features, D). Required if use_cross_attention=True
            cross_attention_mask: Optional mask for cross-attention
            decoder_cache: Optional cached K/V from previous steps
            use_cache: Whether to return updated cache

        Returns:
            Tuple of (final_hidden_states, updated_decoder_cache)
        """
        batch_size = hidden_states.shape[0]
        sequence_length = hidden_states.shape[1]
        device = hidden_states.device

        # Apply positional encoding to input embeddings if using sinusoidal
        if isinstance(self.positional_encoding, SinusoidalPositionalEncoding1D):
            hidden_states = self.positional_encoding(hidden_states)

        # Generate causal mask for self-attention
        # During generation, need to account for cached sequence length
        cache_length = decoder_cache.get_length() if decoder_cache else 0
        total_length = cache_length + sequence_length
        self_attention_mask = self.generate_causal_mask(total_length, device)
        # Slice to get only rows for current queries (last sequence_length rows)
        # Shape: (1, 1, sequence_length, total_length)
        self_attention_mask = self_attention_mask[:, :, -sequence_length:, :]

        # Precompute cross-attention K/V if using cross-attention
        cross_kv_per_layer = None
        if self.use_cross_attention:
            if decoder_cache is not None and decoder_cache.layers[0].cross_attention_keys is not None:
                # Use precomputed cross KV from cache
                cross_kv_per_layer = [
                    (cache.cross_attention_keys, cache.cross_attention_values)
                    for cache in decoder_cache.layers
                ]
            else:
                # Precompute cross KV for all layers
                if encoded_features is None:
                    raise ValueError("encoded_features required when use_cross_attention=True and no cached cross KV")
                cross_kv_per_layer = self.precompute_cross_attention_kv(encoded_features)

        # Initialize or get cache
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

        # Pass through decoder layers
        new_layer_caches = []
        for layer_index, layer in enumerate(self.layers):
            original_layer_cache = layer_caches[layer_index] if layer_caches is not None else None

            # Build layer cache with self KV and optional cross KV
            if original_layer_cache is None:
                empty_shape = (batch_size, self.number_of_key_value_heads, 0, self.head_dimension)
                self_keys = torch.empty(empty_shape, device=device, dtype=hidden_states.dtype)
                self_values = torch.empty(empty_shape, device=device, dtype=hidden_states.dtype)
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
                self_attention_mask=self_attention_mask,
                cross_attention_mask=cross_attention_mask,
                layer_cache=layer_cache,
                use_cache=use_cache,
                positional_encoding=self.positional_encoding,
            )

            if use_cache:
                new_layer_caches.append(new_layer_cache)

        hidden_states = self.final_normalization(hidden_states)
        new_decoder_cache = None
        if use_cache:
            new_decoder_cache = DecoderKVCache(layers=new_layer_caches)

        return hidden_states, new_decoder_cache