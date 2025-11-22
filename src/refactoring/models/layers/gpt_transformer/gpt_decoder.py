"""GPT-style transformer decoder with KV cache support."""

import torch
import torch.nn as nn
import math

from refactoring.models.layers.activation import ActivationFunction
from refactoring.models.layers.constants import AttentionType
from refactoring.models.layers.normalization.ada_norm import AdaNorm
from refactoring.models.layers.normalization.constants import NormalizationType
from refactoring.models.layers.gpt_transformer.decoder_layer import TransformerDecoderLayer
from refactoring.models.layers.gpt_transformer.kv_cache import DecoderKVCache, LayerKVCache, initialize_decoder_cache
from refactoring.models.layers.normalization.factory import create_normalization_layer
from refactoring.models.layers.gpt_transformer.positional_encoding import create_positional_encoding
from refactoring.models.layers.positional_encoding.learned import LearnedPositionalEncoding1D
from refactoring.models.layers.positional_encoding.rotary import RotaryPositionalEncoding
from refactoring.models.layers.positional_encoding.sinusoidal import SinusoidalPositionalEncoding1D
from refactoring.models.layers.normalization.rms_norm import RMSNorm


RESIDUAL_STREAM_FLAG = "SQUARE_ROOT_WEIGHT" # Used for initialization flag

class GPTDecoder(nn.Module):
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
        if attention_type == AttentionType.GROUPED_QUERY.value:
            if number_of_key_value_heads is None:
                raise ValueError("number_of_key_value_heads required for GQA")
            self.number_of_key_value_heads = number_of_key_value_heads
        else:
            self.number_of_key_value_heads = number_of_heads

        self.head_dimension = embedding_dimension // number_of_heads

        self.positional_encoding = None
        if positional_encoding_type is not None:
            self.positional_encoding = create_positional_encoding(
                encoding_type=positional_encoding_type,
                embedding_dimension=embedding_dimension,
                maximum_length=maximum_sequence_length,
                num_heads=number_of_heads,
            )

        self.layers = nn.ModuleList([
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
        ])

        self.final_normalization = create_normalization_layer(
            normalization_type=normalization_type,
            dimension=embedding_dimension,
            epsilon=normalization_epsilon,
        )
        self.apply(self._init_weights)


    def _init_weights(self, module):
        """Initialize the weights."""
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        # > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        # > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        # > -- GPT-2 :: https://openai.com/blog/better-language-models/
        if hasattr(module, RESIDUAL_STREAM_FLAG):  # Residual stream correction
            num_norm_layers = 3 if self.use_cross_attention else 2
            std = self.initializer_range / math.sqrt(num_norm_layers * self.number_of_layers)
        else:
            std = self.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (nn.LayerNorm, RMSNorm, AdaNorm)):
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
            if hasattr(module, 'weight') and module.weight is not None:
                module.weight.data.fill_(1.0)


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

        Note:
            This mask uses the opposite convention as torch.nn.scaled_dot_product_attention.
            This mask uses the same convention as torch.nn.MultiHeadAttention.
        """
        # Create triangular matrix mask with True for future positions
        mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=device, dtype=torch.bool),
            diagonal=1 # `True`s start above the main diagonal
        )
        mask = mask.unsqueeze(0).unsqueeze(0) #(1, 1, seq_len, seq_len)
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
            projected_key = layer.cross_attention.key_projection(encoded_features) # (B, len, embed_dim)
            projected_value = layer.cross_attention.value_projection(encoded_features) # (B, len, embed_dim)
            # Reshape to (B, len, kv_heads, head_dim)
            projected_key = projected_key.view(
                batch_size, num_features, self.number_of_key_value_heads, self.head_dimension
            )
            projected_value = projected_value.view(
                batch_size, num_features, self.number_of_key_value_heads, self.head_dimension
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
            hidden_states: Input token embeddings (B, seq_len, D)
            encoded_features: Encoder visual features (B, num_features, D). Required if self.use_cross_attention=True
            self_attention_mask: Optional custom self-attention mask (B, 1, seq_len, seq_len) where True means masked.
               If None, generates standard triangular causal mask.
            cross_attention_mask: Optional mask for cross-attention, where True means masked position.
            key_padding_mask: Optional key padding mask for padded observation tokens (B, seq_len) where True means masked.
            decoder_cache: Optional cached K/V from previous steps
            use_cache: Whether to return updated cache

        Returns:
            Tuple of (final_hidden_states, updated_decoder_cache)
        """
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        query_length = hidden_states.shape[1]
        if isinstance(self.positional_encoding, (SinusoidalPositionalEncoding1D, LearnedPositionalEncoding1D)):
            hidden_states = self.positional_encoding(hidden_states)

        cache_length = decoder_cache.get_length() if decoder_cache else 0
        key_length = cache_length + query_length
        cached_key_padding_mask = decoder_cache.key_padding_mask if decoder_cache else None
        full_key_padding_mask = None
        if key_padding_mask is not None:
            if cached_key_padding_mask is None:
                full_key_padding_mask = key_padding_mask # (B, seq_len + 0 = key_length)
            else:
                full_key_padding_mask = torch.cat((cached_key_padding_mask, key_padding_mask), dim=1) # (B, total_len)
        else:
            # full_key_padding_mask is None
            if cached_key_padding_mask is not None:
                full_key_padding_mask = torch.cat(
                    (cached_key_padding_mask, torch.zeros(batch_size, query_length, device=device, dtype=torch.bool)), dim=1
                ) # (B, total_len)

        if self_attention_mask is None:
            causal_mask = self.generate_causal_mask(key_length, device)  # (1, 1, key_length, key_length)
            total_mask = causal_mask[:, :, -query_length:, :]  # (1,1,query_length,key_length)
            if full_key_padding_mask is not None:
                padding_mask = full_key_padding_mask.unsqueeze(1).unsqueeze(1)  # (B,1,1,key_length)
                # Broadcasting to (B,1,query_length,key_length)
                total_mask = total_mask | padding_mask
        else:
            # self_attention_mask (B, 1, query_length, query_length)
            # total_mask (B, 1, query_length, key_length)
            total_mask = torch.zeros(batch_size, 1, query_length, key_length, dtype=torch.bool, device=device)
            if full_key_padding_mask is not None:
                padding_mask = full_key_padding_mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, key_length)
                # Broadcasting to (B,1, query_length, key_length)
                total_mask |= padding_mask
            # Slicing the query_length elements in total mask
            # total_mask[:, :, :, cache_length:] has shape (B,1, query_length, query_length)
            total_mask[:, :, :, cache_length:] |= self_attention_mask


        # Precompute cross-attention K/V if using cross-attention
        cross_kv_per_layer = None
        if self.use_cross_attention:
            if decoder_cache is not None and decoder_cache.layers[0].cross_attention_keys is not None:
                cross_kv_per_layer = [
                    (cache.cross_attention_keys, cache.cross_attention_values)
                    for cache in decoder_cache.layers
                ]
            else:
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
            # Compute decoder layers forward pass
            rope_pe = self.positional_encoding if isinstance(self.positional_encoding, RotaryPositionalEncoding) else None
            hidden_states, new_layer_cache = layer(
                hidden_states=hidden_states,
                encoded_features=encoded_features,
                self_attention_mask=total_mask,
                cross_attention_mask=cross_attention_mask,
                layer_cache=layer_cache,
                use_cache=use_cache,
                positional_encoding=rope_pe ,
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