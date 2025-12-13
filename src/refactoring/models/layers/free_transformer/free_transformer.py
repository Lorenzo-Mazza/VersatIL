"""Based on "The Free Transformer" (Fleuret, 2025) - arXiv:2510.17558
https://arxiv.org/abs/2510.17558

The Free Transformer extends decoder transformers by injecting learnable latent
variables into the middle layer, enabling conditional generation through a variational
autoencoder framework.
"""
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from refactoring.models.layers.constants import AttentionType
from refactoring.models.layers.transformer import TransformerDecoderLayer, LayerKVCache, DecoderKVCache, \
    initialize_decoder_cache, create_positional_encoding
from refactoring.models.layers.transformer.autoregressive_decoder import RESIDUAL_STREAM_FLAG
from refactoring.models.layers.transformer.masking import create_full_padding_mask
from refactoring.models.layers.normalization.ada_norm import AdaNorm
from refactoring.models.layers.normalization.constants import NormalizationType
from refactoring.models.layers.normalization.factory import create_normalization_layer
from refactoring.models.layers.normalization.rms_norm import RMSNorm
from refactoring.models.layers.free_transformer.binary_mapper import BinaryMapper
from refactoring.models.layers.positional_encoding.learned import LearnedPositionalEncoding1D
from refactoring.models.layers.positional_encoding.rotary import RotaryPositionalEncoding
from refactoring.models.layers.activation import ActivationFunction
from refactoring.models.layers.positional_encoding.sinusoidal import SinusoidalPositionalEncoding1D



class LatentConditionedDecoderLayer(TransformerDecoderLayer):
    """Transformer decoder layer with optional latent conditioning on key/value inputs.

    Identical to TransformerDecoderLayer except that a projected latent vector can be added
    to the normalized hidden states before key/value projection (query remains unconditioned).
    """

    def __init__(
        self,
        embedding_dimension: int,
        number_of_heads: int,
        latent_dim: int,
        number_of_key_value_heads: int | None = None,
        feedforward_dimension: int | None = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        activation: str = ActivationFunction.SWIGLU.value,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        attention_type: str = AttentionType.GROUPED_QUERY.value,
        bias: bool = True,
        normalization_epsilon: float = 1e-6,
        autoregressive: bool = True,
    ):
        super().__init__(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_key_value_heads=number_of_key_value_heads,
            feedforward_dimension=feedforward_dimension,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation=activation,
            normalization_type=normalization_type,
            attention_type=attention_type,
            use_cross_attention=False,
            bias=bias,
            normalization_epsilon=normalization_epsilon,
            autoregressive=autoregressive,
        )

        self.latent_proj = nn.Linear(latent_dim, embedding_dimension, bias=False)


    def forward(
            self,
            hidden_states: torch.Tensor,
            encoded_features: torch.Tensor | None = None,
            self_attention_mask: torch.Tensor | None = None,
            cross_attention_mask: torch.Tensor | None = None,
            layer_cache: LayerKVCache | None = None,
            use_cache: bool = False,
            positional_encoding: RotaryPositionalEncoding | None = None,
            latent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, LayerKVCache | None]:
        """Forward pass through latent-conditioned decoder layer.

        Args:
            hidden_states: Input embeddings (B, seq_len, embedding_dim).
            encoded_features: Not used here. Kept for signature compatibility.
            self_attention_mask: Optional causal mask for self-attention with shape (B,1, query length, key length)
             where True=masked. If None, no causal masking is applied.
            cross_attention_mask: Not used here. Kept for signature compatibility.
            layer_cache: Optional cached K/V from previous steps.
            use_cache: Whether to return updated cache. Only valid if autoregressive=True.
            positional_encoding: Optional rotary positional encoding module.
            latent: Optional latent conditioning of shape (B, T, latent_dim) or (B, 1, latent_dim).

        Returns:
            Tuple of (output_states, updated_cache),

        Raises:
            ValueError: When use_self_attention_cache=True for non-autoregressive model
        """
        if use_cache and not self.autoregressive:
            raise ValueError("use_self_attention_cache=True only valid for autoregressive models")

        residual = hidden_states
        hidden_states = self.self_attention_normalization(hidden_states)
        if latent is not None:
            R = self.latent_proj(latent)
            if R.shape[1] == 1 and hidden_states.shape[1] > 1: # (B, 1, embedding_dim) -> (B, T, embedding_dim)
                R = R.expand(-1, hidden_states.shape[1], -1)
            kv_residual = residual + R  # X + R
            norm_kv = self.self_attention_normalization(kv_residual)  # norm(X + R) for KV
            attention_output, cache = self.self_attention(
                query_input=hidden_states,
                key_input=norm_kv,
                value_input=norm_kv,
                attention_mask=self_attention_mask,
                layer_cache=layer_cache,
                use_self_attention_cache=use_cache,
                positional_encoding=positional_encoding,
            )
        else:
            attention_output, cache = self.self_attention(
                query_input=hidden_states,
                key_input=hidden_states,
                value_input=hidden_states,
                attention_mask=self_attention_mask,
                layer_cache=layer_cache,
                use_self_attention_cache=use_cache,
                positional_encoding=positional_encoding,
            )

        hidden_states = residual + self.dropout(attention_output)
        residual = hidden_states
        hidden_states = self.feedforward_normalization(hidden_states)
        ff_output = self.feedforward_network(hidden_states)
        hidden_states = residual + self.dropout(ff_output)

        return hidden_states, cache


class FreeTransformerLatentEncoder(nn.Module):
    """Training-only latent encoder that predicts latent codes from mid-decoder features.

    Uses a learned query + cross-attention to the mid-decoder hidden states.
    """

    def __init__(
        self,
        embedding_dimension: int,
        number_of_layers: int = 2,
        number_of_heads: int = 8,
        number_of_key_value_heads: int | None = None,
        feedforward_dimension: int | None = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        activation: str = ActivationFunction.SWIGLU.value,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        attention_type: str = AttentionType.GROUPED_QUERY.value,
        bias: bool = True,
        normalization_epsilon: float = 1e-6,
        use_global_latent: bool = False,
    ):
        super().__init__()
        self.learned_query = nn.Parameter(torch.randn(1, 1, embedding_dimension))
        layer = TransformerDecoderLayer(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_key_value_heads=number_of_key_value_heads,
            feedforward_dimension=feedforward_dimension,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation=activation,
            normalization_type=normalization_type,
            attention_type=attention_type,
            use_cross_attention=True,
            bias=bias,
            normalization_epsilon=normalization_epsilon,
            autoregressive=False,   # non-causal self-attention on the query
        )
        self.use_global_latent = use_global_latent
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(number_of_layers)])
        self.final_normalization = create_normalization_layer(normalization_type, embedding_dimension, normalization_epsilon)

    def forward(
            self,
            mid_features: torch.Tensor,
            mid_features_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the non-causal latent encoder blocks

        A learned query vector is fed into a sequence of Transformer Decoder blocks, where cross-attention
        attends to the mid-decoder features. The output is used to predict the latent embedding.

        Args:
            mid_features: Mid-decoder features (B, T, embedding_dimension) used for cross-attention.
            mid_features_mask: Optional padding mask for mid_features (B, T) with True=padding.

        Returns:
            Output target tensor with shape (B, T, embedding_dimension), representing latent embeddings.
        """
        B, T, D = mid_features.shape
        if self.use_global_latent:
            target = self.learned_query.expand(B, 1, -1)  # (B, 1, embedding_dimension)
        else:
            target = self.learned_query.expand(B, T, -1) # (B, T, embedding_dimension)
        if mid_features_mask is not None:
            # Expand (B, T) -> (B, 1, 1, T) for broadcast in cross-attn (Q=T, K=T)
            mid_features_mask = mid_features_mask.unsqueeze(1).unsqueeze(2)
        for layer in self.layers:
            target, _ = layer(
                hidden_states=target,
                encoded_features=mid_features,
                self_attention_mask=None,
                cross_attention_mask=mid_features_mask,
                use_cache=False,
            )

        return self.final_normalization(target)  # (B, T, embedding_dimension)



class FreeTransformer(nn.Module):
    """Free Transformer model (Fleuret, 2025).

    Contains:
    - FreeTransformerDecoder (main autoregressive decoder with latent injection at middle layer)
    - FreeTransformerLatentEncoder (training-only latent predictor)
    - BinaryMapper (converts latent embedding to discrete one-hot codes + logits for KL)

    Training forward returns (output, kl_logits)
    Inference forward returns output only
    """

    def __init__(
        self,
        latent_bits: int = 16,                     # 2**16 = 65536, as in paper
        latent_dim: int | None = None,             # overrides 2**latent_bits if set
        number_of_decoder_layers: int = 12,
        number_of_encoder_layers: int = 1,
        embedding_dimension: int = 768,
        number_of_heads: int = 12,
        number_of_key_value_heads: int = 2,
        feedforward_dimension: int | None = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        attention_type: str = AttentionType.GROUPED_QUERY.value,
        activation: str = ActivationFunction.SWIGLU.value,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        positional_encoding_type: str | None = None,
        maximum_sequence_length: int = 2048,
        bias: bool = True,
        use_global_latent: bool = False,
        normalization_epsilon: float = 1e-6,
        initializer_range: float = 0.02,
    ):
        super().__init__()

        latent_dim = latent_dim or (1 << latent_bits) # bitwise left-shift operator, equivalent to 2 ** latent_bits
        if number_of_decoder_layers % 2 != 0:
            raise ValueError("number_of_layers must be even")

        self.number_of_decoder_layers = number_of_decoder_layers
        self.number_of_encoder_layers = number_of_encoder_layers
        self.use_global_latent = use_global_latent
        self.embedding_dimension = embedding_dimension
        self.latent_dim = latent_dim
        self.latent_bits = latent_bits
        self.number_of_heads = number_of_heads
        self.maximum_sequence_length = maximum_sequence_length
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
        decoder_layer = TransformerDecoderLayer(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_key_value_heads=number_of_key_value_heads,
            feedforward_dimension=feedforward_dimension,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation=activation,
            normalization_type=normalization_type,
            attention_type=attention_type,
            use_cross_attention=False,  # No cross-attention
            bias=bias,
            normalization_epsilon=normalization_epsilon,
            autoregressive=True, # Causal self-attention
        )
        mid = number_of_decoder_layers // 2
        decoder_layers = []
        for i in range(number_of_decoder_layers):
            if i == mid:
                # Insert latent-conditioned layer at middle
                decoder_layers.append(
                    LatentConditionedDecoderLayer(
                        embedding_dimension=embedding_dimension,
                        number_of_heads=number_of_heads,
                        latent_dim=latent_dim,
                        number_of_key_value_heads=number_of_key_value_heads,
                        feedforward_dimension=feedforward_dimension,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        activation=activation,
                        normalization_type=normalization_type,
                        attention_type=attention_type,
                        bias=bias,
                        normalization_epsilon=normalization_epsilon,
                    )
                )
            else:
                decoder_layers.append(copy.deepcopy(decoder_layer))
        self.decoder_layers = nn.ModuleList(decoder_layers)
        self.final_normalization = create_normalization_layer(
            normalization_type=normalization_type,
            dimension=embedding_dimension,
            epsilon=1e-6,
        )
        self.latent_encoder = FreeTransformerLatentEncoder(
            embedding_dimension=embedding_dimension,
            number_of_layers=number_of_encoder_layers,
            number_of_heads=number_of_heads,
            normalization_type=normalization_type,
            number_of_key_value_heads=number_of_key_value_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation=activation,
            feedforward_dimension=feedforward_dimension,
            attention_type=attention_type,
            bias=bias,
            normalization_epsilon=normalization_epsilon,
            use_global_latent=use_global_latent
        )
        self.binary_mapper = BinaryMapper(
            latent_bits=latent_bits,
            embedding_dimension=embedding_dimension,
        )
        self.apply(self._init_weights)


    def _init_weights(self, module):
        """Initialize the weights."""
        # > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        # > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        # > -- GPT-2 :: https://openai.com/blog/better-language-models/
        number_of_norm_layers = 2 * self.number_of_decoder_layers + 3 * self.number_of_encoder_layers
        if hasattr(module, RESIDUAL_STREAM_FLAG):
            std = self.initializer_range / math.sqrt(number_of_norm_layers)
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


    def forward(
            self,
            hidden_states: torch.Tensor,
            self_attention_mask: torch.Tensor | None = None,
            key_padding_mask: torch.Tensor | None = None,
            decoder_cache: DecoderKVCache | None = None,
            use_cache: bool = False,
            deterministic: bool = False,
            is_inference: bool = False,
            return_latent_embeddings: bool = False,
    ) -> (tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, DecoderKVCache | None] |
          tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, DecoderKVCache | None]):
        """Forward pass through Free transformer with midpoint latent injection.

        Args:
            hidden_states: Input token embeddings (B, query_length, D)
            self_attention_mask: Optional custom self-attention mask (B, 1, query_length, query_length) where True means masked.
               If None, generates standard triangular causal mask.
            key_padding_mask: Optional current key padding mask for padded observation tokens (B, query_length) where True means masked.
            decoder_cache: Optional cached K/V from previous steps
            use_cache: Whether to return updated cache
            deterministic: Whether to use deterministic latent sampling (inference) or stochastic (training)
            is_inference: Whether the model is in inference mode (no logits, sample directly from one-hot vector).
            return_latent_embeddings: Whether to return the latent embeddings from the latent encoder.

        Returns:
            Tuple of (hidden_states, bit_logits, latent_codes, new_cache), where hidden_states has shape (B, query_len, D),
             optional bit_logits has shape (B, query_len, latent_bits), latent_codes has shape (B, query_len, 2**latent_bits),
             and new_cache is a LayerKVCache or None.
            If return_latent_embeddings is True, also returns latent embeddings with shape (B, query_len, D).

        Note:
            If self.use_global_latent is True, bit logits, latent codes and latent embeddings have all shape (B, 1, D).
        """
        if isinstance(self.positional_encoding, (SinusoidalPositionalEncoding1D, LearnedPositionalEncoding1D)):
            hidden_states += self.positional_encoding(hidden_states)
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        query_length = hidden_states.shape[1]
        cache_length = decoder_cache.get_length() if decoder_cache else 0
        cached_key_padding_mask = decoder_cache.key_padding_mask if decoder_cache else None
        total_mask, full_key_padding_mask = create_full_padding_mask(
            key_padding_mask=key_padding_mask,
            cached_key_padding_mask=cached_key_padding_mask,
            self_attention_mask=self_attention_mask,
            batch_size=batch_size,
            query_length=query_length,
            cache_length=cache_length,
            device=device,
        ) # (B, 1, query_length, key_length), (B, key_length), where key_length = cache_length + query_length

        decoder_caches = decoder_cache.layers if decoder_cache is not None else None
        if use_cache and decoder_caches is None:
            decoder_caches = initialize_decoder_cache(
                batch_size=batch_size,
                num_layers=self.number_of_decoder_layers,
                num_heads=self.number_of_key_value_heads,
                head_dimension=self.head_dimension,
                device=device,
                dtype=hidden_states.dtype,
            )
        new_decoder_layer_caches = []
        mid_features = hidden_states
        mid_cache_idx = 0
        rope_pe = self.positional_encoding if isinstance(self.positional_encoding, RotaryPositionalEncoding) else None
        # Forward pass through first half of decoder layers
        for layer in self.decoder_layers[:self.number_of_decoder_layers // 2]:
            cache = decoder_caches[mid_cache_idx] if decoder_caches is not None else None
            mid_features, new_cache = layer(
                hidden_states=mid_features,
                encoded_features=None,
                self_attention_mask=total_mask,
                cross_attention_mask=None,
                layer_cache=cache,
                use_cache=use_cache,
                positional_encoding=rope_pe,
            )
            if use_cache:
                new_decoder_layer_caches.append(new_cache)
            mid_cache_idx += 1

        # Generate latent
        mid_features_mask = key_padding_mask # (B, query_length) or None
        if self.training or not is_inference:
            z = self.latent_encoder(mid_features=mid_features, mid_features_mask=mid_features_mask) # (B, query_length or 1, D)
            # (B, query_length or 1, 2^H), (B, query_length or 1, H)
            latent_codes, bit_logits = self.binary_mapper(z, deterministic=deterministic)
        else:
            z = self.latent_encoder(mid_features=mid_features, mid_features_mask=mid_features_mask) # (B, query_length or 1, D)
            # Uniform prior sample
            if self.use_global_latent:
                query_dim = 1
            else:
                query_dim = query_length
            uniform_indices = torch.randint(0, self.latent_dim, (batch_size, query_dim), device=device, dtype=torch.long)
            latent_codes = F.one_hot(uniform_indices, num_classes=self.latent_dim).float()  # (B, query_length or 1, 2^H)
            bit_logits = None

        # Forward pass through latent-conditioned mid decoder layer
        layer = self.decoder_layers[self.number_of_decoder_layers // 2]
        cache = decoder_caches[mid_cache_idx] if decoder_caches is not None else None
        hidden_states, new_cache = layer(
            hidden_states=mid_features,
            encoded_features=None,
            self_attention_mask=total_mask,
            cross_attention_mask=None,
            layer_cache=cache,
            use_cache=use_cache,
            positional_encoding=rope_pe,
            latent=latent_codes,
        )
        if use_cache:
            new_decoder_layer_caches.append(new_cache)
        mid_cache_idx += 1

        for layer in self.decoder_layers[self.number_of_decoder_layers // 2 +1:]:
            cache = decoder_caches[mid_cache_idx] if decoder_caches is not None else None
            hidden_states, new_cache = layer(
                hidden_states=hidden_states,
                encoded_features=None,
                self_attention_mask=total_mask,
                cross_attention_mask=None,
                layer_cache=cache,
                use_cache=use_cache,
                positional_encoding=rope_pe,
            )
            if use_cache:
                new_decoder_layer_caches.append(new_cache)
            mid_cache_idx += 1

        hidden_states = self.final_normalization(hidden_states) # (B, query_length, D)
        new_decoder_cache = (
                DecoderKVCache(layers=new_decoder_layer_caches, key_padding_mask=full_key_padding_mask)
                if use_cache
                else None
            )
        if return_latent_embeddings:
            return hidden_states, bit_logits, latent_codes, z, new_decoder_cache
        return hidden_states, bit_logits, latent_codes, new_decoder_cache
