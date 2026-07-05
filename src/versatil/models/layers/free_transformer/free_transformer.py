"""Based on "The Free Transformer" (Fleuret, 2025) - arXiv:2510.17558
https://arxiv.org/abs/2510.17558

The Free Transformer extends decoder transformers by injecting learnable latent
variables into the middle layer, enabling conditional generation through a variational
autoencoder framework.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType
from versatil.models.layers.free_transformer.binary_mapper import BinaryMapper
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.normalization.factory import create_normalization_layer
from versatil.models.layers.positional_encoding.rotary import (
    RotaryPositionalEncoding,
)
from versatil.models.layers.transformer.cache.conditioning import (
    ConditioningLayerCache,
)
from versatil.models.layers.transformer.cache.generation import (
    GenerationCache,
    GenerationLayerCache,
    initialize_generation_cache,
)
from versatil.models.layers.transformer.layer.decoder_layer import (
    TransformerDecoderLayer,
)
from versatil.models.layers.transformer.masking import create_full_padding_mask
from versatil.models.layers.transformer.transformer_mixin import TransformerMixin


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
    ):
        """Initialize a decoder layer with midpoint latent K/V conditioning.

        Args:
            embedding_dimension: Decoder hidden dimension.
            number_of_heads: Number of query attention heads.
            latent_dim: Dimension of the one-hot latent code.
            number_of_key_value_heads: Number of key/value heads for GQA.
            feedforward_dimension: Feedforward hidden dimension.
            dropout: Residual dropout probability.
            attention_dropout: Attention dropout probability.
            activation: Feedforward activation name.
            normalization_type: Normalization type name.
            attention_type: Attention implementation name.
            bias: Whether linear projections use bias.
            normalization_epsilon: Epsilon for normalization layers.
        """
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
        )

        self.latent_proj = nn.Linear(latent_dim, embedding_dimension, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoded_features: torch.Tensor | None = None,
        self_attention_mask: torch.Tensor | None = None,
        cross_attention_mask: torch.Tensor | None = None,
        generation_cache: GenerationLayerCache | None = None,
        conditioning_cache: ConditioningLayerCache | None = None,
        positional_encoding: RotaryPositionalEncoding | None = None,
        latent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, GenerationLayerCache | None]:
        """Forward pass through latent-conditioned decoder layer.

        Args:
            hidden_states: Input embeddings (B, seq_len, embedding_dimension).
            encoded_features: Not used. Kept for signature compatibility.
            self_attention_mask: Causal mask (B, 1, query_len, key_len), True = masked.
            cross_attention_mask: Not used. Kept for signature compatibility.
            generation_cache: Cached K/V from the main sequence. When provided,
                an updated cache is returned.
            conditioning_cache: Not used. Kept for signature compatibility.
            positional_encoding: Optional rotary positional encoding module.
            latent: Latent conditioning (B, T, latent_dim) or (B, 1, latent_dim).

        Returns:
            Tuple of (output_states, updated GenerationLayerCache or None).
        """
        attention_block = self.self_attention_block
        normalization = attention_block.normalization
        attention = attention_block.attention

        residual = hidden_states
        hidden_states, gate = normalization(x=hidden_states, condition=None)
        if latent is not None:
            if latent.dim() != 3:
                raise ValueError(
                    f"latent must have shape (B, T, latent_dim), got {latent.shape}."
                )
            if latent.shape[-1] != self.latent_proj.in_features:
                raise ValueError(
                    f"latent last dimension must be {self.latent_proj.in_features}, "
                    f"got {latent.shape[-1]}."
                )
            latent = latent.to(device=hidden_states.device, dtype=hidden_states.dtype)
            projected_latent = self.latent_proj(latent)
            if projected_latent.shape[1] == 1 and hidden_states.shape[1] > 1:
                projected_latent = projected_latent.expand(
                    -1, hidden_states.shape[1], -1
                )
            if projected_latent.shape[1] != hidden_states.shape[1]:
                raise ValueError(
                    "latent sequence length must be 1 or match hidden_states, got "
                    f"{projected_latent.shape[1]} and {hidden_states.shape[1]}."
                )
            kv_residual = residual + projected_latent
            norm_kv, _ = normalization(x=kv_residual, condition=None)
            attention_output, cache = attention(
                query_input=hidden_states,
                key_input=norm_kv,
                value_input=norm_kv,
                attention_mask=self_attention_mask,
                generation_cache=generation_cache,
                positional_encoding=positional_encoding,
            )
        else:
            attention_output, cache = attention(
                query_input=hidden_states,
                key_input=hidden_states,
                value_input=hidden_states,
                attention_mask=self_attention_mask,
                generation_cache=generation_cache,
                positional_encoding=positional_encoding,
            )

        hidden_states = attention_block.apply_residual(residual, attention_output, gate)
        hidden_states = self.feedforward_block(hidden_states=hidden_states)

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
        """Initialize the training posterior latent encoder.

        Args:
            embedding_dimension: Hidden dimension of mid-decoder features.
            number_of_layers: Number of cross-attention encoder layers.
            number_of_heads: Number of query attention heads.
            number_of_key_value_heads: Number of key/value heads for GQA.
            feedforward_dimension: Feedforward hidden dimension.
            dropout: Residual dropout probability.
            attention_dropout: Attention dropout probability.
            activation: Feedforward activation name.
            normalization_type: Normalization type name.
            attention_type: Attention implementation name.
            bias: Whether linear projections use bias.
            normalization_epsilon: Epsilon for normalization layers.
            use_global_latent: Whether to predict one latent for the sequence.
        """
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
        )
        self.use_global_latent = use_global_latent
        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(number_of_layers)]
        )
        self.final_normalization = create_normalization_layer(
            normalization_type, embedding_dimension, normalization_epsilon
        )

    def forward(
        self,
        mid_features: torch.Tensor,
        mid_features_mask: torch.Tensor | None = None,
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
        batch_size, sequence_length, _ = mid_features.shape
        if self.use_global_latent:
            target = self.learned_query.expand(
                batch_size, 1, -1
            )  # (B, 1, embedding_dimension)
        else:
            target = self.learned_query.expand(
                batch_size, sequence_length, -1
            )  # (B, T, embedding_dimension)
        if mid_features_mask is not None:
            # Expand (B, T) -> (B, 1, 1, T) for broadcast in cross-attn (Q=T, K=T)
            mid_features_mask = mid_features_mask.unsqueeze(1).unsqueeze(2)
        for layer in self.layers:
            target, _ = layer(
                hidden_states=target,
                encoded_features=mid_features,
                self_attention_mask=None,
                cross_attention_mask=mid_features_mask,
            )

        return self.final_normalization(target)  # (B, T, embedding_dimension)


class FreeTransformer(TransformerMixin, nn.Module):
    """Free Transformer model (Fleuret, 2025).

    Contains:
    - FreeTransformer (main autoregressive decoder with latent injection at middle layer)
    - FreeTransformerLatentEncoder (training-only latent predictor)
    - BinaryMapper (converts latent embedding to discrete one-hot codes + logits for KL)

    Training forward returns (output, kl_logits)
    Inference forward returns output only
    """

    def __init__(
        self,
        latent_bits: int = 16,  # 2**16 = 65536, as in paper
        latent_dim: int | None = None,  # overrides 2**latent_bits if set
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
        """Initialize the Free Transformer.

        Args:
            latent_bits: Number of binary latent bits.
            latent_dim: Optional explicit latent dimension. Must equal
                ``2 ** latent_bits`` when provided.
            number_of_decoder_layers: Number of autoregressive decoder layers.
            number_of_encoder_layers: Number of latent encoder layers.
            embedding_dimension: Transformer hidden dimension.
            number_of_heads: Number of query attention heads.
            number_of_key_value_heads: Number of key/value heads for GQA.
            feedforward_dimension: Feedforward hidden dimension.
            dropout: Residual dropout probability.
            attention_dropout: Attention dropout probability.
            attention_type: Attention implementation name.
            activation: Feedforward activation name.
            normalization_type: Normalization type name.
            positional_encoding_type: Optional positional encoding type.
            maximum_sequence_length: Maximum sequence length for positional encoding.
            bias: Whether linear projections use bias.
            use_global_latent: Whether to sample one latent for the sequence.
            normalization_epsilon: Epsilon for normalization layers.
            initializer_range: Standard deviation for weight initialization.

        Raises:
            ValueError: If layer counts or latent dimensions are invalid.
        """
        super().__init__()

        if latent_bits <= 0:
            raise ValueError(f"latent_bits must be positive, got {latent_bits}.")
        expected_latent_dim = 1 << latent_bits
        if latent_dim is None:
            latent_dim = expected_latent_dim
        elif latent_dim != expected_latent_dim:
            raise ValueError(
                f"latent_dim must equal 2 ** latent_bits ({expected_latent_dim}), "
                f"got {latent_dim}."
            )
        if number_of_decoder_layers <= 0 or number_of_decoder_layers % 2 != 0:
            raise ValueError(
                "number_of_decoder_layers must be a positive even integer, "
                f"got {number_of_decoder_layers}."
            )
        if number_of_encoder_layers < 0:
            raise ValueError(
                "number_of_encoder_layers must be non-negative, "
                f"got {number_of_encoder_layers}."
            )

        self.number_of_decoder_layers = number_of_decoder_layers
        self.number_of_encoder_layers = number_of_encoder_layers
        self.use_global_latent = use_global_latent
        self.embedding_dimension = embedding_dimension
        self.latent_dim = latent_dim
        self.latent_bits = latent_bits
        self.number_of_heads = number_of_heads
        self.maximum_sequence_length = maximum_sequence_length
        self.initializer_range = initializer_range
        self.number_of_key_value_heads, self.head_dimension = (
            self._resolve_attention_dimensions(
                embedding_dimension=embedding_dimension,
                number_of_heads=number_of_heads,
                number_of_key_value_heads=number_of_key_value_heads,
                attention_type=attention_type,
            )
        )
        self._setup_positional_encoding(
            positional_encoding_type=positional_encoding_type,
            embedding_dimension=embedding_dimension,
            maximum_sequence_length=maximum_sequence_length,
            number_of_heads=number_of_heads,
        )
        decoder_layer = TransformerDecoderLayer(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_key_value_heads=self.number_of_key_value_heads,
            feedforward_dimension=feedforward_dimension,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation=activation,
            normalization_type=normalization_type,
            attention_type=attention_type,
            use_cross_attention=False,  # No cross-attention
            bias=bias,
            normalization_epsilon=normalization_epsilon,
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
                        number_of_key_value_heads=self.number_of_key_value_heads,
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
            epsilon=normalization_epsilon,
        )
        self.latent_encoder = FreeTransformerLatentEncoder(
            embedding_dimension=embedding_dimension,
            number_of_layers=number_of_encoder_layers,
            number_of_heads=number_of_heads,
            normalization_type=normalization_type,
            number_of_key_value_heads=self.number_of_key_value_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation=activation,
            feedforward_dimension=feedforward_dimension,
            attention_type=attention_type,
            bias=bias,
            normalization_epsilon=normalization_epsilon,
            use_global_latent=use_global_latent,
        )
        self.binary_mapper = BinaryMapper(
            latent_bits=latent_bits,
            embedding_dimension=embedding_dimension,
        )
        self.apply(self._init_weights)

    @property
    def _total_residual_streams(self) -> int:
        """Decoder layers have 2 residual streams, encoder layers have 3."""
        return 2 * self.number_of_decoder_layers + 3 * self.number_of_encoder_layers

    def create_empty_generation_cache(
        self,
        batch_size: int,
        device: torch.device,
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
                num_layers=self.number_of_decoder_layers,
                number_of_heads=self.number_of_key_value_heads,
                head_dimension=self.head_dimension,
                device=device,
                dtype=dtype,
            ),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        self_attention_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        generation_cache: GenerationCache | None = None,
        deterministic: bool = False,
        is_inference: bool = False,
        return_latent_embeddings: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, GenerationCache | None]
        | tuple[
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor,
            torch.Tensor,
            GenerationCache | None,
        ]
    ):
        """Forward pass through Free transformer with midpoint latent injection.

        Args:
            hidden_states: Input token embeddings (B, query_length, D)
            self_attention_mask: Optional custom self-attention mask (B, 1, query_length, query_length) where True means masked.
                If None, generates standard triangular causal mask.
            key_padding_mask: Optional current key padding mask for padded observation tokens (B, query_length) where True means masked.
            generation_cache: Cached K/V from previous generation steps. When provided,
                an updated cache is returned.
            deterministic: Whether to use deterministic latent sampling (inference) or stochastic (training)
            is_inference: Whether the model is in inference mode (no logits, sample directly from one-hot vector).
            return_latent_embeddings: Whether to return the latent embeddings from the latent encoder.

        Returns:
            Tuple of (hidden_states, bit_logits, latent_codes, new_cache), where hidden_states has shape (B, query_len, D),
                optional bit_logits has shape (B, query_len, latent_bits), latent_codes has shape (B, query_len, 2**latent_bits),
                and new_cache is a GenerationLayerCache or None.
            If return_latent_embeddings is True, also returns latent embeddings with shape (B, query_len, D).

        Note:
            If self.use_global_latent is True, bit logits, latent codes and
            latent embeddings use one sequence position.
        """
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        query_length = hidden_states.shape[1]
        cache_length = generation_cache.get_length() if generation_cache else 0
        hidden_states, rope_pe = self._apply_positional_encoding(
            hidden_states=hidden_states, offset=cache_length
        )
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

        use_cache = generation_cache is not None
        generation_caches = generation_cache.layers if use_cache else None
        new_decoder_generation_caches = []
        mid_features = hidden_states
        mid_cache_idx = 0
        # Forward pass through first half of decoder layers
        for layer in self.decoder_layers[: self.number_of_decoder_layers // 2]:
            cache = (
                generation_caches[mid_cache_idx]
                if generation_caches is not None
                else None
            )
            mid_features, new_cache = layer(
                hidden_states=mid_features,
                encoded_features=None,
                self_attention_mask=total_mask,
                cross_attention_mask=None,
                generation_cache=cache,
                positional_encoding=rope_pe,
            )
            if use_cache:
                new_decoder_generation_caches.append(new_cache)
            mid_cache_idx += 1

        # Generate latent
        mid_features_mask = key_padding_mask  # (B, query_length) or None
        if self.training or not is_inference:
            z = self.latent_encoder(
                mid_features=mid_features, mid_features_mask=mid_features_mask
            )  # (B, query_length or 1, D)
            # (B, query_length or 1, 2^H), (B, query_length or 1, H)
            latent_codes, bit_logits = self.binary_mapper(
                z, deterministic=deterministic
            )
        else:
            z = self.latent_encoder(
                mid_features=mid_features, mid_features_mask=mid_features_mask
            )  # (B, query_length or 1, D)
            # Uniform prior sample
            query_dim = 1 if self.use_global_latent else query_length
            uniform_indices = torch.randint(
                0,
                self.latent_dim,
                (batch_size, query_dim),
                device=device,
                dtype=torch.long,
            )
            latent_codes = F.one_hot(uniform_indices, num_classes=self.latent_dim).to(
                dtype=mid_features.dtype
            )  # (B, query_length or 1, 2^H)
            bit_logits = None

        # Forward pass through latent-conditioned mid decoder layer
        layer = self.decoder_layers[self.number_of_decoder_layers // 2]
        cache = (
            generation_caches[mid_cache_idx] if generation_caches is not None else None
        )
        hidden_states, new_cache = layer(
            hidden_states=mid_features,
            encoded_features=None,
            self_attention_mask=total_mask,
            cross_attention_mask=None,
            generation_cache=cache,
            positional_encoding=rope_pe,
            latent=latent_codes,
        )
        if use_cache:
            new_decoder_generation_caches.append(new_cache)
        mid_cache_idx += 1

        for layer in self.decoder_layers[self.number_of_decoder_layers // 2 + 1 :]:
            cache = (
                generation_caches[mid_cache_idx]
                if generation_caches is not None
                else None
            )
            hidden_states, new_cache = layer(
                hidden_states=hidden_states,
                encoded_features=None,
                self_attention_mask=total_mask,
                cross_attention_mask=None,
                generation_cache=cache,
                positional_encoding=rope_pe,
            )
            if use_cache:
                new_decoder_generation_caches.append(new_cache)
            mid_cache_idx += 1

        hidden_states = self.final_normalization(hidden_states)  # (B, query_length, D)
        new_generation_cache = (
            GenerationCache(
                layers=new_decoder_generation_caches,
                key_padding_mask=full_key_padding_mask,
            )
            if use_cache
            else None
        )
        if return_latent_embeddings:
            return hidden_states, bit_logits, latent_codes, z, new_generation_cache
        else:
            return hidden_states, bit_logits, latent_codes, new_generation_cache
