"""Conditional bidirectional transformer decoder with latent conditioning."""

import torch
import torch.nn as nn

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.normalization.factory import create_normalization_layer
from versatil.models.layers.transformer.cache.conditioning import (
    ConditioningCache,
    precompute_conditioning,
)
from versatil.models.layers.transformer.layer.decoder_layer import (
    TransformerDecoderLayer,
)
from versatil.models.layers.transformer.transformer_mixin import TransformerMixin


class ConditionalBidirectionalDecoder(TransformerMixin, nn.Module):
    """Bidirectional transformer decoder with conditional modulation.

    Each transformer layer uses adaptive normalization (AdaNorm) to condition
    its representations on a conditioning signal throughout the network.
    Supports optional cross-attention to encoded features.
    """

    def __init__(
        self,
        number_of_layers: int,
        embedding_dimension: int,
        conditioning_dimension: int,
        number_of_heads: int,
        number_of_key_value_heads: int | None = None,
        feedforward_dimension: int | None = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        activation: str = ActivationFunction.SWIGLU.value,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        use_gating: bool = False,
        attention_type: str = AttentionType.GROUPED_QUERY.value,
        positional_encoding_type: str | None = None,
        maximum_sequence_length: int = 2048,
        bias: bool = True,
        normalization_epsilon: float = 1e-6,
        initializer_range: float = 0.02,
        use_cross_attention: bool = True,
        cross_attention_conditioning_dimension: int | None = None,
        cross_attention_normalization_type: str | None = None,
        use_final_normalization: bool = True,
        condition_final_normalization: bool = True,
    ):
        """Initialize conditional bidirectional decoder.

        Args:
            number_of_layers: Number of decoder layers.
            embedding_dimension: Model embedding dimension.
            conditioning_dimension: Dimension of conditioning vector.
            number_of_heads: Number of attention heads.
            number_of_key_value_heads: Number of K/V heads (for GQA).
            feedforward_dimension: FFN hidden dimension.
            dropout: Dropout probability for residual connections.
            attention_dropout: Dropout probability for attention weights.
            activation: Activation function (use ActivationFunction enum values).
            normalization_type: Normalization type for self-attention and FFN.
            use_gating: Whether to use gating in adaptive normalization (AdaLN-Zero).
            attention_type: Type of attention (use AttentionType enum values).
            positional_encoding_type: Type of positional encoding (or None).
            maximum_sequence_length: Maximum sequence length for positional encoding.
            bias: Whether to use bias in linear layers.
            normalization_epsilon: Epsilon for normalization layers.
            initializer_range: Standard deviation for weight initialization.
            use_cross_attention: Whether to include cross-attention blocks.
            cross_attention_conditioning_dimension: Conditioning dimension for
                cross-attention normalization. None means unconditioned cross-attention.
            cross_attention_normalization_type: Normalization type for cross-attention.
                Defaults to normalization_type when None.
            use_final_normalization: Whether to apply final normalization.
            condition_final_normalization: Whether final normalization is conditioned.
                When False, uses plain normalization regardless of condition_dimension.
        """
        super().__init__()
        self.number_of_layers = number_of_layers
        self.embedding_dimension = embedding_dimension
        self.condition_dimension = conditioning_dimension
        self.use_cross_attention = use_cross_attention
        self.maximum_sequence_length = maximum_sequence_length
        self.initializer_range = initializer_range
        self.number_of_heads = number_of_heads
        self.condition_final_normalization = condition_final_normalization
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
                    autoregressive=False,
                    conditioning_dimension=conditioning_dimension,
                    use_gating=use_gating,
                    cross_attention_conditioning_dimension=cross_attention_conditioning_dimension,
                    cross_attention_normalization_type=cross_attention_normalization_type,
                )
                for _ in range(number_of_layers)
            ]
        )
        self.final_normalization = None
        if use_final_normalization:
            final_condition_dim = (
                conditioning_dimension if condition_final_normalization else None
            )
            self.final_normalization = create_normalization_layer(
                normalization_type=cross_attention_normalization_type
                or normalization_type,
                dimension=embedding_dimension,
                epsilon=normalization_epsilon,
                condition_dim=final_condition_dim,
            )
        self.apply(self._init_weights)

    def precompute_conditioning_kv(
        self,
        encoded_features: torch.Tensor,
    ) -> ConditioningCache:
        """Precompute conditioning K/V for all layers for forward pass reuse."""
        return precompute_conditioning(
            layers=self.layers,  # type: ignore[arg-type]
            encoded_features=encoded_features,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        condition: torch.Tensor,
        encoded_features: torch.Tensor | None = None,
        conditioning_cache: ConditioningCache | None = None,
        query_padding_mask: torch.Tensor | None = None,
        memory_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through conditional bidirectional decoder.

        Args:
            hidden_states: Query embeddings (B, query_length, D).
            condition: Conditioning vector (B, conditioning_dimension).
            encoded_features: Encoder features to cross-attend to (B, memory_length, D).
            conditioning_cache: Precomputed K/V for static conditioning. When provided,
                encoded_features is not needed for cross-attention.
            query_padding_mask: Optional padding mask for queries (B, query_length).
            memory_padding_mask: Optional padding mask for memory (B, memory_length).

        Returns:
            Output hidden states (B, query_length, D).

        Raises:
            ValueError: If use_cross_attention=True but neither encoded_features
                nor conditioning_cache is provided.
        """
        if self.use_cross_attention and (
            encoded_features is None and conditioning_cache is None
        ):
            raise ValueError(
                "Either encoded_features or conditioning_cache must be provided "
                "when use_cross_attention=True."
            )

        query_length = hidden_states.shape[1]
        self_attention_mask = None
        if query_padding_mask is not None:
            self_attention_mask = self._expand_padding_mask(
                query_padding_mask, query_length
            )
        cross_attention_mask = None
        if memory_padding_mask is not None:
            cross_attention_mask = self._expand_padding_mask(
                memory_padding_mask, query_length
            )
        hidden_states, rope_pe = self._apply_positional_encoding(hidden_states)
        for layer_index, layer in enumerate(self.layers):
            hidden_states, _ = layer(
                hidden_states=hidden_states,
                encoded_features=encoded_features,
                self_attention_mask=self_attention_mask,
                cross_attention_mask=cross_attention_mask,
                conditioning_cache=conditioning_cache[layer_index]
                if conditioning_cache
                else None,
                positional_encoding=rope_pe,
                conditioning=condition,
            )
        if self.final_normalization is not None:
            if self.condition_final_normalization:
                hidden_states, _ = self.final_normalization(hidden_states, condition)
            else:
                hidden_states = self.final_normalization(hidden_states)
        return hidden_states
