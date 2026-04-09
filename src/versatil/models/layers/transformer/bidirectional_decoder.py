"""Bidirectional transformer decoder for non-autoregressive generation."""

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


class BidirectionalDecoder(TransformerMixin, nn.Module):
    """Bidirectional transformer decoder for non-autoregressive generation.

    All tokens are processed in parallel (no causal mask). Supports optional
    cross-attention to encoded features and conditioning cache for reusing
    precomputed K/V projections across repeated forward calls.
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
        positional_encoding_type: str | None = None,
        maximum_sequence_length: int = 2048,
        bias: bool = True,
        normalization_epsilon: float = 1e-6,
        initializer_range: float = 0.02,
        use_cross_attention: bool = True,
    ):
        """Initialize bidirectional decoder.

        Args:
            number_of_layers: Number of decoder layers.
            embedding_dimension: Model embedding dimension.
            number_of_heads: Number of attention heads.
            number_of_key_value_heads: Number of K/V heads (for GQA).
            feedforward_dimension: FFN hidden dimension.
            dropout: Dropout probability for residual connections.
            attention_dropout: Dropout probability for attention weights.
            activation: Activation function (use ActivationFunction enum values).
            normalization_type: Type of normalization (use NormalizationType enum values).
            attention_type: Type of attention (use AttentionType enum values).
            positional_encoding_type: Type of positional encoding (or None).
            maximum_sequence_length: Maximum sequence length for positional encoding.
            bias: Whether to use bias in linear layers.
            normalization_epsilon: Epsilon for normalization layers.
            initializer_range: Standard deviation for weight initialization.
            use_cross_attention: Whether to include cross-attention blocks.
        """
        super().__init__()
        self.number_of_layers = number_of_layers
        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        self.use_cross_attention = use_cross_attention
        self.maximum_sequence_length = maximum_sequence_length
        self.initializer_range = initializer_range
        self.number_of_residual_blocks = (
            3 if use_cross_attention else 2
        )  # Self-Attention (+ Cross-Attention) + Feedforward
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
        encoded_features: torch.Tensor | None = None,
        conditioning_cache: ConditioningCache | None = None,
        query_padding_mask: torch.Tensor | None = None,
        memory_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through bidirectional decoder.

        Args:
            hidden_states: Query embeddings (B, query_length, D).
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
            )
        hidden_states = self.final_normalization(hidden_states)
        return hidden_states
