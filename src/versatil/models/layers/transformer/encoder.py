"""Bidirectional transformer encoder for sequence encoding."""

import torch
import torch.nn as nn

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.normalization.factory import create_normalization_layer
from versatil.models.layers.transformer.layer.encoder_layer import (
    TransformerEncoderLayer,
)
from versatil.models.layers.transformer.transformer_mixin import TransformerMixin


class TransformerEncoder(TransformerMixin, nn.Module):
    """Bidirectional transformer encoder for sequence encoding.

    Processes all tokens in parallel with bidirectional self-attention.
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
    ):
        """Initialize transformer encoder.

        Args:
            number_of_layers: Number of encoder layers.
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
        """
        super().__init__()
        self.number_of_layers = number_of_layers
        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        self.maximum_sequence_length = maximum_sequence_length
        self.initializer_range = initializer_range
        self.number_of_residual_blocks = 2  # Self-Attention + Feedforward
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
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embedding_dimension=embedding_dimension,
                    number_of_heads=number_of_heads,
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
                for _ in range(number_of_layers)
            ]
        )
        self.final_normalization = create_normalization_layer(
            normalization_type=normalization_type,
            dimension=embedding_dimension,
            epsilon=normalization_epsilon,
        )
        self.apply(self._init_weights)

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through transformer encoder.

        Args:
            hidden_states: Input embeddings (B, seq_length, D).
            padding_mask: Optional padding mask (B, seq_length).

        Returns:
            Output hidden states (B, seq_length, D).
        """
        sequence_length = hidden_states.shape[1]
        attention_mask = None
        if padding_mask is not None:
            attention_mask = self._expand_padding_mask(padding_mask, sequence_length)
        hidden_states, rope_pe = self._apply_positional_encoding(hidden_states)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                positional_encoding=rope_pe,
            )
        hidden_states = self.final_normalization(hidden_states)
        return hidden_states
