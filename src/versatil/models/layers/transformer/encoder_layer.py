"""Transformer encoder layer for bidirectional self-attention."""

import torch
import torch.nn as nn

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.normalization.factory import create_normalization_layer
from versatil.models.layers.positional_encoding.rotary import RotaryPositionalEncoding
from versatil.models.layers.transformer.attention import CachedAttention


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with self-attention and feed-forward network."""

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
        bias: bool = True,
        normalization_epsilon: float = 1e-6,
    ):
        """Initialize Transformer encoder layer.

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
            bias: Whether to use bias in linear layers
            normalization_epsilon: Epsilon for normalization layers
        """
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
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
        activation_enum = ActivationFunction(activation)
        if activation_enum.is_gated:
            self.feedforward_network = nn.Sequential(
                activation_enum.to_torch_activation()(
                    input_dim=embedding_dimension,
                    hidden_dim=feedforward_dimension,
                    bias=bias,
                ),
                nn.Dropout(dropout),
                nn.Linear(feedforward_dimension, embedding_dimension, bias=bias),
            )
        else:
            self.feedforward_network = nn.Sequential(
                nn.Linear(embedding_dimension, feedforward_dimension, bias=bias),
                activation_enum.to_torch_activation()(),
                nn.Dropout(dropout),
                nn.Linear(feedforward_dimension, embedding_dimension, bias=bias),
            )
        self.feedforward_normalization = create_normalization_layer(
            normalization_type=normalization_type,
            dimension=embedding_dimension,
            epsilon=normalization_epsilon,
        )
        # Flag for initialization (GPT2 style)
        self.feedforward_network[-1].SQUARE_ROOT_WEIGHT = True
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        positional_encoding: RotaryPositionalEncoding | None = None,
    ) -> torch.Tensor:
        """Forward pass through encoder layer.

        Args:
            hidden_states: Input embeddings (B, seq_len, D)
            attention_mask: Optional mask (B, 1, seq_len, seq_len) where True means masked.
            positional_encoding: Optional rotary positional encoding module

        Returns:
            Output hidden states (B, seq_len, D)
        """
        residual = hidden_states
        hidden_states = self.self_attention_normalization(hidden_states)
        self_attention_output, _ = self.self_attention(
            query_input=hidden_states,
            key_input=hidden_states,
            value_input=hidden_states,
            attention_mask=attention_mask,
            positional_encoding=positional_encoding,
        )
        hidden_states = residual + self.dropout(self_attention_output)
        residual = hidden_states
        hidden_states = self.feedforward_normalization(hidden_states)
        feedforward_output = self.feedforward_network(hidden_states)
        hidden_states = residual + self.dropout(feedforward_output)
        return hidden_states
