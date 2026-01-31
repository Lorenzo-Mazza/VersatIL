"""Conditional transformer decoder layer with AdaLN/FiLM support."""
from typing import Literal

import torch
import torch.nn as nn

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType, ConditioningType
from versatil.models.layers.modulation.conditional_modulation import (
    ConditionalModulation,
)
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.normalization.factory import create_normalization_layer
from versatil.models.layers.positional_encoding.rotary import RotaryPositionalEncoding
from versatil.models.layers.swiglu import SwiGLU
from versatil.models.layers.transformer.attention import CachedAttention


class ConditionalTransformerDecoderLayer(nn.Module):
    """Transformer decoder layer with conditional modulation support.

    Extends TransformerDecoderLayer to add conditioning via AdaLN or FiLM
    after each sub-layer's normalization.

    Architecture:
        x = x + SelfAttn(Modulate(Norm(x), condition))
        x = x + CrossAttn(Modulate(Norm(x), condition), encoded_features)
        x = x + FFN(Modulate(Norm(x), condition))
    """

    def __init__(
        self,
        embedding_dimension: int,
        condition_dimension: int,
        number_of_heads: int,
        number_of_key_value_heads: int | None = None,
        feedforward_dimension: int | None = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        activation: str = ActivationFunction.SWIGLU.value,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        attention_type: str = AttentionType.GROUPED_QUERY.value,
        conditioning_type: str = ConditioningType.ADALN.value,
        use_cross_attention: bool = True,
        bias: bool = True,
        normalization_epsilon: float = 1e-6,
        modulation_init_strategy: Literal["identity", "xavier", "zero"] = "identity",
    ):
        """Initialize conditional transformer decoder layer.

        Args:
            embedding_dimension: Model embedding dimension
            condition_dimension: Dimension of conditioning vector (e.g., latent dim)
            number_of_heads: Number of attention heads
            number_of_key_value_heads: Number of K/V heads (for GQA)
            feedforward_dimension: FFN hidden dimension (defaults to 4 * embedding_dimension)
            dropout: Dropout probability for residual connections
            attention_dropout: Dropout probability for attention weights
            activation: Activation function (use ActivationFunction enum values)
            normalization_type: Type of normalization (use NormalizationType enum values)
            attention_type: Type of attention (use AttentionType enum values)
            conditioning_type: Type of conditioning - "adaln" or "film"
            use_cross_attention: Whether to use cross-attention
            bias: Whether to use bias in linear layers
            normalization_epsilon: Epsilon for normalization layers
            modulation_init_strategy: Initialization for modulation ("identity", "xavier", "zero")
        """
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.condition_dimension = condition_dimension
        self.number_of_heads = number_of_heads
        self.use_cross_attention = use_cross_attention
        self.conditioning_type = ConditioningType(conditioning_type)
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
        self.self_attention_modulation = ConditionalModulation(
            condition_dim=condition_dimension,
            feature_dim=embedding_dimension,
            use_shift=True,
            init_strategy=modulation_init_strategy,
        )
        if use_cross_attention:
            self.cross_attention = CachedAttention(
                embedding_dimension=embedding_dimension,
                number_of_heads=number_of_heads,
                number_of_key_value_heads=number_of_key_value_heads,
                dropout=attention_dropout,
                bias=bias,
                attention_type=attention_type,
            )
            self.cross_attention_normalization = create_normalization_layer(
                normalization_type=normalization_type,
                dimension=embedding_dimension,
                epsilon=normalization_epsilon,
            )
            self.cross_attention_modulation = ConditionalModulation(
                condition_dim=condition_dimension,
                feature_dim=embedding_dimension,
                use_shift=True,
                init_strategy=modulation_init_strategy,
            )
        else:
            self.cross_attention = None
            self.cross_attention_normalization = None
            self.cross_attention_modulation = None

        if activation == ActivationFunction.SWIGLU.value:
            self.feedforward_network = nn.Sequential(
                SwiGLU(
                    input_dim=embedding_dimension,
                    hidden_dim=feedforward_dimension,
                    bias=bias,
                ),
                nn.Dropout(dropout),
                nn.Linear(feedforward_dimension, embedding_dimension, bias=bias),
            )
        else:
            activation_class = ActivationFunction(activation).to_torch_activation()
            self.feedforward_network = nn.Sequential(
                nn.Linear(embedding_dimension, feedforward_dimension, bias=bias),
                activation_class(),
                nn.Dropout(dropout),
                nn.Linear(feedforward_dimension, embedding_dimension, bias=bias),
            )

        self.feedforward_normalization = create_normalization_layer(
            normalization_type=normalization_type,
            dimension=embedding_dimension,
            epsilon=normalization_epsilon,
        )
        self.feedforward_modulation = ConditionalModulation(
            condition_dim=condition_dimension,
            feature_dim=embedding_dimension,
            use_shift=True,
            init_strategy=modulation_init_strategy,
        )
        self.feedforward_network[-1].SQUARE_ROOT_WEIGHT = True
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        condition: torch.Tensor,
        encoded_features: torch.Tensor | None = None,
        self_attention_mask: torch.Tensor | None = None,
        cross_attention_mask: torch.Tensor | None = None,
        positional_encoding: RotaryPositionalEncoding | None = None,
    ) -> torch.Tensor:
        """Forward pass with conditional modulation.

        Args:
            hidden_states: Input embeddings (B, seq_len, D)
            condition: Conditioning vector (B, condition_dim)
            encoded_features: Encoded visual features (B, num_features, D)
            self_attention_mask: Optional mask for self-attention (B, 1, q_len, k_len)
            cross_attention_mask: Optional mask for cross-attention (B, 1, q_len, k_len)
            positional_encoding: Optional rotary positional encoding module

        Returns:
            Output hidden states (B, seq_len, D)
        """
        residual = hidden_states
        hidden_states = self.self_attention_normalization(hidden_states)
        hidden_states = self.self_attention_modulation(hidden_states, condition)
        self_attention_output, _ = self.self_attention(
            query_input=hidden_states,
            key_input=hidden_states,
            value_input=hidden_states,
            attention_mask=self_attention_mask,
            layer_cache=None,
            use_self_attention_cache=False,
            use_cross_attention_cache=False,
            positional_encoding=positional_encoding,
        )
        hidden_states = residual + self.dropout(self_attention_output)
        if self.use_cross_attention:
            if encoded_features is None:
                raise ValueError(
                    "encoded_features required when use_cross_attention=True"
                )
            if self.cross_attention_modulation is None or self.cross_attention is None:
                raise ValueError(
                    "cross_attention_modulation and cross_attention modules are required when use_cross_attention=True"
                )
            residual = hidden_states
            hidden_states = self.cross_attention_normalization(hidden_states)
            hidden_states = self.cross_attention_modulation(hidden_states, condition)
            cross_attention_output, _ = self.cross_attention(
                query_input=hidden_states,
                key_input=encoded_features,
                value_input=encoded_features,
                attention_mask=cross_attention_mask,
                layer_cache=None,
                use_self_attention_cache=False,
                use_cross_attention_cache=False,
            )
            hidden_states = residual + self.dropout(cross_attention_output)

        residual = hidden_states
        hidden_states = self.feedforward_normalization(hidden_states)
        hidden_states = self.feedforward_modulation(hidden_states, condition)
        feedforward_output = self.feedforward_network(hidden_states)
        hidden_states = residual + self.dropout(feedforward_output)
        return hidden_states
