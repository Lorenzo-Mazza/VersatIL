"""DiT decoder layer with cross-attention for conditioning (PixArt-style)."""

import torch
import torch.nn as nn

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType
from versatil.models.layers.normalization.ada_norm import AdaNorm
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.normalization.factory import create_normalization_layer
from versatil.models.layers.positional_encoding.rotary import RotaryPositionalEncoding
from versatil.models.layers.swiglu import SwiGLU
from versatil.models.layers.transformer import CachedAttention


class CrossConditioningDecoderLayer(nn.Module):
    """DiT decoder layer with cross-attention for conditioning (PixArt-style)."""

    def __init__(
        self,
        embedding_dimension: int,
        timestep_dimension: int,
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
        use_gating: bool = True,
    ):
        """Initialize the cross-conditioning decoder layer.

        Args:
            embedding_dimension: Hidden dimension of the transformer.
            timestep_dimension: Dimension of the timestep embedding.
            number_of_heads: Number of attention heads.
            number_of_key_value_heads: Number of key/value heads (for grouped query attention).
            feedforward_dimension: Dimension of the feedforward network.
            dropout: Dropout rate.
            attention_dropout: Dropout rate for attention weights.
            activation: Activation function name.
            normalization_type: Type of normalization to use.
            attention_type: Type of attention to use.
            bias: Whether to use bias in linear layers.
            normalization_epsilon: Epsilon value for normalization layers.
            use_gating: Whether to use gating in AdaNorm (AdaLN-Zero style).
        """
        super().__init__()
        self.use_gating = use_gating
        if feedforward_dimension is None:
            feedforward_dimension = 4 * embedding_dimension
        base_normalization_layer = create_normalization_layer(
            normalization_type=normalization_type,
            dimension=embedding_dimension,
            epsilon=normalization_epsilon,
        )
        self.self_attention_normalization = AdaNorm(
            base_norm=base_normalization_layer,
            condition_dim=timestep_dimension,
            feature_dim=embedding_dimension,
            use_gate=use_gating,
        )
        self.self_attention = CachedAttention(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_key_value_heads=number_of_key_value_heads,
            dropout=attention_dropout,
            bias=bias,
            attention_type=attention_type,
        )
        self.self_attention_dropout = nn.Dropout(dropout)
        self.cross_attention_normalization = create_normalization_layer(
            normalization_type=normalization_type,
            dimension=embedding_dimension,
            epsilon=normalization_epsilon,
        )
        self.cross_attention = CachedAttention(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_key_value_heads=number_of_key_value_heads,
            dropout=attention_dropout,
            bias=bias,
            attention_type=attention_type,
        )
        self.cross_attention_dropout = nn.Dropout(dropout)
        self.feedforward_normalization = AdaNorm(
            base_norm=create_normalization_layer(
                normalization_type=normalization_type,
                dimension=embedding_dimension,
                epsilon=normalization_epsilon,
            ),
            condition_dim=timestep_dimension,
            feature_dim=embedding_dimension,
            use_gate=use_gating,
        )
        if activation == ActivationFunction.SWIGLU.value:
            self.feedforward_network = nn.Sequential(
                SwiGLU(embedding_dimension, feedforward_dimension),
                nn.Dropout(dropout),
                nn.Linear(feedforward_dimension, embedding_dimension),
            )
        else:
            activation_function = ActivationFunction(activation).to_torch_activation()()
            self.feedforward_network = nn.Sequential(
                nn.Linear(embedding_dimension, feedforward_dimension),
                activation_function,
                nn.Dropout(dropout),
                nn.Linear(feedforward_dimension, embedding_dimension),
            )
        self.feedforward_dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning_embedding: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        self_attention_mask: torch.Tensor | None = None,
        cross_attention_mask: torch.Tensor | None = None,
        positional_encoding: RotaryPositionalEncoding | None = None,
    ) -> torch.Tensor:
        """Forward pass through the decoder layer.

        Args:
            hidden_states: Input tokens (B, T, D).
            conditioning_embedding: Timestep conditioning embedding (B, D).
            encoder_hidden_states: Encoder output for cross-attention (B, S, D).
            self_attention_mask: Self-attention mask (B, 1, T, T) where True means masked.
            cross_attention_mask: Cross-attention mask (B, 1, T, S) where True means masked.
            positional_encoding: Optional rotary positional encoding module.

        Returns:
            Output tokens (B, T, D).
        """
        residual = hidden_states
        if self.use_gating:
            hidden_states, gate_self_attention = self.self_attention_normalization(
                x=hidden_states, condition=conditioning_embedding
            )
        else:
            hidden_states = self.self_attention_normalization(
                x=hidden_states, condition=conditioning_embedding
            )
            gate_self_attention = 1.0
        hidden_states, _ = self.self_attention(
            query_input=hidden_states,
            key_input=hidden_states,
            value_input=hidden_states,
            attention_mask=self_attention_mask,
            positional_encoding=positional_encoding,
        )
        hidden_states = residual + gate_self_attention * self.self_attention_dropout(
            hidden_states
        )

        residual = hidden_states
        hidden_states = self.cross_attention_normalization(hidden_states)
        hidden_states, _ = self.cross_attention(
            query_input=hidden_states,
            key_input=encoder_hidden_states,
            value_input=encoder_hidden_states,
            attention_mask=cross_attention_mask,
        )
        hidden_states = residual + self.cross_attention_dropout(hidden_states)
        residual = hidden_states
        if self.use_gating:
            hidden_states, gate_feedforward = self.feedforward_normalization(
                x=hidden_states, condition=conditioning_embedding
            )
        else:
            hidden_states = self.feedforward_normalization(
                x=hidden_states, condition=conditioning_embedding
            )
            gate_feedforward = 1.0
        hidden_states = self.feedforward_network(hidden_states)
        hidden_states = residual + gate_feedforward * self.feedforward_dropout(
            hidden_states
        )
        return hidden_states
