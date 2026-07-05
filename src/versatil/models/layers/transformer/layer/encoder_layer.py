"""Transformer encoder layer inspired by the original "Attention is All You Need" paper, with
bidirectional self-attention and optional conditioning."""

import torch
import torch.nn as nn

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.normalization.factory import create_block_normalization
from versatil.models.layers.positional_encoding.rotary import RotaryPositionalEncoding
from versatil.models.layers.transformer.attention.cached_attention import (
    CachedAttention,
)
from versatil.models.layers.transformer.block.feedforward import (
    FeedforwardBlock,
    build_feedforward,
)
from versatil.models.layers.transformer.block.self_attention import (
    SelfAttentionBlock,
)


class TransformerEncoderLayer(nn.Module):
    """Self-attention + feedforward blocks.

    Note:
        Supports optional conditioning when constructed with adaptive
        normalization types.
    """

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
        conditioning_dimension: int | None = None,
        use_gating: bool = False,
    ):
        """Initialize Transformer encoder layer.

        Args:
            embedding_dimension: Model embedding dimension.
            number_of_heads: Number of attention heads.
            number_of_key_value_heads: Number of K/V heads (for GQA).
            feedforward_dimension: FFN hidden dimension (defaults to 4 * embedding_dimension).
            dropout: Dropout probability for residual connections.
            attention_dropout: Dropout probability for attention weights.
            activation: Activation function (use ActivationFunction enum values).
            normalization_type: Type of normalization (use NormalizationType enum values).
            attention_type: Type of attention (use AttentionType enum values).
            bias: Whether to use bias in linear layers.
            normalization_epsilon: Epsilon for normalization layers.
            conditioning_dimension: Conditioning dimension for adaptive normalization.
                Required when normalization_type is adaptive.
            use_gating: Whether to use gating in adaptive normalization (AdaLN-Zero).
        """
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        if feedforward_dimension is None:
            feedforward_dimension = 4 * embedding_dimension
        self.self_attention_block = SelfAttentionBlock(
            attention=CachedAttention(
                embedding_dimension=embedding_dimension,
                number_of_heads=number_of_heads,
                number_of_key_value_heads=number_of_key_value_heads,
                dropout=attention_dropout,
                bias=bias,
                attention_type=attention_type,
            ),
            normalization=create_block_normalization(
                normalization_type=normalization_type,
                dimension=embedding_dimension,
                epsilon=normalization_epsilon,
                conditioning_dimension=conditioning_dimension,
                use_gating=use_gating,
            ),
            dropout=dropout,
        )
        self.feedforward_block = FeedforwardBlock(
            feedforward=build_feedforward(
                embedding_dimension=embedding_dimension,
                feedforward_dimension=feedforward_dimension,
                activation=activation,
                dropout=dropout,
                bias=bias,
            ),
            normalization=create_block_normalization(
                normalization_type=normalization_type,
                dimension=embedding_dimension,
                epsilon=normalization_epsilon,
                conditioning_dimension=conditioning_dimension,
                use_gating=use_gating,
            ),
            dropout=dropout,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        positional_encoding: RotaryPositionalEncoding | None = None,
    ) -> torch.Tensor:
        """Forward pass through encoder layer.

        Args:
            hidden_states: Input embeddings (B, S, D).
            conditioning: Conditioning vector for adaptive normalization (B, C).
                Ignored when constructed with plain normalization.
            attention_mask: Optional mask (B, 1, S, S) where True means masked.
            positional_encoding: Optional rotary positional encoding module.

        Returns:
            Output hidden states (B, S, D).
        """
        hidden_states, _ = self.self_attention_block(
            hidden_states=hidden_states,
            conditioning=conditioning,
            attention_mask=attention_mask,
            positional_encoding=positional_encoding,
        )
        hidden_states = self.feedforward_block(
            hidden_states=hidden_states,
            conditioning=conditioning,
        )
        return hidden_states
