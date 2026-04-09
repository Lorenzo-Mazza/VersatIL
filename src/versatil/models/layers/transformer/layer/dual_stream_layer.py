"""Dual-stream transformer layer: joint attention + per-stream feedforward."""

import torch
import torch.nn as nn

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.normalization.factory import create_block_normalization
from versatil.models.layers.positional_encoding.rotary import RotaryPositionalEncoding
from versatil.models.layers.transformer.attention.joint_attention import JointAttention
from versatil.models.layers.transformer.blocks.dual_stream_attention import (
    DualStreamAttentionBlock,
)
from versatil.models.layers.transformer.blocks.feedforward import (
    FeedforwardBlock,
    build_feedforward,
)


class DualStreamLayer(nn.Module):
    """Joint attention over two streams followed by per-stream feedforward.

    Both streams share attention through joint K/V concatenation but have
    independent normalization and feedforward networks.
    """

    def __init__(
        self,
        embedding_dimension: int,
        number_of_heads: int,
        conditioning_dimension: int | None = None,
        feedforward_dimension: int | None = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        activation: str = ActivationFunction.SWIGLU.value,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        normalization_epsilon: float = 1e-6,
        use_query_key_norm: bool = True,
        use_gating: bool = True,
        bias: bool = True,
    ):
        """Initialize DualStreamLayer.

        Args:
            embedding_dimension: Hidden dimension for both streams.
            number_of_heads: Number of attention heads.
            conditioning_dimension: Dimension of conditioning vector for adaptive norm.
            feedforward_dimension: FFN hidden dimension (defaults to 4 * embedding_dimension).
            dropout: Dropout rate for residual connections.
            attention_dropout: Dropout rate for attention weights.
            activation: Activation function for FFN.
            normalization_type: Normalization type.
            normalization_epsilon: Epsilon for normalization layers.
            use_query_key_norm: Whether to apply QK-normalization.
            use_gating: Whether to use gating in adaptive normalization.
            bias: Whether to use bias in linear layers.
        """
        super().__init__()
        if feedforward_dimension is None:
            feedforward_dimension = 4 * embedding_dimension
        self.attention_block = DualStreamAttentionBlock(
            joint_attention=JointAttention(
                primary_embedding_dimension=embedding_dimension,
                number_of_heads=number_of_heads,
                dropout=attention_dropout,
                use_query_key_norm=use_query_key_norm,
                normalization_epsilon=normalization_epsilon,
                bias=bias,
            ),
            attention_normalization_primary=create_block_normalization(
                normalization_type=normalization_type,
                dimension=embedding_dimension,
                epsilon=normalization_epsilon,
                condition_dim=conditioning_dimension,
                use_gating=use_gating,
            ),
            attention_normalization_secondary=create_block_normalization(
                normalization_type=normalization_type,
                dimension=embedding_dimension,
                epsilon=normalization_epsilon,
                condition_dim=conditioning_dimension,
                use_gating=use_gating,
            ),
            dropout=dropout,
        )
        self.feedforward_block_primary = FeedforwardBlock(
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
                condition_dim=conditioning_dimension,
                use_gating=use_gating,
            ),
            dropout=dropout,
        )
        self.feedforward_block_secondary = FeedforwardBlock(
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
                condition_dim=conditioning_dimension,
                use_gating=use_gating,
            ),
            dropout=dropout,
        )

    def forward(
        self,
        hidden_states_primary: torch.Tensor,
        hidden_states_secondary: torch.Tensor,
        conditioning: torch.Tensor | None = None,
        attention_mask_primary: torch.Tensor | None = None,
        attention_mask_secondary: torch.Tensor | None = None,
        joint_attention_mask: torch.Tensor | None = None,
        positional_encoding_primary: RotaryPositionalEncoding | None = None,
        positional_encoding_secondary: RotaryPositionalEncoding | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through dual-stream layer.

        Args:
            hidden_states_primary: Primary stream tokens (B, S, D).
            hidden_states_secondary: Secondary stream tokens (B, T, D).
            conditioning: Conditioning vector for adaptive normalization (B, C).
            attention_mask_primary: Padding mask (B, S), True = masked.
            attention_mask_secondary: Padding mask (B, T), True = masked.
            joint_attention_mask: Pre-built joint mask (B, 1, S+T, S+T).
            positional_encoding_primary: Optional RoPE for primary stream.
            positional_encoding_secondary: Optional RoPE for secondary stream.

        Returns:
            Tuple of (primary_output (B, S, D), secondary_output (B, T, D)).
        """
        hidden_states_primary, hidden_states_secondary = self.attention_block(
            hidden_states_primary=hidden_states_primary,
            hidden_states_secondary=hidden_states_secondary,
            conditioning=conditioning,
            attention_mask_primary=attention_mask_primary,
            attention_mask_secondary=attention_mask_secondary,
            joint_attention_mask=joint_attention_mask,
            positional_encoding_primary=positional_encoding_primary,
            positional_encoding_secondary=positional_encoding_secondary,
        )
        hidden_states_primary = self.feedforward_block_primary(
            hidden_states=hidden_states_primary, conditioning=conditioning
        )
        hidden_states_secondary = self.feedforward_block_secondary(
            hidden_states=hidden_states_secondary, conditioning=conditioning
        )
        return hidden_states_primary, hidden_states_secondary
