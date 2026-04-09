"""Precomputed dual-stream layer: joint attention (precomputed primary) + secondary FFN."""

import torch
import torch.nn as nn

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.normalization.factory import create_block_normalization
from versatil.models.layers.positional_encoding.rotary import RotaryPositionalEncoding
from versatil.models.layers.transformer.attention.precomputed_primary_joint_attention import (
    PrecomputedPrimaryJointAttention,
)
from versatil.models.layers.transformer.block.feedforward import (
    FeedforwardBlock,
    build_feedforward,
)
from versatil.models.layers.transformer.block.precomputed_dual_stream_attention import (
    PrecomputedDualStreamAttentionBlock,
)


class PrecomputedDualStreamLayer(nn.Module):
    """Joint attention with precomputed primary Q/K/V, plus secondary feedforward.

    The primary stream provides pre-projected Q/K/V from an external source.
    Only the secondary stream has learnable normalization and feedforward.
    """

    def __init__(
        self,
        primary_embedding_dimension: int,
        secondary_embedding_dimension: int,
        number_of_heads: int,
        number_of_key_value_heads: int,
        head_dimension: int,
        secondary_feedforward_dimension: int,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        conditioning_dimension: int | None = None,
        use_gating: bool = False,
        dropout: float = 0.1,
        activation: str = ActivationFunction.SILU.value,
        bias: bool = False,
        use_query_key_norm: bool = False,
    ):
        """Initialize PrecomputedDualStreamLayer.

        Args:
            primary_embedding_dimension: Primary stream embedding dimension.
            secondary_embedding_dimension: Secondary stream embedding dimension.
            number_of_heads: Number of attention heads.
            number_of_key_value_heads: Number of K/V heads.
            head_dimension: Dimension per attention head.
            secondary_feedforward_dimension: FFN hidden dimension for secondary stream.
            normalization_type: Normalization type for secondary stream.
            conditioning_dimension: Conditioning dimension for adaptive normalization.
            use_gating: Whether to use gating in adaptive normalization.
            dropout: Dropout rate for residual connections.
            activation: Activation function for FFN.
            bias: Whether to use bias in linear layers.
            use_query_key_norm: Whether to apply QK-normalization.
        """
        super().__init__()
        self.attention_block = PrecomputedDualStreamAttentionBlock(
            joint_attention=PrecomputedPrimaryJointAttention(
                primary_embedding_dimension=primary_embedding_dimension,
                number_of_heads=number_of_heads,
                secondary_embedding_dimension=secondary_embedding_dimension,
                number_of_key_value_heads=number_of_key_value_heads,
                head_dimension=head_dimension,
                dropout=dropout,
                use_query_key_norm=use_query_key_norm,
                bias=bias,
            ),
            attention_normalization_secondary=create_block_normalization(
                normalization_type=normalization_type,
                dimension=secondary_embedding_dimension,
                condition_dim=conditioning_dimension,
                use_gating=use_gating,
            ),
            dropout=dropout,
        )
        self.feedforward_block_secondary = FeedforwardBlock(
            feedforward=build_feedforward(
                embedding_dimension=secondary_embedding_dimension,
                feedforward_dimension=secondary_feedforward_dimension,
                activation=activation,
                dropout=dropout,
                bias=bias,
            ),
            normalization=create_block_normalization(
                normalization_type=normalization_type,
                dimension=secondary_embedding_dimension,
                condition_dim=conditioning_dimension,
                use_gating=use_gating,
            ),
            dropout=dropout,
        )

    def forward(
        self,
        precomputed_primary: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        hidden_states_secondary: torch.Tensor,
        conditioning: torch.Tensor | None = None,
        attention_mask_primary: torch.Tensor | None = None,
        attention_mask_secondary: torch.Tensor | None = None,
        joint_attention_mask: torch.Tensor | None = None,
        positional_encoding_secondary: RotaryPositionalEncoding | None = None,
        precomputed_secondary_rope: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with precomputed primary Q/K/V.

        Args:
            precomputed_primary: Pre-projected primary (Q, K, V) tuple,
                each shaped (B, H/KV_H, S, D_head).
            hidden_states_secondary: Secondary stream tokens (B, T, D).
            conditioning: Conditioning vector for adaptive normalization (B, C).
            attention_mask_primary: Padding mask (B, S), True = masked.
            attention_mask_secondary: Padding mask (B, T), True = masked.
            joint_attention_mask: Pre-built joint mask (B, 1, S+T, S+T).
            positional_encoding_secondary: Optional RoPE module for secondary stream.
            precomputed_secondary_rope: Pre-computed (cos, sin) for secondary positions.

        Returns:
            Tuple of (raw primary attention output (B, S, H*D_head),
            processed secondary output (B, T, D)).
        """
        attention_output_primary, hidden_states_secondary = self.attention_block(
            precomputed_primary=precomputed_primary,
            hidden_states_secondary=hidden_states_secondary,
            conditioning=conditioning,
            attention_mask_primary=attention_mask_primary,
            attention_mask_secondary=attention_mask_secondary,
            joint_attention_mask=joint_attention_mask,
            positional_encoding_secondary=positional_encoding_secondary,
            precomputed_secondary_rope=precomputed_secondary_rope,
        )
        hidden_states_secondary = self.feedforward_block_secondary(
            hidden_states=hidden_states_secondary, conditioning=conditioning
        )
        return attention_output_primary, hidden_states_secondary
