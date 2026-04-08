"""
VLA-specific interleaved attention layers for interaction between action expert network and
    Vision Language Backbone.
"""

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
from versatil.models.layers.transformer.attention.precomputed_primary_joint_attention import (
    PrecomputedPrimaryJointAttention,
)
from versatil.models.layers.transformer.blocks.feedforward import (
    FeedforwardBlock,
    build_feedforward,
)
from versatil.models.layers.transformer.blocks.precomputed_cross_attention import (
    PrecomputedCrossAttentionBlock,
)
from versatil.models.layers.transformer.blocks.precomputed_dual_stream_attention import (
    PrecomputedDualStreamAttentionBlock,
)


class VLACrossAttentionLayer(nn.Module):
    """Projects VLM key/value states to expert dimension, then cross-attends + FFN.

    Bridges two embedding spaces (VLM and expert) via learned K/V
    projections, followed by precomputed cross-attention and a
    feedforward block.
    """

    def __init__(
        self,
        expert_embedding_dimension: int,
        vlm_key_value_dimension: int,
        expert_number_of_heads: int,
        expert_number_of_key_value_heads: int,
        expert_head_dimension: int,
        expert_feedforward_dimension: int,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        dropout: float = 0.1,
        activation: str = ActivationFunction.SILU.value,
    ):
        super().__init__()
        expert_key_value_dimension = (
            expert_number_of_key_value_heads * expert_head_dimension
        )
        self.key_projection = nn.Linear(
            vlm_key_value_dimension, expert_key_value_dimension, bias=False
        )
        self.value_projection = nn.Linear(
            vlm_key_value_dimension, expert_key_value_dimension, bias=False
        )
        if expert_number_of_key_value_heads == expert_number_of_heads:
            attention_type = AttentionType.MULTI_HEAD.value
        else:
            attention_type = AttentionType.GROUPED_QUERY.value
        self.cross_attention_block = PrecomputedCrossAttentionBlock(
            attention=CachedAttention(
                embedding_dimension=expert_embedding_dimension,
                number_of_heads=expert_number_of_heads,
                number_of_key_value_heads=expert_number_of_key_value_heads,
                head_dimension=expert_head_dimension,
                dropout=dropout,
                bias=False,
                attention_type=attention_type,
            ),
            normalization=create_block_normalization(
                normalization_type=normalization_type,
                dimension=expert_embedding_dimension,
            ),
            dropout=dropout,
        )
        self.feedforward_block = FeedforwardBlock(
            feedforward=build_feedforward(
                embedding_dimension=expert_embedding_dimension,
                feedforward_dimension=expert_feedforward_dimension,
                activation=activation,
                dropout=dropout,
                bias=False,
            ),
            normalization=create_block_normalization(
                normalization_type=normalization_type,
                dimension=expert_embedding_dimension,
            ),
            dropout=dropout,
        )

    def forward(
        self,
        expert_hidden_states: torch.Tensor,
        vlm_key_states: torch.Tensor,
        vlm_value_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        precomputed_query_rope: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Project VLM K/V to expert dimension, cross-attend with optional RoPE, then FFN.

        Args:
            expert_hidden_states: Expert tokens (B, S, D_expert).
            vlm_key_states: VLM keys (B, P, vlm_kv_dim).
            vlm_value_states: VLM values (B, P, vlm_kv_dim).
            attention_mask: Optional mask (B, 1, S, P).
            precomputed_query_rope: Precomputed (cos, sin) for expert query positions.

        Returns:
            Updated expert hidden states (B, S, D_expert).
        """
        projected_keys = self.key_projection(vlm_key_states)  # (B, P, kv_dim)
        projected_values = self.value_projection(vlm_value_states)  # (B, P, kv_dim)
        expert_hidden_states = self.cross_attention_block(
            hidden_states=expert_hidden_states,
            keys=projected_keys,
            values=projected_values,
            attention_mask=attention_mask,
            precomputed_query_rope=precomputed_query_rope,
        )
        expert_hidden_states = self.feedforward_block(
            hidden_states=expert_hidden_states,
        )
        return expert_hidden_states


class VLAJointAttentionLayer(nn.Module):
    """Joint self-attention layer between VLM backbone and expert network.

    The VLM primary stream provides precomputed Q/K/V; only the expert
    secondary stream has learnable normalization and feedforward.
    Supports optional adaptive normalization for timestep conditioning.
    """

    def __init__(
        self,
        vlm_embedding_dimension: int,
        expert_embedding_dimension: int,
        number_of_heads: int,
        number_of_key_value_heads: int,
        head_dimension: int,
        expert_feedforward_dimension: int,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        condition_dim: int | None = None,
        use_gating: bool = False,
        dropout: float = 0.1,
        activation: str = ActivationFunction.SILU.value,
        bias: bool = False,
        use_query_key_norm: bool = False,
    ):
        super().__init__()
        self.block = PrecomputedDualStreamAttentionBlock(
            joint_attention=PrecomputedPrimaryJointAttention(
                primary_embedding_dimension=vlm_embedding_dimension,
                number_of_heads=number_of_heads,
                secondary_embedding_dimension=expert_embedding_dimension,
                number_of_key_value_heads=number_of_key_value_heads,
                head_dimension=head_dimension,
                dropout=dropout,
                use_query_key_norm=use_query_key_norm,
                bias=bias,
            ),
            attention_normalization_secondary=create_block_normalization(
                normalization_type=normalization_type,
                dimension=expert_embedding_dimension,
                condition_dim=condition_dim,
                use_gating=use_gating,
            ),
            feedforward_block_secondary=FeedforwardBlock(
                feedforward=build_feedforward(
                    embedding_dimension=expert_embedding_dimension,
                    feedforward_dimension=expert_feedforward_dimension,
                    activation=activation,
                    dropout=dropout,
                    bias=bias,
                ),
                normalization=create_block_normalization(
                    normalization_type=normalization_type,
                    dimension=expert_embedding_dimension,
                    condition_dim=condition_dim,
                    use_gating=use_gating,
                ),
                dropout=dropout,
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
        """Joint attention between precomputed VLM stream and expert stream.

        Args:
            precomputed_primary: VLM (Q, K, V) tuple, each (B, H, S, D_head).
            hidden_states_secondary: Expert tokens (B, T, D_expert).
            conditioning: Conditioning vector for adaptive normalization (B, C).
            attention_mask_primary: VLM padding mask (B, S), True = masked.
            attention_mask_secondary: Expert padding mask (B, T), True = masked.
            joint_attention_mask: Pre-built joint mask (B, 1, S+T, S+T).
            positional_encoding_secondary: Optional RoPE module for expert stream.
            precomputed_secondary_rope: Pre-computed (cos, sin) for expert positions.

        Returns:
            Tuple of (raw VLM attention output (B, S, H*D_head),
            processed expert output (B, T, D_expert)).
        """
        return self.block(
            precomputed_primary=precomputed_primary,
            hidden_states_secondary=hidden_states_secondary,
            conditioning=conditioning,
            attention_mask_primary=attention_mask_primary,
            attention_mask_secondary=attention_mask_secondary,
            joint_attention_mask=joint_attention_mask,
            positional_encoding_secondary=positional_encoding_secondary,
            precomputed_secondary_rope=precomputed_secondary_rope,
        )
