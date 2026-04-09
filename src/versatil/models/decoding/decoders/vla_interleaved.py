"""VLA-specific cross-attention layer for interaction between action expert and VLM backbone."""

import torch
import torch.nn as nn

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.normalization.factory import create_block_normalization
from versatil.models.layers.transformer.attention.cached_attention import (
    CachedAttention,
)
from versatil.models.layers.transformer.blocks.feedforward import (
    FeedforwardBlock,
    build_feedforward,
)
from versatil.models.layers.transformer.blocks.precomputed_cross_attention import (
    PrecomputedCrossAttentionBlock,
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
