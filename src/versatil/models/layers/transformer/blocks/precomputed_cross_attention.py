"""Cross-attention block with precomputed K/V and optional query RoPE."""

import torch

from versatil.models.layers.normalization.typedefs import BlockNormalization
from versatil.models.layers.positional_encoding.rotary import RotaryPositionalEncoding
from versatil.models.layers.transformer.attention.cached_attention import (
    CachedAttention,
)
from versatil.models.layers.transformer.blocks.base import TransformerBlock


class PrecomputedCrossAttentionBlock(TransformerBlock):
    """Norm -> query projection -> optional RoPE -> cross-attention -> gated residual.

    Accepts precomputed K/V tensors (already in head-split format) and
    only projects queries.
    """

    def __init__(
        self,
        attention: CachedAttention,
        normalization: BlockNormalization,
        dropout: float = 0.1,
    ):
        super().__init__(normalization=normalization, dropout=dropout)
        self.attention = attention

    def _reshape_to_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape flat projected tensor (B, S, kv_dim) to (B, KV_H, S, D_head)."""
        batch_size, sequence_length, _ = tensor.shape
        return tensor.view(
            batch_size,
            sequence_length,
            self.attention.number_of_key_value_heads,
            self.attention.head_dimension,
        ).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        conditioning: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        precomputed_query_rope: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Norm -> cross-attention with precomputed K/V -> gated residual.

        Args:
            hidden_states: Query input (B, T, D).
            keys: Precomputed keys (B, S, kv_dim).
            values: Precomputed values (B, S, kv_dim).
            conditioning: Conditioning vector for AdaNorm (B, C). Ignored by UnconditionedNorm.
            attention_mask: Bool mask (B, 1, T, S), True = masked.
            precomputed_query_rope: Precomputed (cos, sin) for query positions.
                Applied via half-rotation after query projection.

        Returns:
            Output hidden states (B, T, D).
        """
        residual = hidden_states
        hidden_states, gate = self.normalization(
            x=hidden_states, condition=conditioning
        )
        keys = self._reshape_to_heads(keys)
        values = self._reshape_to_heads(values)
        queries = self.attention.compute_query(hidden_states)  # (B, H, T, D_head)
        if precomputed_query_rope is not None:
            cos, sin = precomputed_query_rope
            queries = RotaryPositionalEncoding.apply_rotation_half(queries, sin, cos)
        attention_output = self.attention.compute_attention(
            queries=queries,
            keys=keys,
            values=values,
            attention_mask=attention_mask,
        )
        hidden_states = self.apply_residual(residual, attention_output, gate)
        return hidden_states
