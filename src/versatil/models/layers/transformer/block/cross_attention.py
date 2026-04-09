"""Cross-attention block: norm -> cross-attention -> gated residual."""

import torch

from versatil.models.layers.normalization.typedefs import BlockNormalization
from versatil.models.layers.transformer.attention.cached_attention import (
    CachedAttention,
)
from versatil.models.layers.transformer.block.base import TransformerBlock
from versatil.models.layers.transformer.cache.conditioning import ConditioningLayerCache


class CrossAttentionBlock(TransformerBlock):
    """Norm -> cross-attention to encoder hidden states -> gated residual."""

    def __init__(
        self,
        attention: CachedAttention,
        normalization: BlockNormalization,
        dropout: float = 0.1,
    ):
        super().__init__(normalization=normalization, dropout=dropout)
        self.attention = attention

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        conditioning: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        conditioning_cache: ConditioningLayerCache | None = None,
    ) -> torch.Tensor:
        """Norm -> cross-attention -> gated residual.

        Args:
            hidden_states: Query input (B, T, D).
            encoder_hidden_states: Encoder output for K/V projection (B, S, D).
                Can be None when using conditioning_cache.
            conditioning: Conditioning vector for AdaNorm (B, C). Ignored by UnconditionedNorm.
            attention_mask: Bool mask (B, 1, T, S), True = masked.
            conditioning_cache: Precomputed K/V for static conditioning.

        Returns:
            Output hidden states (B, T, D).

        Raises:
            ValueError: If both encoder_hidden_states and conditioning_cache are None.
        """
        if encoder_hidden_states is None and conditioning_cache is None:
            raise ValueError(
                "Either encoder_hidden_states or conditioning_cache must be provided"
            )
        residual = hidden_states
        hidden_states, gate = self.normalization(
            x=hidden_states, condition=conditioning
        )
        attention_output, _ = self.attention(
            query_input=hidden_states,
            key_input=encoder_hidden_states if conditioning_cache is None else None,
            value_input=encoder_hidden_states if conditioning_cache is None else None,
            attention_mask=attention_mask,
            conditioning_cache=conditioning_cache,
        )
        hidden_states = self.apply_residual(residual, attention_output, gate)
        return hidden_states
