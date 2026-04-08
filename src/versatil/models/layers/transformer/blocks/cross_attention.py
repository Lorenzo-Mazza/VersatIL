"""Cross-attention block: norm -> cross-attention -> gated residual."""

import torch

from versatil.models.layers.normalization.typedefs import BlockNormalization
from versatil.models.layers.transformer.attention.cached_attention import (
    CachedAttention,
)
from versatil.models.layers.transformer.blocks.base import TransformerBlock
from versatil.models.layers.transformer.kv_cache import LayerKVCache


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
        layer_cache: LayerKVCache | None = None,
    ) -> tuple[torch.Tensor, LayerKVCache | None]:
        """Norm -> cross-attention -> gated residual.

        Note:
            On the first call, ``encoder_hidden_states`` are projected to K/V.
            On subsequent calls, if ``layer_cache`` contains precomputed cross-attention
            K/V, those are reused and ``encoder_hidden_states`` can be None.

        Args:
            hidden_states: Query input (B, T, D).
            encoder_hidden_states: Encoder output for K/V projection (B, S, D).
                Can be None when using cached cross-attention K/V.
            conditioning: Conditioning vector for AdaNorm (B, C). Ignored by UnconditionedNorm.
            attention_mask: Bool mask (B, 1, T, S), True = masked.
            layer_cache: Cached cross-attention K/V from a previous call.

        Returns:
            Tuple of (output hidden states (B, T, D), updated cache or None).
        """
        residual = hidden_states
        hidden_states, gate = self.normalization(
            x=hidden_states, condition=conditioning
        )
        use_cross_cache = (
            layer_cache is not None and layer_cache.cross_attention_keys is not None
        )
        attention_output, new_cache = self.attention(
            query_input=hidden_states,
            key_input=encoder_hidden_states if not use_cross_cache else None,
            value_input=encoder_hidden_states if not use_cross_cache else None,
            attention_mask=attention_mask,
            layer_cache=layer_cache,
            use_self_attention_cache=False,
            use_cross_attention_cache=use_cross_cache,
        )
        hidden_states = self.apply_residual(residual, attention_output, gate)
        return hidden_states, new_cache
