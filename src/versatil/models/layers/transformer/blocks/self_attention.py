"""Self-attention block: norm -> self-attention -> gated residual."""

import torch

from versatil.models.layers.normalization.typedefs import BlockNormalization
from versatil.models.layers.positional_encoding.rotary import RotaryPositionalEncoding
from versatil.models.layers.transformer.attention.cached_attention import (
    CachedAttention,
)
from versatil.models.layers.transformer.blocks.base import TransformerBlock
from versatil.models.layers.transformer.kv_cache import LayerKVCache


class SelfAttentionBlock(TransformerBlock):
    """Norm -> self-attention -> gated residual."""

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
        conditioning: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        positional_encoding: RotaryPositionalEncoding | None = None,
        layer_cache: LayerKVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, LayerKVCache | None]:
        """Norm -> self-attention -> gated residual.

        Args:
            hidden_states: Input embeddings (B, S, D).
            conditioning: Conditioning vector for AdaNorm (B, C). Ignored by UnconditionedNorm.
            attention_mask: Bool mask (B, 1, S, S), True = masked.
            positional_encoding: Optional RoPE module.
            layer_cache: Cached K/V from previous autoregressive steps.
            use_cache: Whether to return updated K/V cache.

        Returns:
            Tuple of (output hidden states (B, S, D), updated cache or None).
        """
        residual = hidden_states
        hidden_states, gate = self.normalization(
            x=hidden_states, condition=conditioning
        )
        attention_output, new_cache = self.attention(
            query_input=hidden_states,
            key_input=hidden_states,
            value_input=hidden_states,
            attention_mask=attention_mask,
            positional_encoding=positional_encoding,
            layer_cache=layer_cache,
            use_self_attention_cache=use_cache,
            use_cross_attention_cache=False,
        )
        hidden_states = self.apply_residual(residual, attention_output, gate)
        return hidden_states, new_cache
