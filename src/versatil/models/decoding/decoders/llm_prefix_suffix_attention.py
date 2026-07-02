"""Attention helpers for LLM decoders with prefix and action suffix tokens."""

import torch

from versatil.models.decoding.action_masking import make_attention_mask


class LLMPrefixSuffixAttentionMixin:
    """Build attention masks for single-stream LLM prefix/suffix sequences."""

    causal_prefix: bool

    @staticmethod
    def _all_false_or_none(mask: torch.Tensor | None) -> torch.Tensor | None:
        """Drop padding masks that contain no padded positions."""
        if mask is not None and not mask.any():
            return None
        return mask

    def _build_causal_attention_mask(
        self,
        padding_mask: torch.Tensor | None,
        tokens: torch.Tensor,
    ) -> torch.Tensor | None:
        """Build a standard causal-model padding mask."""
        if padding_mask is None:
            return None
        return (~padding_mask).to(device=tokens.device, dtype=torch.long)  # (B, L)

    @staticmethod
    def _build_prefix_attention_mask(
        prefix_tokens: torch.Tensor,
        suffix_tokens: torch.Tensor,
        prefix_mask: torch.Tensor | None,
        causal_suffix: bool,
    ) -> torch.Tensor:
        """Build a bidirectional-prefix mask over action suffix tokens."""
        masked_attention_mask, _ = make_attention_mask(
            feature_tokens=prefix_tokens,
            action_tokens=suffix_tokens,
            feature_token_mask=prefix_mask,
            causal_actions=causal_suffix,
        )
        return ~masked_attention_mask  # (B, 1, P+S_suffix, P+S_suffix)

    def _build_attention_mask(
        self,
        padding_mask: torch.Tensor | None,
        tokens: torch.Tensor,
        prefix_length: int,
        causal_suffix: bool = True,
    ) -> torch.Tensor | None:
        """Build the configured attention mask for prefix-plus-suffix tokens."""
        if self.causal_prefix:
            return self._build_causal_attention_mask(
                padding_mask=padding_mask,
                tokens=tokens,
            )
        sequence_length = tokens.shape[1]
        if not 0 < prefix_length <= sequence_length:
            raise ValueError(
                "prefix_length must be in [1, sequence_length], got "
                f"prefix_length={prefix_length}, sequence_length={sequence_length}."
            )
        prefix_tokens = tokens[:, :prefix_length, :]  # (B, P, D)
        suffix_tokens = tokens[:, prefix_length:, :]  # (B, S_suffix, D)
        prefix_mask = (
            padding_mask[:, :prefix_length] if padding_mask is not None else None
        )
        # Always return the explicit 4D mask: HF decoder-only models treat a
        # None mask as fully causal, which contradicts the bidirectional
        # prefix this branch encodes (even when no position is masked).
        return self._build_prefix_attention_mask(
            prefix_tokens=prefix_tokens,
            suffix_tokens=suffix_tokens,
            prefix_mask=prefix_mask,
            causal_suffix=causal_suffix,
        )

    @staticmethod
    def _append_unmasked_tokens(
        padding_mask: torch.Tensor | None,
        tokens: torch.Tensor,
    ) -> torch.Tensor | None:
        """Extend a prefix padding mask with unmasked suffix tokens."""
        if padding_mask is None:
            return None
        token_mask = torch.zeros(
            tokens.shape[:2],
            dtype=torch.bool,
            device=tokens.device,
        )  # (B, S_suffix)
        return torch.cat([padding_mask, token_mask], dim=1)  # (B, P+S_suffix)

    def _build_prefix_suffix_inputs(
        self,
        prefix_tokens: torch.Tensor,
        suffix_tokens: torch.Tensor,
        prefix_mask: torch.Tensor | None,
        causal_suffix: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Concatenate prefix and suffix tokens and build their attention mask."""
        full_token_sequence = torch.cat(
            [
                prefix_tokens,  # (B, P, D)
                suffix_tokens,  # (B, S_suffix, D)
            ],
            dim=1,
        )  # (B, P+S_suffix, D)
        full_padding_mask = self._append_unmasked_tokens(
            padding_mask=prefix_mask,
            tokens=suffix_tokens,
        )  # (B, P+S_suffix) or None
        attention_mask = self._build_attention_mask(
            padding_mask=full_padding_mask,
            tokens=full_token_sequence,
            prefix_length=prefix_tokens.shape[1],
            causal_suffix=causal_suffix,
        )
        return full_token_sequence, attention_mask
