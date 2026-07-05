"""Reusable helpers for cached autoregressive decoders."""

from dataclasses import dataclass

import torch
from transformers.cache_utils import Cache

from versatil.models.decoding.generative_language_models.base import (
    CausalLanguageModelOutput,
)
from versatil.models.layers.transformer.cache.generation import GenerationCache

type PastKeyValues = Cache | GenerationCache | tuple[tuple[torch.Tensor, ...], ...]


@dataclass
class CachedAutoregressiveGenerationState:
    """State carried by cached autoregressive generation loops.

    Args:
        step_index: Number of generated steps already decoded.
        sequence_length: Prefix plus generated sequence length currently in cache.
        past_key_values: Cached transformer key/value state.
        next_inputs: Decoder-specific next-step input tensor. Concrete decoders
            decide whether it contains token IDs, embeddings, or continuous values.
        attention_mask: Optional cache-aware attention mask.
        cache_position: Optional cache position tensor for HuggingFace models.
        position_ids: Optional position IDs for ``next_inputs`` with matching
            shape up to the last dimension, e.g. ``(B, 1)`` for one token.
        completed_sequence_mask: Optional boolean mask with shape ``(B,)`` where
            ``True`` marks samples that already met their stop condition.
    """

    step_index: int
    sequence_length: int
    past_key_values: PastKeyValues
    next_inputs: torch.Tensor
    attention_mask: torch.Tensor | None = None
    cache_position: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None
    completed_sequence_mask: torch.Tensor | None = None


class AutoregressiveDecoderMixin:
    """Cached autoregressive generation loop."""

    def _decode_next_autoregressive_step(
        self,
        state: CachedAutoregressiveGenerationState,
    ) -> tuple[torch.Tensor | CausalLanguageModelOutput, PastKeyValues]:
        """Decode one incremental autoregressive step."""
        raise NotImplementedError

    def _sample_next_autoregressive_output(
        self,
        step_output: torch.Tensor | CausalLanguageModelOutput,
    ) -> torch.Tensor:
        """Sample or choose the next generated value from one step output."""
        raise NotImplementedError

    def _prepare_next_autoregressive_inputs(
        self,
        generated_output: torch.Tensor,
    ) -> torch.Tensor:
        """Convert a generated value into the next-step model input."""
        raise NotImplementedError

    def _finalize_autoregressive_outputs(
        self,
        generated_outputs: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Convert generated step values into decoder predictions."""
        raise NotImplementedError

    def _get_completed_sequence_mask(
        self,
        generated_output: torch.Tensor,
        state: CachedAutoregressiveGenerationState,
    ) -> torch.Tensor | None:
        """Return an updated per-sample stop mask, if the decoder has one."""
        return state.completed_sequence_mask

    def _advance_autoregressive_attention_mask(
        self,
        state: CachedAutoregressiveGenerationState,
        generated_output: torch.Tensor,
    ) -> torch.Tensor | None:
        """Return the attention mask for the next cached generation step."""
        return state.attention_mask

    def _advance_autoregressive_cache_position(
        self,
        state: CachedAutoregressiveGenerationState,
        next_inputs: torch.Tensor,
    ) -> torch.Tensor | None:
        """Return the cache position for the next cached generation step."""
        if state.cache_position is None:
            return None
        return state.cache_position + next_inputs.shape[1]

    def _advance_autoregressive_position_ids(
        self,
        state: CachedAutoregressiveGenerationState,
        next_inputs: torch.Tensor,
    ) -> torch.Tensor | None:
        """Advance cached position IDs to match ``next_inputs`` shape ``(B, S)``."""
        if state.position_ids is None:
            return None
        return state.position_ids + next_inputs.shape[1]

    def _run_cached_autoregressive_generation(
        self,
        initial_state: CachedAutoregressiveGenerationState,
        max_generation_steps: int,
        initial_step_output: torch.Tensor | CausalLanguageModelOutput | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run cached autoregressive generation from a prepared state.

        Args:
            initial_state: State after any required prefix prefill.
            max_generation_steps: Maximum number of generated values.
            initial_step_output: Optional output from prefix prefill. Use this
                when the first generated value is sampled from the prefix output.

        Returns:
            Decoder-specific generated prediction dictionary.
        """
        generated_outputs = []
        state = initial_state
        step_output = initial_step_output
        for _ in range(max_generation_steps):
            if step_output is None:
                step_output, past_key_values = self._decode_next_autoregressive_step(
                    state=state,
                )
            else:
                past_key_values = state.past_key_values

            generated_output = self._sample_next_autoregressive_output(
                step_output=step_output,
            )
            generated_outputs.append(generated_output)
            completed_sequence_mask = self._get_completed_sequence_mask(
                generated_output=generated_output,
                state=state,
            )
            if completed_sequence_mask is not None and completed_sequence_mask.all():
                break

            next_inputs = self._prepare_next_autoregressive_inputs(
                generated_output=generated_output,
            )
            state = CachedAutoregressiveGenerationState(
                step_index=state.step_index + 1,
                sequence_length=state.sequence_length + next_inputs.shape[1],
                past_key_values=past_key_values,
                next_inputs=next_inputs,
                attention_mask=self._advance_autoregressive_attention_mask(
                    state=state,
                    generated_output=generated_output,
                ),
                cache_position=self._advance_autoregressive_cache_position(
                    state=state,
                    next_inputs=next_inputs,
                ),
                position_ids=self._advance_autoregressive_position_ids(
                    state=state,
                    next_inputs=next_inputs,
                ),
                completed_sequence_mask=completed_sequence_mask,
            )
            step_output = None

        return self._finalize_autoregressive_outputs(
            generated_outputs=generated_outputs,
        )
