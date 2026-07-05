"""Tests for versatil.models.decoding.decoders.autoregressive_mixin module."""

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
import torch

from versatil.models.decoding.constants import DecoderOutputKey
from versatil.models.decoding.decoders.autoregressive_mixin import (
    AutoregressiveDecoderMixin,
    CachedAutoregressiveGenerationState,
)
from versatil.models.layers.transformer.cache.generation import GenerationCache

BATCH_SIZE = 2
PREFIX_TOKEN_COUNT = 3
ACTION_TOKEN_COUNT = 2
EMBEDDING_DIMENSION = 4
VOCABULARY_SIZE = 7


class ConcreteAutoregressiveDecoder(AutoregressiveDecoderMixin):
    def __init__(self) -> None:
        self.decoded_states = []
        self.sampled_step_outputs = []
        self.prepared_generated_outputs = []
        self.decode_outputs = []
        self.sample_outputs = []
        self.prepared_inputs = []
        self.completed_sequence_masks = []

    def _decode_next_autoregressive_step(
        self,
        state: CachedAutoregressiveGenerationState,
    ) -> tuple[torch.Tensor, GenerationCache]:
        self.decoded_states.append(state)
        step_output, past_key_values = self.decode_outputs.pop(0)
        return step_output, past_key_values

    def _sample_next_autoregressive_output(
        self,
        step_output: torch.Tensor,
    ) -> torch.Tensor:
        self.sampled_step_outputs.append(step_output)
        return self.sample_outputs.pop(0)

    def _prepare_next_autoregressive_inputs(
        self,
        generated_output: torch.Tensor,
    ) -> torch.Tensor:
        self.prepared_generated_outputs.append(generated_output)
        return self.prepared_inputs.pop(0)

    def _get_completed_sequence_mask(
        self,
        generated_output: torch.Tensor,
        state: CachedAutoregressiveGenerationState,
    ) -> torch.Tensor | None:
        if self.completed_sequence_masks:
            return self.completed_sequence_masks.pop(0)
        return state.completed_sequence_mask

    def _finalize_autoregressive_outputs(
        self,
        generated_outputs: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {
            DecoderOutputKey.PREDICTED_ACTION_TOKENS.value: torch.cat(
                generated_outputs,
                dim=1,
            )
        }


@pytest.fixture
def autoregressive_decoder_factory() -> Callable[..., ConcreteAutoregressiveDecoder]:
    def factory() -> ConcreteAutoregressiveDecoder:
        return ConcreteAutoregressiveDecoder()

    return factory


@pytest.mark.unit
def test_run_cached_autoregressive_generation_samples_initial_prefill_output(
    autoregressive_decoder_factory: Callable[..., ConcreteAutoregressiveDecoder],
) -> None:
    decoder = autoregressive_decoder_factory()
    initial_cache = MagicMock(spec=GenerationCache)
    updated_cache = MagicMock(spec=GenerationCache)
    initial_step_output = torch.ones(BATCH_SIZE, 1, EMBEDDING_DIMENSION)
    decoded_step_output = torch.full((BATCH_SIZE, 1, EMBEDDING_DIMENSION), 2.0)
    first_token = torch.tensor([[1], [2]])
    end_token = torch.full((BATCH_SIZE, 1), VOCABULARY_SIZE - 1)
    next_inputs = torch.full((BATCH_SIZE, 1, EMBEDDING_DIMENSION), 3.0)
    decoder.sample_outputs = [first_token, end_token]
    decoder.prepared_inputs = [next_inputs]
    decoder.decode_outputs = [(decoded_step_output, updated_cache)]
    decoder.completed_sequence_masks = [
        torch.zeros(BATCH_SIZE, dtype=torch.bool),
        torch.ones(BATCH_SIZE, dtype=torch.bool),
    ]
    initial_state = CachedAutoregressiveGenerationState(
        step_index=0,
        sequence_length=PREFIX_TOKEN_COUNT,
        past_key_values=initial_cache,
        next_inputs=torch.empty(BATCH_SIZE, 0, dtype=torch.long),
    )

    predictions = decoder._run_cached_autoregressive_generation(
        initial_state=initial_state,
        max_generation_steps=ACTION_TOKEN_COUNT,
        initial_step_output=initial_step_output,
    )

    assert len(decoder.decoded_states) == 1
    decoded_state = decoder.decoded_states[0]
    assert decoded_state.step_index == 1
    assert decoded_state.sequence_length == PREFIX_TOKEN_COUNT + 1
    assert decoded_state.past_key_values is initial_cache
    torch.testing.assert_close(decoded_state.next_inputs, next_inputs)
    torch.testing.assert_close(
        decoder.sampled_step_outputs[0],
        initial_step_output,
    )
    torch.testing.assert_close(
        decoder.sampled_step_outputs[1],
        decoded_step_output,
    )
    torch.testing.assert_close(decoder.prepared_generated_outputs[0], first_token)
    torch.testing.assert_close(
        predictions[DecoderOutputKey.PREDICTED_ACTION_TOKENS.value],
        torch.cat([first_token, end_token], dim=1),
    )


@pytest.mark.unit
def test_run_cached_autoregressive_generation_stops_without_decoding_after_completion(
    autoregressive_decoder_factory: Callable[..., ConcreteAutoregressiveDecoder],
) -> None:
    decoder = autoregressive_decoder_factory()
    initial_step_output = torch.ones(BATCH_SIZE, 1, EMBEDDING_DIMENSION)
    end_token = torch.full((BATCH_SIZE, 1), VOCABULARY_SIZE - 1)
    decoder.sample_outputs = [end_token]
    decoder.completed_sequence_masks = [torch.ones(BATCH_SIZE, dtype=torch.bool)]
    initial_state = CachedAutoregressiveGenerationState(
        step_index=0,
        sequence_length=PREFIX_TOKEN_COUNT,
        past_key_values=MagicMock(spec=GenerationCache),
        next_inputs=torch.empty(BATCH_SIZE, 0, dtype=torch.long),
    )

    predictions = decoder._run_cached_autoregressive_generation(
        initial_state=initial_state,
        max_generation_steps=ACTION_TOKEN_COUNT,
        initial_step_output=initial_step_output,
    )

    assert decoder.decoded_states == []
    assert decoder.prepared_generated_outputs == []
    torch.testing.assert_close(
        predictions[DecoderOutputKey.PREDICTED_ACTION_TOKENS.value],
        end_token,
    )
