"""Tests for versatil.models.decoding.decoders.autoregressive_mixin module."""

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
import torch

from versatil.models.decoding.constants import DecoderOutputKey
from versatil.models.decoding.decoders.autoregressive_mixin import (
    AutoregressiveDecoderMixin,
    AutoregressivePrefix,
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


@pytest.fixture
def generation_state_factory() -> Callable[..., CachedAutoregressiveGenerationState]:
    def factory(batch_size: int) -> CachedAutoregressiveGenerationState:
        return CachedAutoregressiveGenerationState(
            step_index=0,
            sequence_length=PREFIX_TOKEN_COUNT,
            past_key_values=MagicMock(spec=GenerationCache),
            next_inputs=torch.zeros(
                batch_size, 1, EMBEDDING_DIMENSION
            ),  # (batch, 1, embedding)
        )

    return factory


@pytest.fixture
def bounded_generation_decoder_factory(
    autoregressive_decoder_factory: Callable[..., ConcreteAutoregressiveDecoder],
) -> Callable[..., ConcreteAutoregressiveDecoder]:
    def factory(
        sampled_tokens: list[list[int]], completed_masks: list[list[bool]]
    ) -> ConcreteAutoregressiveDecoder:
        decoder = autoregressive_decoder_factory()
        batch_size = len(sampled_tokens[0])
        for step_tokens, completed_mask in zip(
            sampled_tokens, completed_masks, strict=True
        ):
            decoder.sample_outputs.append(
                torch.tensor(step_tokens).unsqueeze(dim=1)  # (batch,) -> (batch, 1)
            )
            decoder.completed_sequence_masks.append(
                torch.tensor(completed_mask, dtype=torch.bool)  # (batch,)
            )
            decoder.decode_outputs.append(
                (
                    torch.zeros(
                        batch_size, 1, EMBEDDING_DIMENSION
                    ),  # (batch, 1, embedding)
                    MagicMock(spec=GenerationCache),
                )
            )
            decoder.prepared_inputs.append(
                torch.zeros(batch_size, 1, EMBEDDING_DIMENSION)  # (batch, 1, embedding)
            )
        return decoder

    return factory


@pytest.fixture
def generation_interface_factory(
    generation_state_factory: Callable[..., CachedAutoregressiveGenerationState],
) -> Callable[..., MagicMock]:
    def factory(batch_size: int, max_generation_steps: int) -> MagicMock:
        decoder = MagicMock(spec=AutoregressiveDecoderMixin)
        decoder.prefill.return_value = AutoregressivePrefix(
            state=generation_state_factory(batch_size=batch_size),
            max_generation_steps=max_generation_steps,
            first_output=torch.zeros(
                batch_size, 1, EMBEDDING_DIMENSION
            ),  # (batch, 1, embedding)
        )
        decoder._run_cached_autoregressive_generation.return_value = {
            DecoderOutputKey.PREDICTED_ACTION_TOKENS.value: torch.zeros(
                batch_size, max_generation_steps, dtype=torch.long
            )  # (batch, generated_length)
        }
        return decoder

    return factory


@pytest.mark.unit
@pytest.mark.parametrize("fixed_length", [False, True])
def test_generate_passes_prefix_state_and_generation_mode_to_loop(
    generation_interface_factory: Callable[..., MagicMock],
    flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    fixed_length: bool,
) -> None:
    decoder = generation_interface_factory(batch_size=2, max_generation_steps=4)
    features = flat_feature_factory(batch_size=2, feature_dim=EMBEDDING_DIMENSION)

    predictions = AutoregressiveDecoderMixin.generate(
        self=decoder, features=features, fixed_length=fixed_length
    )  # tokens: (batch, generated_length)

    decoder.prefill.assert_called_once_with(
        features=features, fixed_length=fixed_length
    )
    prefix = decoder.prefill.return_value
    decoder._run_cached_autoregressive_generation.assert_called_once_with(
        initial_state=prefix.state,
        max_generation_steps=4,
        initial_step_output=prefix.first_output,
        fixed_length=fixed_length,
    )
    torch.testing.assert_close(
        predictions,
        decoder._run_cached_autoregressive_generation.return_value,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "sampled_tokens, completed_masks, expected_tokens",
    [
        (
            [[6, 1], [2, 2], [3, 6], [4, 4]],
            [[True, False], [True, False], [True, True], [True, True]],
            [[6, 6, 6, 6], [1, 2, 6, 6]],
        ),
        (
            [[1, 6], [2, 2], [6, 3], [4, 4]],
            [[False, True], [False, True], [True, True], [True, True]],
            [[1, 2, 6, 6], [6, 6, 6, 6]],
        ),
    ],
)
def test_fixed_length_generation_repeats_each_samples_eos_through_final_step(
    bounded_generation_decoder_factory: Callable[..., ConcreteAutoregressiveDecoder],
    generation_state_factory: Callable[..., CachedAutoregressiveGenerationState],
    sampled_tokens: list[list[int]],
    completed_masks: list[list[bool]],
    expected_tokens: list[list[int]],
) -> None:
    decoder = bounded_generation_decoder_factory(
        sampled_tokens=sampled_tokens, completed_masks=completed_masks
    )

    predictions = decoder._run_cached_autoregressive_generation(
        initial_state=generation_state_factory(batch_size=2),
        max_generation_steps=4,
        fixed_length=True,
    )  # tokens: (batch, generated_length)

    expected = torch.tensor(expected_tokens)  # (batch, generated_length)
    torch.testing.assert_close(
        predictions[DecoderOutputKey.PREDICTED_ACTION_TOKENS.value], expected
    )
    assert len(decoder.decoded_states) == 4
    for step_index, prepared_tokens in enumerate(decoder.prepared_generated_outputs):
        expected_step = expected[:, step_index : step_index + 1]  # (batch, 1)
        torch.testing.assert_close(prepared_tokens, expected_step)


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
