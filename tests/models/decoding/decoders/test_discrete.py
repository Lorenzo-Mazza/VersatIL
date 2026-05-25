"""Tests for versatil.models.decoding.decoders.discrete module."""

import re
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from versatil.data.constants import SampleKey
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.data.tokenization import ActionTokenizer, Tokenizer
from versatil.models.decoding.action_heads.single_output import ActionHead
from versatil.models.decoding.constants import ActionHeadLayout, DecoderOutputKey
from versatil.models.decoding.decoders.base import DecoderInput
from versatil.models.decoding.decoders.discrete import DiscreteDecoder

BATCH_SIZE = 2
ACTION_TOKEN_COUNT = 2
EMBEDDING_DIMENSION = 4
ACTION_HEAD_INPUT_DIMENSION = 6
VOCABULARY_SIZE = 7
INITIALIZER_RANGE = 0.02


class ConcreteDiscreteActionTokenDecoder(DiscreteDecoder):
    action_head_layout: ActionHeadLayout = ActionHeadLayout.VOCABULARY

    def __init__(
        self,
        action_head: ActionHead,
        action_space: ActionSpace,
        observation_space: ObservationSpace,
        embedding_dimension: int,
        deterministic: bool,
        temperature: float,
        learnable_temperature: bool,
    ) -> None:
        super().__init__(
            decoder_input=DecoderInput(keys=[]),
            observation_space=observation_space,
            action_space=action_space,
            action_heads={DecoderOutputKey.ACTION_LOGITS.value: action_head},
            device="cpu",
            observation_horizon=1,
            prediction_horizon=ACTION_TOKEN_COUNT,
            temperature=temperature,
            learnable_temperature=learnable_temperature,
            deterministic=deterministic,
        )
        self.embedding_dimension = embedding_dimension
        self._init_action_bos_embedding(
            embedding_dimension=embedding_dimension,
            initializer_range=INITIALIZER_RANGE,
        )

    def _action_token_initializer_range(self) -> float:
        return INITIALIZER_RANGE

    def _action_token_embedding_dimension(self) -> int:
        return self.embedding_dimension

    def forward(
        self,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        return {}


class MissingEmbeddingDimensionDiscreteDecoder(DiscreteDecoder):
    action_head_layout: ActionHeadLayout = ActionHeadLayout.VOCABULARY

    def __init__(
        self,
        action_head: ActionHead,
        action_space: ActionSpace,
        observation_space: ObservationSpace,
    ) -> None:
        super().__init__(
            decoder_input=DecoderInput(keys=[]),
            observation_space=observation_space,
            action_space=action_space,
            action_heads={DecoderOutputKey.ACTION_LOGITS.value: action_head},
            device="cpu",
            observation_horizon=1,
            prediction_horizon=ACTION_TOKEN_COUNT,
            temperature=1.0,
            learnable_temperature=False,
            deterministic=True,
        )

    def _action_token_initializer_range(self) -> float:
        return INITIALIZER_RANGE

    def forward(
        self,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        return {}


@pytest.fixture
def tokenizer_factory() -> Callable[..., MagicMock]:
    def factory(
        vocabulary_size: int = VOCABULARY_SIZE,
        end_token_id: int = VOCABULARY_SIZE - 1,
    ) -> MagicMock:
        tokenizer = MagicMock(spec=Tokenizer)
        action_tokenizer = MagicMock(spec=ActionTokenizer)
        action_tokenizer.vocab_size = vocabulary_size
        action_tokenizer.eos_token_id = end_token_id
        tokenizer.action_tokenizer = action_tokenizer
        return tokenizer

    return factory


@pytest.fixture
def action_head_factory() -> Callable[..., MagicMock]:
    def factory(input_dimension: int = EMBEDDING_DIMENSION) -> MagicMock:
        action_head = MagicMock(spec=ActionHead)
        output_projection = MagicMock(spec=nn.Linear)
        output_projection.in_features = input_dimension
        action_head.output_proj = output_projection
        action_head.output_dim = 1
        action_head.set_output_dim.side_effect = lambda dim: setattr(
            action_head,
            "output_dim",
            dim,
        )
        return action_head

    return factory


@pytest.fixture
def discrete_decoder_factory(
    action_head_factory: Callable[..., MagicMock],
    mock_action_space_factory: Callable[..., MagicMock],
    mock_observation_space_factory: Callable[..., MagicMock],
) -> Callable[..., ConcreteDiscreteActionTokenDecoder]:
    def factory(
        action_head_input_dimension: int = EMBEDDING_DIMENSION,
        embedding_dimension: int = EMBEDDING_DIMENSION,
        deterministic: bool = True,
        temperature: float = 1.0,
        learnable_temperature: bool = False,
    ) -> ConcreteDiscreteActionTokenDecoder:
        return ConcreteDiscreteActionTokenDecoder(
            action_head=action_head_factory(
                input_dimension=action_head_input_dimension
            ),
            action_space=mock_action_space_factory(),
            observation_space=mock_observation_space_factory(),
            embedding_dimension=embedding_dimension,
            deterministic=deterministic,
            temperature=temperature,
            learnable_temperature=learnable_temperature,
        )

    return factory


@pytest.fixture
def tokenized_actions_factory() -> Callable[..., dict[str, torch.Tensor]]:
    def factory(
        batch_size: int = BATCH_SIZE,
        action_token_count: int = ACTION_TOKEN_COUNT,
    ) -> dict[str, torch.Tensor]:
        token_ids = torch.arange(
            batch_size * action_token_count,
            dtype=torch.long,
        ).reshape(batch_size, action_token_count)
        return {SampleKey.TOKENIZED_ACTIONS.value: token_ids}

    return factory


@pytest.mark.unit
@pytest.mark.parametrize(
    "learnable_temperature, expected_requires_grad",
    [
        (False, False),
        (True, True),
    ],
)
def test_init_stores_sampling_state(
    discrete_decoder_factory: Callable[..., ConcreteDiscreteActionTokenDecoder],
    learnable_temperature: bool,
    expected_requires_grad: bool,
) -> None:
    decoder = discrete_decoder_factory(
        temperature=0.5,
        learnable_temperature=learnable_temperature,
    )

    torch.testing.assert_close(decoder.temperature, torch.tensor(0.5))
    assert decoder.temperature.requires_grad is expected_requires_grad
    assert decoder.token_embedding is None
    assert decoder.vocab_size is None


@pytest.mark.unit
def test_init_action_bos_embedding_creates_batch_expandable_parameter(
    discrete_decoder_factory: Callable[..., ConcreteDiscreteActionTokenDecoder],
) -> None:
    decoder = discrete_decoder_factory(embedding_dimension=EMBEDDING_DIMENSION)

    expanded = decoder._expand_action_bos_embedding(
        batch_size=BATCH_SIZE,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    assert decoder.action_bos_embedding.shape == (1, 1, EMBEDDING_DIMENSION)
    assert decoder.action_bos_embedding.requires_grad
    assert expanded.shape == (BATCH_SIZE, 1, EMBEDDING_DIMENSION)
    assert expanded.dtype == torch.float64


@pytest.mark.unit
def test_action_token_embedding_dimension_returns_embedding_dimension(
    discrete_decoder_factory: Callable[..., ConcreteDiscreteActionTokenDecoder],
) -> None:
    decoder = discrete_decoder_factory(embedding_dimension=EMBEDDING_DIMENSION)

    assert decoder._action_token_embedding_dimension() == EMBEDDING_DIMENSION


@pytest.mark.unit
def test_action_token_embedding_dimension_requires_subclass_contract(
    action_head_factory: Callable[..., MagicMock],
    mock_action_space_factory: Callable[..., MagicMock],
    mock_observation_space_factory: Callable[..., MagicMock],
) -> None:
    decoder = MissingEmbeddingDimensionDiscreteDecoder(
        action_head=action_head_factory(input_dimension=EMBEDDING_DIMENSION),
        action_space=mock_action_space_factory(),
        observation_space=mock_observation_space_factory(),
    )
    expected_message = (
        "MissingEmbeddingDimensionDiscreteDecoder must define the "
        "action-token embedding dimension."
    )

    with pytest.raises(NotImplementedError, match=re.escape(expected_message)):
        decoder._action_token_embedding_dimension()


@pytest.mark.unit
@pytest.mark.parametrize("tokenizer_value", [None, "without_action_tokenizer"])
def test_set_tokenizer_rejects_missing_action_tokenizer(
    discrete_decoder_factory: Callable[..., ConcreteDiscreteActionTokenDecoder],
    tokenizer_value: str | None,
) -> None:
    decoder = discrete_decoder_factory()
    tokenizer = None
    if tokenizer_value == "without_action_tokenizer":
        tokenizer = MagicMock(spec=Tokenizer)
        tokenizer.action_tokenizer = None
    expected_message = (
        "ConcreteDiscreteActionTokenDecoder requires a tokenizer for "
        "tokenized action prediction."
    )

    with pytest.raises(ValueError, match=re.escape(expected_message)):
        decoder.set_tokenizer(tokenizer=tokenizer)


@pytest.mark.unit
def test_set_tokenizer_uses_direct_embedding_when_head_width_matches_embedding_width(
    discrete_decoder_factory: Callable[..., ConcreteDiscreteActionTokenDecoder],
    tokenizer_factory: Callable[..., MagicMock],
) -> None:
    decoder = discrete_decoder_factory(action_head_input_dimension=EMBEDDING_DIMENSION)

    decoder.set_tokenizer(tokenizer=tokenizer_factory())

    action_head = decoder.action_heads[DecoderOutputKey.ACTION_LOGITS.value]
    assert decoder.vocab_size == VOCABULARY_SIZE
    assert decoder.token_embedding.num_embeddings == VOCABULARY_SIZE
    assert decoder.token_embedding.embedding_dim == EMBEDDING_DIMENSION
    assert action_head.output_dim == VOCABULARY_SIZE
    assert action_head.output_proj.in_features == EMBEDDING_DIMENSION
    assert action_head.output_proj.out_features == VOCABULARY_SIZE
    assert (
        action_head.output_proj.weight.data_ptr()
        == decoder.token_embedding.weight.data_ptr()
    )
    assert decoder.tokenizer.eos_token_id == VOCABULARY_SIZE - 1


@pytest.mark.unit
def test_set_tokenizer_adds_projection_when_head_width_differs_from_embedding_width(
    discrete_decoder_factory: Callable[..., ConcreteDiscreteActionTokenDecoder],
    tokenizer_factory: Callable[..., MagicMock],
) -> None:
    decoder = discrete_decoder_factory(
        action_head_input_dimension=ACTION_HEAD_INPUT_DIMENSION,
        embedding_dimension=EMBEDDING_DIMENSION,
    )

    decoder.set_tokenizer(tokenizer=tokenizer_factory())

    action_head = decoder.action_heads[DecoderOutputKey.ACTION_LOGITS.value]
    token_input_embedding = decoder.token_embedding[0]
    token_projection = decoder.token_embedding[1]
    assert token_input_embedding.num_embeddings == VOCABULARY_SIZE
    assert token_input_embedding.embedding_dim == ACTION_HEAD_INPUT_DIMENSION
    assert token_projection.in_features == ACTION_HEAD_INPUT_DIMENSION
    assert token_projection.out_features == EMBEDDING_DIMENSION
    assert action_head.output_proj.in_features == ACTION_HEAD_INPUT_DIMENSION
    assert action_head.output_proj.out_features == VOCABULARY_SIZE
    assert (
        action_head.output_proj.weight.data_ptr()
        == token_input_embedding.weight.data_ptr()
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "has_token_embedding, has_tokenizer, vocabulary_size",
    [
        (False, True, VOCABULARY_SIZE),
        (True, False, VOCABULARY_SIZE),
        (True, True, None),
    ],
)
def test_validate_action_tokenizer_is_set_rejects_incomplete_tokenizer_state(
    discrete_decoder_factory: Callable[..., ConcreteDiscreteActionTokenDecoder],
    has_token_embedding: bool,
    has_tokenizer: bool,
    vocabulary_size: int | None,
) -> None:
    decoder = discrete_decoder_factory()
    decoder.token_embedding = (
        MagicMock(spec=nn.Embedding) if has_token_embedding else None
    )
    decoder.tokenizer = MagicMock(spec=ActionTokenizer) if has_tokenizer else None
    decoder.vocab_size = vocabulary_size
    expected_message = (
        "ConcreteDiscreteActionTokenDecoder requires set_tokenizer() "
        "to be called before forward."
    )

    with pytest.raises(ValueError, match=re.escape(expected_message)):
        decoder._validate_action_tokenizer_is_set()


@pytest.mark.unit
def test_validate_action_tokenizer_is_set_accepts_complete_tokenizer_state(
    discrete_decoder_factory: Callable[..., ConcreteDiscreteActionTokenDecoder],
) -> None:
    decoder = discrete_decoder_factory()
    decoder.token_embedding = MagicMock(spec=nn.Embedding)
    decoder.tokenizer = MagicMock(spec=ActionTokenizer)
    decoder.vocab_size = VOCABULARY_SIZE

    decoder._validate_action_tokenizer_is_set()


@pytest.mark.unit
def test_get_target_token_ids_returns_tokenized_action_ids(
    discrete_decoder_factory: Callable[..., ConcreteDiscreteActionTokenDecoder],
    tokenized_actions_factory: Callable[..., dict[str, torch.Tensor]],
) -> None:
    decoder = discrete_decoder_factory()
    actions = tokenized_actions_factory(batch_size=BATCH_SIZE)

    target_token_ids = decoder._get_target_token_ids(
        actions=actions,
        batch_size=BATCH_SIZE,
    )

    torch.testing.assert_close(
        target_token_ids,
        actions[SampleKey.TOKENIZED_ACTIONS.value],
    )


@pytest.mark.unit
def test_get_target_token_ids_rejects_missing_tokenized_actions(
    discrete_decoder_factory: Callable[..., ConcreteDiscreteActionTokenDecoder],
) -> None:
    decoder = discrete_decoder_factory()
    expected_message = (
        "ConcreteDiscreteActionTokenDecoder training requires "
        f"'{SampleKey.TOKENIZED_ACTIONS.value}' in actions."
    )

    with pytest.raises(ValueError, match=re.escape(expected_message)):
        decoder._get_target_token_ids(actions={}, batch_size=BATCH_SIZE)


@pytest.mark.unit
@pytest.mark.parametrize(
    "target_token_ids, expected_message",
    [
        (
            torch.zeros(BATCH_SIZE, ACTION_TOKEN_COUNT, 1, dtype=torch.long),
            f"'{SampleKey.TOKENIZED_ACTIONS.value}' must have shape "
            "(B, token_length), got torch.Size([2, 2, 1]).",
        ),
        (
            torch.zeros(BATCH_SIZE + 1, ACTION_TOKEN_COUNT, dtype=torch.long),
            f"'{SampleKey.TOKENIZED_ACTIONS.value}' batch size must match "
            "feature batch size 2, got 3.",
        ),
    ],
)
def test_get_target_token_ids_rejects_invalid_tokenized_action_shape(
    discrete_decoder_factory: Callable[..., ConcreteDiscreteActionTokenDecoder],
    target_token_ids: torch.Tensor,
    expected_message: str,
) -> None:
    decoder = discrete_decoder_factory()
    actions = {SampleKey.TOKENIZED_ACTIONS.value: target_token_ids}

    with pytest.raises(ValueError, match=re.escape(expected_message)):
        decoder._get_target_token_ids(actions=actions, batch_size=BATCH_SIZE)


@pytest.mark.unit
def test_sample_next_action_token_returns_argmax_when_deterministic(
    discrete_decoder_factory: Callable[..., ConcreteDiscreteActionTokenDecoder],
) -> None:
    decoder = discrete_decoder_factory(deterministic=True)
    logits = torch.tensor([[[0.1, 0.9, 0.2]], [[0.8, 0.1, 0.3]]])

    next_token = decoder._sample_next_action_token(logits=logits)

    torch.testing.assert_close(next_token, torch.tensor([[1], [0]]))


@pytest.mark.unit
def test_sample_next_action_token_uses_temperature_scaled_probabilities(
    discrete_decoder_factory: Callable[..., ConcreteDiscreteActionTokenDecoder],
) -> None:
    decoder = discrete_decoder_factory(deterministic=False, temperature=0.5)
    logits = torch.tensor([[[0.1, 0.9, 0.2]], [[0.8, 0.1, 0.3]]])
    expected_probabilities = torch.softmax(logits / 0.5, dim=-1).squeeze(1)
    sampled_token = torch.tensor([[2], [1]])

    with patch(
        "versatil.models.decoding.decoders.discrete.torch.multinomial",
        autospec=True,
        return_value=sampled_token,
    ) as multinomial_mock:
        next_token = decoder._sample_next_action_token(logits=logits)

    torch.testing.assert_close(next_token, sampled_token)
    torch.testing.assert_close(
        multinomial_mock.call_args.args[0], expected_probabilities
    )
    assert multinomial_mock.call_args.kwargs["num_samples"] == 1
