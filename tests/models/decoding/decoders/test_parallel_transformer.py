"""Tests for versatil.models.decoding.decoders.parallel_transformer module."""

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from versatil.data.task import ActionSpace, ObservationSpace
from versatil.models.decoding.action_heads.single_output import ActionHead
from versatil.models.decoding.constants import ActionHeadLayout
from versatil.models.decoding.decoders.base import DecoderInput
from versatil.models.decoding.decoders.parallel_transformer import (
    BaseParallelTransformerDecoder,
)
from versatil.models.decoding.transformer_input_builder import TransformerInputBuilder
from versatil.models.layers.constants import PositionalEncodingType
from versatil.models.layers.positional_encoding.learned import (
    LearnedPositionalEncoding1D,
)
from versatil.models.layers.positional_encoding.sinusoidal import (
    SinusoidalPositionalEncoding1D,
)

BATCH_SIZE = 2
PREDICTION_HORIZON = 4
EMBEDDING_DIMENSION = 8
TOKEN_COUNT = 3


class ConcreteParallelTransformerDecoder(BaseParallelTransformerDecoder):
    action_head_layout = ActionHeadLayout.NONE

    def __init__(
        self,
        observation_horizon: int = 1,
    ) -> None:
        super().__init__(
            decoder_input=DecoderInput(keys=[], requires_actions=False),
            action_space=MagicMock(spec=ActionSpace),
            action_heads={},
            observation_space=MagicMock(spec=ObservationSpace),
            prediction_horizon=PREDICTION_HORIZON,
            observation_horizon=observation_horizon,
            device="cpu",
            embedding_dimension=EMBEDDING_DIMENSION,
        )

    def forward(
        self,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return no predictions; tests exercise base helpers directly."""
        return {}


@pytest.fixture
def parallel_transformer_decoder_factory() -> Callable[
    ..., ConcreteParallelTransformerDecoder
]:
    def factory(
        observation_horizon: int = 1,
    ) -> ConcreteParallelTransformerDecoder:
        return ConcreteParallelTransformerDecoder(
            observation_horizon=observation_horizon,
        )

    return factory


@pytest.fixture
def observation_tokens_factory() -> Callable[..., torch.Tensor]:
    def factory() -> torch.Tensor:
        return torch.ones(BATCH_SIZE, TOKEN_COUNT, EMBEDDING_DIMENSION)

    return factory


@pytest.mark.unit
class TestParallelTransformerInputBuilder:
    @pytest.mark.parametrize(
        "observation_horizon, expects_temporal_encoding",
        [
            (1, False),
            (3, True),
        ],
    )
    def test_builds_standard_input_sequence_builder(
        self,
        parallel_transformer_decoder_factory: Callable[
            ..., ConcreteParallelTransformerDecoder
        ],
        observation_horizon: int,
        expects_temporal_encoding: bool,
    ) -> None:
        decoder = parallel_transformer_decoder_factory(
            observation_horizon=observation_horizon,
        )
        input_sequence_builder = MagicMock(spec=TransformerInputBuilder)
        temporal_encoding = MagicMock(spec=LearnedPositionalEncoding1D)
        flat_encoding = MagicMock(spec=LearnedPositionalEncoding1D)
        learned_encoding_side_effects = (
            [temporal_encoding, flat_encoding]
            if expects_temporal_encoding
            else [flat_encoding]
        )

        with (
            patch(
                "versatil.models.decoding.decoders.parallel_transformer.TransformerInputBuilder",
                autospec=True,
                return_value=input_sequence_builder,
            ) as input_sequence_builder_class,
            patch(
                "versatil.models.decoding.decoders.parallel_transformer.LearnedPositionalEncoding1D",
                autospec=True,
                side_effect=learned_encoding_side_effects,
            ) as learned_encoding_class,
            patch(
                "versatil.models.decoding.decoders.parallel_transformer.SinusoidalPositionalEncoding2D",
                autospec=True,
            ) as spatial_encoding_class,
        ):
            decoder._build_parallel_input_sequence_builder(
                exclude_keys=["latent"],
            )

        spatial_encoding_class.assert_called_once_with(
            embedding_dimension=EMBEDDING_DIMENSION,
            normalize=True,
        )
        if expects_temporal_encoding:
            assert learned_encoding_class.call_count == 2
            temporal_layer = temporal_encoding
        else:
            assert learned_encoding_class.call_count == 1
            temporal_layer = None
        input_sequence_builder_class.assert_called_once_with(
            embedding_dim=EMBEDDING_DIMENSION,
            has_time_dim=observation_horizon > 1,
            spatial_positional_encoding_layer=spatial_encoding_class.return_value,
            flat_positional_encoding_layer=flat_encoding,
            temporal_positional_encoding_layer=temporal_layer,
            exclude_keys=["latent"],
        )

    def test_builds_input_sequence_builder_with_sinusoidal_flat_encoding(
        self,
        parallel_transformer_decoder_factory: Callable[
            ..., ConcreteParallelTransformerDecoder
        ],
    ) -> None:
        decoder = parallel_transformer_decoder_factory(observation_horizon=3)
        input_sequence_builder = MagicMock(spec=TransformerInputBuilder)
        temporal_encoding = MagicMock(spec=LearnedPositionalEncoding1D)
        flat_encoding = MagicMock(spec=SinusoidalPositionalEncoding1D)

        with (
            patch(
                "versatil.models.decoding.decoders.parallel_transformer.TransformerInputBuilder",
                autospec=True,
                return_value=input_sequence_builder,
            ) as input_sequence_builder_class,
            patch(
                "versatil.models.decoding.decoders.parallel_transformer.LearnedPositionalEncoding1D",
                autospec=True,
                return_value=temporal_encoding,
            ) as learned_encoding_class,
            patch(
                "versatil.models.decoding.decoders.parallel_transformer.SinusoidalPositionalEncoding1D",
                autospec=True,
                return_value=flat_encoding,
            ) as flat_encoding_class,
            patch(
                "versatil.models.decoding.decoders.parallel_transformer.SinusoidalPositionalEncoding2D",
                autospec=True,
            ) as spatial_encoding_class,
        ):
            decoder._build_parallel_input_sequence_builder(
                flat_positional_encoding_type=PositionalEncodingType.SINUSOIDAL.value,
            )

        learned_encoding_class.assert_called_once_with(
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        flat_encoding_class.assert_called_once_with(
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        input_sequence_builder_class.assert_called_once_with(
            embedding_dim=EMBEDDING_DIMENSION,
            has_time_dim=True,
            spatial_positional_encoding_layer=spatial_encoding_class.return_value,
            flat_positional_encoding_layer=flat_encoding,
            temporal_positional_encoding_layer=temporal_encoding,
            exclude_keys=None,
        )

    def test_rejects_unknown_flat_positional_encoding_type(
        self,
        parallel_transformer_decoder_factory: Callable[
            ..., ConcreteParallelTransformerDecoder
        ],
    ) -> None:
        decoder = parallel_transformer_decoder_factory()

        with pytest.raises(
            ValueError,
            match="flat_positional_encoding_type must be one of",
        ):
            decoder._build_flat_positional_encoding(
                flat_positional_encoding_type="invalid",
            )


@pytest.mark.unit
class TestParallelTransformerQueries:
    def test_build_query_embedding_uses_prediction_horizon_and_embedding_dimension(
        self,
        parallel_transformer_decoder_factory: Callable[
            ..., ConcreteParallelTransformerDecoder
        ],
    ) -> None:
        decoder = parallel_transformer_decoder_factory()

        query_embedding = decoder._build_parallel_query_embedding()

        assert query_embedding.num_embeddings == PREDICTION_HORIZON
        assert query_embedding.embedding_dim == EMBEDDING_DIMENSION

    def test_expand_query_embedding_adds_batch_dimension(
        self,
        parallel_transformer_decoder_factory: Callable[
            ..., ConcreteParallelTransformerDecoder
        ],
    ) -> None:
        decoder = parallel_transformer_decoder_factory()
        query_embedding = nn.Embedding(PREDICTION_HORIZON, EMBEDDING_DIMENSION)

        expanded_query = decoder._expand_parallel_query_embedding(
            query_embedding=query_embedding,
            batch_size=BATCH_SIZE,
        )

        assert expanded_query.shape == (
            BATCH_SIZE,
            PREDICTION_HORIZON,
            EMBEDDING_DIMENSION,
        )
        torch.testing.assert_close(expanded_query[0], query_embedding.weight)

    def test_expand_query_tensor_adds_batch_dimension(
        self,
        parallel_transformer_decoder_factory: Callable[
            ..., ConcreteParallelTransformerDecoder
        ],
    ) -> None:
        decoder = parallel_transformer_decoder_factory()
        query = torch.ones(PREDICTION_HORIZON, EMBEDDING_DIMENSION)

        expanded_query = decoder._expand_parallel_query_tensor(
            query=query,
            batch_size=BATCH_SIZE,
        )

        assert expanded_query.shape == (
            BATCH_SIZE,
            PREDICTION_HORIZON,
            EMBEDDING_DIMENSION,
        )
        torch.testing.assert_close(expanded_query[0], query)


@pytest.mark.unit
class TestParallelTransformerForwardHelpers:
    @pytest.mark.parametrize("add_positional_encodings", [False, True])
    def test_build_observation_tokens_uses_input_sequence_builder(
        self,
        parallel_transformer_decoder_factory: Callable[
            ..., ConcreteParallelTransformerDecoder
        ],
        observation_tokens_factory: Callable[..., torch.Tensor],
        add_positional_encodings: bool,
    ) -> None:
        decoder = parallel_transformer_decoder_factory()
        features = {"rgb": torch.ones(BATCH_SIZE, EMBEDDING_DIMENSION)}
        observation_tokens = observation_tokens_factory()
        positional_encodings = torch.full_like(observation_tokens, 2.0)
        padding_mask = torch.zeros(BATCH_SIZE, TOKEN_COUNT, dtype=torch.bool)
        input_sequence_builder = MagicMock(spec=TransformerInputBuilder)
        input_sequence_builder.return_value = (
            observation_tokens,
            positional_encodings,
            padding_mask,
        )

        built_tokens, built_padding_mask = decoder._build_parallel_observation_tokens(
            input_sequence_builder=input_sequence_builder,
            features=features,
            add_positional_encodings=add_positional_encodings,
        )

        input_sequence_builder.assert_called_once_with(features)
        expected_tokens = (
            observation_tokens + positional_encodings
            if add_positional_encodings
            else observation_tokens
        )
        torch.testing.assert_close(built_tokens, expected_tokens)
        torch.testing.assert_close(built_padding_mask, padding_mask)

    def test_apply_action_heads_calls_each_head_with_action_embeddings(
        self,
        parallel_transformer_decoder_factory: Callable[
            ..., ConcreteParallelTransformerDecoder
        ],
    ) -> None:
        decoder = parallel_transformer_decoder_factory()
        action_embeddings = torch.ones(
            BATCH_SIZE,
            PREDICTION_HORIZON,
            EMBEDDING_DIMENSION,
        )
        position_output = torch.ones(BATCH_SIZE, PREDICTION_HORIZON, 3)
        gripper_output = torch.ones(BATCH_SIZE, PREDICTION_HORIZON, 1)
        position_head = MagicMock(spec=ActionHead)
        gripper_head = MagicMock(spec=ActionHead)
        position_head.return_value = position_output
        gripper_head.return_value = gripper_output
        action_heads = MagicMock(spec=nn.ModuleDict)
        action_heads.items.return_value = [
            ("position_action", position_head),
            ("gripper_action", gripper_head),
        ]
        decoder.__dict__["action_heads"] = action_heads

        predictions = decoder._apply_action_heads(action_embeddings=action_embeddings)

        position_head.assert_called_once()
        gripper_head.assert_called_once()
        torch.testing.assert_close(position_head.call_args.args[0], action_embeddings)
        torch.testing.assert_close(gripper_head.call_args.args[0], action_embeddings)
        torch.testing.assert_close(predictions["position_action"], position_output)
        torch.testing.assert_close(predictions["gripper_action"], gripper_output)
