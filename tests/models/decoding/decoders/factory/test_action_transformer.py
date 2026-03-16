"""Tests for versatil.models.decoding.decoders.factory.action_transformer module."""
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
import torch

from versatil.models.decoding.action_heads.single_output import ActionHead
from versatil.models.decoding.constants import FeatureType
from versatil.models.decoding.decoders.base import ActionDecoder
from versatil.models.decoding.decoders.factory.action_transformer import (
    ActionTransformer,
)
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType, PositionalEncodingType
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.positional_encoding.learned import (
    LearnedPositionalEncoding1D,
)
from versatil.models.layers.transformer.bidirectional_decoder import (
    BidirectionalDecoder,
)
from versatil.models.decoding.transformer_input_builder import TransformerInputBuilder


EMBEDDING_DIMENSION = 32
NUMBER_OF_HEADS = 2
NUMBER_OF_LAYERS = 1
FEEDFORWARD_DIMENSION = 64
SPATIAL_HEIGHT = 4
SPATIAL_WIDTH = 4
BATCH_SIZE = 2
POSITION_DIM = 3


@pytest.fixture
def action_transformer_factory(
    mock_action_space_factory: Callable[..., MagicMock],
    mock_observation_space_factory: Callable[..., MagicMock],
    action_heads_factory: Callable[..., dict[str, ActionHead]],
) -> Callable[..., ActionTransformer]:
    """Factory for ActionTransformer instances with small dimensions."""

    def factory(
        input_keys: list[str] | None = None,
        position_dim: int = POSITION_DIM,
        has_orientation: bool = False,
        orientation_dim: int = 0,
        has_gripper: bool = False,
        gripper_dim: int = 0,
        observation_horizon: int = 1,
        prediction_horizon: int = 4,
        embedding_dimension: int = EMBEDDING_DIMENSION,
        number_of_heads: int = NUMBER_OF_HEADS,
        number_of_key_value_heads: int | None = None,
        feedforward_dimension: int | None = FEEDFORWARD_DIMENSION,
        number_of_layers: int = NUMBER_OF_LAYERS,
        activation: str = ActivationFunction.GELU.value,
        normalization_type: str = NormalizationType.LAYER_NORM.value,
        attention_type: str = AttentionType.MULTI_HEAD.value,
        dropout_rate: float = 0.1,
        attention_dropout: float = 0.0,
        positional_encoding_type: str | None = PositionalEncodingType.ROPE.value,
        device: str = "cpu",
    ) -> ActionTransformer:
        if input_keys is None:
            input_keys = ["rgb_features"]
        action_space = mock_action_space_factory(
            position_dim=position_dim,
            has_orientation=has_orientation,
            orientation_dim=orientation_dim,
            has_gripper=has_gripper,
            gripper_dim=gripper_dim,
        )
        action_heads = action_heads_factory(
            action_space=action_space,
            input_dim=embedding_dimension,
        )
        observation_space = mock_observation_space_factory()
        return ActionTransformer(
            input_keys=input_keys,
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_key_value_heads=number_of_key_value_heads,
            feedforward_dimension=feedforward_dimension,
            number_of_layers=number_of_layers,
            activation=activation,
            normalization_type=normalization_type,
            attention_type=attention_type,
            dropout_rate=dropout_rate,
            attention_dropout=attention_dropout,
            positional_encoding_type=positional_encoding_type,
        )

    return factory


class TestActionTransformerInitialization:

    def test_inherits_from_action_decoder(
        self,
        action_transformer_factory: Callable[..., ActionTransformer],
    ):
        decoder = action_transformer_factory()
        assert isinstance(decoder, ActionDecoder)

    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    @pytest.mark.parametrize("number_of_layers", [1, 2])
    @pytest.mark.parametrize("activation", [ActivationFunction.GELU.value, ActivationFunction.SWIGLU.value])
    @pytest.mark.parametrize("normalization_type", [NormalizationType.LAYER_NORM.value, NormalizationType.RMS_NORM.value])
    def test_stores_configuration(
        self,
        action_transformer_factory: Callable[..., ActionTransformer],
        embedding_dimension: int,
        number_of_layers: int,
        activation: str,
        normalization_type: str,
    ):
        decoder = action_transformer_factory(
            embedding_dimension=embedding_dimension,
            number_of_layers=number_of_layers,
            number_of_heads=NUMBER_OF_HEADS,
            feedforward_dimension=FEEDFORWARD_DIMENSION,
            activation=activation,
            normalization_type=normalization_type,
            attention_type=AttentionType.MULTI_HEAD.value,
            dropout_rate=0.05,
            attention_dropout=0.02,
            positional_encoding_type=PositionalEncodingType.ROPE.value,
        )
        assert decoder.embedding_dimension == embedding_dimension
        assert decoder.number_of_layers == number_of_layers
        assert decoder.number_of_heads == NUMBER_OF_HEADS
        assert decoder.feedforward_dimension == FEEDFORWARD_DIMENSION
        assert decoder.activation == activation
        assert decoder.normalization_type == normalization_type
        assert decoder.attention_type == AttentionType.MULTI_HEAD.value
        assert decoder.dropout_rate == 0.05
        assert decoder.attention_dropout == 0.02
        assert decoder.positional_encoding_type == PositionalEncodingType.ROPE.value

    def test_creates_components(
        self,
        action_transformer_factory: Callable[..., ActionTransformer],
    ):
        decoder = action_transformer_factory()
        assert isinstance(decoder.input_sequence_builder, TransformerInputBuilder)
        assert isinstance(decoder.learnable_query, torch.nn.Embedding)
        assert isinstance(decoder.action_decoder, BidirectionalDecoder)

    def test_decoder_input_requires_spatial(
        self,
        action_transformer_factory: Callable[..., ActionTransformer],
    ):
        decoder = action_transformer_factory()
        assert FeatureType.SPATIAL.value in decoder.decoder_input.required_types
        assert decoder.decoder_input.requires_actions is False

    def test_learnable_query_shape(
        self,
        action_transformer_factory: Callable[..., ActionTransformer],
    ):
        prediction_horizon = 6
        embedding_dimension = 32
        decoder = action_transformer_factory(
            prediction_horizon=prediction_horizon,
            embedding_dimension=embedding_dimension,
        )
        assert decoder.learnable_query.weight.shape == (
            prediction_horizon,
            embedding_dimension,
        )


class TestActionTransformerForward:

    def test_output_keys_match_action_heads(
        self,
        action_transformer_factory: Callable[..., ActionTransformer],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = action_transformer_factory()
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        predictions = decoder(features=features)
        assert set(predictions.keys()) == set(decoder.action_heads.keys())

    @pytest.mark.parametrize("prediction_horizon", [4, 8])
    def test_output_shape(
        self,
        action_transformer_factory: Callable[..., ActionTransformer],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
        prediction_horizon: int,
    ):
        decoder = action_transformer_factory(
            prediction_horizon=prediction_horizon,
        )
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        predictions = decoder(features=features)
        for action_key in decoder.action_heads:
            expected_dim = decoder.action_heads[action_key].output_dim
            assert predictions[action_key].shape == (
                BATCH_SIZE,
                prediction_horizon,
                expected_dim,
            )

    def test_with_multiple_action_heads(
        self,
        action_transformer_factory: Callable[..., ActionTransformer],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = action_transformer_factory(
            position_dim=3,
            has_orientation=True,
            orientation_dim=4,
            has_gripper=True,
            gripper_dim=1,
        )
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        predictions = decoder(features=features)
        assert "position_action" in predictions
        assert "orientation_action" in predictions
        assert "gripper_action" in predictions
        assert predictions["position_action"].shape == (BATCH_SIZE, 4, 3)
        assert predictions["orientation_action"].shape == (BATCH_SIZE, 4, 4)
        assert predictions["gripper_action"].shape == (BATCH_SIZE, 4, 1)

    def test_with_multiple_spatial_features_changes_output(
        self,
        action_transformer_factory: Callable[..., ActionTransformer],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = action_transformer_factory(
            input_keys=["left_features", "right_features"],
        )
        decoder.eval()
        features_both = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
            feature_keys=["left_features", "right_features"],
        )
        with torch.no_grad():
            predictions_both = decoder(features=features_both)
        features_swapped = {
            "left_features": features_both["right_features"],
            "right_features": features_both["left_features"],
        }
        with torch.no_grad():
            predictions_swapped = decoder(features=features_swapped)
        for action_key in decoder.action_heads:
            assert not torch.equal(
                predictions_both[action_key],
                predictions_swapped[action_key],
            )

    def test_forward_ignores_actions_argument(
        self,
        action_transformer_factory: Callable[..., ActionTransformer],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = action_transformer_factory()
        decoder.eval()
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        dummy_actions = noisy_actions_factory()
        predictions_without = decoder(features=features)
        predictions_with = decoder(features=features, actions=dummy_actions)
        assert set(predictions_without.keys()) == set(predictions_with.keys())
        for key in predictions_without:
            assert torch.equal(predictions_without[key], predictions_with[key])


class TestActionTransformerTemporalObservation:

    @pytest.mark.parametrize("observation_horizon, expects_temporal_pe", [
        (1, False),
        (3, True),
    ])
    def test_temporal_pe_created_based_on_observation_horizon(
        self,
        action_transformer_factory: Callable[..., ActionTransformer],
        observation_horizon: int,
        expects_temporal_pe: bool,
    ):
        decoder = action_transformer_factory(observation_horizon=observation_horizon)
        layer = decoder.input_sequence_builder.temporal_positional_encoding_layer
        if expects_temporal_pe:
            assert isinstance(layer, LearnedPositionalEncoding1D)
        else:
            assert layer is None

