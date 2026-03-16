"""Tests for versatil.models.decoding.decoders.factory.act module."""
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
import torch

from versatil.models.decoding.action_heads.single_output import ActionHead
from versatil.models.decoding.constants import FeatureType, LatentKey
from versatil.models.decoding.decoders.base import ActionDecoder
from versatil.models.decoding.decoders.factory.act import ACT
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.detr_transformer.transformer import Transformer
from versatil.models.layers.positional_encoding.learned import (
    LearnedPositionalEncoding1D,
)
from versatil.models.decoding.transformer_input_builder import TransformerInputBuilder


EMBEDDING_DIMENSION = 32
NUMBER_OF_HEADS = 2
NUMBER_OF_ENCODER_LAYERS = 1
NUMBER_OF_DECODER_LAYERS = 1
FEEDFORWARD_DIMENSION = 64
SPATIAL_HEIGHT = 4
SPATIAL_WIDTH = 4
BATCH_SIZE = 2
POSITION_DIM = 3


@pytest.fixture
def act_factory(
    mock_action_space_factory: Callable[..., MagicMock],
    mock_observation_space_factory: Callable[..., MagicMock],
    action_heads_factory: Callable[..., dict[str, ActionHead]],
) -> Callable[..., ACT]:
    """Factory for ACT instances with small dimensions."""

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
        feedforward_dimension: int = FEEDFORWARD_DIMENSION,
        number_of_encoder_layers: int = NUMBER_OF_ENCODER_LAYERS,
        number_of_decoder_layers: int = NUMBER_OF_DECODER_LAYERS,
        activation: str = ActivationFunction.RELU.value,
        dropout_rate: float = 0.1,
        normalize_before: bool = False,
        device: str = "cpu",
    ) -> ACT:
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
        return ACT(
            input_keys=input_keys,
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            feedforward_dimension=feedforward_dimension,
            number_of_encoder_layers=number_of_encoder_layers,
            number_of_decoder_layers=number_of_decoder_layers,
            activation=activation,
            dropout_rate=dropout_rate,
            normalize_before=normalize_before,
        )

    return factory


class TestACTInitialization:

    def test_inherits_from_action_decoder(
        self,
        act_factory: Callable[..., ACT],
    ):
        decoder = act_factory()
        assert isinstance(decoder, ActionDecoder)

    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    @pytest.mark.parametrize("number_of_encoder_layers", [1, 2])
    @pytest.mark.parametrize("number_of_decoder_layers", [1, 2])
    @pytest.mark.parametrize("normalize_before", [True, False])
    def test_stores_configuration(
        self,
        act_factory: Callable[..., ACT],
        embedding_dimension: int,
        number_of_encoder_layers: int,
        number_of_decoder_layers: int,
        normalize_before: bool,
    ):
        decoder = act_factory(
            embedding_dimension=embedding_dimension,
            number_of_encoder_layers=number_of_encoder_layers,
            number_of_heads=NUMBER_OF_HEADS,
            feedforward_dimension=FEEDFORWARD_DIMENSION,
            number_of_decoder_layers=number_of_decoder_layers,
            activation=ActivationFunction.RELU.value,
            dropout_rate=0.05,
            normalize_before=normalize_before,
        )
        assert decoder.embedding_dimension == embedding_dimension
        assert decoder.number_of_encoder_layers == number_of_encoder_layers
        assert decoder.number_of_heads == NUMBER_OF_HEADS
        assert decoder.feedforward_dimension == FEEDFORWARD_DIMENSION
        assert decoder.number_of_decoder_layers == number_of_decoder_layers
        assert decoder.activation == ActivationFunction.RELU.value
        assert decoder.dropout_rate == 0.05
        assert decoder.normalize_before is normalize_before

    def test_creates_detr_transformer(
        self,
        act_factory: Callable[..., ACT],
    ):
        decoder = act_factory()
        assert isinstance(decoder.input_sequence_builder, TransformerInputBuilder)
        assert isinstance(decoder.action_decoder, Transformer)
        assert isinstance(decoder.learnable_query, torch.nn.Embedding)

    def test_decoder_input_requires_spatial(
        self,
        act_factory: Callable[..., ACT],
    ):
        decoder = act_factory()
        assert FeatureType.SPATIAL.value in decoder.decoder_input.required_types
        assert decoder.decoder_input.requires_actions is False

    def test_learnable_query_shape(
        self,
        act_factory: Callable[..., ACT],
    ):
        prediction_horizon = 6
        embedding_dimension = 32
        decoder = act_factory(
            prediction_horizon=prediction_horizon,
            embedding_dimension=embedding_dimension,
        )
        assert decoder.learnable_query.weight.shape == (
            prediction_horizon,
            embedding_dimension,
        )


class TestACTForward:

    def test_output_keys_match_action_heads(
        self,
        act_factory: Callable[..., ACT],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = act_factory()
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
        act_factory: Callable[..., ACT],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
        prediction_horizon: int,
    ):
        decoder = act_factory(
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
        act_factory: Callable[..., ACT],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = act_factory(
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
        act_factory: Callable[..., ACT],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = act_factory(
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

    def test_forward_with_latent_in_features_changes_output(
        self,
        act_factory: Callable[..., ACT],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
        input_tensor_factory: Callable[..., torch.Tensor],
    ):
        decoder = act_factory()
        decoder.eval()
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        with torch.no_grad():
            predictions_without = decoder(features=features)
        features[LatentKey.POSTERIOR_LATENT.value] = input_tensor_factory(
            batch_size=BATCH_SIZE,
            input_dimension=EMBEDDING_DIMENSION,
        )
        with torch.no_grad():
            predictions_with = decoder(features=features)
        for action_key in decoder.action_heads:
            assert not torch.equal(
                predictions_without[action_key],
                predictions_with[action_key],
            )

    def test_forward_ignores_actions_argument(
        self,
        act_factory: Callable[..., ACT],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = act_factory()
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


class TestACTTemporalObservation:

    @pytest.mark.parametrize("observation_horizon, expects_temporal_pe", [
        (1, False),
        (3, True),
    ])
    def test_temporal_pe_created_based_on_observation_horizon(
        self,
        act_factory: Callable[..., ACT],
        observation_horizon: int,
        expects_temporal_pe: bool,
    ):
        decoder = act_factory(observation_horizon=observation_horizon)
        layer = decoder.input_sequence_builder.temporal_positional_encoding_layer
        if expects_temporal_pe:
            assert isinstance(layer, LearnedPositionalEncoding1D)
        else:
            assert layer is None


class TestACTDecodeActions:

    def test_decode_actions_output_shape(
        self,
        act_factory: Callable[..., ACT],
        input_tensor_factory: Callable[..., torch.Tensor],
    ):
        prediction_horizon = 4
        decoder = act_factory(prediction_horizon=prediction_horizon)
        sequence_length = SPATIAL_HEIGHT * SPATIAL_WIDTH
        input_tokens = input_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=sequence_length,
            input_dimension=EMBEDDING_DIMENSION,
        )
        positional_encodings = input_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=sequence_length,
            input_dimension=EMBEDDING_DIMENSION,
        )
        action_embeddings = decoder._decode_actions(
            input_tokens=input_tokens,
            positional_encodings=positional_encodings,
            padding_mask=None,
        )
        assert action_embeddings.shape == (
            BATCH_SIZE,
            prediction_horizon,
            EMBEDDING_DIMENSION,
        )

    def test_decode_actions_with_padding_mask_changes_output(
        self,
        act_factory: Callable[..., ACT],
        input_tensor_factory: Callable[..., torch.Tensor],
    ):
        prediction_horizon = 4
        decoder = act_factory(prediction_horizon=prediction_horizon)
        decoder.eval()
        sequence_length = SPATIAL_HEIGHT * SPATIAL_WIDTH
        input_tokens = input_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=sequence_length,
            input_dimension=EMBEDDING_DIMENSION,
        )
        positional_encodings = input_tensor_factory(
            batch_size=BATCH_SIZE,
            sequence_length=sequence_length,
            input_dimension=EMBEDDING_DIMENSION,
        )
        with torch.no_grad():
            output_without_mask = decoder._decode_actions(
                input_tokens=input_tokens,
                positional_encodings=positional_encodings,
                padding_mask=None,
            )
        padding_mask = torch.zeros(
            BATCH_SIZE, sequence_length, dtype=torch.bool
        )
        padding_mask[:, -2:] = True
        with torch.no_grad():
            output_with_mask = decoder._decode_actions(
                input_tokens=input_tokens,
                positional_encodings=positional_encodings,
                padding_mask=padding_mask,
            )
        assert not torch.equal(output_without_mask, output_with_mask)
