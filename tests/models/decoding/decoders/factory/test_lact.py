"""Tests for versatil.models.decoding.decoders.factory.lact module."""

import re
from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from versatil.models.decoding.action_heads.single_output import ActionHead
from versatil.models.decoding.constants import LatentKey
from versatil.models.decoding.decoders.base import ActionDecoder
from versatil.models.decoding.decoders.factory.lact import LACT
from versatil.models.feature_meta import FeatureType
from versatil.models.layers.transformer.conditional_bidirectional_decoder import (
    ConditionalBidirectionalDecoder,
)

EMBEDDING_DIMENSION = 32
LATENT_DIMENSION = 16
NUMBER_OF_HEADS = 2
NUMBER_OF_LAYERS = 1
PREDICTION_HORIZON = 4
OBSERVATION_HORIZON = 1
BATCH_SIZE = 2
SPATIAL_CHANNELS = 32
SPATIAL_HEIGHT = 7
SPATIAL_WIDTH = 7
POSITION_DIM = 3


@pytest.fixture
def lact_decoder_factory(
    mock_action_space_factory: Callable[..., MagicMock],
    mock_observation_space_factory: Callable[..., MagicMock],
    action_heads_factory: Callable[..., dict[str, ActionHead]],
) -> Callable[..., LACT]:
    """Factory for LACT instances with small dimensions."""

    def factory(
        input_keys: list[str] | None = None,
        position_dim: int = POSITION_DIM,
        has_orientation: bool = False,
        orientation_dim: int = 0,
        has_gripper: bool = False,
        gripper_dim: int = 0,
        embedding_dimension: int = EMBEDDING_DIMENSION,
        latent_dimension: int = LATENT_DIMENSION,
        number_of_heads: int = NUMBER_OF_HEADS,
        number_of_layers: int = NUMBER_OF_LAYERS,
        observation_horizon: int = OBSERVATION_HORIZON,
        prediction_horizon: int = PREDICTION_HORIZON,
    ) -> LACT:
        if input_keys is None:
            input_keys = ["rgb_features", LatentKey.POSTERIOR_LATENT.value]
        action_space = mock_action_space_factory(
            position_dim=position_dim,
            has_orientation=has_orientation,
            orientation_dim=orientation_dim,
            has_gripper=has_gripper,
            gripper_dim=gripper_dim,
        )
        observation_space = mock_observation_space_factory()
        action_heads = action_heads_factory(
            action_space=action_space,
            input_dim=embedding_dimension,
        )
        return LACT(
            input_keys=input_keys,
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device="cpu",
            latent_dimension=latent_dimension,
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_layers=number_of_layers,
        )

    return factory


@pytest.fixture
def spatial_features_with_latent_factory(
    spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    rng: np.random.Generator,
) -> Callable[..., dict[str, torch.Tensor]]:
    """Factory for spatial features with a latent key included."""

    def factory(
        batch_size: int = BATCH_SIZE,
        channels: int = SPATIAL_CHANNELS,
        height: int = SPATIAL_HEIGHT,
        width: int = SPATIAL_WIDTH,
        latent_dimension: int = LATENT_DIMENSION,
        feature_keys: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        if feature_keys is None:
            feature_keys = ["rgb_features"]
        features = spatial_feature_factory(
            batch_size=batch_size,
            channels=channels,
            height=height,
            width=width,
            feature_keys=feature_keys,
        )
        features[LatentKey.POSTERIOR_LATENT.value] = torch.from_numpy(
            rng.standard_normal((batch_size, latent_dimension)).astype(np.float32)
        )
        return features

    return factory


class TestLACTInitialization:
    def test_inherits_from_action_decoder(
        self,
        lact_decoder_factory: Callable[..., LACT],
    ):
        decoder = lact_decoder_factory()
        assert isinstance(decoder, ActionDecoder)

    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    @pytest.mark.parametrize("latent_dimension", [8, 16])
    @pytest.mark.parametrize("number_of_heads", [2, 4])
    @pytest.mark.parametrize("number_of_layers", [1, 2])
    def test_stores_configuration(
        self,
        lact_decoder_factory: Callable[..., LACT],
        embedding_dimension: int,
        latent_dimension: int,
        number_of_heads: int,
        number_of_layers: int,
    ):
        decoder = lact_decoder_factory(
            embedding_dimension=embedding_dimension,
            latent_dimension=latent_dimension,
            number_of_heads=number_of_heads,
            number_of_layers=number_of_layers,
        )
        assert decoder.embedding_dimension == embedding_dimension
        assert decoder.latent_dimension == latent_dimension
        assert decoder.number_of_heads == number_of_heads
        assert decoder.number_of_layers == number_of_layers

    def test_creates_cross_conditioning_decoder(
        self,
        lact_decoder_factory: Callable[..., LACT],
    ):
        decoder = lact_decoder_factory()
        assert isinstance(decoder.action_decoder, ConditionalBidirectionalDecoder)

    def test_cross_conditioning_decoder_uses_latent_as_condition_dimension(
        self,
        lact_decoder_factory: Callable[..., LACT],
    ):
        latent_dimension = 24
        decoder = lact_decoder_factory(latent_dimension=latent_dimension)
        first_layer = decoder.action_decoder.layers[0]
        assert (
            first_layer.self_attention_block.normalization.condition_dim
            == latent_dimension
        )
        assert (
            first_layer.feedforward_block.normalization.condition_dim
            == latent_dimension
        )

    def test_excludes_latent_from_tokenization(
        self,
        lact_decoder_factory: Callable[..., LACT],
    ):
        decoder = lact_decoder_factory()
        assert (
            LatentKey.POSTERIOR_LATENT.value
            in decoder.input_sequence_builder.exclude_keys
        )

    def test_decoder_input_specification(
        self,
        lact_decoder_factory: Callable[..., LACT],
    ):
        decoder = lact_decoder_factory()
        assert FeatureType.SPATIAL.value in decoder.decoder_input.required_types
        assert decoder.decoder_input.requires_actions is False
        assert (
            decoder.decoder_input.conditioning_key == LatentKey.POSTERIOR_LATENT.value
        )
        assert (
            LatentKey.POSTERIOR_LATENT.value
            in decoder.decoder_input.conditioning_required
        )


class TestLACTForward:
    def test_raises_without_latent_in_features(
        self,
        lact_decoder_factory: Callable[..., LACT],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = lact_decoder_factory()
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=SPATIAL_CHANNELS,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"LACT requires '{LatentKey.POSTERIOR_LATENT.value}' in features. "
                f"Make sure to use a variational algorithm that provides latent embeddings. "
                f"Available features: {list(features.keys())}"
            ),
        ):
            decoder(features=features)

    def test_output_keys_match_action_heads(
        self,
        lact_decoder_factory: Callable[..., LACT],
        spatial_features_with_latent_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = lact_decoder_factory()
        features = spatial_features_with_latent_factory()
        output = decoder(features=features)
        assert set(output.keys()) == set(decoder.action_heads.keys())

    def test_output_shape(
        self,
        lact_decoder_factory: Callable[..., LACT],
        spatial_features_with_latent_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = lact_decoder_factory()
        features = spatial_features_with_latent_factory()
        output = decoder(features=features)
        for action_key in decoder.action_heads:
            expected_dim = decoder.action_heads[action_key].output_dim
            assert output[action_key].shape == (
                BATCH_SIZE,
                PREDICTION_HORIZON,
                expected_dim,
            )

    @pytest.mark.parametrize(
        "latent_shape, expected_message",
        [
            (
                (BATCH_SIZE, LATENT_DIMENSION, 1),
                f"LACT latent '{LatentKey.POSTERIOR_LATENT.value}' must have "
                "shape (B, latent_dimension), got "
                f"torch.Size([{BATCH_SIZE}, {LATENT_DIMENSION}, 1]).",
            ),
            (
                (BATCH_SIZE + 1, LATENT_DIMENSION),
                "LACT latent batch size must match observation batch size "
                f"{BATCH_SIZE}, got {BATCH_SIZE + 1}.",
            ),
            (
                (BATCH_SIZE, LATENT_DIMENSION + 1),
                f"LACT latent dimension must be {LATENT_DIMENSION}, "
                f"got {LATENT_DIMENSION + 1}.",
            ),
        ],
    )
    def test_raises_for_invalid_latent_shape(
        self,
        lact_decoder_factory: Callable[..., LACT],
        spatial_features_with_latent_factory: Callable[..., dict[str, torch.Tensor]],
        latent_shape: tuple[int, ...],
        expected_message: str,
    ):
        decoder = lact_decoder_factory()
        features = spatial_features_with_latent_factory()
        features[LatentKey.POSTERIOR_LATENT.value] = torch.zeros(latent_shape)
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            decoder(features=features)

    def test_with_multiple_action_heads(
        self,
        lact_decoder_factory: Callable[..., LACT],
        spatial_features_with_latent_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        position_dim = 3
        orientation_dim = 4
        gripper_dim = 1
        decoder = lact_decoder_factory(
            position_dim=position_dim,
            has_orientation=True,
            orientation_dim=orientation_dim,
            has_gripper=True,
            gripper_dim=gripper_dim,
        )
        features = spatial_features_with_latent_factory()
        output = decoder(features=features)
        assert set(output.keys()) == {
            "position_action",
            "orientation_action",
            "gripper_action",
        }
        assert output["position_action"].shape == (
            BATCH_SIZE,
            PREDICTION_HORIZON,
            position_dim,
        )
        assert output["orientation_action"].shape == (
            BATCH_SIZE,
            PREDICTION_HORIZON,
            orientation_dim,
        )
        assert output["gripper_action"].shape == (
            BATCH_SIZE,
            PREDICTION_HORIZON,
            gripper_dim,
        )

    def test_adaln_zero_init_makes_output_latent_independent(
        self,
        lact_decoder_factory: Callable[..., LACT],
        spatial_features_with_latent_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        # LACT uses ConditionalBidirectionalDecoder with AdaLN-Zero modulation.
        # At initialization, scale=0 and shift=0 so the latent has no effect
        # on the output — this ensures stable early training.
        decoder = lact_decoder_factory()
        decoder.eval()
        features_latent_a = spatial_features_with_latent_factory()
        features_latent_b = {
            key: tensor.clone() for key, tensor in features_latent_a.items()
        }
        features_latent_b[LatentKey.POSTERIOR_LATENT.value] = torch.zeros_like(
            features_latent_b[LatentKey.POSTERIOR_LATENT.value]
        )
        with torch.no_grad():
            output_a = decoder(features=features_latent_a)
            output_b = decoder(features=features_latent_b)
        for action_key in decoder.action_heads:
            torch.testing.assert_close(output_a[action_key], output_b[action_key])
