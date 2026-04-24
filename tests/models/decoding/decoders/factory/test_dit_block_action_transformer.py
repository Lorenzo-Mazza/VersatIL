"""Tests for versatil.models.decoding.decoders.factory.dit_block_action_transformer module."""

import re
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
import torch

from versatil.models.decoding.action_heads.single_output import ActionHead
from versatil.models.decoding.constants import DecoderOutputKey
from versatil.models.decoding.decoders.base import ActionDecoder
from versatil.models.decoding.decoders.factory.dit_block_action_transformer import (
    DiTBlockActionTransformer,
)
from versatil.models.feature_meta import FeatureType
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType, PositionalEncodingType
from versatil.models.layers.normalization.constants import NormalizationType

EMBEDDING_DIMENSION = 32
NUMBER_OF_HEADS = 2
NUMBER_OF_ENCODER_LAYERS = 1
NUMBER_OF_DECODER_LAYERS = 1
FEEDFORWARD_DIMENSION = 64
TIMESTEP_EMBEDDING_DIMENSION = 16
MAX_SEQUENCE_LENGTH = 64
PREDICTION_HORIZON = 4
OBSERVATION_HORIZON = 1
BATCH_SIZE = 2
FEATURE_DIMENSION = 32
POSITION_DIM = 3


@pytest.fixture
def dit_decoder_factory(
    mock_action_space_factory: Callable[..., MagicMock],
    mock_observation_space_factory: Callable[..., MagicMock],
    action_head_factory: Callable[..., ActionHead],
) -> Callable[..., DiTBlockActionTransformer]:
    """Factory for DiTBlockActionTransformer with small dimensions."""

    def factory(
        embedding_dimension: int = EMBEDDING_DIMENSION,
        number_of_encoder_layers: int = NUMBER_OF_ENCODER_LAYERS,
        number_of_decoder_layers: int = NUMBER_OF_DECODER_LAYERS,
        number_of_heads: int = NUMBER_OF_HEADS,
        feedforward_dimension: int = FEEDFORWARD_DIMENSION,
        timestep_embedding_dimension: int = TIMESTEP_EMBEDDING_DIMENSION,
        max_sequence_length: int = MAX_SEQUENCE_LENGTH,
        prediction_horizon: int = PREDICTION_HORIZON,
        observation_horizon: int = OBSERVATION_HORIZON,
        position_dim: int = POSITION_DIM,
        activation: str = ActivationFunction.GELU.value,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        attention_type: str = AttentionType.MULTI_HEAD.value,
        positional_encoding_type: str = PositionalEncodingType.ROPE.value,
        dropout_rate: float = 0.0,
        attention_dropout: float = 0.0,
        use_gating: bool = True,
        input_keys: list[str] | None = None,
    ) -> DiTBlockActionTransformer:
        if input_keys is None:
            input_keys = ["rgb_features"]
        action_space = mock_action_space_factory(position_dim=position_dim)
        observation_space = mock_observation_space_factory()
        action_heads = {
            key: action_head_factory(input_dim=embedding_dimension)
            for key in action_space.actions_metadata
        }
        return DiTBlockActionTransformer(
            input_keys=input_keys,
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device="cpu",
            max_sequence_length=max_sequence_length,
            embedding_dimension=embedding_dimension,
            timestep_embedding_dimension=timestep_embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_encoder_layers=number_of_encoder_layers,
            number_of_decoder_layers=number_of_decoder_layers,
            feedforward_dimension=feedforward_dimension,
            activation=activation,
            normalization_type=normalization_type,
            attention_type=attention_type,
            dropout_rate=dropout_rate,
            attention_dropout=attention_dropout,
            positional_encoding_type=positional_encoding_type,
            use_gating=use_gating,
        )

    return factory


class TestDiTBlockActionTransformerInitialization:
    def test_inherits_from_action_decoder(
        self,
        dit_decoder_factory: Callable[..., DiTBlockActionTransformer],
    ):
        decoder = dit_decoder_factory()
        assert isinstance(decoder, ActionDecoder)

    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    @pytest.mark.parametrize("number_of_encoder_layers", [1, 2])
    def test_stores_configuration(
        self,
        dit_decoder_factory: Callable[..., DiTBlockActionTransformer],
        embedding_dimension: int,
        number_of_encoder_layers: int,
    ):
        decoder = dit_decoder_factory(
            embedding_dimension=embedding_dimension,
            number_of_encoder_layers=number_of_encoder_layers,
        )
        assert decoder.embedding_dimension == embedding_dimension
        assert decoder.number_of_encoder_layers == number_of_encoder_layers

    def test_creates_dit_block_transformer(
        self,
        dit_decoder_factory: Callable[..., DiTBlockActionTransformer],
    ):
        decoder = dit_decoder_factory()
        assert hasattr(decoder, "transformer")
        assert decoder.transformer is not None

    def test_creates_noisy_input_projection(
        self,
        dit_decoder_factory: Callable[..., DiTBlockActionTransformer],
    ):
        decoder = dit_decoder_factory()
        assert hasattr(decoder, "noisy_input_projection")
        assert decoder.noisy_input_projection is not None

    def test_caching_initially_disabled(
        self,
        dit_decoder_factory: Callable[..., DiTBlockActionTransformer],
    ):
        decoder = dit_decoder_factory()
        assert decoder._caching_enabled is False
        assert decoder._encoder_cache is None

    def test_decoder_input_raises_for_spatial(
        self,
        dit_decoder_factory: Callable[..., DiTBlockActionTransformer],
    ):
        decoder = dit_decoder_factory()
        assert FeatureType.SPATIAL.value in decoder.decoder_input.raises_for_types

    def test_decoder_input_requires_actions(
        self,
        dit_decoder_factory: Callable[..., DiTBlockActionTransformer],
    ):
        decoder = dit_decoder_factory()
        assert decoder.decoder_input.requires_actions is True

    def test_action_heads_blocks_cleared(
        self,
        mock_action_space_factory: Callable[..., MagicMock],
        mock_observation_space_factory: Callable[..., MagicMock],
    ):
        action_space = mock_action_space_factory(position_dim=POSITION_DIM)
        head = ActionHead(input_dim=EMBEDDING_DIMENSION)
        dummy_block = torch.nn.Linear(EMBEDDING_DIMENSION, EMBEDDING_DIMENSION)
        dummy_block.output_dim = EMBEDDING_DIMENSION
        head.blocks = torch.nn.ModuleList([dummy_block])
        action_heads = {"position_action": head}
        decoder = DiTBlockActionTransformer(
            input_keys=["rgb_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=mock_observation_space_factory(),
            observation_horizon=OBSERVATION_HORIZON,
            prediction_horizon=PREDICTION_HORIZON,
            device="cpu",
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
            timestep_embedding_dimension=TIMESTEP_EMBEDDING_DIMENSION,
            number_of_heads=NUMBER_OF_HEADS,
            number_of_encoder_layers=NUMBER_OF_ENCODER_LAYERS,
            number_of_decoder_layers=NUMBER_OF_DECODER_LAYERS,
            feedforward_dimension=FEEDFORWARD_DIMENSION,
            activation=ActivationFunction.GELU.value,
            normalization_type=NormalizationType.RMS_NORM.value,
        )
        for action_key in decoder.action_heads:
            assert len(decoder.action_heads[action_key].blocks) == 0


class TestDiTBlockActionTransformerForward:
    def test_raises_without_actions(
        self,
        dit_decoder_factory: Callable[..., DiTBlockActionTransformer],
        flat_features_with_timestep_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = dit_decoder_factory()
        features = flat_features_with_timestep_factory(feature_dim=FEATURE_DIMENSION)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "DiTBlockActionTransformer requires 'actions' parameter. "
                "The algorithm should provide noisy actions during forward pass."
            ),
        ):
            decoder(features=features, actions=None)

    def test_raises_without_timestep(
        self,
        dit_decoder_factory: Callable[..., DiTBlockActionTransformer],
        flat_features_with_timestep_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = dit_decoder_factory()
        features = flat_features_with_timestep_factory(
            feature_dim=FEATURE_DIMENSION, include_timestep=False
        )
        actions = noisy_actions_factory()
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Missing '{DecoderOutputKey.TIMESTEP.value}' in features dict. "
                "The algorithm should inject timesteps into features."
            ),
        ):
            decoder(features=features, actions=actions)

    def test_output_keys_match_action_heads(
        self,
        dit_decoder_factory: Callable[..., DiTBlockActionTransformer],
        flat_features_with_timestep_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = dit_decoder_factory()
        features = flat_features_with_timestep_factory(feature_dim=FEATURE_DIMENSION)
        actions = noisy_actions_factory()
        outputs = decoder(features=features, actions=actions)
        assert set(outputs.keys()) == set(actions.keys())

    def test_output_shape(
        self,
        dit_decoder_factory: Callable[..., DiTBlockActionTransformer],
        flat_features_with_timestep_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = dit_decoder_factory()
        features = flat_features_with_timestep_factory(feature_dim=FEATURE_DIMENSION)
        actions = noisy_actions_factory()
        outputs = decoder(features=features, actions=actions)
        for action_key, output_tensor in outputs.items():
            assert output_tensor.shape == (
                BATCH_SIZE,
                PREDICTION_HORIZON,
                decoder.action_heads[action_key].output_dim,
            )

    def test_timestep_squeeze_from_two_dimensions(
        self,
        dit_decoder_factory: Callable[..., DiTBlockActionTransformer],
        flat_features_with_timestep_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = dit_decoder_factory()
        features = flat_features_with_timestep_factory(feature_dim=FEATURE_DIMENSION)
        features[DecoderOutputKey.TIMESTEP.value] = features[
            DecoderOutputKey.TIMESTEP.value
        ].unsqueeze(-1)
        actions = noisy_actions_factory()
        outputs = decoder(features=features, actions=actions)
        assert all(tensor.shape[0] == BATCH_SIZE for tensor in outputs.values())

    def test_forward_does_not_mutate_features(
        self,
        dit_decoder_factory: Callable[..., DiTBlockActionTransformer],
        flat_features_with_timestep_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = dit_decoder_factory()
        decoder.eval()
        features = flat_features_with_timestep_factory(feature_dim=FEATURE_DIMENSION)
        timestep = features[DecoderOutputKey.TIMESTEP.value]
        actions = noisy_actions_factory()
        with torch.no_grad():
            decoder(features=features, actions=actions)
            decoder(features=features, actions=actions)
        assert features[DecoderOutputKey.TIMESTEP.value] is timestep

    def test_adaln_zero_init_makes_output_timestep_independent(
        self,
        dit_decoder_factory: Callable[..., DiTBlockActionTransformer],
        flat_features_with_timestep_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        # AdaLN-Zero initializes modulation scale=0, shift=0 and the final output
        # linear to zeros. At init, the network output must be identical regardless
        # of timestep — this is the DiT design that ensures stable early training.
        decoder = dit_decoder_factory()
        decoder.eval()
        features_t0 = flat_features_with_timestep_factory(feature_dim=FEATURE_DIMENSION)
        features_t0[DecoderOutputKey.TIMESTEP.value] = torch.zeros(
            BATCH_SIZE, dtype=torch.long
        )
        features_t99 = {key: tensor.clone() for key, tensor in features_t0.items()}
        features_t99[DecoderOutputKey.TIMESTEP.value] = torch.full(
            (BATCH_SIZE,), 99, dtype=torch.long
        )
        actions = noisy_actions_factory()
        with torch.no_grad():
            output_t0 = decoder(features=features_t0, actions=actions)
            output_t99 = decoder(features=features_t99, actions=actions)
        for action_key in actions:
            torch.testing.assert_close(output_t0[action_key], output_t99[action_key])


class TestDiTBlockActionTransformerCaching:
    def test_enable_cache(
        self,
        dit_decoder_factory: Callable[..., DiTBlockActionTransformer],
    ):
        decoder = dit_decoder_factory()
        decoder.enable_encoder_cache()
        assert decoder._caching_enabled is True
        assert decoder._encoder_cache is None

    def test_disable_cache(
        self,
        dit_decoder_factory: Callable[..., DiTBlockActionTransformer],
    ):
        decoder = dit_decoder_factory()
        decoder.enable_encoder_cache()
        decoder.disable_encoder_cache()
        assert decoder._caching_enabled is False
        assert decoder._encoder_cache is None

    def test_cache_stores_encoder_output(
        self,
        dit_decoder_factory: Callable[..., DiTBlockActionTransformer],
        flat_features_with_timestep_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = dit_decoder_factory()
        decoder.enable_encoder_cache()
        features = flat_features_with_timestep_factory(feature_dim=FEATURE_DIMENSION)
        actions = noisy_actions_factory()
        decoder(features=features, actions=actions)
        assert decoder._encoder_cache is not None

    def test_cache_reused_on_second_forward(
        self,
        dit_decoder_factory: Callable[..., DiTBlockActionTransformer],
        flat_features_with_timestep_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = dit_decoder_factory()
        decoder.enable_encoder_cache()
        features_first = flat_features_with_timestep_factory(
            feature_dim=FEATURE_DIMENSION
        )
        actions = noisy_actions_factory()
        decoder(features=features_first, actions=actions)
        first_cache = decoder._encoder_cache
        # Second forward with new features but same cache
        features_second = flat_features_with_timestep_factory(
            feature_dim=FEATURE_DIMENSION
        )
        decoder(features=features_second, actions=actions)
        second_cache = decoder._encoder_cache
        # Cache object from first forward should be reused, not recomputed
        assert first_cache is second_cache

    def test_cached_forward_produces_identical_output_to_uncached(
        self,
        dit_decoder_factory: Callable[..., DiTBlockActionTransformer],
        flat_features_with_timestep_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = dit_decoder_factory()
        decoder.eval()
        actions = noisy_actions_factory()
        base_features = flat_features_with_timestep_factory(
            feature_dim=FEATURE_DIMENSION
        )
        features_uncached = {k: v.clone() for k, v in base_features.items()}
        with torch.no_grad():
            output_uncached = decoder(features=features_uncached, actions=actions)
        decoder.enable_encoder_cache()
        features_cached_first = {k: v.clone() for k, v in base_features.items()}
        with torch.no_grad():
            output_first_cached = decoder(
                features=features_cached_first, actions=actions
            )
        features_cached_second = {k: v.clone() for k, v in base_features.items()}
        with torch.no_grad():
            output_second_cached = decoder(
                features=features_cached_second, actions=actions
            )
        for action_key in actions:
            torch.testing.assert_close(
                output_uncached[action_key],
                output_first_cached[action_key],
            )
            torch.testing.assert_close(
                output_first_cached[action_key],
                output_second_cached[action_key],
            )
