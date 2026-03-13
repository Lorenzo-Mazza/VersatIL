"""Tests for versatil.models.decoding.decoders.factory.diffusion_action_transformer module."""
import re
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
import torch

from versatil.models.decoding.action_heads.single_output import ActionHead
from versatil.models.decoding.constants import DecoderOutputKey, DiTType
from versatil.models.decoding.decoders.base import ActionDecoder
from versatil.models.decoding.decoders.factory.diffusion_action_transformer import (
    DiffusionActionTransformer,
)
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType, PositionalEncodingType
from versatil.models.layers.diffusion_transformer import (
    CrossAttentionDiT,
    MMDiTTransformer,
)
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.positional_encoding.learned import (
    LearnedPositionalEncoding1D,
)
from versatil.models.layers import MLP


EMBEDDING_DIMENSION = 32
NUMBER_OF_HEADS = 2
NUMBER_OF_LAYERS = 1
FEEDFORWARD_DIMENSION = 64
TIMESTEP_EMBEDDING_DIMENSION = 16
MAX_SEQUENCE_LENGTH = 64
PREDICTION_HORIZON = 4
OBSERVATION_HORIZON = 1
BATCH_SIZE = 2
SPATIAL_HEIGHT = 4
SPATIAL_WIDTH = 4
POSITION_DIM = 3


@pytest.fixture
def diffusion_transformer_factory(
    mock_action_space_factory: Callable[..., MagicMock],
    mock_observation_space_factory: Callable[..., MagicMock],
    action_head_factory: Callable[..., ActionHead],
) -> Callable[..., DiffusionActionTransformer]:
    """Factory for DiffusionActionTransformer with small dimensions."""

    def factory(
        embedding_dimension: int = EMBEDDING_DIMENSION,
        number_of_layers: int = NUMBER_OF_LAYERS,
        number_of_heads: int = NUMBER_OF_HEADS,
        feedforward_dimension: int = FEEDFORWARD_DIMENSION,
        timestep_embedding_dimension: int = TIMESTEP_EMBEDDING_DIMENSION,
        max_sequence_length: int = MAX_SEQUENCE_LENGTH,
        prediction_horizon: int = PREDICTION_HORIZON,
        observation_horizon: int = OBSERVATION_HORIZON,
        position_dim: int = POSITION_DIM,
        has_orientation: bool = False,
        orientation_dim: int = 0,
        has_gripper: bool = False,
        gripper_dim: int = 0,
        diffusion_transformer_type: str = DiTType.CROSS_ATTENTION.value,
        activation: str = ActivationFunction.GELU.value,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        attention_type: str = AttentionType.MULTI_HEAD.value,
        positional_encoding_type: str = PositionalEncodingType.ROPE.value,
        dropout_rate: float = 0.0,
        attention_dropout: float = 0.0,
        use_gating: bool = True,
        input_keys: list[str] | None = None,
        action_heads: dict[str, ActionHead] | None = None,
    ) -> DiffusionActionTransformer:
        if input_keys is None:
            input_keys = ["rgb_features"]
        action_space = mock_action_space_factory(
            position_dim=position_dim,
            has_orientation=has_orientation,
            orientation_dim=orientation_dim,
            has_gripper=has_gripper,
            gripper_dim=gripper_dim,
        )
        if action_heads is None:
            action_heads = {
                key: action_head_factory(input_dim=embedding_dimension)
                for key in action_space.actions_metadata
            }
        observation_space = mock_observation_space_factory()
        return DiffusionActionTransformer(
            input_keys=input_keys,
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device="cpu",
            diffusion_transformer_type=diffusion_transformer_type,
            max_sequence_length=max_sequence_length,
            embedding_dimension=embedding_dimension,
            timestep_embedding_dimension=timestep_embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_layers=number_of_layers,
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



class TestDiffusionActionTransformerInitialization:

    def test_inherits_from_action_decoder(
        self,
        diffusion_transformer_factory: Callable[..., DiffusionActionTransformer],
    ):
        decoder = diffusion_transformer_factory()
        assert isinstance(decoder, ActionDecoder)

    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    @pytest.mark.parametrize("number_of_layers", [1, 2])
    @pytest.mark.parametrize("use_gating", [True, False])
    @pytest.mark.parametrize("diffusion_transformer_type", [DiTType.CROSS_ATTENTION.value, DiTType.MMDIT.value])
    def test_stores_configuration(
        self,
        diffusion_transformer_factory: Callable[..., DiffusionActionTransformer],
        embedding_dimension: int,
        number_of_layers: int,
        use_gating: bool,
        diffusion_transformer_type: str,
    ):
        decoder = diffusion_transformer_factory(
            embedding_dimension=embedding_dimension,
            number_of_layers=number_of_layers,
            number_of_heads=NUMBER_OF_HEADS,
            feedforward_dimension=FEEDFORWARD_DIMENSION,
            timestep_embedding_dimension=TIMESTEP_EMBEDDING_DIMENSION,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            activation=ActivationFunction.GELU.value,
            normalization_type=NormalizationType.RMS_NORM.value,
            attention_type=AttentionType.MULTI_HEAD.value,
            dropout_rate=0.05,
            attention_dropout=0.02,
            positional_encoding_type=PositionalEncodingType.ROPE.value,
            use_gating=use_gating,
            diffusion_transformer_type=diffusion_transformer_type,
        )
        assert decoder.embedding_dimension == embedding_dimension
        assert decoder.number_of_layers == number_of_layers
        assert decoder.number_of_heads == NUMBER_OF_HEADS
        assert decoder.feedforward_dimension == FEEDFORWARD_DIMENSION
        assert decoder.timestep_embedding_dimension == TIMESTEP_EMBEDDING_DIMENSION
        assert decoder.max_sequence_length == MAX_SEQUENCE_LENGTH
        assert decoder.activation == ActivationFunction.GELU.value
        assert decoder.normalization_type == NormalizationType.RMS_NORM.value
        assert decoder.attention_type == AttentionType.MULTI_HEAD.value
        assert decoder.dropout_rate == 0.05
        assert decoder.attention_dropout == 0.02
        assert decoder.positional_encoding_type == PositionalEncodingType.ROPE.value
        assert decoder.use_gating is use_gating
        assert decoder.diffusion_transformer_type == diffusion_transformer_type

    def test_creates_cross_attention_dit(
        self,
        diffusion_transformer_factory: Callable[..., DiffusionActionTransformer],
    ):
        decoder = diffusion_transformer_factory(
            diffusion_transformer_type=DiTType.CROSS_ATTENTION.value,
        )
        assert isinstance(decoder.transformer, CrossAttentionDiT)

    def test_creates_mmdit(
        self,
        diffusion_transformer_factory: Callable[..., DiffusionActionTransformer],
    ):
        decoder = diffusion_transformer_factory(
            diffusion_transformer_type=DiTType.MMDIT.value,
        )
        assert isinstance(decoder.transformer, MMDiTTransformer)

    def test_invalid_dit_type_raises(
        self,
        diffusion_transformer_factory: Callable[..., DiffusionActionTransformer],
    ):
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Unsupported diffusion_transformer_type: invalid_type. "
                f"Supported types: {[DiTType.CROSS_ATTENTION.value, DiTType.MMDIT.value]}. "
                f"Use DiTBlockActionTransformer for type {DiTType.DIT_BLOCK.value}."
            ),
        ):
            diffusion_transformer_factory(
                diffusion_transformer_type="invalid_type",
            )

    def test_decoder_input_requires_actions(
        self,
        diffusion_transformer_factory: Callable[..., DiffusionActionTransformer],
    ):
        decoder = diffusion_transformer_factory()
        assert decoder.decoder_input.requires_actions is True

    def test_creates_noisy_input_projection(
        self,
        diffusion_transformer_factory: Callable[..., DiffusionActionTransformer],
    ):
        decoder = diffusion_transformer_factory()
        assert hasattr(decoder, "noisy_input_projection")
        assert isinstance(decoder.noisy_input_projection, MLP)

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
        decoder = DiffusionActionTransformer(
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
            number_of_layers=NUMBER_OF_LAYERS,
            feedforward_dimension=FEEDFORWARD_DIMENSION,
            activation=ActivationFunction.GELU.value,
        )
        for action_key in decoder.action_heads:
            assert len(decoder.action_heads[action_key].blocks) == 0


class TestDiffusionActionTransformerForward:

    def test_raises_without_actions(
        self,
        diffusion_transformer_factory: Callable[..., DiffusionActionTransformer],
        spatial_features_with_timestep_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = diffusion_transformer_factory()
        features = spatial_features_with_timestep_factory(
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                "DiffusionActionTransformer requires 'actions' parameter. "
                "The algorithm should provide noisy actions during forward pass."
            ),
        ):
            decoder(features=features, actions=None)

    def test_raises_without_timestep(
        self,
        diffusion_transformer_factory: Callable[..., DiffusionActionTransformer],
        spatial_features_with_timestep_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = diffusion_transformer_factory()
        features = spatial_features_with_timestep_factory(
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
            include_timestep=False,
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
        diffusion_transformer_factory: Callable[..., DiffusionActionTransformer],
        spatial_features_with_timestep_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = diffusion_transformer_factory()
        features = spatial_features_with_timestep_factory(
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        actions = noisy_actions_factory()
        outputs = decoder(features=features, actions=actions)
        assert set(outputs.keys()) == set(actions.keys())

    @pytest.mark.parametrize("prediction_horizon", [4, 8])
    def test_output_shape(
        self,
        diffusion_transformer_factory: Callable[..., DiffusionActionTransformer],
        spatial_features_with_timestep_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
        prediction_horizon: int,
    ):
        decoder = diffusion_transformer_factory(
            prediction_horizon=prediction_horizon,
        )
        features = spatial_features_with_timestep_factory(
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        actions = noisy_actions_factory(prediction_horizon=prediction_horizon)
        outputs = decoder(features=features, actions=actions)
        for action_key, output_tensor in outputs.items():
            assert output_tensor.shape == (
                BATCH_SIZE,
                prediction_horizon,
                decoder.action_heads[action_key].output_dim,
            )

    def test_timestep_squeeze_from_two_dimensions(
        self,
        diffusion_transformer_factory: Callable[..., DiffusionActionTransformer],
        spatial_features_with_timestep_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = diffusion_transformer_factory()
        features = spatial_features_with_timestep_factory(
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        features[DecoderOutputKey.TIMESTEP.value] = features[
            DecoderOutputKey.TIMESTEP.value
        ].unsqueeze(-1)
        actions = noisy_actions_factory()
        outputs = decoder(features=features, actions=actions)
        assert all(
            tensor.shape[0] == BATCH_SIZE for tensor in outputs.values()
        )

    def test_with_multiple_action_heads(
        self,
        diffusion_transformer_factory: Callable[..., DiffusionActionTransformer],
        spatial_features_with_timestep_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        position_dim = 3
        orientation_dim = 4
        gripper_dim = 1
        decoder = diffusion_transformer_factory(
            position_dim=position_dim,
            has_orientation=True,
            orientation_dim=orientation_dim,
            has_gripper=True,
            gripper_dim=gripper_dim,
        )
        features = spatial_features_with_timestep_factory(
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        actions = noisy_actions_factory(
            action_keys_to_dims={
                "gripper_action": gripper_dim,
                "orientation_action": orientation_dim,
                "position_action": position_dim,
            },
        )
        outputs = decoder(features=features, actions=actions)
        assert "position_action" in outputs
        assert "orientation_action" in outputs
        assert "gripper_action" in outputs
        assert outputs["position_action"].shape == (BATCH_SIZE, PREDICTION_HORIZON, 3)
        assert outputs["orientation_action"].shape == (BATCH_SIZE, PREDICTION_HORIZON, 4)
        assert outputs["gripper_action"].shape == (BATCH_SIZE, PREDICTION_HORIZON, 1)


class TestDiffusionActionTransformerTemporal:

    def test_observation_horizon_greater_than_one_creates_temporal_pe(
        self,
        diffusion_transformer_factory: Callable[..., DiffusionActionTransformer],
    ):
        decoder = diffusion_transformer_factory(observation_horizon=3)
        assert decoder.input_builder.temporal_positional_encoding_layer is not None
        assert isinstance(
            decoder.input_builder.temporal_positional_encoding_layer,
            LearnedPositionalEncoding1D,
        )

    def test_observation_horizon_equal_to_one_has_no_temporal_pe(
        self,
        diffusion_transformer_factory: Callable[..., DiffusionActionTransformer],
    ):
        decoder = diffusion_transformer_factory(observation_horizon=1)
        assert decoder.input_builder.temporal_positional_encoding_layer is None
