"""Tests for versatil.models.decoding.decoders.factory.conditional_action_unet module."""

import re
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
import torch

from versatil.models.decoding.action_heads.single_output import ActionHead
from versatil.models.decoding.constants import DecoderOutputKey
from versatil.models.decoding.decoders.base import ActionDecoder
from versatil.models.decoding.decoders.factory.conditional_action_unet import (
    ConditionalActionUNet,
)
from versatil.models.feature_meta import FeatureType

EMBEDDING_DIMENSION = 32
DOWN_DIMENSIONS = [32, 64]
NUM_GROUPS = 4
KERNEL_SIZE = 3
PREDICTION_HORIZON = 4
OBSERVATION_HORIZON = 1
BATCH_SIZE = 2
FEATURE_DIMENSION = 48
POSITION_DIM = 3


@pytest.fixture
def unet_decoder_factory(
    mock_action_space_factory: Callable[..., MagicMock],
    mock_observation_space_factory: Callable[..., MagicMock],
    action_heads_factory: Callable[..., dict[str, ActionHead]],
) -> Callable[..., ConditionalActionUNet]:
    """Factory for ConditionalActionUNet instances with small dimensions."""

    def factory(
        input_keys: list[str] | None = None,
        position_dim: int = POSITION_DIM,
        has_orientation: bool = False,
        orientation_dim: int = 0,
        has_gripper: bool = False,
        gripper_dim: int = 0,
        embedding_dimension: int = EMBEDDING_DIMENSION,
        down_dimensions: list[int] | None = None,
        kernel_size: int = KERNEL_SIZE,
        num_groups: int = NUM_GROUPS,
        use_local_conditioning: bool = False,
        condition_predict_scale: bool = False,
        observation_horizon: int = OBSERVATION_HORIZON,
        prediction_horizon: int = PREDICTION_HORIZON,
    ) -> ConditionalActionUNet:
        if input_keys is None:
            input_keys = ["rgb_features"]
        if down_dimensions is None:
            down_dimensions = list(DOWN_DIMENSIONS)
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
        return ConditionalActionUNet(
            input_keys=input_keys,
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device="cpu",
            embedding_dimension=embedding_dimension,
            down_dimensions=down_dimensions,
            kernel_size=kernel_size,
            num_groups=num_groups,
            use_local_conditioning=use_local_conditioning,
            condition_predict_scale=condition_predict_scale,
        )

    return factory


class TestConditionalActionUNetInitialization:
    def test_inherits_from_action_decoder(
        self,
        unet_decoder_factory: Callable[..., ConditionalActionUNet],
    ):
        decoder = unet_decoder_factory()
        assert isinstance(decoder, ActionDecoder)

    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    @pytest.mark.parametrize("down_dimensions", [[32, 64], [16, 32, 64]])
    @pytest.mark.parametrize("kernel_size", [3, 5])
    @pytest.mark.parametrize("condition_predict_scale", [True, False])
    def test_stores_configuration(
        self,
        unet_decoder_factory: Callable[..., ConditionalActionUNet],
        embedding_dimension: int,
        down_dimensions: list[int],
        kernel_size: int,
        condition_predict_scale: bool,
    ):
        decoder = unet_decoder_factory(
            embedding_dimension=embedding_dimension,
            down_dimensions=down_dimensions,
            kernel_size=kernel_size,
            condition_predict_scale=condition_predict_scale,
            use_local_conditioning=False,
        )
        assert decoder.embedding_dimension == embedding_dimension
        assert decoder.down_dimensions == down_dimensions
        assert decoder.kernel_size == kernel_size
        assert decoder.condition_predict_scale is condition_predict_scale
        assert decoder.use_local_conditioning is False

    def test_unet_not_initialized_before_forward(
        self,
        unet_decoder_factory: Callable[..., ConditionalActionUNet],
    ):
        decoder = unet_decoder_factory()
        assert decoder._unet is None

    def test_device_tracker_is_persisted(
        self,
        unet_decoder_factory: Callable[..., ConditionalActionUNet],
    ):
        decoder = unet_decoder_factory()
        assert "_device_tracker" in decoder.state_dict()

    def test_local_conditioning_raises_not_implemented(
        self,
        unet_decoder_factory: Callable[..., ConditionalActionUNet],
    ):
        with pytest.raises(
            NotImplementedError,
            match=re.escape(
                "Local conditioning is not yet implemented. "
                "Use global conditioning (obs_as_global_cond=True) for now."
            ),
        ):
            unet_decoder_factory(use_local_conditioning=True)

    def test_decoder_input_rejects_spatial_features(
        self,
        unet_decoder_factory: Callable[..., ConditionalActionUNet],
    ):
        decoder = unet_decoder_factory()
        assert FeatureType.SPATIAL.value in decoder.decoder_input.raises_for_types

    def test_decoder_input_requires_actions(
        self,
        unet_decoder_factory: Callable[..., ConditionalActionUNet],
    ):
        decoder = unet_decoder_factory()
        assert decoder.decoder_input.requires_actions is True

    def test_action_heads_blocks_cleared_when_present(
        self,
        mock_action_space_factory: Callable[..., MagicMock],
        mock_observation_space_factory: Callable[..., MagicMock],
    ):
        action_space = mock_action_space_factory(position_dim=POSITION_DIM)
        observation_space = mock_observation_space_factory()
        head_with_blocks = ActionHead(input_dim=EMBEDDING_DIMENSION)
        head_with_blocks.blocks = torch.nn.ModuleList(
            [torch.nn.Linear(EMBEDDING_DIMENSION, EMBEDDING_DIMENSION)]
        )
        action_heads = {"position_action": head_with_blocks}
        decoder = ConditionalActionUNet(
            input_keys=["rgb_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=OBSERVATION_HORIZON,
            prediction_horizon=PREDICTION_HORIZON,
            device="cpu",
            embedding_dimension=EMBEDDING_DIMENSION,
            down_dimensions=list(DOWN_DIMENSIONS),
            kernel_size=KERNEL_SIZE,
            num_groups=NUM_GROUPS,
        )
        assert len(decoder.action_heads["position_action"].blocks) == 0


class TestConditionalActionUNetForward:
    def test_raises_without_actions(
        self,
        unet_decoder_factory: Callable[..., ConditionalActionUNet],
        flat_features_with_timestep_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = unet_decoder_factory()
        features = flat_features_with_timestep_factory(feature_dim=FEATURE_DIMENSION)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "ConditionalActionUNet requires 'actions' parameter. "
                "The algorithm should provide noisy actions during forward pass."
            ),
        ):
            decoder(features=features, actions=None)

    def test_raises_without_timestep_in_features(
        self,
        unet_decoder_factory: Callable[..., ConditionalActionUNet],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = unet_decoder_factory()
        features = flat_feature_factory(
            batch_size=BATCH_SIZE,
            feature_dim=FEATURE_DIMENSION,
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

    def test_lazy_initializes_unet_on_first_forward(
        self,
        unet_decoder_factory: Callable[..., ConditionalActionUNet],
        flat_features_with_timestep_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = unet_decoder_factory()
        assert decoder._unet is None
        features = flat_features_with_timestep_factory(feature_dim=FEATURE_DIMENSION)
        actions = noisy_actions_factory()
        decoder(features=features, actions=actions)
        assert decoder._unet is not None

    def test_load_state_dict_initializes_lazy_unet_from_checkpoint(
        self,
        unet_decoder_factory: Callable[..., ConditionalActionUNet],
        flat_features_with_timestep_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        source_decoder = unet_decoder_factory()
        source_decoder.eval()
        features = flat_features_with_timestep_factory(feature_dim=FEATURE_DIMENSION)
        actions = noisy_actions_factory()
        with torch.no_grad():
            source_output = source_decoder(features=features, actions=actions)

        target_decoder = unet_decoder_factory()
        target_decoder.eval()
        assert target_decoder._unet is None
        target_decoder.load_state_dict(source_decoder.state_dict())
        with torch.no_grad():
            target_output = target_decoder(features=features, actions=actions)

        for action_key in source_output:
            torch.testing.assert_close(
                target_output[action_key], source_output[action_key]
            )

    def test_output_keys_match_action_heads(
        self,
        unet_decoder_factory: Callable[..., ConditionalActionUNet],
        flat_features_with_timestep_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = unet_decoder_factory()
        features = flat_features_with_timestep_factory(feature_dim=FEATURE_DIMENSION)
        actions = noisy_actions_factory()
        output = decoder(features=features, actions=actions)
        assert set(output.keys()) == set(actions.keys())

    def test_output_shape(
        self,
        unet_decoder_factory: Callable[..., ConditionalActionUNet],
        flat_features_with_timestep_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = unet_decoder_factory()
        features = flat_features_with_timestep_factory(feature_dim=FEATURE_DIMENSION)
        actions = noisy_actions_factory()
        output = decoder(features=features, actions=actions)
        for action_key, action_tensor in actions.items():
            assert output[action_key].shape == action_tensor.shape

    def test_timestep_2d_squeezed_to_1d(
        self,
        unet_decoder_factory: Callable[..., ConditionalActionUNet],
        flat_features_with_timestep_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = unet_decoder_factory()
        features = flat_features_with_timestep_factory(
            feature_dim=FEATURE_DIMENSION,
            timestep_shape=(BATCH_SIZE, 1),
        )
        actions = noisy_actions_factory()
        output = decoder(features=features, actions=actions)
        for action_key, action_tensor in actions.items():
            assert output[action_key].shape == action_tensor.shape

    def test_forward_does_not_mutate_features(
        self,
        unet_decoder_factory: Callable[..., ConditionalActionUNet],
        flat_features_with_timestep_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = unet_decoder_factory()
        decoder.eval()
        features = flat_features_with_timestep_factory(feature_dim=FEATURE_DIMENSION)
        timestep = features[DecoderOutputKey.TIMESTEP.value]
        actions = noisy_actions_factory()
        with torch.no_grad():
            decoder(features=features, actions=actions)
            decoder(features=features, actions=actions)
        assert features[DecoderOutputKey.TIMESTEP.value] is timestep

    def test_with_multiple_action_heads(
        self,
        unet_decoder_factory: Callable[..., ConditionalActionUNet],
        flat_features_with_timestep_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        position_dim = 3
        orientation_dim = 4
        gripper_dim = 1
        decoder = unet_decoder_factory(
            position_dim=position_dim,
            has_orientation=True,
            orientation_dim=orientation_dim,
            has_gripper=True,
            gripper_dim=gripper_dim,
        )
        features = flat_features_with_timestep_factory(feature_dim=FEATURE_DIMENSION)
        actions = noisy_actions_factory(
            action_keys_to_dims={
                "gripper_action": gripper_dim,
                "orientation_action": orientation_dim,
                "position_action": position_dim,
            },
        )
        output = decoder(features=features, actions=actions)
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

    def test_zero_init_modulation_makes_output_timestep_independent(
        self,
        unet_decoder_factory: Callable[..., ConditionalActionUNet],
        flat_features_with_timestep_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        # FiLM modulation layers are zero-initialized, so the conditioning bias
        # is zero and the output is timestep-independent at initialization.
        decoder = unet_decoder_factory()
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
