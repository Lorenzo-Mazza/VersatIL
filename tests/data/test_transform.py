"""Tests for data transformation and normalization functions."""
import pytest
import torch
from unittest.mock import MagicMock
from versatil.data.transform import (
    normalize_sample,
    normalize_observation,
    normalize_actions,
    unnormalize_actions,
    tokenize_sample,
    tokenize_observation,
    tokenize_actions,
    detokenize_actions,
)
from versatil.data.constants import (
    Cameras,
    GripperType,
    ObsKey,
    ProprioceptiveType,
    ProprioKey,
    SampleKey,
)
from versatil.data.normalization.normalizer import LinearNormalizer


@pytest.fixture
def mock_observation_space():
    """Factory for creating mock observation spaces."""
    def factory(
        camera_keys=None,
        use_proprio_base_frame=False,
        use_proprio_camera_frame=False,
        use_language=False,
        use_gripper_state=False,
        gripper_type=GripperType.BINARY.value,
    ):
        obs_space = MagicMock()
        obs_space.camera_keys = camera_keys if camera_keys is not None else []
        obs_space.use_proprio_base_frame = use_proprio_base_frame
        obs_space.use_proprio_camera_frame = use_proprio_camera_frame
        obs_space.use_language = use_language
        obs_space.use_gripper_state = use_gripper_state
        obs_space.gripper_type = gripper_type
        return obs_space
    return factory


@pytest.fixture
def mock_action_space():
    """Factory for creating mock action spaces."""
    def factory(
        has_position=True,
        position_dim=3,
        has_orientation=False,
        orientation_dim=0,
        has_gripper=False,
        gripper_type=GripperType.BINARY.value,
        gripper_dim=1,
    ):
        action_space = MagicMock()
        action_space.has_position = has_position
        action_space.position_dim = position_dim
        action_space.has_orientation = has_orientation
        action_space.orientation_dim = orientation_dim
        action_space.has_gripper = has_gripper
        action_space.gripper_type = gripper_type
        action_space.gripper_dim = gripper_dim
        return action_space
    return factory


@pytest.fixture
def mock_normalizer():
    """Factory for creating mock normalizers."""
    def factory():
        normalizer = MagicMock(spec=LinearNormalizer)
        normalizer.__getitem__ = MagicMock(return_value=normalizer)
        normalizer.normalize = MagicMock(side_effect=lambda x: x)
        normalizer.unnormalize = MagicMock(side_effect=lambda x: x)
        return normalizer
    return factory


@pytest.fixture
def mock_tokenizer():
    """Factory for creating mock tokenizers."""
    def factory(has_obs_tokenizer=True, has_action_tokenizer=True):
        tokenizer = MagicMock()

        if has_obs_tokenizer:
            obs_tokenizer = MagicMock()
            obs_tokenizer.observation_keys = [ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value]
            obs_tokenizer.tokenize = MagicMock(return_value={
                SampleKey.TOKENIZED_OBSERVATIONS.value: torch.randn(10, 128),
                SampleKey.IS_PAD_OBSERVATION.value: torch.zeros(10, dtype=torch.bool),
            })
            tokenizer.observation_tokenizer = obs_tokenizer
        else:
            tokenizer.observation_tokenizer = None

        if has_action_tokenizer:
            action_tokenizer = MagicMock()
            action_tokenizer.encode = MagicMock(return_value={
                SampleKey.TOKENIZED_ACTIONS.value: torch.randint(0, 256, (16,)),
                SampleKey.IS_PAD_ACTION.value: torch.zeros(16, dtype=torch.bool),
            })
            action_tokenizer.decode = MagicMock(return_value=torch.randn(16, 7).numpy())
            tokenizer.action_tokenizer = action_tokenizer
        else:
            tokenizer.action_tokenizer = None

        return tokenizer
    return factory


@pytest.fixture
def sample_factory():
    """Factory for creating sample dictionaries."""
    def factory(
        include_language=False,
        include_gripper_obs=False,
        pred_horizon=16,
        obs_horizon=1,
    ):
        observation = {
            Cameras.LEFT.value: torch.randn(obs_horizon, 3, 64, 64),
            ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value: torch.randn(obs_horizon, 7),
        }

        if include_language:
            observation[ObsKey.LANGUAGE.value] = torch.randint(0, 1000, (obs_horizon, 77))

        if include_gripper_obs:
            observation[ProprioKey.GRIPPER_STATE.value] = torch.randint(0, 2, (obs_horizon, 1)).float()

        actions = {
            ProprioceptiveType.POSITION.value: torch.randn(pred_horizon, 3),
            ProprioceptiveType.ORIENTATION.value: torch.randn(pred_horizon, 4),
            ProprioceptiveType.GRIPPER.value: torch.randint(0, 2, (pred_horizon, 1)).float(),
        }

        return {
            SampleKey.OBSERVATION.value: observation,
            SampleKey.ACTION.value: actions,
        }
    return factory


@pytest.mark.unit
class TestNormalizeObservation:
    """Test normalize_observation function."""

    def test_normalize_basic_observations(self, mock_normalizer, mock_observation_space):
        """Test normalizing basic observations without special keys."""
        normalizer = mock_normalizer()
        obs_space = mock_observation_space(
            camera_keys=[Cameras.LEFT.value],
            use_proprio_base_frame=True,
            use_language=False,
            use_gripper_state=False,
        )

        observation = {
            Cameras.LEFT.value: torch.randn(1, 3, 64, 64),
            ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value: torch.randn(1, 7),
        }

        result = normalize_observation(observation, normalizer, obs_space)

        assert Cameras.LEFT.value in result
        assert ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value in result
        normalizer.normalize.assert_called_once()

    def test_normalize_preserves_language(self, mock_normalizer, mock_observation_space):
        """Test that language observations are not normalized."""
        normalizer = mock_normalizer()
        obs_space = mock_observation_space(
            camera_keys=[Cameras.LEFT.value],
            use_language=True,
        )

        language_tensor = torch.randint(0, 1000, (1, 77))
        observation = {
            Cameras.LEFT.value: torch.randn(1, 3, 64, 64),
            ObsKey.LANGUAGE.value: language_tensor,
        }

        result = normalize_observation(observation, normalizer, obs_space)

        assert ObsKey.LANGUAGE.value in result
        assert torch.equal(result[ObsKey.LANGUAGE.value], language_tensor)

    def test_normalize_preserves_binary_gripper(self, mock_normalizer, mock_observation_space):
        """Test that binary gripper states are not normalized."""
        normalizer = mock_normalizer()
        obs_space = mock_observation_space(
            camera_keys=[Cameras.LEFT.value],
            use_gripper_state=True,
            gripper_type=GripperType.BINARY.value,
        )

        gripper_tensor = torch.randint(0, 2, (1, 1)).float()
        observation = {
            Cameras.LEFT.value: torch.randn(1, 3, 64, 64),
            ProprioKey.GRIPPER_STATE.value: gripper_tensor,
        }

        result = normalize_observation(observation, normalizer, obs_space)

        assert ProprioKey.GRIPPER_STATE.value in result
        assert torch.equal(result[ProprioKey.GRIPPER_STATE.value], gripper_tensor)

    def test_normalize_continuous_gripper(self, mock_normalizer, mock_observation_space):
        """Test that continuous gripper states are normalized."""
        normalizer = mock_normalizer()
        obs_space = mock_observation_space(
            camera_keys=[Cameras.LEFT.value],
            use_gripper_state=True,
            gripper_type=GripperType.CONTINUOUS.value,
        )

        observation = {
            Cameras.LEFT.value: torch.randn(1, 3, 64, 64),
            ProprioKey.GRIPPER_STATE.value: torch.rand(1, 1),
        }

        result = normalize_observation(observation, normalizer, obs_space)

        assert ProprioKey.GRIPPER_STATE.value in result
        normalizer.normalize.assert_called_once()


@pytest.mark.unit
class TestNormalizeActions:
    """Test normalize_actions function."""

    @pytest.mark.parametrize("has_pos,has_ori,has_grip", [
        (True, False, False),
        (True, True, False),
        (True, True, True),
        (False, True, True),
        (True, False, True),
    ])
    def test_normalize_action_combinations(self, mock_normalizer, has_pos, has_ori, has_grip, mock_action_space):
        """Test normalizing different action combinations."""
        normalizer = mock_normalizer()
        action_space = mock_action_space(
            has_position=has_pos,
            has_orientation=has_ori,
            has_gripper=has_grip,
            gripper_type=GripperType.CONTINUOUS.value if has_grip else GripperType.BINARY.value,
        )

        actions = {}
        if has_pos:
            actions[ProprioceptiveType.POSITION.value] = torch.randn(16, 3)
        if has_ori:
            actions[ProprioceptiveType.ORIENTATION.value] = torch.randn(16, 4)
        if has_grip:
            actions[ProprioceptiveType.GRIPPER.value] = torch.rand(16, 1)

        result = normalize_actions(actions, normalizer, action_space)

        for key in actions:
            assert key in result


    def test_normalize_binary_gripper_not_normalized(self, mock_normalizer, mock_action_space):
        """Test that binary gripper actions are not normalized."""
        normalizer = mock_normalizer()
        action_space = mock_action_space(
            has_position=False,
            has_orientation=False,
            has_gripper=True,
            gripper_type=GripperType.BINARY.value,
        )

        gripper_tensor = torch.randint(0, 2, (16, 1)).float()
        actions = {ProprioceptiveType.GRIPPER.value: gripper_tensor}

        result = normalize_actions(actions, normalizer, action_space)

        assert torch.equal(result[ProprioceptiveType.GRIPPER.value], gripper_tensor)


@pytest.mark.unit
class TestUnnormalizeActions:
    """Test unnormalize_actions function."""

    def test_unnormalize_position_actions(self, mock_normalizer, mock_action_space):
        """Test unnormalizing position actions."""
        normalizer = mock_normalizer()
        action_space = mock_action_space(
            has_position=True,
            has_orientation=False,
            has_gripper=False,
        )

        normalized_actions = {ProprioceptiveType.POSITION.value: torch.randn(16, 3)}

        result = unnormalize_actions(normalized_actions, normalizer, action_space)

        assert ProprioceptiveType.POSITION.value in result

    def test_unnormalize_full_actions(self, mock_normalizer, mock_action_space):
        """Test unnormalizing all action types."""
        normalizer = mock_normalizer()
        action_space = mock_action_space(
            has_position=True,
            has_orientation=True,
            has_gripper=True,
            gripper_type=GripperType.CONTINUOUS.value,
        )

        normalized_actions = {
            ProprioceptiveType.POSITION.value: torch.randn(16, 3),
            ProprioceptiveType.ORIENTATION.value: torch.randn(16, 4),
            ProprioceptiveType.GRIPPER.value: torch.rand(16, 1),
        }

        result = unnormalize_actions(normalized_actions, normalizer, action_space)

        assert ProprioceptiveType.POSITION.value in result
        assert ProprioceptiveType.ORIENTATION.value in result
        assert ProprioceptiveType.GRIPPER.value in result

    def test_unnormalize_binary_gripper_unchanged(self, mock_normalizer, mock_action_space):
        """Test that binary gripper actions are not unnormalized."""
        normalizer = mock_normalizer()
        action_space = mock_action_space(
            has_position=False,
            has_orientation=False,
            has_gripper=True,
            gripper_type=GripperType.BINARY.value,
        )

        gripper_tensor = torch.randint(0, 2, (16, 1)).float()
        normalized_actions = {ProprioceptiveType.GRIPPER.value: gripper_tensor}

        result = unnormalize_actions(normalized_actions, normalizer, action_space)

        assert torch.equal(result[ProprioceptiveType.GRIPPER.value], gripper_tensor)


@pytest.mark.unit
class TestNormalizeSample:
    """Test normalize_sample function."""

    def test_normalize_sample_basic(self, mock_normalizer, sample_factory, mock_observation_space, mock_action_space):
        """Test normalizing a complete sample."""
        normalizer = mock_normalizer()
        sample = sample_factory()

        obs_space = mock_observation_space(
            camera_keys=[Cameras.LEFT.value],
            use_proprio_base_frame=True,
        )
        action_space = mock_action_space(
            has_position=True,
            has_orientation=True,
            has_gripper=True,
        )

        result = normalize_sample(sample, normalizer, obs_space, action_space)

        assert SampleKey.OBSERVATION.value in result
        assert SampleKey.ACTION.value in result
        assert Cameras.LEFT.value in result[SampleKey.OBSERVATION.value]
        assert ProprioceptiveType.POSITION.value in result[SampleKey.ACTION.value]

    def test_normalize_sample_does_not_modify_original(self, mock_normalizer, sample_factory, mock_observation_space, mock_action_space):
        """Test that normalize_sample doesn't modify the original sample."""
        normalizer = mock_normalizer()
        sample = sample_factory()
        original_obs_keys = set(sample[SampleKey.OBSERVATION.value].keys())
        original_action_keys = set(sample[SampleKey.ACTION.value].keys())

        obs_space = mock_observation_space(
            camera_keys=[Cameras.LEFT.value],
            use_proprio_base_frame=True,
        )
        action_space = mock_action_space(
            has_position=True,
            has_orientation=True,
            has_gripper=True,
        )

        normalize_sample(sample, normalizer, obs_space, action_space)

        assert set(sample[SampleKey.OBSERVATION.value].keys()) == original_obs_keys
        assert set(sample[SampleKey.ACTION.value].keys()) == original_action_keys


@pytest.mark.unit
class TestTokenizeObservation:
    """Test tokenize_observation function."""

    def test_tokenize_observation_basic(self, mock_tokenizer):
        """Test tokenizing observations."""
        tokenizer_obj = mock_tokenizer(has_action_tokenizer=False)
        obs_tokenizer = tokenizer_obj.observation_tokenizer

        observation = {
            Cameras.LEFT.value: torch.randn(1, 3, 64, 64),
            ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value: torch.randn(1, 7),
        }

        result = tokenize_observation(observation, obs_tokenizer)

        assert SampleKey.TOKENIZED_OBSERVATIONS.value in result
        assert SampleKey.IS_PAD_OBSERVATION.value in result
        assert Cameras.LEFT.value in result

    def test_tokenize_observation_missing_key_raises(self, mock_tokenizer):
        """Test that missing observation key raises KeyError."""
        tokenizer_obj = mock_tokenizer(has_action_tokenizer=False)
        obs_tokenizer = tokenizer_obj.observation_tokenizer
        obs_tokenizer.observation_keys = ["missing_key"]

        observation = {
            Cameras.LEFT.value: torch.randn(1, 3, 64, 64),
        }

        with pytest.raises(KeyError, match="Observation key.*not found"):
            tokenize_observation(observation, obs_tokenizer)


@pytest.mark.unit
class TestTokenizeActions:
    """Test tokenize_actions function."""

    def test_tokenize_actions_basic(self, mock_tokenizer):
        """Test tokenizing actions."""
        tokenizer_obj = mock_tokenizer(has_obs_tokenizer=False)
        action_tokenizer = tokenizer_obj.action_tokenizer

        actions = {
            ProprioceptiveType.POSITION.value: torch.randn(16, 3),
            ProprioceptiveType.ORIENTATION.value: torch.randn(16, 4),
            ProprioceptiveType.GRIPPER.value: torch.randint(0, 2, (16, 1)).float(),
        }

        result = tokenize_actions(actions, action_tokenizer)

        assert SampleKey.TOKENIZED_ACTIONS.value in result
        assert SampleKey.IS_PAD_ACTION.value in result
        action_tokenizer.encode.assert_called_once()

    def test_tokenize_actions_with_padding_mask(self, mock_tokenizer):
        """Test tokenizing actions with padding mask."""
        tokenizer_obj = mock_tokenizer(has_obs_tokenizer=False)
        action_tokenizer = tokenizer_obj.action_tokenizer

        pad_mask = torch.zeros(16, dtype=torch.bool)
        pad_mask[-5:] = True

        actions = {
            ProprioceptiveType.POSITION.value: torch.randn(16, 3),
            SampleKey.IS_PAD_ACTION.value: pad_mask,
        }

        result = tokenize_actions(actions, action_tokenizer)

        assert SampleKey.IS_PAD_ACTION.value in result
        action_tokenizer.encode.assert_called_once()

    def test_tokenize_actions_handles_1d_tensors(self, mock_tokenizer):
        """Test that 1D action tensors are unsqueezed."""
        tokenizer_obj = mock_tokenizer(has_obs_tokenizer=False)
        action_tokenizer = tokenizer_obj.action_tokenizer

        actions = {
            ProprioceptiveType.GRIPPER.value: torch.randint(0, 2, (16,)).float(),
        }

        result = tokenize_actions(actions, action_tokenizer)

        assert SampleKey.TOKENIZED_ACTIONS.value in result
        action_tokenizer.encode.assert_called_once()


@pytest.mark.unit
class TestDetokenizeActions:
    """Test detokenize_actions function."""

    def test_detokenize_actions_basic(self, mock_tokenizer, mock_action_space):
        """Test detokenizing actions."""
        tokenizer_obj = mock_tokenizer(has_obs_tokenizer=False)
        action_tokenizer = tokenizer_obj.action_tokenizer

        action_space = mock_action_space(
            has_position=True,
            position_dim=3,
            has_orientation=True,
            orientation_dim=4,
            has_gripper=False,
        )

        action_tokens = torch.randint(0, 256, (2, 16))

        result = detokenize_actions(action_tokens, action_tokenizer, action_space)

        assert ProprioceptiveType.POSITION.value in result
        assert ProprioceptiveType.ORIENTATION.value in result
        assert result[ProprioceptiveType.POSITION.value].shape[-1] == 3
        assert result[ProprioceptiveType.ORIENTATION.value].shape[-1] == 4

    def test_detokenize_actions_with_binary_gripper(self, mock_tokenizer, mock_action_space):
        """Test detokenizing actions with binary gripper."""
        tokenizer_obj = mock_tokenizer(has_obs_tokenizer=False)
        action_tokenizer = tokenizer_obj.action_tokenizer

        action_space = mock_action_space(
            has_position=True,
            position_dim=3,
            has_orientation=False,
            has_gripper=True,
            gripper_type=GripperType.BINARY.value,
        )

        action_tokens = torch.randint(0, 256, (2, 16))

        result = detokenize_actions(action_tokens, action_tokenizer, action_space)

        assert ProprioceptiveType.GRIPPER.value in result
        assert result[ProprioceptiveType.GRIPPER.value].dtype == torch.long

    def test_detokenize_actions_with_continuous_gripper(self, mock_tokenizer, mock_action_space):
        """Test detokenizing actions with continuous gripper."""
        tokenizer_obj = mock_tokenizer(has_obs_tokenizer=False)
        action_tokenizer = tokenizer_obj.action_tokenizer

        action_space = mock_action_space(
            has_position=True,
            position_dim=3,
            has_orientation=False,
            has_gripper=True,
            gripper_type=GripperType.CONTINUOUS.value,
        )

        action_tokens = torch.randint(0, 256, (2, 16))

        result = detokenize_actions(action_tokens, action_tokenizer, action_space)

        assert ProprioceptiveType.GRIPPER.value in result
        assert result[ProprioceptiveType.GRIPPER.value].dtype == torch.float32

    def test_detokenize_handles_3d_tokens(self, mock_tokenizer, mock_action_space):
        """Test that 3D action tokens (B, H, 1) are squeezed."""
        tokenizer_obj = mock_tokenizer(has_obs_tokenizer=False)
        action_tokenizer = tokenizer_obj.action_tokenizer

        action_space = mock_action_space(
            has_position=True,
            position_dim=3,
            has_orientation=False,
            has_gripper=False,
        )

        action_tokens = torch.randint(0, 256, (2, 16, 1))

        result = detokenize_actions(action_tokens, action_tokenizer, action_space)

        assert ProprioceptiveType.POSITION.value in result


@pytest.mark.unit
class TestTokenizeSample:
    """Test tokenize_sample function."""

    def test_tokenize_sample_both_tokenizers(self, mock_tokenizer, sample_factory):
        """Test tokenizing sample with both obs and action tokenizers."""
        tokenizer = mock_tokenizer(has_obs_tokenizer=True, has_action_tokenizer=True)
        sample = sample_factory()

        result = tokenize_sample(sample, tokenizer)

        assert SampleKey.TOKENIZED_OBSERVATIONS.value in result[SampleKey.OBSERVATION.value]
        assert SampleKey.TOKENIZED_ACTIONS.value in result[SampleKey.ACTION.value]

    def test_tokenize_sample_obs_only(self, mock_tokenizer, sample_factory):
        """Test tokenizing sample with only observation tokenizer."""
        tokenizer = mock_tokenizer(has_obs_tokenizer=True, has_action_tokenizer=False)
        sample = sample_factory()

        result = tokenize_sample(sample, tokenizer)

        assert SampleKey.TOKENIZED_OBSERVATIONS.value in result[SampleKey.OBSERVATION.value]
        assert SampleKey.TOKENIZED_ACTIONS.value not in result[SampleKey.ACTION.value]

    def test_tokenize_sample_action_only(self, mock_tokenizer, sample_factory):
        """Test tokenizing sample with only action tokenizer."""
        tokenizer = mock_tokenizer(has_obs_tokenizer=False, has_action_tokenizer=True)
        sample = sample_factory()

        result = tokenize_sample(sample, tokenizer)

        assert SampleKey.TOKENIZED_OBSERVATIONS.value not in result[SampleKey.OBSERVATION.value]
        assert SampleKey.TOKENIZED_ACTIONS.value in result[SampleKey.ACTION.value]

    def test_tokenize_sample_no_tokenizers(self, mock_tokenizer, sample_factory):
        """Test tokenizing sample with no tokenizers."""
        tokenizer = mock_tokenizer(has_obs_tokenizer=False, has_action_tokenizer=False)
        sample = sample_factory()

        result = tokenize_sample(sample, tokenizer)

        assert SampleKey.TOKENIZED_OBSERVATIONS.value not in result[SampleKey.OBSERVATION.value]
        assert SampleKey.TOKENIZED_ACTIONS.value not in result[SampleKey.ACTION.value]


@pytest.mark.unit
class TestIntegration:
    """Integration tests for transform pipeline."""

    def test_full_normalization_pipeline(self, mock_normalizer, sample_factory, mock_observation_space, mock_action_space):
        """Test complete normalization pipeline."""
        normalizer = mock_normalizer()
        sample = sample_factory(include_language=True, include_gripper_obs=True)

        obs_space = mock_observation_space(
            camera_keys=[Cameras.LEFT.value],
            use_proprio_base_frame=True,
            use_language=True,
            use_gripper_state=True,
            gripper_type=GripperType.BINARY.value,
        )
        action_space = mock_action_space(
            has_position=True,
            has_orientation=True,
            has_gripper=True,
            gripper_type=GripperType.BINARY.value,
        )

        result = normalize_sample(sample, normalizer, obs_space, action_space)

        assert ObsKey.LANGUAGE.value in result[SampleKey.OBSERVATION.value]
        assert ProprioKey.GRIPPER_STATE.value in result[SampleKey.OBSERVATION.value]
        assert ProprioceptiveType.GRIPPER.value in result[SampleKey.ACTION.value]

    def test_normalize_unnormalize_round_trip(self, mock_normalizer, mock_action_space):
        """Test that normalize and unnormalize are inverse operations."""
        normalizer = mock_normalizer()
        action_space = mock_action_space(
            has_position=True,
            has_orientation=True,
            has_gripper=True,
            gripper_type=GripperType.CONTINUOUS.value,
        )

        original_actions = {
            ProprioceptiveType.POSITION.value: torch.randn(16, 3),
            ProprioceptiveType.ORIENTATION.value: torch.randn(16, 4),
            ProprioceptiveType.GRIPPER.value: torch.rand(16, 1),
        }

        normalized = normalize_actions(original_actions, normalizer, action_space)
        unnormalized = unnormalize_actions(normalized, normalizer, action_space)

        assert set(unnormalized.keys()) == set(original_actions.keys())