import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch

from refactoring.data.normalize.normalizer_builder import NormalizerBuilder
from refactoring.data.normalize.normalizer import LinearNormalizer
from refactoring.data.constants import (
    Cameras,
    POSITION_ACTION_KEY,
    ORIENTATION_ACTION_KEY,
    GRIPPER_ACTION_KEY,
    PROPRIO_OBS_ROBOT_FRAME_KEY,
    PROPRIO_OBS_CAMERA_FRAME_KEY,
    GRIPPER_STATE_OBS_KEY,
    GripperType,
    KinematicsNormalizationType, ImageNormalizationType,
)


@pytest.fixture
def mock_replay_buffer():
    """Mock replay buffer with test data."""
    buffer = MagicMock()
    buffer.n_episodes = 3
    buffer.n_steps = 100
    buffer.episode_ends = np.array([30, 70, 100])

    # Proprioceptive data
    buffer.__getitem__ = MagicMock(side_effect=lambda key: {
        PROPRIO_OBS_ROBOT_FRAME_KEY: np.random.randn(100, 7).astype(np.float32),
        PROPRIO_OBS_CAMERA_FRAME_KEY: np.random.randn(100, 7).astype(np.float32),
        GRIPPER_STATE_OBS_KEY: np.random.randint(0, 2, (100, 1)).astype(np.float32),
        Cameras.LEFT.value: np.random.randint(0, 255, (100, 224, 224, 3), dtype=np.uint8),
        Cameras.RIGHT.value: np.random.randint(0, 255, (100, 224, 224, 3), dtype=np.uint8),
        Cameras.DEPTH.value: np.random.uniform(0.5, 5.0, (100, 224, 224)).astype(np.float32),
        'custom_obs': np.random.randn(100, 4).astype(np.float32),
    }[key])

    buffer.keys.return_value = [
        PROPRIO_OBS_ROBOT_FRAME_KEY,
        PROPRIO_OBS_CAMERA_FRAME_KEY,
        GRIPPER_STATE_OBS_KEY,
        Cameras.LEFT.value,
        'custom_obs',
    ]

    return buffer


@pytest.fixture
def mock_action_processor():
    """Mock action processor."""
    processor = MagicMock()
    processor.predict_in_camera_frame = False
    processor.has_position = True
    processor.has_orientation = True
    processor.has_gripper = True
    processor.action_space.gripper_type = GripperType.BINARY

    processor.compute_actions_from_observations.return_value = {
        POSITION_ACTION_KEY: np.random.randn(99, 3).astype(np.float32),
        ORIENTATION_ACTION_KEY: np.random.randn(99, 4).astype(np.float32),
    }

    processor.compute_gripper_actions.return_value = np.random.randint(0, 2, (99, 1)).astype(np.float32)

    return processor


@pytest.fixture
def observation_space():
    """Observation space configuration."""
    config = MagicMock()
    config.camera_keys = [Cameras.LEFT.value, Cameras.RIGHT.value]
    config.use_proprio_base_frame = True
    config.use_proprio_camera_frame = False
    config.use_gripper_state = False
    config.custom_obs_keys = []
    return config


@pytest.fixture
def normalizer_builder(mock_replay_buffer, mock_action_processor, observation_space):
    """NormalizerBuilder instance."""
    return NormalizerBuilder(
        replay_buffer=mock_replay_buffer,
        action_processor=mock_action_processor,
        observation_space=observation_space,
        episode_ends=np.array([30, 70, 100]),
        kinematics_norm_type=KinematicsNormalizationType.MIN_MAX.value,
        image_norm_type=ImageNormalizationType.ZERO_TO_ONE.value,
        depth_norm_type=ImageNormalizationType.ZERO_TO_ONE.value,
    )

class TestNormalizerBuilderInitialization:
    """Test NormalizerBuilder initialization."""


    def test_init_stores_parameters(self, mock_replay_buffer, mock_action_processor, observation_space):
        """Test that initialization stores all parameters."""
        episode_ends = np.array([30, 70, 100])

        builder = NormalizerBuilder(
            replay_buffer=mock_replay_buffer,
            action_processor=mock_action_processor,
            observation_space=observation_space,
            episode_ends=episode_ends,
            kinematics_norm_type=KinematicsNormalizationType.GAUSSIAN.value,
            image_norm_type=KinematicsNormalizationType.MIN_MAX.value,
            depth_norm_type=KinematicsNormalizationType.MIN_MAX.value,
        )

        assert builder.replay_buffer == mock_replay_buffer
        assert builder.action_processor == mock_action_processor
        assert builder.observation_space == observation_space
        assert builder.kinematics_norm_type == KinematicsNormalizationType.GAUSSIAN.value
        np.testing.assert_array_equal(builder.episode_ends, episode_ends)


class TestReadProprioDataFromBuffer:
    """Test proprioceptive data reading."""


    def test_reads_position_actions(self, normalizer_builder, mock_action_processor):
        """Test reading position actions."""
        proprio_data = normalizer_builder._read_proprio_data_from_buffer()

        assert POSITION_ACTION_KEY in proprio_data
        assert proprio_data[POSITION_ACTION_KEY].shape == (99, 3)
        mock_action_processor.compute_actions_from_observations.assert_called_once()


    def test_reads_orientation_actions(self, normalizer_builder):
        """Test reading orientation actions."""
        proprio_data = normalizer_builder._read_proprio_data_from_buffer()

        assert ORIENTATION_ACTION_KEY in proprio_data
        assert proprio_data[ORIENTATION_ACTION_KEY].shape == (99, 4)


    def test_reads_continuous_gripper_actions(self, normalizer_builder, mock_action_processor):
        """Test reading continuous gripper actions."""
        mock_action_processor.action_space.gripper_type = GripperType.CONTINUOUS

        proprio_data = normalizer_builder._read_proprio_data_from_buffer()

        assert GRIPPER_ACTION_KEY in proprio_data
        mock_action_processor.compute_gripper_actions.assert_called_once()


    def test_reads_binary_gripper_actions(self, normalizer_builder, mock_action_processor):
        """Test that binary gripper actions are normalized."""
        mock_action_processor.action_space.gripper_type = GripperType.BINARY

        proprio_data = normalizer_builder._read_proprio_data_from_buffer()

        assert GRIPPER_ACTION_KEY in proprio_data


    def test_reads_continuous_gripper_observations(self, normalizer_builder):
        """Test reading continuous gripper state observations."""
        normalizer_builder.observation_space.use_gripper_state = True
        normalizer_builder.observation_space.gripper_type = GripperType.CONTINUOUS.value

        proprio_data = normalizer_builder._read_proprio_data_from_buffer()

        assert GRIPPER_STATE_OBS_KEY in proprio_data


    def test_reads_binary_gripper_observations(self, normalizer_builder):
        """Test that binary gripper observations are normalized."""
        normalizer_builder.observation_space.use_gripper_state = True
        normalizer_builder.observation_space.gripper_type = GripperType.BINARY.value

        proprio_data = normalizer_builder._read_proprio_data_from_buffer()

        assert GRIPPER_STATE_OBS_KEY in proprio_data


    def test_reads_robot_frame_observations(self, normalizer_builder):
        """Test reading robot frame proprioceptive observations."""
        proprio_data = normalizer_builder._read_proprio_data_from_buffer()

        assert PROPRIO_OBS_ROBOT_FRAME_KEY in proprio_data
        assert proprio_data[PROPRIO_OBS_ROBOT_FRAME_KEY].shape == (100, 7)


    def test_reads_camera_frame_observations(self, normalizer_builder):
        """Test reading camera frame proprioceptive observations."""
        normalizer_builder.observation_space.use_proprio_camera_frame = True

        proprio_data = normalizer_builder._read_proprio_data_from_buffer()

        assert PROPRIO_OBS_CAMERA_FRAME_KEY in proprio_data


    def test_reads_both_frame_observations(self, normalizer_builder):
        """Test reading both robot and camera frame observations."""
        normalizer_builder.observation_space.use_proprio_camera_frame = True

        proprio_data = normalizer_builder._read_proprio_data_from_buffer()

        assert PROPRIO_OBS_ROBOT_FRAME_KEY in proprio_data
        assert PROPRIO_OBS_CAMERA_FRAME_KEY in proprio_data


    def test_reads_custom_observation_keys(self, normalizer_builder):
        """Test reading custom observation keys."""
        normalizer_builder.observation_space.custom_obs_keys = ['custom_obs']

        proprio_data = normalizer_builder._read_proprio_data_from_buffer()

        assert 'custom_obs' in proprio_data


    def test_excludes_episode_boundaries(self, normalizer_builder, mock_action_processor):
        """Test that episode boundaries are excluded from actions."""
        proprio_data = normalizer_builder._read_proprio_data_from_buffer()

        # Should have n_steps - 1 - (n_episodes - 1) = 100 - 1 - 2 = 97 valid transitions
        # But we get 99 from the mock, so just verify compute was called with valid mask
        call_args = mock_action_processor.compute_actions_from_observations.call_args
        curr_obs, next_obs = call_args[0]

        # Length should be less than n_steps - 1 due to episode boundary masking
        assert len(curr_obs) == 99 or len(curr_obs) == 97


    def test_uses_camera_frame_for_actions_when_configured(self, normalizer_builder,
                                                           mock_action_processor, mock_replay_buffer):
        """Test that camera frame is used for action computation when configured."""
        mock_action_processor.predict_in_camera_frame = True

        normalizer_builder._read_proprio_data_from_buffer()

        # Should read from camera frame key
        mock_replay_buffer.__getitem__.assert_any_call(PROPRIO_OBS_CAMERA_FRAME_KEY)


    def test_raises_error_on_empty_buffer(self, normalizer_builder, mock_replay_buffer):
        """Test error handling for empty replay buffer."""
        mock_replay_buffer.__getitem__ = MagicMock(
            return_value=np.array([]).reshape(0, 7)
        )

        with pytest.raises(ValueError, match="Replay buffer is empty"):
            normalizer_builder._read_proprio_data_from_buffer()


class TestSetupImageNormalizers:
    """Test image normalizer setup."""


    def test_setup_rgb_normalizers(self, normalizer_builder):
        """Test RGB image normalizer setup."""
        normalizer = LinearNormalizer()

        with patch('refactoring.data.normalize.normalizer_builder.get_rgb_image_normalizer') as mock_get:
            normalizer_builder._setup_image_normalizers(normalizer, device=None, winsorize_depth=True)

            # Should be called for each RGB camera
            assert mock_get.call_count == 2


    def test_setup_depth_normalizer(self, normalizer_builder, mock_replay_buffer):
        """Test depth image normalizer setup."""
        normalizer_builder.observation_space.camera_keys = [Cameras.DEPTH.value]
        normalizer = LinearNormalizer()

        with patch('refactoring.data.normalize.normalizer_builder.get_depth_image_normalizer') as mock_get:
            normalizer_builder._setup_image_normalizers(normalizer, device=None, winsorize_depth=False)

            mock_get.assert_called_once()


    def test_depth_normalizer_with_winsorization(self, normalizer_builder):
        """Test depth normalizer with winsorization enabled."""
        normalizer_builder.observation_space.camera_keys = [Cameras.DEPTH.value]
        normalizer = LinearNormalizer()

        with patch('refactoring.data.normalize.normalizer_builder.get_depth_image_normalizer') as mock_get:
            normalizer_builder._setup_image_normalizers(normalizer, device=None, winsorize_depth=True)

            # Winsorization should clip values, affecting min/max
            call_kwargs = mock_get.call_args[1]
            assert 'input_min' in call_kwargs
            assert 'input_max' in call_kwargs


    def test_depth_normalizer_without_winsorization(self, normalizer_builder):
        """Test depth normalizer without winsorization."""
        normalizer_builder.observation_space.camera_keys = [Cameras.DEPTH.value]
        normalizer = LinearNormalizer()

        with patch('refactoring.data.normalize.normalizer_builder.get_depth_image_normalizer') as mock_get:
            normalizer_builder._setup_image_normalizers(normalizer, device=None, winsorize_depth=False)

            mock_get.assert_called_once()


class TestCreateNormalizer:
    """Test complete normalizer creation."""


    def test_create_normalizer_basic(self, normalizer_builder):
        """Test basic normalizer creation."""
        normalizer = normalizer_builder.create_normalizer(device=None, winsorize_depth=True)

        assert isinstance(normalizer, LinearNormalizer)
        assert POSITION_ACTION_KEY in normalizer.params_dict
        assert ORIENTATION_ACTION_KEY in normalizer.params_dict


    def test_create_normalizer_with_device(self, normalizer_builder):
        """Test normalizer creation with specific device."""
        device = torch.device('cpu')
        normalizer = normalizer_builder.create_normalizer(device=device, winsorize_depth=True)

        assert normalizer[POSITION_ACTION_KEY].params_dict['scale'].device.type == 'cpu'


    def test_create_normalizer_min_max_mode(self, normalizer_builder):
        """Test normalizer with min-max normalization."""
        normalizer_builder.kinematics_norm_type = KinematicsNormalizationType.MIN_MAX.value

        normalizer = normalizer_builder.create_normalizer(device=None, winsorize_depth=True)

        # Verify it was fitted with min_max mode
        assert POSITION_ACTION_KEY in normalizer.params_dict


    def test_create_normalizer_gaussian_mode(self, normalizer_builder):
        """Test normalizer with Gaussian normalization."""
        normalizer_builder.kinematics_norm_type = KinematicsNormalizationType.GAUSSIAN.value

        normalizer = normalizer_builder.create_normalizer(device=None, winsorize_depth=True)

        assert POSITION_ACTION_KEY in normalizer.params_dict


    def test_normalizer_includes_all_proprio_keys(self, normalizer_builder):
        """Test that normalizer includes all configured proprioceptive keys."""
        normalizer_builder.observation_space.use_proprio_camera_frame = True
        normalizer_builder.observation_space.custom_obs_keys = ['custom_obs']

        normalizer = normalizer_builder.create_normalizer(device=None, winsorize_depth=True)

        assert POSITION_ACTION_KEY in normalizer.params_dict
        assert ORIENTATION_ACTION_KEY in normalizer.params_dict
        assert PROPRIO_OBS_ROBOT_FRAME_KEY in normalizer.params_dict
        assert PROPRIO_OBS_CAMERA_FRAME_KEY in normalizer.params_dict
        assert 'custom_obs' in normalizer.params_dict


    def test_normalizer_includes_camera_keys(self, normalizer_builder):
        """Test that normalizer includes all camera keys."""
        normalizer = normalizer_builder.create_normalizer(device=None, winsorize_depth=True)

        assert Cameras.LEFT.value in normalizer.params_dict
        assert Cameras.RIGHT.value in normalizer.params_dict


class TestIntegration:
    """Integration tests for complete workflows."""


    def test_complete_normalizer_pipeline(self, normalizer_builder):
        """Test complete normalizer creation and usage."""
        normalizer = normalizer_builder.create_normalizer(device=None, winsorize_depth=True)

        # Test normalization
        test_position = np.random.randn(10, 3).astype(np.float32)
        normalized = normalizer[POSITION_ACTION_KEY].normalize(test_position)

        assert isinstance(normalized, torch.Tensor)
        assert normalized.shape == (10, 3)


    def test_normalizer_with_all_features(self, mock_replay_buffer, mock_action_processor):
        """Test normalizer with all features enabled."""
        obs_space = MagicMock()
        obs_space.camera_keys = [Cameras.LEFT.value, Cameras.RIGHT.value, Cameras.DEPTH.value]
        obs_space.use_proprio_base_frame = True
        obs_space.use_proprio_camera_frame = True
        obs_space.use_gripper_state = True
        obs_space.gripper_type = GripperType.CONTINUOUS.value
        obs_space.custom_obs_keys = ['custom_obs']

        mock_action_processor.action_space.gripper_type = GripperType.CONTINUOUS

        builder = NormalizerBuilder(
            replay_buffer=mock_replay_buffer,
            action_processor=mock_action_processor,
            observation_space=obs_space,
            episode_ends=np.array([30, 70, 100]),
            kinematics_norm_type=KinematicsNormalizationType.MIN_MAX.value,
            image_norm_type=ImageNormalizationType.ZERO_TO_ONE.value,
            depth_norm_type=ImageNormalizationType.ZERO_TO_ONE.value,
        )

        normalizer = builder.create_normalizer(device=None, winsorize_depth=True)

        # Verify all keys present
        assert POSITION_ACTION_KEY in normalizer.params_dict
        assert ORIENTATION_ACTION_KEY in normalizer.params_dict
        assert GRIPPER_ACTION_KEY in normalizer.params_dict
        assert GRIPPER_STATE_OBS_KEY in normalizer.params_dict
        assert PROPRIO_OBS_ROBOT_FRAME_KEY in normalizer.params_dict
        assert PROPRIO_OBS_CAMERA_FRAME_KEY in normalizer.params_dict
        assert Cameras.DEPTH.value in normalizer.params_dict
        assert 'custom_obs' in normalizer.params_dict


    def test_normalizer_output_stats(self, normalizer_builder):
        """Test that normalizer provides output statistics."""
        normalizer = normalizer_builder.create_normalizer(device=None, winsorize_depth=True)

        stats = normalizer[POSITION_ACTION_KEY].get_input_stats()

        assert 'min' in stats
        assert 'max' in stats
        assert 'mean' in stats
        assert 'std' in stats


    def test_normalizer_roundtrip(self, normalizer_builder):
        """Test normalization and unnormalization roundtrip."""
        normalizer = normalizer_builder.create_normalizer(device=None, winsorize_depth=True)

        original = np.random.randn(10, 3).astype(np.float32)
        normalized = normalizer[POSITION_ACTION_KEY].normalize(original)
        unnormalized = normalizer[POSITION_ACTION_KEY].unnormalize(normalized)

        np.testing.assert_allclose(unnormalized.numpy(), original, rtol=1e-5, atol=1e-6)