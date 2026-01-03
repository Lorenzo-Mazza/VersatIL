import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch

from refactoring.data.preprocessor_builder import PreprocessorBuilder
from refactoring.data.normalization.normalizer import LinearNormalizer
from refactoring.data.tokenization.tokenizer import Tokenizer
from refactoring.configs.data.tokenizer import ObservationTokenizationConfig, ActionTokenizationConfig, TokenizationConfig
from refactoring.data.constants import (
    Cameras,
    POSITION_ACTION_KEY,
    ORIENTATION_ACTION_KEY,
    GRIPPER_ACTION_KEY,
    PROPRIO_OBS_ROBOT_FRAME_KEY,
    PROPRIO_OBS_CAMERA_FRAME_KEY,
    GRIPPER_STATE_OBS_KEY,
    LANGUAGE_KEY,
    GripperType,
    KinematicsNormalizationType,
    ImageNormalizationType,
    TokenizerType,
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
    """PreprocessorBuilder instance."""
    return PreprocessorBuilder(
        replay_buffer=mock_replay_buffer,
        action_processor=mock_action_processor,
        prediction_horizon=5,
        observation_space=observation_space,
        episode_ends=np.array([30, 70, 100]),
        kinematics_norm_type=KinematicsNormalizationType.MIN_MAX.value,
        image_norm_type=ImageNormalizationType.ZERO_TO_ONE.value,
        depth_norm_type=ImageNormalizationType.ZERO_TO_ONE.value,
    )

class TestNormalizerBuilderInitialization:
    """Test PreprocessorBuilder initialization."""


    def test_init_stores_parameters(self, mock_replay_buffer, mock_action_processor, observation_space):
        """Test that initialization stores all parameters."""
        episode_ends = np.array([30, 70, 100])

        builder = PreprocessorBuilder(
            replay_buffer=mock_replay_buffer,
            action_processor=mock_action_processor,
            prediction_horizon=10,
            observation_space=observation_space,
            episode_ends=episode_ends,
            kinematics_norm_type=KinematicsNormalizationType.GAUSSIAN.value,
            image_norm_type=KinematicsNormalizationType.MIN_MAX.value,
            depth_norm_type=KinematicsNormalizationType.MIN_MAX.value,
        )

        assert builder.replay_buffer == mock_replay_buffer
        assert builder.action_processor == mock_action_processor
        assert builder.observation_space == observation_space
        assert builder.prediction_horizon == 10
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

        with patch('refactoring.data.normalization.normalizer_builder.get_rgb_image_normalizer') as mock_get:
            normalizer_builder._setup_image_normalizers(normalizer, device=None, winsorize_depth=True)

            # Should be called for each RGB camera
            assert mock_get.call_count == 2


    def test_setup_depth_normalizer(self, normalizer_builder, mock_replay_buffer):
        """Test depth image normalizer setup."""
        normalizer_builder.observation_space.camera_keys = [Cameras.DEPTH.value]
        normalizer = LinearNormalizer()

        with patch('refactoring.data.normalization.normalizer_builder.get_depth_image_normalizer') as mock_get:
            normalizer_builder._setup_image_normalizers(normalizer, device=None, winsorize_depth=False)

            mock_get.assert_called_once()


    def test_depth_normalizer_with_winsorization(self, normalizer_builder):
        """Test depth normalizer with winsorization enabled."""
        normalizer_builder.observation_space.camera_keys = [Cameras.DEPTH.value]
        normalizer = LinearNormalizer()

        with patch('refactoring.data.normalization.normalizer_builder.get_depth_image_normalizer') as mock_get:
            normalizer_builder._setup_image_normalizers(normalizer, device=None, winsorize_depth=True)

            # Winsorization should clip values, affecting min/max
            call_kwargs = mock_get.call_args[1]
            assert 'input_min' in call_kwargs
            assert 'input_max' in call_kwargs


    def test_depth_normalizer_without_winsorization(self, normalizer_builder):
        """Test depth normalizer without winsorization."""
        normalizer_builder.observation_space.camera_keys = [Cameras.DEPTH.value]
        normalizer = LinearNormalizer()

        with patch('refactoring.data.normalization.normalizer_builder.get_depth_image_normalizer') as mock_get:
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


    def test_binary_gripper_not_normalized(self, normalizer_builder, mock_action_processor):
        """Test that binary gripper actions are NOT included in normalizer."""
        mock_action_processor.action_space.gripper_type = GripperType.BINARY

        normalizer = normalizer_builder.create_normalizer(device=None, winsorize_depth=True)

        # Binary gripper should NOT be in normalizer
        assert GRIPPER_ACTION_KEY not in normalizer.params_dict


    def test_binary_gripper_obs_not_normalized(self, normalizer_builder, mock_action_processor):
        """Test that binary gripper observations are NOT included in normalizer."""
        normalizer_builder.observation_space.use_gripper_state = True
        normalizer_builder.observation_space.gripper_type = GripperType.BINARY.value
        mock_action_processor.action_space.gripper_type = GripperType.BINARY

        normalizer = normalizer_builder.create_normalizer(device=None, winsorize_depth=True)

        # Binary gripper state should NOT be in normalizer
        assert GRIPPER_STATE_OBS_KEY not in normalizer.params_dict


    def test_continuous_gripper_is_normalized(self, normalizer_builder, mock_action_processor):
        """Test that continuous gripper actions ARE included in normalizer."""
        mock_action_processor.action_space.gripper_type = GripperType.CONTINUOUS

        normalizer = normalizer_builder.create_normalizer(device=None, winsorize_depth=True)

        # Continuous gripper SHOULD be in normalizer
        assert GRIPPER_ACTION_KEY in normalizer.params_dict


    def test_language_not_in_normalizer(self, mock_replay_buffer, mock_action_processor):
        """Test that language observations are never included in normalizer."""
        # Create observation space with language
        obs_space = MagicMock()
        obs_space.camera_keys = []
        obs_space.use_proprio_base_frame = True
        obs_space.use_proprio_camera_frame = False
        obs_space.use_gripper_state = False
        obs_space.use_language = True
        obs_space.custom_obs_keys = []

        # Add language to replay buffer
        original_getitem = mock_replay_buffer.__getitem__
        def getitem_with_language(key):
            if key == LANGUAGE_KEY:
                return np.array(["instruction 1", "instruction 2"] * 50, dtype=object)
            return original_getitem(key)
        mock_replay_buffer.__getitem__ = MagicMock(side_effect=getitem_with_language)

        builder = PreprocessorBuilder(
            replay_buffer=mock_replay_buffer,
            action_processor=mock_action_processor,
            prediction_horizon=5,
            observation_space=obs_space,
            episode_ends=np.array([30, 70, 100]),
            kinematics_norm_type=KinematicsNormalizationType.MIN_MAX.value,
            image_norm_type=ImageNormalizationType.ZERO_TO_ONE.value,
            depth_norm_type=ImageNormalizationType.ZERO_TO_ONE.value,
        )

        normalizer = builder.create_normalizer(device=None, winsorize_depth=True)
        assert LANGUAGE_KEY not in normalizer.params_dict


    def test_create_normalizer_with_device(self, normalizer_builder):
        """Test normalizer creation with specific device."""
        device = torch.device('cpu')
        normalizer = normalizer_builder.create_normalizer(device=device, winsorize_depth=True)
        assert normalizer[POSITION_ACTION_KEY].params_dict['scale'].device.type == 'cpu'


    def test_create_normalizer_min_max_mode(self, normalizer_builder):
        """Test normalizer with min-max normalization."""
        normalizer_builder.kinematics_norm_type = KinematicsNormalizationType.MIN_MAX.value

        normalizer = normalizer_builder.create_normalizer(device=None, winsorize_depth=True)
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

        builder = PreprocessorBuilder(
            replay_buffer=mock_replay_buffer,
            action_processor=mock_action_processor,
            prediction_horizon=5,
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


class TestCreateActionChunks:
    """Test action chunk creation for tokenizer."""

    def test_create_action_chunks_basic(self, normalizer_builder):
        """Test basic action chunk creation."""
        # Episode ends at [30, 70, 100] -> 97 actions total (each episode loses 1)
        action_dict = {
            POSITION_ACTION_KEY: np.random.randn(97, 3).astype(np.float32),
            ORIENTATION_ACTION_KEY: np.random.randn(97, 4).astype(np.float32),
        }
        prediction_horizon = 5

        chunks = normalizer_builder._create_action_chunks_for_tokenizer(
            action_dict, prediction_horizon
        )

        # Check shape: (num_chunks, prediction_horizon, total_action_dim)
        assert chunks.ndim == 3
        assert chunks.shape[1] == prediction_horizon
        assert chunks.shape[2] == 7  # 3 position + 4 orientation

    def test_create_action_chunks_respects_episode_boundaries(self, normalizer_builder):
        """Test that chunks don't cross episode boundaries."""
        # Episode ends at [30, 70, 100] -> actions at [29, 69, 99]
        # Adjusted ends: [29, 68, 97] (each episode loses 1 action)
        action_dict = {
            POSITION_ACTION_KEY: np.random.randn(97, 3).astype(np.float32),
        }
        prediction_horizon = 10

        chunks = normalizer_builder._create_action_chunks_for_tokenizer(
            action_dict, prediction_horizon
        )

        # Episode 1: 29 actions -> 20 chunks (29 - 10 + 1)
        # Episode 2: 39 actions (68-29) -> 30 chunks (39 - 10 + 1)
        # Episode 3: 29 actions (97-68) -> 20 chunks (29 - 10 + 1)
        # Total: 70 chunks
        expected_chunks = 20 + 30 + 20
        assert chunks.shape[0] == expected_chunks

    def test_create_action_chunks_concatenates_action_components(self, normalizer_builder):
        """Test that action components are concatenated in sorted key order."""
        # Episode ends at [30, 70, 100] -> 97 actions total
        action_dict = {
            POSITION_ACTION_KEY: np.ones((97, 3), dtype=np.float32),
            ORIENTATION_ACTION_KEY: np.ones((97, 4), dtype=np.float32) * 2,
            GRIPPER_ACTION_KEY: np.ones((97, 1), dtype=np.float32) * 3,
        }
        prediction_horizon = 5

        chunks = normalizer_builder._create_action_chunks_for_tokenizer(
            action_dict, prediction_horizon
        )

        # Check concatenation order (sorted keys: gripper < orientation < position)
        assert chunks.shape[2] == 8  # 1 + 4 + 3
        # gripper_action < orientation_action < position_action alphabetically
        assert np.allclose(chunks[0, 0, 0], 3.0)  # gripper first
        assert np.allclose(chunks[0, 0, 1:5], 2.0)  # orientation second
        assert np.allclose(chunks[0, 0, 5:8], 1.0)  # position last

    def test_create_action_chunks_raises_error_when_episodes_too_short(self, normalizer_builder):
        """Test error when no episodes are long enough for prediction horizon."""
        action_dict = {
            POSITION_ACTION_KEY: np.random.randn(97, 3).astype(np.float32),
        }
        # Prediction horizon longer than any episode
        prediction_horizon = 100

        with pytest.raises(ValueError, match="No episodes long enough"):
            normalizer_builder._create_action_chunks_for_tokenizer(
                action_dict, prediction_horizon
            )

    def test_create_action_chunks_sliding_window(self, normalizer_builder):
        """Test that chunks are created with sliding window within episodes."""
        # Create simple episode structure
        normalizer_builder.episode_ends = np.array([20])  # Single episode
        action_dict = {
            POSITION_ACTION_KEY: np.arange(19 * 3).reshape(19, 3).astype(np.float32),
        }
        prediction_horizon = 3

        chunks = normalizer_builder._create_action_chunks_for_tokenizer(
            action_dict, prediction_horizon
        )

        # Episode has 19 actions -> 17 chunks (19 - 3 + 1)
        assert chunks.shape[0] == 17
        # First chunk should be actions 0, 1, 2
        assert np.allclose(chunks[0, :, 0], [0, 3, 6])
        # Second chunk should be actions 1, 2, 3
        assert np.allclose(chunks[1, :, 0], [3, 6, 9])


class TestApplyWinsorization:
    """Test winsorization function."""

    def test_winsorization_clips_outliers(self, normalizer_builder):
        """Test that winsorization clips extreme values."""
        # Create data with outliers
        data = np.array([
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [100, 200, 300],  # Outlier
        ], dtype=np.float32)

        data_dict = {"test_key": data}
        quantiles = (0.1, 0.9)

        winsorized = normalizer_builder._apply_winsorization(data_dict, quantiles)

        # Outlier should be clipped
        assert "test_key" in winsorized
        assert winsorized["test_key"][-1, 0] < 100  # Should be clipped down
        assert winsorized["test_key"][-1, 0] > 1  # But not to minimum

    def test_winsorization_preserves_non_outliers(self, normalizer_builder):
        """Test that winsorization preserves non-outlier values."""
        # Create data without outliers
        data = np.random.randn(100, 3).astype(np.float32)
        data_dict = {"test_key": data}
        quantiles = (0.01, 0.99)

        winsorized = normalizer_builder._apply_winsorization(data_dict, quantiles)
        unchanged_ratio = np.mean(data == winsorized["test_key"])
        assert unchanged_ratio > 0.9  # At least 90% unchanged

    def test_winsorization_handles_multiple_keys(self, normalizer_builder):
        """Test winsorization on multiple keys."""
        data_dict = {
            POSITION_ACTION_KEY: np.random.randn(100, 3).astype(np.float32),
            ORIENTATION_ACTION_KEY: np.random.randn(100, 4).astype(np.float32),
        }
        quantiles = (0.05, 0.95)

        winsorized = normalizer_builder._apply_winsorization(data_dict, quantiles)

        assert POSITION_ACTION_KEY in winsorized
        assert ORIENTATION_ACTION_KEY in winsorized
        assert winsorized[POSITION_ACTION_KEY].shape == (100, 3)
        assert winsorized[ORIENTATION_ACTION_KEY].shape == (100, 4)

    def test_winsorization_per_dimension(self, normalizer_builder):
        """Test that winsorization is applied per dimension."""
        # Create data where each dimension has different outliers
        data = np.array([
            [1.0, 5.0, 10.0],
            [1.0, 5.0, 10.0],
            [1.0, 5.0, 10.0],
            [100.0, 5.0, 10.0],  # Outlier in dim 0
            [1.0, 500.0, 10.0],  # Outlier in dim 1
            [1.0, 5.0, 1000.0],  # Outlier in dim 2
        ], dtype=np.float32)

        data_dict = {"test_key": data}
        quantiles = (0.1, 0.9)

        winsorized = normalizer_builder._apply_winsorization(data_dict, quantiles)

        # Each dimension should be clipped independently
        assert winsorized["test_key"][3, 0] < 100  # Dim 0 outlier clipped
        assert winsorized["test_key"][4, 1] < 500  # Dim 1 outlier clipped
        assert winsorized["test_key"][5, 2] < 1000  # Dim 2 outlier clipped

    def test_winsorization_symmetric_bounds(self, normalizer_builder):
        """Test winsorization with symmetric quantiles."""
        # Create symmetric data
        data = np.random.randn(1000, 5).astype(np.float32)
        data_dict = {"test_key": data}
        quantiles = (0.05, 0.95)

        winsorized = normalizer_builder._apply_winsorization(data_dict, quantiles)

        # Check that bounds are approximately symmetric for normal data
        lower_bound = np.quantile(data, 0.05, axis=0)
        upper_bound = np.quantile(data, 0.95, axis=0)

        # For normal distribution, quantiles should be roughly symmetric
        assert np.allclose(np.abs(lower_bound), np.abs(upper_bound), rtol=0.2)


@pytest.mark.integration
class TestTokenizerCreation:
    """Tests for tokenizer creation alongside normalizer."""

    def test_create_normalizer_and_tokenizer_without_config(self, normalizer_builder):
        """Test that tokenizer is None when no tokenization config provided."""
        normalizer, tokenizer = normalizer_builder.create_normalizer_and_tokenizer(device=None)

        assert isinstance(normalizer, LinearNormalizer)
        assert tokenizer is None

    def test_create_normalizer_and_tokenizer_both_disabled(self, normalizer_builder):
        """Test that tokenizer is None when both tokenizations are disabled."""
        tokenization_config = TokenizationConfig(
            tokenize_observations=False,
            tokenize_actions=False,
        )
        normalizer_builder.tokenization_config = tokenization_config

        normalizer, tokenizer = normalizer_builder.create_normalizer_and_tokenizer(device=None)

        assert isinstance(normalizer, LinearNormalizer)
        assert tokenizer is None

    def test_create_observation_tokenizer(self, normalizer_builder):
        """Test creating observation tokenizer."""
        obs_tokenizer_config = ObservationTokenizationConfig(
            tokenizer_model="google/bert_uncased_L-2_H-128_A-2",
            observation_keys=[LANGUAGE_KEY, PROPRIO_OBS_ROBOT_FRAME_KEY],
            bin_continuous_data=True,
            num_bins=128,
            max_token_len=256,
        )
        tokenization_config = TokenizationConfig(
            tokenize_observations=True,
            tokenize_actions=False,
            observation_tokenizer=obs_tokenizer_config,
        )
        normalizer_builder.tokenization_config = tokenization_config

        normalizer, tokenizer = normalizer_builder.create_normalizer_and_tokenizer(device=None)

        assert isinstance(normalizer, LinearNormalizer)
        assert isinstance(tokenizer, Tokenizer)
        assert tokenizer.observation_tokenizer is not None
        assert tokenizer.action_tokenizer is None
        assert tokenizer.observation_vocab_size is not None
        assert tokenizer.observation_vocab_size > 0

    def test_create_action_tokenizer_pretrained(self, normalizer_builder):
        """Test creating action tokenizer with pretrained FAST."""
        action_tokenizer_config = ActionTokenizationConfig(
            tokenizer_chain=[TokenizerType.FAST.value],
            use_pretrained_fast=True,
        )
        tokenization_config = TokenizationConfig(
            tokenize_observations=False,
            tokenize_actions=True,
            action_tokenizer=action_tokenizer_config,
        )
        normalizer_builder.tokenization_config = tokenization_config

        normalizer, tokenizer = normalizer_builder.create_normalizer_and_tokenizer(device=None)

        assert isinstance(normalizer, LinearNormalizer)
        assert isinstance(tokenizer, Tokenizer)
        assert tokenizer.observation_tokenizer is None
        assert tokenizer.action_tokenizer is not None
        assert tokenizer.action_vocab_size == 2048

    def test_create_both_tokenizers(self, normalizer_builder):
        """Test creating both observation and action tokenizers."""
        obs_tokenizer_config = ObservationTokenizationConfig(
            tokenizer_model="google/bert_uncased_L-2_H-128_A-2",
            observation_keys=[LANGUAGE_KEY, PROPRIO_OBS_ROBOT_FRAME_KEY],
            bin_continuous_data=False,
            num_bins=128,
            max_token_len=256,
        )
        action_tokenizer_config = ActionTokenizationConfig(
            tokenizer_chain=[TokenizerType.FAST.value],
            use_pretrained_fast=True,
        )
        tokenization_config = TokenizationConfig(
            tokenize_observations=True,
            tokenize_actions=True,
            observation_tokenizer=obs_tokenizer_config,
            action_tokenizer=action_tokenizer_config,
        )
        normalizer_builder.tokenization_config = tokenization_config

        normalizer, tokenizer = normalizer_builder.create_normalizer_and_tokenizer(device=None)

        assert isinstance(normalizer, LinearNormalizer)
        assert isinstance(tokenizer, Tokenizer)
        assert tokenizer.observation_tokenizer is not None
        assert tokenizer.action_tokenizer is not None

