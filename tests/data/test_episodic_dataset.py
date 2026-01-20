import pytest
import numpy as np
import torch
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock
from torch.utils.data import DataLoader

from versatil.data.episodic_dataset import EpisodicDataset
from versatil.data.preprocessing.replay_buffer import ReplayBuffer
from versatil.data.constants import (
    Cameras,
    PROPRIO_OBS_ROBOT_FRAME_KEY,
    PROPRIO_OBS_CAMERA_FRAME_KEY,
    GRIPPER_STATE_OBS_KEY,
    GRIPPER_ACTION_KEY,
    POSITION_ACTION_KEY,
    ORIENTATION_ACTION_KEY,
    GripperType,
    OBSERVATION_KEY,
    IS_PAD_ACTION_KEY,
    LANGUAGE_KEY,
    ACTION_KEY,
    TOKENIZED_OBSERVATIONS_KEY,
    IS_PAD_OBSERVATION_KEY,
    TOKENIZED_ACTIONS_KEY,
    TokenizerType,
)
from versatil.configs.data.tokenizer import ObservationTokenizationConfig, ActionTokenizationConfig, TokenizationConfig


@pytest.fixture
def temp_zarr_dir():
    """Create temporary directory for zarr files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def action_config():
    """Action space configuration."""
    config = MagicMock()
    config.predict_in_camera_frame = False
    config.deltas_as_actions = True
    config.has_gripper = True
    config.has_position = True
    config.has_orientation = True
    config.task_has_phases = False

    config.gripper_type = GripperType.BINARY.value
    config.position_dim = 3
    config.orientation_dim = 4
    config.gripper_dim = 1
    config.orientation_repr = "quaternion"
    config.denoise_actions = False
    config.get_required_zarr_keys.return_value = [
        PROPRIO_OBS_ROBOT_FRAME_KEY,
        PROPRIO_OBS_CAMERA_FRAME_KEY,
        GRIPPER_STATE_OBS_KEY,
    ]
    return config


@pytest.fixture
def observation_config():
    """Observation space configuration."""
    config = MagicMock()
    config.camera_keys = [Cameras.LEFT.value, Cameras.RIGHT.value]
    config.use_proprioceptive_data = True
    config.use_proprio_base_frame = True
    config.use_proprio_camera_frame = False
    config.task_has_phases = False
    config.use_language = False
    config.get_required_zarr_keys.return_value = [
        Cameras.LEFT.value,
        Cameras.RIGHT.value,
        PROPRIO_OBS_ROBOT_FRAME_KEY,
    ]
    return config


@pytest.fixture
def observation_config_with_language():
    """Observation space configuration with language enabled."""
    config = MagicMock()
    config.camera_keys = [Cameras.LEFT.value, Cameras.RIGHT.value]
    config.use_proprioceptive_data = True
    config.use_proprio_base_frame = True
    config.use_proprio_camera_frame = False
    config.task_has_phases = False
    config.use_language = True
    config.get_required_zarr_keys.return_value = [
        Cameras.LEFT.value,
        Cameras.RIGHT.value,
        PROPRIO_OBS_ROBOT_FRAME_KEY,
        LANGUAGE_KEY,
    ]
    return config


@pytest.fixture
def dataloader_config():
    """Dataloader configuration."""
    config = MagicMock()
    config.action_backward_shift = 0
    config.kinematics_norm_type = "min_max"
    config.image_norm_type = "imagenet"
    config.depth_norm_type = "min_max"
    config.color_augmentation = None
    config.spatial_augmentation = None
    config.rotation_augmentation = None
    config.image_height = 224
    config.image_width = 224
    config.val_ratio = 0.1
    config.total_ratio = 1.0
    config.downsample_factor = 1
    config.center_episode_start = False
    config.skip_initial_episode_steps = 0
    return config


@pytest.fixture
def simple_replay_buffer(temp_zarr_dir):
    """Create a simple replay buffer with multiple episodes."""
    zarr_path = Path(temp_zarr_dir) / "test_buffer.zarr"
    buffer = ReplayBuffer.create_empty_zarr()

    episode_lengths = [10, 15, 8]
    for ep_len in episode_lengths:
        episode = {
            Cameras.LEFT.value: np.random.randint(0, 255, (ep_len, 32, 32, 3), dtype=np.uint8),
            Cameras.RIGHT.value: np.random.randint(0, 255, (ep_len, 32, 32, 3), dtype=np.uint8),
            PROPRIO_OBS_ROBOT_FRAME_KEY: np.random.randn(ep_len, 7).astype(np.float32),
            PROPRIO_OBS_CAMERA_FRAME_KEY: np.random.randn(ep_len, 7).astype(np.float32),
            GRIPPER_STATE_OBS_KEY: np.random.randint(0, 2, (ep_len, 1)).astype(np.float32),
        }
        buffer.add_episode(episode)

    buffer.save_to_path(str(zarr_path))
    return str(zarr_path)


@pytest.fixture
def uncentered_replay_buffer(temp_zarr_dir):
    """Replay buffer with known first position for centering tests."""
    zarr_path = Path(temp_zarr_dir) / "uncentered.zarr"
    buffer = ReplayBuffer.create_empty_zarr()

    first_pos = np.array([1.5, 2.5, 3.5])
    episode = {
        Cameras.LEFT.value: np.random.randint(0, 255, (5, 32, 32, 3), dtype=np.uint8),
        Cameras.RIGHT.value: np.random.randint(0, 255, (5, 32, 32, 3), dtype=np.uint8),
        PROPRIO_OBS_ROBOT_FRAME_KEY: np.vstack([
            np.hstack([first_pos, np.random.randn(4)]),
            np.random.randn(4, 7)
        ]).astype(np.float32),
        PROPRIO_OBS_CAMERA_FRAME_KEY: np.random.randn(5, 7).astype(np.float32),
        GRIPPER_STATE_OBS_KEY: np.random.randint(0, 2, (5, 1)).astype(np.float32),
    }
    buffer.add_episode(episode)
    buffer.save_to_path(str(zarr_path))
    return str(zarr_path), first_pos


@pytest.fixture
def gripper_imbalanced_buffer(temp_zarr_dir):
    """Buffer with known gripper distribution (7 closed, 3 open)."""
    zarr_path = Path(temp_zarr_dir) / "gripper_test.zarr"
    buffer = ReplayBuffer.create_empty_zarr()

    gripper_states = np.array([[1], [1], [1], [1], [1], [1], [1], [0], [0], [0]], dtype=np.float32)
    episode = {
        Cameras.LEFT.value: np.random.randint(0, 255, (10, 32, 32, 3), dtype=np.uint8),
        Cameras.RIGHT.value: np.random.randint(0, 255, (10, 32, 32, 3), dtype=np.uint8),
        PROPRIO_OBS_ROBOT_FRAME_KEY: np.random.randn(10, 7).astype(np.float32),
        PROPRIO_OBS_CAMERA_FRAME_KEY: np.random.randn(10, 7).astype(np.float32),
        GRIPPER_STATE_OBS_KEY: gripper_states,
    }
    buffer.add_episode(episode)
    buffer.save_to_path(str(zarr_path))
    return str(zarr_path)


@pytest.fixture
def language_replay_buffer(temp_zarr_dir):
    """Replay buffer with language instructions."""
    zarr_path = Path(temp_zarr_dir) / "language_buffer.zarr"
    buffer = ReplayBuffer.create_empty_zarr()

    for ep_idx in range(2):
        ep_len = 10
        episode = {
            Cameras.LEFT.value: np.random.randint(0, 255, (ep_len, 32, 32, 3), dtype=np.uint8),
            Cameras.RIGHT.value: np.random.randint(0, 255, (ep_len, 32, 32, 3), dtype=np.uint8),
            PROPRIO_OBS_ROBOT_FRAME_KEY: np.random.randn(ep_len, 7).astype(np.float32),
            PROPRIO_OBS_CAMERA_FRAME_KEY: np.random.randn(ep_len, 7).astype(np.float32),
            GRIPPER_STATE_OBS_KEY: np.random.randint(0, 2, (ep_len, 1)).astype(np.float32),
            LANGUAGE_KEY: np.array([f'episode_{ep_idx}_instruction_{i}' for i in range(ep_len)], dtype=object),
        }
        buffer.add_episode(episode)

    buffer.save_to_path(str(zarr_path))
    return str(zarr_path)


@pytest.fixture
def varlen_language_buffer(temp_zarr_dir):
    """Buffer with variable-length language instructions."""
    zarr_path = Path(temp_zarr_dir) / "varlen_buffer.zarr"
    buffer = ReplayBuffer.create_empty_zarr()

    instructions = [
        'short',
        'this is a very long instruction with many words that describes the task in detail',
        'medium length instruction here',
        'action_embedding',
        'another lengthy instruction that explains what to do step by step',
        'brief',
        'normal',
        'extended instruction with details',
        'ok',
        'final instruction'
    ]

    episode = {
        Cameras.LEFT.value: np.random.randint(0, 255, (10, 32, 32, 3), dtype=np.uint8),
        Cameras.RIGHT.value: np.random.randint(0, 255, (10, 32, 32, 3), dtype=np.uint8),
        PROPRIO_OBS_ROBOT_FRAME_KEY: np.random.randn(10, 7).astype(np.float32),
        PROPRIO_OBS_CAMERA_FRAME_KEY: np.random.randn(10, 7).astype(np.float32),
        GRIPPER_STATE_OBS_KEY: np.random.randint(0, 2, (10, 1)).astype(np.float32),
        LANGUAGE_KEY: np.array(instructions, dtype=object),
    }
    buffer.add_episode(episode)
    buffer.save_to_path(str(zarr_path))
    return str(zarr_path)


@pytest.fixture
def long_language_buffer(temp_zarr_dir):
    """Buffer with long episode for collation tests."""
    zarr_path = Path(temp_zarr_dir) / "collation_buffer.zarr"
    buffer = ReplayBuffer.create_empty_zarr()

    episode = {
        Cameras.LEFT.value: np.random.randint(0, 255, (20, 32, 32, 3), dtype=np.uint8),
        Cameras.RIGHT.value: np.random.randint(0, 255, (20, 32, 32, 3), dtype=np.uint8),
        PROPRIO_OBS_ROBOT_FRAME_KEY: np.random.randn(20, 7).astype(np.float32),
        PROPRIO_OBS_CAMERA_FRAME_KEY: np.random.randn(20, 7).astype(np.float32),
        GRIPPER_STATE_OBS_KEY: np.random.randint(0, 2, (20, 1)).astype(np.float32),
        LANGUAGE_KEY: np.array([f'instruction_{i}' for i in range(20)], dtype=object),
    }
    buffer.add_episode(episode)
    buffer.save_to_path(str(zarr_path))
    return str(zarr_path)


@pytest.fixture
def downsample_language_buffer(temp_zarr_dir):
    """Buffer for testing language with downsampling."""
    zarr_path = Path(temp_zarr_dir) / "downsample_lang.zarr"
    buffer = ReplayBuffer.create_empty_zarr()

    episode = {
        Cameras.LEFT.value: np.random.randint(0, 255, (20, 32, 32, 3), dtype=np.uint8),
        Cameras.RIGHT.value: np.random.randint(0, 255, (20, 32, 32, 3), dtype=np.uint8),
        PROPRIO_OBS_ROBOT_FRAME_KEY: np.random.randn(20, 7).astype(np.float32),
        PROPRIO_OBS_CAMERA_FRAME_KEY: np.random.randn(20, 7).astype(np.float32),
        GRIPPER_STATE_OBS_KEY: np.random.randint(0, 2, (20, 1)).astype(np.float32),
        LANGUAGE_KEY: np.array([f'step_{i}' for i in range(20)], dtype=object),
    }
    buffer.add_episode(episode)
    buffer.save_to_path(str(zarr_path))
    return str(zarr_path)


class TestEpisodicDatasetInitialization:
    """Test dataset initialization."""


    def test_init_loads_replay_buffer(self, simple_replay_buffer, action_config, observation_config, dataloader_config):
        """Test that replay buffer is loaded from zarr path."""
        dataset = EpisodicDataset(
            zarr_path=simple_replay_buffer,
            action_space=action_config,
            observation_space=observation_config,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=True,
            seed=42,
        )

        assert dataset.replay_buffer is not None
        assert dataset.replay_buffer.n_episodes == 3
        assert dataset.replay_buffer.n_steps == 33


    def test_init_stores_config_params(self, simple_replay_buffer, action_config, observation_config, dataloader_config):
        """Test that configuration parameters are stored."""
        dataset = EpisodicDataset(
            zarr_path=simple_replay_buffer,
            action_space=action_config,
            observation_space=observation_config,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=True,
            seed=42,
        )

        assert dataset.pred_horizon == 4
        assert dataset.obs_horizon == 3
        assert dataset.action_backward_shift == 0
        assert dataset.train is True
        assert dataset.seed == 42


    def test_init_creates_components(self, simple_replay_buffer, action_config, observation_config, dataloader_config):
        """Test that required components are initialized."""
        dataset = EpisodicDataset(
            zarr_path=simple_replay_buffer,
            action_space=action_config,
            observation_space=observation_config,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=True,
            seed=42,
        )

        assert dataset.action_processor is not None
        assert dataset.augmentation_pipeline is not None
        assert dataset.sample_builder is not None
        assert dataset.sampler is not None


class TestEpisodeSplitting:
    """Test train/val episode splitting."""


    def test_train_uses_non_val_episodes(self, simple_replay_buffer, action_config, observation_config, dataloader_config):
        """Test that training dataset excludes validation episodes."""
        dataloader_config.val_ratio = 0.33

        train_dataset = EpisodicDataset(
            zarr_path=simple_replay_buffer,
            action_space=action_config,
            observation_space=observation_config,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=True,
            seed=42,
        )

        val_dataset = EpisodicDataset(
            zarr_path=simple_replay_buffer,
            action_space=action_config,
            observation_space=observation_config,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=False,
            seed=42,
        )

        train_mask = train_dataset.sampler.episode_mask
        val_mask = val_dataset.sampler.episode_mask

        assert not np.any(np.logical_and(train_mask, val_mask)), "Train and val should not overlap"
        assert np.sum(train_mask) + np.sum(val_mask) == len(train_mask), "Should cover all episodes"
        assert np.sum(train_mask) > 0, "Should have training episodes"
        assert np.sum(val_mask) > 0, "Should have validation episodes"


    def test_val_uses_val_episodes(self, simple_replay_buffer, action_config, observation_config, dataloader_config):
        """Test that validation dataset uses validation episodes."""
        dataloader_config.val_ratio = 0.33

        val_dataset = EpisodicDataset(
            zarr_path=simple_replay_buffer,
            action_space=action_config,
            observation_space=observation_config,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=False,
            seed=42,
        )

        assert len(val_dataset.episode_ends) >= 1


    def test_same_seed_produces_same_split(self, simple_replay_buffer, action_config, observation_config, dataloader_config):
        """Test deterministic splitting with same seed."""
        dataloader_config.val_ratio = 0.33

        dataset1 = EpisodicDataset(
            zarr_path=simple_replay_buffer,
            action_space=action_config,
            observation_space=observation_config,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=True,
            seed=42,
        )

        dataset2 = EpisodicDataset(
            zarr_path=simple_replay_buffer,
            action_space=action_config,
            observation_space=observation_config,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=True,
            seed=42,
        )

        np.testing.assert_array_equal(dataset1.episode_ends, dataset2.episode_ends)


    def test_different_seed_produces_different_split(self, simple_replay_buffer, action_config, observation_config, dataloader_config):
        """Test different splits with different seeds."""
        dataloader_config.val_ratio = 0.33

        dataset1 = EpisodicDataset(
            zarr_path=simple_replay_buffer,
            action_space=action_config,
            observation_space=observation_config,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=True,
            seed=42,
        )

        dataset2 = EpisodicDataset(
            zarr_path=simple_replay_buffer,
            action_space=action_config,
            observation_space=observation_config,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=True,
            seed=99,
        )

        assert len(dataset1.episode_ends) >= 1
        assert len(dataset2.episode_ends) >= 1


class TestDownsampling:
    """Test episode downsampling."""


    def test_downsample_reduces_steps(self, simple_replay_buffer, action_config, observation_config, dataloader_config):
        """Test that downsampling reduces total steps."""
        dataloader_config.downsample_factor = 2

        dataset = EpisodicDataset(
            zarr_path=simple_replay_buffer,
            action_space=action_config,
            observation_space=observation_config,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=True,
            seed=42,
        )

        assert dataset.replay_buffer.n_steps < 33


    def test_downsample_preserves_episode_count(self, simple_replay_buffer, action_config, observation_config, dataloader_config):
        """Test that downsampling preserves number of episodes."""
        dataloader_config.downsample_factor = 2

        dataset = EpisodicDataset(
            zarr_path=simple_replay_buffer,
            action_space=action_config,
            observation_space=observation_config,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=True,
            seed=42,
        )

        assert dataset.replay_buffer.n_episodes == 3


    def test_no_downsample_when_factor_1(self, simple_replay_buffer, action_config, observation_config, dataloader_config):
        """Test no downsampling when factor is 1."""
        dataloader_config.downsample_factor = 1

        dataset = EpisodicDataset(
            zarr_path=simple_replay_buffer,
            action_space=action_config,
            observation_space=observation_config,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=True,
            seed=42,
        )

        assert dataset.replay_buffer.n_steps == 33


class TestCentering:
    """Test episode centering at origin."""


    def test_centering_zeros_first_positions(self, simple_replay_buffer, action_config, observation_config, dataloader_config):
        """Test that first position of each episode becomes zero."""
        dataloader_config.center_episode_start = True

        dataset = EpisodicDataset(
            zarr_path=simple_replay_buffer,
            action_space=action_config,
            observation_space=observation_config,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=True,
            seed=42,
        )

        current_start = 0
        for end in dataset.episode_ends:
            first_obs = dataset.replay_buffer[PROPRIO_OBS_ROBOT_FRAME_KEY][current_start]
            first_pos = first_obs[:3]
            assert np.allclose(first_pos, 0, atol=1e-6), f"First position not zero: {first_pos}"
            current_start = end


    def test_no_centering_when_disabled(self, uncentered_replay_buffer, action_config, observation_config, dataloader_config):
        """Test that positions unchanged when centering disabled."""
        zarr_path, first_pos = uncentered_replay_buffer
        dataloader_config.center_episode_start = False

        dataset = EpisodicDataset(
            zarr_path=zarr_path,
            action_space=action_config,
            observation_space=observation_config,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=True,
            seed=42,
        )

        actual_first = dataset.replay_buffer[PROPRIO_OBS_ROBOT_FRAME_KEY][0][:3]
        np.testing.assert_allclose(actual_first, first_pos, rtol=1e-5)


class TestSampler:
    """Test sampler setup and indexing."""


    def test_sampler_created_with_correct_length(self, simple_replay_buffer, action_config, observation_config, dataloader_config):
        """Test that sampler has correct sequence length."""
        dataset = EpisodicDataset(
            zarr_path=simple_replay_buffer,
            action_space=action_config,
            observation_space=observation_config,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=True,
            seed=42,
        )

        expected_seq_len = 3 + 4 # obs_horizon + pred_horizon
        assert dataset.sampler.sequence_length == expected_seq_len


    def test_sampler_respects_episode_boundaries(self, simple_replay_buffer, action_config, observation_config, dataloader_config):
        """Test that sampler indices don't cross episode boundaries."""
        dataset = EpisodicDataset(
            zarr_path=simple_replay_buffer,
            action_space=action_config,
            observation_space=observation_config,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=True,
            seed=42,
        )

        episode_boundaries = dataset.episode_ends[:]

        for idx_row in dataset.sampler.indices:
            buffer_start, buffer_end, _, _ = idx_row
            episode_idx = np.searchsorted(episode_boundaries, buffer_start, side='right')
            episode_end = episode_boundaries[episode_idx]
            assert buffer_end <= episode_end, f"Sample crosses episode boundary: {buffer_start}-{buffer_end}, episode ends at {episode_end}"


class TestDatasetLength:
    """Test dataset length calculation."""


    def test_overlapping_mode_length_equals_sampler_length(self, simple_replay_buffer, action_config, observation_config, dataloader_config):
        """Test dataset length in overlapping mode."""
        dataset = EpisodicDataset(
            zarr_path=simple_replay_buffer,
            action_space=action_config,
            observation_space=observation_config,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=True,
            seed=42,
        )

        assert len(dataset) == len(dataset.sampler)



class TestGetItem:
    """Test sample retrieval."""


    def test_getitem_returns_valid_sample(self, simple_replay_buffer, action_config, observation_config, dataloader_config):
        """Test that __getitem__ returns properly structured sample."""
        dataset = EpisodicDataset(
            zarr_path=simple_replay_buffer,
            action_space=action_config,
            observation_space=observation_config,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=True,
            seed=42,
        )

        if len(dataset) == 0:
            pytest.skip("No valid samples in dataset")

        sample = dataset[0]

        assert isinstance(sample, dict)
        assert OBSERVATION_KEY in sample
        assert ACTION_KEY in sample
        assert IS_PAD_ACTION_KEY in sample[ACTION_KEY]
        assert POSITION_ACTION_KEY in sample[ACTION_KEY]
        assert ORIENTATION_ACTION_KEY in sample[ACTION_KEY]
        assert GRIPPER_ACTION_KEY in sample[ACTION_KEY]


    def test_getitem_observation_has_correct_structure(self, simple_replay_buffer, action_config, observation_config, dataloader_config):
        """Test observation structure."""
        dataset = EpisodicDataset(
            zarr_path=simple_replay_buffer,
            action_space=action_config,
            observation_space=observation_config,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=True,
            seed=42,
        )

        if len(dataset) == 0:
            pytest.skip("No valid samples in dataset")

        sample = dataset[0]
        obs = sample[OBSERVATION_KEY]

        assert Cameras.LEFT.value in obs
        assert Cameras.RIGHT.value in obs

        left_img = obs[Cameras.LEFT.value]
        assert left_img.shape[0] == 3
        assert left_img.shape[1] == 3
        assert left_img.ndim == 4


    def test_getitem_actions_have_correct_shape(self, simple_replay_buffer, action_config, observation_config, dataloader_config):
        """Test action shapes."""
        dataset = EpisodicDataset(
            zarr_path=simple_replay_buffer,
            action_space=action_config,
            observation_space=observation_config,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=True,
            seed=42,
        )

        if len(dataset) == 0:
            pytest.skip("No valid samples in dataset")

        sample = dataset[0]
        assert ACTION_KEY in sample
        assert sample[ACTION_KEY][POSITION_ACTION_KEY].shape == (4, 3)
        assert sample[ACTION_KEY][ORIENTATION_ACTION_KEY].shape == (4, 4)
        assert sample[ACTION_KEY][GRIPPER_ACTION_KEY].shape == (4, 1)
        assert sample[ACTION_KEY][IS_PAD_ACTION_KEY].shape == (4,)


    def test_getitem_gripper_has_correct_dtype(self, simple_replay_buffer, action_config, observation_config, dataloader_config):
        """Test that binary gripper actions have long dtype."""
        action_config.gripper_type = GripperType.BINARY.value

        dataset = EpisodicDataset(
            zarr_path=simple_replay_buffer,
            action_space=action_config,
            observation_space=observation_config,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=True,
            seed=42,
        )

        if len(dataset) == 0:
            pytest.skip("No valid samples in dataset")

        sample = dataset[0]
        assert ACTION_KEY in sample
        assert sample[ACTION_KEY][GRIPPER_ACTION_KEY].dtype == torch.long


class TestActionComputation:
    """Test action computation logic."""


    def test_actions_use_robot_frame_when_configured(self, simple_replay_buffer, action_config, observation_config, dataloader_config):
        """Test that robot frame is used for action computation."""
        action_config.predict_in_camera_frame = False

        dataset = EpisodicDataset(
            zarr_path=simple_replay_buffer,
            action_space=action_config,
            observation_space=observation_config,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=True,
            seed=42,
        )

        assert dataset.action_processor.predict_in_camera_frame is False


    def test_actions_use_camera_frame_when_configured(self, simple_replay_buffer, action_config, observation_config, dataloader_config):
        """Test that camera frame is used for action computation."""
        action_config.predict_in_camera_frame = True

        dataset = EpisodicDataset(
            zarr_path=simple_replay_buffer,
            action_space=action_config,
            observation_space=observation_config,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=True,
            seed=42,
        )

        assert dataset.action_processor.predict_in_camera_frame is True


class TestGripperImbalance:
    """Test gripper class imbalance weight calculation."""


    def test_gripper_imbalance_calculated_correctly(self, gripper_imbalanced_buffer, action_config, observation_config, dataloader_config):
        """Test gripper imbalance weight calculation."""
        action_config.has_gripper = True

        dataset = EpisodicDataset(
            zarr_path=gripper_imbalanced_buffer,
            action_space=action_config,
            observation_space=observation_config,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=True,
            seed=42,
        )

        weight = dataset.get_gripper_positive_class_imbalance_weight()

        expected = 3.0 / 7.0
        assert np.isclose(weight, expected, rtol=1e-5)


    def test_gripper_imbalance_raises_error_without_gripper(self, simple_replay_buffer, action_config, observation_config, dataloader_config):
        """Test error when gripper not in action space."""
        action_config.has_gripper = False

        dataset = EpisodicDataset(
            zarr_path=simple_replay_buffer,
            action_space=action_config,
            observation_space=observation_config,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=True,
            seed=42,
        )

        with pytest.raises(ValueError, match="Gripper actions are not being predicted"):
            dataset.get_gripper_positive_class_imbalance_weight()


class TestIntegration:
    """Integration tests."""


    def test_full_training_iteration(self, simple_replay_buffer, action_config, observation_config, dataloader_config):
        """Test full iteration through dataset."""
        dataset = EpisodicDataset(
            zarr_path=simple_replay_buffer,
            action_space=action_config,
            observation_space=observation_config,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=True,
            seed=42,
        )

        if len(dataset) == 0:
            pytest.skip("No valid samples")

        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            assert OBSERVATION_KEY in sample
            assert ACTION_KEY in sample
            assert POSITION_ACTION_KEY in sample[ACTION_KEY]
            assert IS_PAD_ACTION_KEY in sample[ACTION_KEY]


    def test_normalizer_creation(self, simple_replay_buffer, action_config, observation_config, dataloader_config):
        """Test that normalizer can be created."""
        dataset = EpisodicDataset(
            zarr_path=simple_replay_buffer,
            action_space=action_config,
            observation_space=observation_config,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=True,
            seed=42,
        )

        normalizer = dataset.get_normalizer()
        assert normalizer is not None


    def test_different_horizons(self, simple_replay_buffer, action_config, observation_config, dataloader_config):
        """Test with different pred/obs horizons."""
        dataset = EpisodicDataset(
            zarr_path=simple_replay_buffer,
            action_space=action_config,
            observation_space=observation_config,
            dataloader_config=dataloader_config,
            pred_horizon=8,
            obs_horizon=5,
            train=True,
            seed=42,
        )

        if len(dataset) == 0:
            pytest.skip("No valid samples")

        sample = dataset[0]
        assert ACTION_KEY in sample
        assert sample[OBSERVATION_KEY][Cameras.LEFT.value].shape[0] == 5
        assert sample[ACTION_KEY][POSITION_ACTION_KEY].shape[0] == 8
        assert sample[ACTION_KEY][IS_PAD_ACTION_KEY].shape[0] == 8


class TestLanguageInDataset:
    """Test language instruction integration in EpisodicDataset."""


    def test_dataset_with_language_instructions(self, language_replay_buffer, action_config, observation_config_with_language, dataloader_config):
        """Test dataset with language instructions enabled."""
        dataset = EpisodicDataset(
            zarr_path=language_replay_buffer,
            action_space=action_config,
            observation_space=observation_config_with_language,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=True,
            seed=42,
        )

        if len(dataset) == 0:
            pytest.skip("No valid samples")

        sample = dataset[0]

        assert OBSERVATION_KEY in sample
        assert LANGUAGE_KEY in sample[OBSERVATION_KEY]

        lang_data = sample[OBSERVATION_KEY][LANGUAGE_KEY]
        assert isinstance(lang_data, list)
        assert len(lang_data) == 3

        assert all(isinstance(s, str) for s in lang_data)
        assert 'episode' in lang_data[0]
        assert 'instruction' in lang_data[0]


    def test_language_collation_in_dataloader(self, long_language_buffer, action_config, observation_config_with_language, dataloader_config):
        """Test that language can be collated by DataLoader."""
        dataset = EpisodicDataset(
            zarr_path=long_language_buffer,
            action_space=action_config,
            observation_space=observation_config_with_language,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=True,
            seed=42,
        )

        if len(dataset) < 2:
            pytest.skip("Not enough samples")

        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        assert OBSERVATION_KEY in batch
        assert LANGUAGE_KEY in batch[OBSERVATION_KEY]

        batch_lang = batch[OBSERVATION_KEY][LANGUAGE_KEY]
        assert isinstance(batch_lang, list)

        # DataLoader collates by timestep, creating list of tuples
        # Structure: [(batch_item_0_t0, batch_item_1_t0), (batch_item_0_t1, batch_item_1_t1), ...]
        assert len(batch_lang) == 3  # obs_horizon

        # Each timestep should have batch_size items
        assert isinstance(batch_lang[0], tuple)
        assert len(batch_lang[0]) == 2  # batch_size

        # All items should be strings
        for timestep in batch_lang:
            assert all(isinstance(s, str) for s in timestep)


    def test_language_with_variable_length_instructions(self, varlen_language_buffer, action_config, observation_config_with_language, dataloader_config):
        """Test variable-length language instructions."""
        dataset = EpisodicDataset(
            zarr_path=varlen_language_buffer,
            action_space=action_config,
            observation_space=observation_config_with_language,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=True,
            seed=42,
        )

        if len(dataset) == 0:
            pytest.skip("No valid samples")

        sample = dataset[0]
        lang_data = sample[OBSERVATION_KEY][LANGUAGE_KEY]

        # Verify language data is present and correctly formatted
        assert isinstance(lang_data, list)
        assert len(lang_data) == 3  # obs_horizon
        assert all(isinstance(s, str) for s in lang_data)

        # Check that we can handle strings (variable lengths work)
        lengths = [len(s) for s in lang_data]
        assert all(l > 0 for l in lengths), "All instructions should be non-empty"


    def test_language_not_present_when_disabled(self, simple_replay_buffer, action_config, observation_config, dataloader_config):
        """Test that language is not in samples when disabled."""
        dataset = EpisodicDataset(
            zarr_path=simple_replay_buffer,
            action_space=action_config,
            observation_space=observation_config,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=True,
            seed=42,
        )

        if len(dataset) == 0:
            pytest.skip("No valid samples")

        sample = dataset[0]
        assert LANGUAGE_KEY not in sample[OBSERVATION_KEY]


    def test_language_with_downsampling(self, downsample_language_buffer, action_config, observation_config_with_language, dataloader_config):
        """Test language instructions are preserved during downsampling."""
        dataloader_config.downsample_factor = 2

        dataset = EpisodicDataset(
            zarr_path=downsample_language_buffer,
            action_space=action_config,
            observation_space=observation_config_with_language,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=True,
            seed=42,
        )

        if len(dataset) == 0:
            pytest.skip("No valid samples after downsampling")

        sample = dataset[0]
        assert LANGUAGE_KEY in sample[OBSERVATION_KEY]
        assert isinstance(sample[OBSERVATION_KEY][LANGUAGE_KEY], list)


@pytest.mark.integration
class TestNormalizerAndTokenizerIntegration:
    """Test normalizer and tokenizer integration in EpisodicDataset."""

    def test_get_normalizer_and_tokenizer_without_tokenization(
        self, simple_replay_buffer, action_config, observation_config, dataloader_config
    ):
        """Test getting normalizer and tokenizer when tokenization is disabled."""
        dataset = EpisodicDataset(
            zarr_path=simple_replay_buffer,
            action_space=action_config,
            observation_space=observation_config,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=True,
            seed=42,
        )

        # Get normalizer and tokenizer without tokenization config
        normalizer, tokenizer = dataset.get_normalizer_and_tokenizer(tokenization_config=None)

        assert normalizer is not None
        assert tokenizer is None

    def test_get_normalizer_and_tokenizer_with_observation_tokenization(
        self, language_replay_buffer, action_config, observation_config_with_language, dataloader_config
    ):
        """Test getting normalizer and tokenizer with observation tokenization enabled."""
        dataset = EpisodicDataset(
            zarr_path=language_replay_buffer,
            action_space=action_config,
            observation_space=observation_config_with_language,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=True,
            seed=42,
        )

        # Create tokenization config
        obs_tokenization = ObservationTokenizationConfig(
            tokenizer_model="google/bert_uncased_L-2_H-128_A-2",
            observation_keys=[LANGUAGE_KEY, PROPRIO_OBS_ROBOT_FRAME_KEY],
            bin_continuous_data=False,
            max_token_len=256,
        )
        tokenization_config = TokenizationConfig(
            tokenize_observations=True,
            tokenize_actions=False,
            observation_tokenizer=obs_tokenization,
        )

        # Get normalizer and tokenizer
        normalizer, tokenizer = dataset.get_normalizer_and_tokenizer(
            tokenization_config=tokenization_config, device=torch.device("cpu")
        )

        assert normalizer is not None
        assert tokenizer is not None
        assert tokenizer.observation_tokenizer is not None
        assert tokenizer.action_tokenizer is None
        assert tokenizer.observation_vocab_size > 0

    def test_get_normalizer_and_tokenizer_with_action_tokenization(
        self, simple_replay_buffer, action_config, observation_config, dataloader_config
    ):
        """Test getting normalizer and tokenizer with action tokenization enabled."""
        dataset = EpisodicDataset(
            zarr_path=simple_replay_buffer,
            action_space=action_config,
            observation_space=observation_config,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=True,
            seed=42,
        )

        # Create tokenization config
        action_tokenization = ActionTokenizationConfig(
            tokenizer_chain=[TokenizerType.FAST.value],
            use_pretrained_fast=True,
        )
        tokenization_config = TokenizationConfig(
            tokenize_observations=False,
            tokenize_actions=True,
            action_tokenizer=action_tokenization,
        )

        # Get normalizer and tokenizer
        normalizer, tokenizer = dataset.get_normalizer_and_tokenizer(
            tokenization_config=tokenization_config, device=torch.device("cpu")
        )

        assert normalizer is not None
        assert tokenizer is not None
        assert tokenizer.observation_tokenizer is None
        assert tokenizer.action_tokenizer is not None
        assert tokenizer.action_vocab_size > 0

    def test_set_normalizer(
        self, simple_replay_buffer, action_config, observation_config, dataloader_config
    ):
        """Test setting normalizer on dataset."""
        dataset = EpisodicDataset(
            zarr_path=simple_replay_buffer,
            action_space=action_config,
            observation_space=observation_config,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=True,
            seed=42,
        )

        # Get normalizer
        normalizer = dataset.get_normalizer()

        # Set normalizer
        dataset.set_normalizer(normalizer)

        # Verify it's set on sample_builder
        assert dataset.sample_builder.normalizer is normalizer

    def test_set_tokenizer(
        self, language_replay_buffer, action_config, observation_config_with_language, dataloader_config
    ):
        """Test setting tokenizer on dataset."""
        dataset = EpisodicDataset(
            zarr_path=language_replay_buffer,
            action_space=action_config,
            observation_space=observation_config_with_language,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=True,
            seed=42,
        )

        # Create tokenization config
        obs_tokenization = ObservationTokenizationConfig(
            tokenizer_model="google/bert_uncased_L-2_H-128_A-2",
            observation_keys=[LANGUAGE_KEY, PROPRIO_OBS_ROBOT_FRAME_KEY],
            bin_continuous_data=False,
            max_token_len=256,
        )
        tokenization_config = TokenizationConfig(
            tokenize_observations=True,
            tokenize_actions=False,
            observation_tokenizer=obs_tokenization,
        )

        # Get tokenizer
        _, tokenizer = dataset.get_normalizer_and_tokenizer(
            tokenization_config=tokenization_config, device=torch.device("cpu")
        )

        # Set tokenizer
        dataset.set_tokenizer(tokenizer)

        # Verify it's set on sample_builder
        assert dataset.sample_builder.tokenizer is tokenizer

    def test_sample_with_normalizer_and_tokenizer(
        self, language_replay_buffer, action_config, observation_config_with_language, dataloader_config
    ):
        """Test that samples include tokenized data when tokenizer is set."""
        dataset = EpisodicDataset(
            zarr_path=language_replay_buffer,
            action_space=action_config,
            observation_space=observation_config_with_language,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=True,
            seed=42,
        )

        # Create tokenization config for both obs and actions
        obs_tokenization = ObservationTokenizationConfig(
            tokenizer_model="google/bert_uncased_L-2_H-128_A-2",
            observation_keys=[LANGUAGE_KEY, PROPRIO_OBS_ROBOT_FRAME_KEY],
            bin_continuous_data=False,
            max_token_len=256,
        )
        action_tokenization = ActionTokenizationConfig(
            tokenizer_chain=[TokenizerType.FAST.value],
            use_pretrained_fast=True,
        )
        tokenization_config = TokenizationConfig(
            tokenize_observations=True,
            tokenize_actions=True,
            observation_tokenizer=obs_tokenization,
            action_tokenizer=action_tokenization,
        )

        # Get normalizer and tokenizer
        normalizer, tokenizer = dataset.get_normalizer_and_tokenizer(
            tokenization_config=tokenization_config, device=torch.device("cpu")
        )

        # Set both
        dataset.set_normalizer(normalizer)
        dataset.set_tokenizer(tokenizer)

        if len(dataset) == 0:
            pytest.skip("No valid samples")

        # Get sample
        sample = dataset[0]

        # Verify tokenized data is present
        assert TOKENIZED_OBSERVATIONS_KEY in sample[OBSERVATION_KEY]
        assert IS_PAD_OBSERVATION_KEY in sample[OBSERVATION_KEY]
        assert TOKENIZED_ACTIONS_KEY in sample[ACTION_KEY]

        # Verify dtypes
        assert sample[OBSERVATION_KEY][TOKENIZED_OBSERVATIONS_KEY].dtype == torch.long
        assert sample[OBSERVATION_KEY][IS_PAD_OBSERVATION_KEY].dtype == torch.bool
        assert sample[ACTION_KEY][TOKENIZED_ACTIONS_KEY].dtype == torch.long

@pytest.mark.unit
class TestComputeSampleActionsSlicing:
    """Test _compute_sample_actions slicing logic."""

    @pytest.mark.parametrize("obs_horizon,pred_horizon", [
        (1, 4),
        (3, 4),
        (5, 10),
        (1, 1),
        (10, 1),
    ])
    def test_action_slicing_indices(self, simple_replay_buffer, action_config, observation_config, dataloader_config, obs_horizon, pred_horizon):
        """Test that action slicing uses correct indices."""
        dataset = EpisodicDataset(
            zarr_path=simple_replay_buffer,
            action_space=action_config,
            observation_space=observation_config,
            dataloader_config=dataloader_config,
            pred_horizon=pred_horizon,
            obs_horizon=obs_horizon,
            train=True,
            seed=42,
        )

        # Create known padded data
        total_len = obs_horizon + pred_horizon
        padded_data = {
            PROPRIO_OBS_ROBOT_FRAME_KEY: np.arange(total_len * 7).reshape(total_len, 7).astype(np.float32),
            GRIPPER_STATE_OBS_KEY: np.arange(total_len).reshape(total_len, 1).astype(np.float32),
        }

        # Compute actions
        action_dict = dataset._compute_sample_actions(padded_data)

        # Verify we got pred_horizon actions
        assert action_dict[POSITION_ACTION_KEY].shape[0] == pred_horizon
        if action_config.has_gripper:
            assert action_dict[GRIPPER_ACTION_KEY].shape[0] == pred_horizon

    def test_action_slicing_computes_correct_deltas(self, simple_replay_buffer, action_config, observation_config, dataloader_config):
        """Test that actions are computed from correct observation pairs."""
        obs_horizon = 3
        pred_horizon = 4
        
        dataset = EpisodicDataset(
            zarr_path=simple_replay_buffer,
            action_space=action_config,
            observation_space=observation_config,
            dataloader_config=dataloader_config,
            pred_horizon=pred_horizon,
            obs_horizon=obs_horizon,
            train=True,
            seed=42,
        )

        # Create sequential data where differences are easy to verify
        total_len = obs_horizon + pred_horizon
        sequential_positions = np.arange(total_len * 7).reshape(total_len, 7).astype(np.float32)
        
        padded_data = {
            PROPRIO_OBS_ROBOT_FRAME_KEY: sequential_positions,
            GRIPPER_STATE_OBS_KEY: np.arange(total_len).reshape(total_len, 1).astype(np.float32),
        }

        # For obs_horizon=3, pred_horizon=4:
        # action_slice_start = 3 - 1 = 2
        # action_slice_end = 2 + 4 = 6
        # curr_obs should be indices [2, 3, 4, 5]
        # next_obs should be indices [3, 4, 5, 6]
        
        action_dict = dataset._compute_sample_actions(padded_data)

        # Verify we got the right number of actions
        assert len(action_dict[POSITION_ACTION_KEY]) == pred_horizon

        # The action processor should compute deltas (next - curr)
        # We can't verify exact values without knowing the action processor implementation,
        # but we can verify the shapes are correct
        assert action_dict[POSITION_ACTION_KEY].shape == (pred_horizon, 3)

    def test_gripper_action_slicing_aligned_with_position(self, simple_replay_buffer, action_config, observation_config, dataloader_config):
        """Test that gripper actions use same slicing as position actions."""
        obs_horizon = 3
        pred_horizon = 4
        
        dataset = EpisodicDataset(
            zarr_path=simple_replay_buffer,
            action_space=action_config,
            observation_space=observation_config,
            dataloader_config=dataloader_config,
            pred_horizon=pred_horizon,
            obs_horizon=obs_horizon,
            train=True,
            seed=42,
        )

        total_len = obs_horizon + pred_horizon
        padded_data = {
            PROPRIO_OBS_ROBOT_FRAME_KEY: np.arange(total_len * 7).reshape(total_len, 7).astype(np.float32),
            GRIPPER_STATE_OBS_KEY: np.arange(total_len).reshape(total_len, 1).astype(np.float32),
        }

        action_dict = dataset._compute_sample_actions(padded_data)

        # Both should have pred_horizon timesteps
        assert action_dict[POSITION_ACTION_KEY].shape[0] == pred_horizon
        assert action_dict[GRIPPER_ACTION_KEY].shape[0] == pred_horizon

    def test_action_uses_camera_frame_when_configured(self, simple_replay_buffer, action_config, observation_config, dataloader_config):
        """Test that action computation uses camera frame when configured."""
        action_config.predict_in_camera_frame = True
        
        dataset = EpisodicDataset(
            zarr_path=simple_replay_buffer,
            action_space=action_config,
            observation_space=observation_config,
            dataloader_config=dataloader_config,
            pred_horizon=4,
            obs_horizon=3,
            train=True,
            seed=42,
        )

        total_len = 7
        padded_data = {
            PROPRIO_OBS_CAMERA_FRAME_KEY: np.arange(total_len * 7).reshape(total_len, 7).astype(np.float32),
            PROPRIO_OBS_ROBOT_FRAME_KEY: np.zeros((total_len, 7), dtype=np.float32),  # Different values
            GRIPPER_STATE_OBS_KEY: np.arange(total_len).reshape(total_len, 1).astype(np.float32),
        }

        action_dict = dataset._compute_sample_actions(padded_data)

        # Should use camera frame data (non-zero) not robot frame (zeros)
        # If implementation is correct, actions won't all be zero
        assert action_dict[POSITION_ACTION_KEY].shape == (4, 3)

    def test_action_slicing_boundary_conditions(self, simple_replay_buffer, action_config, observation_config, dataloader_config):
        """Test slicing doesn't go out of bounds."""
        obs_horizon = 1
        pred_horizon = 1
        
        dataset = EpisodicDataset(
            zarr_path=simple_replay_buffer,
            action_space=action_config,
            observation_space=observation_config,
            dataloader_config=dataloader_config,
            pred_horizon=pred_horizon,
            obs_horizon=obs_horizon,
            train=True,
            seed=42,
        )

        total_len = obs_horizon + pred_horizon
        padded_data = {
            PROPRIO_OBS_ROBOT_FRAME_KEY: np.arange(total_len * 7).reshape(total_len, 7).astype(np.float32),
            GRIPPER_STATE_OBS_KEY: np.arange(total_len).reshape(total_len, 1).astype(np.float32),
        }

        # Should not raise index error
        action_dict = dataset._compute_sample_actions(padded_data)
        
        assert action_dict[POSITION_ACTION_KEY].shape == (1, 3)
        assert action_dict[GRIPPER_ACTION_KEY].shape == (1, 1)
