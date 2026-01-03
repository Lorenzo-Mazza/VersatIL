import pytest
import numpy as np
import torch
from unittest.mock import MagicMock

from refactoring.data.sample_builder import SampleBuilder
from refactoring.data.tokenization.tokenizer import Tokenizer
from refactoring.data.tokenization.observation_tokenizer import ObservationTokenizer
from refactoring.data.tokenization.action_tokenizer import ActionTokenizer
from refactoring.data.constants import (
    Cameras,
    OBSERVATION_KEY,
    PHASE_LABEL_KEY,
    IS_PAD_ACTION_KEY,
    IS_PAD_OBSERVATION_KEY,
    PROPRIO_OBS_ROBOT_FRAME_KEY,
    PROPRIO_OBS_CAMERA_FRAME_KEY,
    POSITION_ACTION_KEY,
    ORIENTATION_ACTION_KEY,
    GRIPPER_ACTION_KEY,
    TOKENIZED_ACTIONS_KEY,
    TOKENIZED_OBSERVATIONS_KEY,
    GripperType,
    LANGUAGE_KEY,
    ACTION_KEY,
    TokenizerType,
)


@pytest.fixture
def action_config():
    """Action space configuration."""
    config = MagicMock()
    config.predict_in_camera_frame = False
    config.deltas_as_actions = True
    config.has_gripper = True
    config.gripper_type = GripperType.BINARY.value
    config.task_has_phases = False
    return config


@pytest.fixture
def observation_config():
    """Observation space configuration."""
    config = MagicMock()
    config.camera_keys = [Cameras.LEFT.value, Cameras.RIGHT.value]
    config.use_proprio_base_frame = True
    config.use_proprio_camera_frame = False
    config.use_language = False
    return config


@pytest.fixture
def mock_augmentation_pipeline():
    """Mock augmentation pipeline."""
    pipeline = MagicMock()
    pipeline.setup_rotation.return_value = (0, None)  # No rotation by default
    pipeline.apply_rgb_augmentations.side_effect = lambda x, angle: x
    pipeline.apply_depth_augmentations.side_effect = lambda x, angle: x
    pipeline.rotate_proprio_data.side_effect = lambda x, R: x
    return pipeline


@pytest.fixture
def mock_action_processor():
    """Mock action processor."""
    processor = MagicMock()
    processor.rotate_actions.side_effect = lambda action_dict, R: action_dict
    return processor


@pytest.fixture
def sample_builder(action_config, observation_config, mock_augmentation_pipeline, mock_action_processor):
    """SampleBuilder instance."""
    return SampleBuilder(
        action_space=action_config,
        observation_space=observation_config,
        obs_horizon=3,
        pred_horizon=4,
        action_backward_shift=0,
        augmentation_pipeline=mock_augmentation_pipeline,
        action_processor=mock_action_processor,
    )


@pytest.fixture
def padded_data():
    """Padded episode data."""
    return {
        Cameras.LEFT.value: np.ones((10, 32, 32, 3), dtype=np.uint8) * 128,
        Cameras.RIGHT.value: np.ones((10, 32, 32, 3), dtype=np.uint8) * 64,
        PROPRIO_OBS_ROBOT_FRAME_KEY: np.arange(70, dtype=np.float32).reshape(10, 7),
        PROPRIO_OBS_CAMERA_FRAME_KEY: np.arange(70, dtype=np.float32).reshape(10, 7),
    }


@pytest.fixture
def action_dict():
    """Action dictionary."""
    return {
        POSITION_ACTION_KEY: np.ones((4, 3), dtype=np.float32) * 0.1,
        ORIENTATION_ACTION_KEY: np.ones((4, 4), dtype=np.float32) * 0.2,
        GRIPPER_ACTION_KEY: np.array([[1], [0], [1], [0]], dtype=np.float32),
    }


@pytest.fixture
def sampler_indices():
    """Sampler indices array."""
    # (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
    indices = np.zeros((10, 4), dtype=np.int64)
    for i in range(10):
        indices[i] = [i, i + 6, 2, 5]
    return indices


@pytest.fixture
def dummy_normalizer():
    """Dummy normalizer that acts as pass-through."""
    class DummyNormalizer(torch.nn.Module):
        """Dummy normalizer that acts as pass-through (no normalization)."""

        def __init__(self):
            super().__init__()
            self.params_dict = torch.nn.ParameterDict()

        def __getitem__(self, key):
            """Support subscripting to match LinearNormalizer API."""
            return self

        def normalize(self, x):
            """Pass-through normalization (identity function)."""
            return x

        def unnormalize(self, x):
            """Pass-through unnormalization (identity function)."""
            return x

        def unnormalize_actions(self, x):
            """Pass-through action unnormalization (identity function)."""
            return x

    return DummyNormalizer()


class TestSampleBuilderInitialization:
    """Test SampleBuilder initialization."""


    def test_init_basic(self, action_config, observation_config, mock_augmentation_pipeline, mock_action_processor):
        """Test basic initialization."""
        builder = SampleBuilder(
            action_space=action_config,
            observation_space=observation_config,
            obs_horizon=5,
            pred_horizon=10,
            action_backward_shift=1,
            augmentation_pipeline=mock_augmentation_pipeline,
            action_processor=mock_action_processor,
        )

        assert builder.obs_horizon == 5
        assert builder.pred_horizon == 10
        assert builder.action_backward_shift == 1
        assert builder.action_space == action_config
        assert builder.observation_space == observation_config


class TestBuildSample:
    """Test complete sample building."""


    def test_build_sample_basic(self, sample_builder, padded_data, action_dict, sampler_indices):
        """Test basic sample building."""
        sample = sample_builder.build_sample(
            padded_data=padded_data,
            action_dict=action_dict,
            start_idx=0,
            sampler_indices=sampler_indices,
        )

        assert OBSERVATION_KEY in sample
        assert ACTION_KEY in sample
        assert POSITION_ACTION_KEY in sample[ACTION_KEY]
        assert ORIENTATION_ACTION_KEY in sample[ACTION_KEY]
        assert GRIPPER_ACTION_KEY in sample[ACTION_KEY]
        assert IS_PAD_ACTION_KEY in sample[ACTION_KEY]


    def test_build_sample_with_rotation(self, sample_builder, padded_data, action_dict, sampler_indices):
        """Test sample building with rotation augmentation."""
        sample_builder.augmentation_pipeline.setup_rotation.return_value = (45.0, np.eye(3))
        sample_builder.action_space.predict_in_camera_frame = True

        sample = sample_builder.build_sample(
            padded_data=padded_data,
            action_dict=action_dict,
            start_idx=0,
            sampler_indices=sampler_indices,
        )

        # Should call rotate_actions when angle != 0 and in camera frame
        sample_builder.action_processor.rotate_actions.assert_called_once()


    def test_build_sample_no_rotation_in_robot_frame(self, sample_builder, padded_data, action_dict, sampler_indices):
        """Test that rotation is not applied in robot frame."""
        sample_builder.augmentation_pipeline.setup_rotation.return_value = (45.0, np.eye(3))
        sample_builder.action_space.predict_in_camera_frame = False

        sample = sample_builder.build_sample(
            padded_data=padded_data,
            action_dict=action_dict,
            start_idx=0,
            sampler_indices=sampler_indices,
        )

        # Should NOT call rotate_actions when in robot frame
        sample_builder.action_processor.rotate_actions.assert_not_called()


    def test_build_sample_with_phases(self, sample_builder, padded_data, action_dict, sampler_indices):
        """Test sample building with phase labels."""
        sample_builder.action_space.predict_task_phases = True
        padded_data[PHASE_LABEL_KEY] = np.array([[0], [1], [1], [2], [2], [3], [3], [4], [4], [0]], dtype=np.int64)

        sample = sample_builder.build_sample(
            padded_data=padded_data,
            action_dict=action_dict,
            start_idx=0,
            sampler_indices=sampler_indices,
        )

        assert PHASE_LABEL_KEY in sample[ACTION_KEY]
        assert sample[ACTION_KEY][PHASE_LABEL_KEY].dtype == torch.long


    def test_build_sample_without_proprio(self, sample_builder, padded_data, action_dict, sampler_indices):
        """Test sample building without proprioceptive data."""
        sample_builder.observation_space.use_proprioceptive_data = False

        sample = sample_builder.build_sample(
            padded_data=padded_data,
            action_dict=action_dict,
            start_idx=0,
            sampler_indices=sampler_indices,
        )

        assert PROPRIO_OBS_CAMERA_FRAME_KEY not in sample[OBSERVATION_KEY]
        assert PROPRIO_OBS_ROBOT_FRAME_KEY not in sample[OBSERVATION_KEY]


class TestAddImages:
    """Test image processing and addition."""


    def test_add_rgb_images(self, sample_builder, padded_data):
        """Test RGB image processing."""
        sample = {OBSERVATION_KEY: {}}
        sample_builder._get_sample_images(sample, padded_data, angle=0)

        assert Cameras.LEFT.value in sample[OBSERVATION_KEY]
        assert Cameras.RIGHT.value in sample[OBSERVATION_KEY]

        # Check shape: (T, C, H, W)
        left_img = sample[OBSERVATION_KEY][Cameras.LEFT.value]
        assert left_img.shape == (3, 3, 32, 32)  # obs_horizon=3

        # Check normalization [0, 255] -> [0, 1]
        assert left_img.dtype == torch.float32
        assert left_img.max() <= 1.0
        assert left_img.min() >= 0.0


    def test_add_depth_images(self, sample_builder, padded_data):
        """Test depth image processing."""
        sample_builder.observation_space.camera_keys = [Cameras.DEPTH.value]
        padded_data[Cameras.DEPTH.value] = np.ones((10, 32, 32), dtype=np.float32) * 2.5

        sample = {OBSERVATION_KEY: {}}
        sample_builder._get_sample_images(sample, padded_data, angle=0)

        assert Cameras.DEPTH.value in sample[OBSERVATION_KEY]

        # Check shape: (T, 1, H, W)
        depth_img = sample[OBSERVATION_KEY][Cameras.DEPTH.value]
        assert depth_img.shape == (3, 1, 32, 32)
        assert depth_img.dtype == torch.float32


    def test_add_images_with_rotation(self, sample_builder, padded_data):
        """Test image processing with rotation."""
        sample = {OBSERVATION_KEY: {}}
        sample_builder._get_sample_images(sample, padded_data, angle=45.0)

        # Should call augmentation pipeline with angle
        assert sample_builder.augmentation_pipeline.apply_rgb_augmentations.call_count == 2


    def test_add_images_respects_obs_horizon(self, sample_builder, padded_data):
        """Test that only obs_horizon frames are included."""
        sample_builder.obs_horizon = 2

        sample = {OBSERVATION_KEY: {}}
        sample_builder._get_sample_images(sample, padded_data, angle=0)

        left_img = sample[OBSERVATION_KEY][Cameras.LEFT.value]
        assert left_img.shape[0] == 2  # Only 2 timesteps


class TestAddProprioceptive:
    """Test proprioceptive data addition."""


    def test_add_proprio_robot_frame(self, sample_builder, padded_data):
        """Test adding robot frame proprioceptive data."""
        sample = {OBSERVATION_KEY: {}}
        sample_builder._add_proprioceptive(sample, padded_data, angle=0, rotation_matrix=None)

        assert PROPRIO_OBS_ROBOT_FRAME_KEY in sample[OBSERVATION_KEY]
        # Camera frame should NOT be present since use_proprio_camera_frame=False in fixture
        assert PROPRIO_OBS_CAMERA_FRAME_KEY not in sample[OBSERVATION_KEY]

        proprio = sample[OBSERVATION_KEY][PROPRIO_OBS_ROBOT_FRAME_KEY]
        assert proprio.shape == (3, 7)  # obs_horizon=3, dim=7
        assert proprio.dtype == torch.float32


    def test_add_proprio_camera_frame(self, sample_builder, padded_data):
        """Test adding camera frame proprioceptive data."""
        sample_builder.observation_space.use_proprio_camera_frame = True

        sample = {OBSERVATION_KEY: {}}
        sample_builder._add_proprioceptive(sample, padded_data, angle=0, rotation_matrix=None)
        assert PROPRIO_OBS_CAMERA_FRAME_KEY in sample[OBSERVATION_KEY]


    def test_add_proprio_camera_frame_with_rotation(self, sample_builder, padded_data):
        """Test camera frame rotation."""
        sample_builder.observation_space.use_proprio_camera_frame = True
        rotation_matrix = np.eye(3, dtype=np.float32)

        sample = {OBSERVATION_KEY: {}}
        sample_builder._add_proprioceptive(sample, padded_data, angle=45.0, rotation_matrix=rotation_matrix)

        # Should call rotate_proprio_data
        sample_builder.augmentation_pipeline.rotate_proprio_data.assert_called_once()


    def test_add_proprio_camera_frame_no_rotation_when_angle_zero(self, sample_builder, padded_data):
        """Test that camera frame is not rotated when angle is zero."""
        sample_builder.observation_space.use_proprio_camera_frame = True

        sample = {OBSERVATION_KEY: {}}
        sample_builder._add_proprioceptive(sample, padded_data, angle=0, rotation_matrix=None)

        # Should NOT call rotate_proprio_data when angle=0
        sample_builder.augmentation_pipeline.rotate_proprio_data.assert_not_called()


    def test_add_proprio_both_frames(self, sample_builder, padded_data):
        """Test adding both robot and camera frames."""
        sample_builder.observation_space.use_proprio_camera_frame = True

        sample = {OBSERVATION_KEY: {}}
        sample_builder._add_proprioceptive(sample, padded_data, angle=0, rotation_matrix=None)
        assert PROPRIO_OBS_ROBOT_FRAME_KEY in sample[OBSERVATION_KEY]
        assert PROPRIO_OBS_CAMERA_FRAME_KEY in sample[OBSERVATION_KEY]


class TestAddPhaseLabels:
    """Test phase label addition."""


    def test_add_phase_labels(self, sample_builder, padded_data):
        """Test adding phase labels."""
        padded_data[PHASE_LABEL_KEY] = np.array([[0], [1], [1], [2], [2], [3], [3], [4], [4], [0]], dtype=np.int64)

        sample = {ACTION_KEY:{}}
        sample_builder._add_phase_labels(sample, padded_data)

        assert PHASE_LABEL_KEY in sample[ACTION_KEY]
        assert sample[ACTION_KEY][PHASE_LABEL_KEY].dtype == torch.long


    def test_phase_labels_with_backward_shift(self, sample_builder, padded_data):
        """Test phase labels with action backward shift."""
        sample_builder.action_backward_shift = 1
        padded_data[PHASE_LABEL_KEY] = np.array([[0], [1], [1], [2], [2], [3], [3], [4], [4], [0]], dtype=np.int64)

        sample = {ACTION_KEY:{}}
        sample_builder._add_phase_labels(sample, padded_data)

        assert sample[ACTION_KEY][PHASE_LABEL_KEY].shape == torch.Size([4, 1])


class TestAddPaddingMask:
    """Test padding mask computation."""


    def test_padding_mask_end_of_episode(self, sample_builder):
        """Test padding mask at episode end with deltas."""
        indices = np.array([[0, 10, 2, 5]], dtype=np.int64)  # valid range [2, 5)
        sample_builder.pred_horizon = 6

        sample = {ACTION_KEY:{}}
        sample_builder._compute_action_padding_mask(sample, start_idx=0, sampler_indices=indices)

        # action_slice_start = 2
        # action_positions = [2, 3, 4, 5, 6, 7]
        # For deltas: need both position AND next position valid
        # Valid range is [2, 5), so positions 2,3,4 are valid
        is_pad = sample[ACTION_KEY][IS_PAD_ACTION_KEY].numpy()
        assert not is_pad[0]  # Position 2 valid, next (3) valid
        assert not is_pad[1]  # Position 3 valid, next (4) valid
        assert is_pad[2]  # Position 4 valid, but next (5) >= 5 (invalid)
        assert is_pad[3]  # Position 5 >= 5 (both invalid)
        assert is_pad[4]  # Position 6 >= 5 (both invalid)


    def test_padding_mask_delta_vs_absolute_distinction(self, sample_builder):
        """Test that deltas require both positions valid, absolute only next."""
        # Key: make action_position invalid but next position valid
        indices = np.array([[0, 10, 3, 6]], dtype=np.int64)  # valid range [3, 6)

        # action_slice_start = 2
        # action_positions = [2, 3, 4, 5]

        # Delta actions: need both current AND next
        sample_builder.action_space.deltas_as_actions = True
        sample = {ACTION_KEY:{}}
        sample_builder._compute_action_padding_mask(sample, start_idx=0, sampler_indices=indices)
        delta_pad = sample[ACTION_KEY][IS_PAD_ACTION_KEY].numpy().copy()

        # Absolute actions: only need next
        sample_builder.action_space.deltas_as_actions = False
        sample = {ACTION_KEY:{}}
        sample_builder._compute_action_padding_mask(sample, start_idx=0, sampler_indices=indices)
        absolute_pad = sample[ACTION_KEY][IS_PAD_ACTION_KEY].numpy()

        # Position 0: action_position=2 (invalid), next=3 (valid at boundary)
        assert delta_pad[0] == True  # Delta: 2 < 3, so padded
        assert absolute_pad[0] == False  # Absolute: only needs 3, which is valid

        # Position 1: action_position=3 (valid), next=4 (valid)
        assert delta_pad[1] == False
        assert absolute_pad[1] == False

        # Position 3: action_position=5 (valid), next=6 (invalid, at boundary)
        assert delta_pad[3] == True
        assert absolute_pad[3] == True  # Both padded when next is invalid


    def test_padding_mask_start_of_episode_absolute_vs_delta(self, sample_builder):
        """Test boundary behavior at episode start."""
        indices = np.array([[0, 10, 3, 8]], dtype=np.int64)

        # Test with deltas
        sample_builder.action_space.deltas_as_actions = True
        sample = {ACTION_KEY:{}}
        sample_builder._compute_action_padding_mask(sample, start_idx=0, sampler_indices=indices)
        delta_result = sample[ACTION_KEY][IS_PAD_ACTION_KEY].numpy()

        # Test with absolute
        sample_builder.action_space.deltas_as_actions = False
        sample = {ACTION_KEY:{}}
        sample_builder._compute_action_padding_mask(sample, start_idx=0, sampler_indices=indices)
        absolute_result = sample[ACTION_KEY][IS_PAD_ACTION_KEY].numpy()

        # action_positions = [2, 3, 4, 5]
        # Position 2 < 3 (start), so current is invalid but next (3) is valid
        assert delta_result[0] == True  # Needs current (2), which is invalid
        assert absolute_result[0] == False  # Only needs next (3), which is valid


    def test_padding_mask_deltas_no_padding(self, sample_builder, sampler_indices):
        """Test padding mask for deltas with no padding needed."""
        sample = {ACTION_KEY:{}}
        sample_builder._compute_action_padding_mask(sample, start_idx=0, sampler_indices=sampler_indices)

        assert IS_PAD_ACTION_KEY in sample[ACTION_KEY]
        assert sample[ACTION_KEY][IS_PAD_ACTION_KEY].dtype == torch.bool
        assert sample[ACTION_KEY][IS_PAD_ACTION_KEY].shape == torch.Size([4])  # pred_horizon=4

        # Check specific values
        is_pad = sample[ACTION_KEY][IS_PAD_ACTION_KEY].numpy()
        # action_slice_start = 2, action_positions = [2, 3, 4, 5]
        # sample_start_idx=2, sample_end_idx=5 (valid range is [2, 5))
        # For deltas: both current and next positions must be valid
        expected = np.array([False, False, True, True])  # position 4,5 are out of range so they are padded
        np.testing.assert_array_equal(is_pad, expected)


    def test_padding_mask_absolute_no_padding(self, sample_builder, sampler_indices):
        """Test padding mask for absolute actions with no padding needed."""
        sample_builder.action_space.deltas_as_actions = False

        sample = {ACTION_KEY:{}}
        sample_builder._compute_action_padding_mask(sample, start_idx=0, sampler_indices=sampler_indices)

        # With absolute actions, only next position needs to be valid
        # action_positions = [2, 3, 4, 5]
        # next positions = [3, 4, 5, 6]
        # sample_end_idx=5, so positions 3,4 are valid, 5,6 are padding
        expected = np.array([False, False, True, True])
        np.testing.assert_array_equal(sample[ACTION_KEY][IS_PAD_ACTION_KEY].numpy(), expected)


    def test_padding_mask_with_backward_shift(self, sample_builder, sampler_indices):
        """Test padding mask with action backward shift."""
        sample_builder.action_backward_shift = 1

        sample = {ACTION_KEY:{}}
        sample_builder._compute_action_padding_mask(sample, start_idx=0, sampler_indices=sampler_indices)

        # action_slice_start = 3 - 1 - 1 = 1
        # action_positions = [1, 2, 3, 4]
        assert sample[ACTION_KEY][IS_PAD_ACTION_KEY].shape == torch.Size([4])  # pred_horizon=4


class TestGetActionSliceStart:
    """Test action slice start computation."""


    def test_get_action_slice_start_no_shift(self, sample_builder):
        """Test action slice start without backward shift."""
        start = sample_builder._get_action_slice_start()
        assert start == 2  # obs_horizon - 1 = 3 - 1


    def test_get_action_slice_start_with_shift(self, sample_builder):
        """Test action slice start with backward shift."""
        sample_builder.action_backward_shift = 1
        start = sample_builder._get_action_slice_start()
        assert start == 2  # obs_horizon - 1 = 3 - 1


    def test_get_action_slice_start_large_horizon(self, sample_builder):
        """Test action slice start with large obs horizon."""
        sample_builder.obs_horizon = 10
        start = sample_builder._get_action_slice_start()
        assert start == 9  # 10 - 1 - 0



class TestActionConversion:
    """Test action dictionary to tensor conversion."""


    def test_binary_gripper_converted_to_long(self, sample_builder, padded_data, sampler_indices):
        """Test binary gripper actions are converted to long."""
        action_dict = {
            POSITION_ACTION_KEY: np.ones((4, 3), dtype=np.float32),
            GRIPPER_ACTION_KEY: np.array([[1], [0], [1], [0]], dtype=np.float32),
        }

        sample = sample_builder.build_sample(
            padded_data=padded_data,
            action_dict=action_dict,
            start_idx=0,
            sampler_indices=sampler_indices,
        )

        assert sample[ACTION_KEY][GRIPPER_ACTION_KEY].dtype == torch.long


    def test_continuous_gripper_converted_to_float(self, sample_builder, padded_data, sampler_indices):
        """Test continuous gripper actions are converted to float."""
        sample_builder.action_space.gripper_type = GripperType.CONTINUOUS.value

        action_dict = {
            POSITION_ACTION_KEY: np.ones((4, 3), dtype=np.float32),
            GRIPPER_ACTION_KEY: np.array([[0.5], [0.3], [0.8], [0.1]], dtype=np.float32),
        }

        sample = sample_builder.build_sample(
            padded_data=padded_data,
            action_dict=action_dict,
            start_idx=0,
            sampler_indices=sampler_indices,
        )

        assert sample[ACTION_KEY][GRIPPER_ACTION_KEY].dtype == torch.float32


    def test_position_actions_converted_to_float(self, sample_builder, padded_data, action_dict, sampler_indices):
        """Test position actions are converted to float."""
        sample = sample_builder.build_sample(
            padded_data=padded_data,
            action_dict=action_dict,
            start_idx=0,
            sampler_indices=sampler_indices,
        )

        assert sample[ACTION_KEY][POSITION_ACTION_KEY].dtype == torch.float32
        assert sample[ACTION_KEY][ORIENTATION_ACTION_KEY].dtype == torch.float32


class TestIntegration:
    """Integration tests for complete workflows."""


    def test_complete_sample_structure(self, sample_builder, padded_data, action_dict, sampler_indices):
        """Test complete sample has correct structure."""
        sample = sample_builder.build_sample(
            padded_data=padded_data,
            action_dict=action_dict,
            start_idx=0,
            sampler_indices=sampler_indices,
        )

        # Check top-level keys
        assert OBSERVATION_KEY in sample
        assert ACTION_KEY in sample
        assert POSITION_ACTION_KEY in sample[ACTION_KEY]
        assert ORIENTATION_ACTION_KEY in sample[ACTION_KEY]
        assert GRIPPER_ACTION_KEY in sample[ACTION_KEY]
        assert IS_PAD_ACTION_KEY in sample[ACTION_KEY]

        # Check observation structure
        assert isinstance(sample[OBSERVATION_KEY], dict)
        assert Cameras.LEFT.value in sample[OBSERVATION_KEY]
        assert Cameras.RIGHT.value in sample[OBSERVATION_KEY]
        assert PROPRIO_OBS_ROBOT_FRAME_KEY in sample[OBSERVATION_KEY]
        # Camera frame should NOT be present since use_proprio_camera_frame=False in fixture
        assert PROPRIO_OBS_CAMERA_FRAME_KEY not in sample[OBSERVATION_KEY]

        # Check all tensors
        for key in [POSITION_ACTION_KEY, ORIENTATION_ACTION_KEY, IS_PAD_ACTION_KEY]:
            assert isinstance(sample[ACTION_KEY][key], torch.Tensor)


    def test_different_obs_pred_horizons(self, action_config, observation_config,
                                         mock_augmentation_pipeline, mock_action_processor,
                                         padded_data, action_dict, sampler_indices):
        """Test with different observation and prediction horizons."""
        builder = SampleBuilder(
            action_space=action_config,
            observation_space=observation_config,
            obs_horizon=5,
            pred_horizon=8,
            action_backward_shift=0,
            augmentation_pipeline=mock_augmentation_pipeline,
            action_processor=mock_action_processor,
        )

        # Need to adjust action_dict and padded_data for new horizons
        action_dict_large = {
            POSITION_ACTION_KEY: np.ones((8, 3), dtype=np.float32),
            ORIENTATION_ACTION_KEY: np.ones((8, 4), dtype=np.float32),
            GRIPPER_ACTION_KEY: np.ones((8, 1), dtype=np.float32),
        }

        sample = builder.build_sample(
            padded_data=padded_data,
            action_dict=action_dict_large,
            start_idx=0,
            sampler_indices=sampler_indices,
        )

        # Check horizons
        assert sample[OBSERVATION_KEY][Cameras.LEFT.value].shape[0] == 5  # obs_horizon
        assert sample[ACTION_KEY][POSITION_ACTION_KEY].shape[0] == 8  # pred_horizon
        assert sample[ACTION_KEY][IS_PAD_ACTION_KEY].shape[0] == 8


    def test_sample_with_all_features(self, action_config, observation_config,
                                      mock_augmentation_pipeline, mock_action_processor,
                                      padded_data, action_dict, sampler_indices):
        """Test sample with all features enabled."""
        observation_config.use_proprio_camera_frame = True
        observation_config.camera_keys = [Cameras.LEFT.value, Cameras.RIGHT.value, Cameras.DEPTH.value]

        padded_data[Cameras.DEPTH.value] = np.ones((10, 32, 32), dtype=np.float32)
        padded_data[PHASE_LABEL_KEY] = np.array([[0], [1], [1], [2], [2], [3], [3], [4], [4], [0]], dtype=np.int64)

        mock_augmentation_pipeline.setup_rotation.return_value = (30.0, np.eye(3))
        action_config.predict_in_camera_frame = True
        action_config.task_has_phases = True

        builder = SampleBuilder(
            action_space=action_config,
            observation_space=observation_config,
            obs_horizon=3,
            pred_horizon=4,
            action_backward_shift=0,
            augmentation_pipeline=mock_augmentation_pipeline,
            action_processor=mock_action_processor,
        )

        sample = builder.build_sample(
            padded_data=padded_data,
            action_dict=action_dict,
            start_idx=0,
            sampler_indices=sampler_indices,
        )

        # Check all features present
        assert Cameras.DEPTH.value in sample[OBSERVATION_KEY]
        assert PROPRIO_OBS_CAMERA_FRAME_KEY in sample[OBSERVATION_KEY]
        assert PHASE_LABEL_KEY in sample[ACTION_KEY]

        # Verify rotation was called
        mock_action_processor.rotate_actions.assert_called_once()


class TestLanguageInSampleBuilder:
    """Test language instruction handling in SampleBuilder."""


    def test_add_language_converts_to_list(self, sample_builder, padded_data):
        """Test that language is converted from numpy array to list."""
        sample_builder.observation_space.use_language = True
        padded_data[LANGUAGE_KEY] = np.array([
            'pick up the red cube',
            'place it on the table',
            'return to home position',
            'grasp the object carefully',
            'move slowly to the right',
            'open the gripper',
            'close the gripper',
            'push the button',
            'pull the lever',
            'complete the task'
        ], dtype=object)

        sample = {OBSERVATION_KEY: {}}
        sample_builder._add_additional_observation_keys(sample, padded_data)

        # Should be in observations
        assert LANGUAGE_KEY in sample[OBSERVATION_KEY]

        # Should be a list, not numpy array
        lang_data = sample[OBSERVATION_KEY][LANGUAGE_KEY]
        assert isinstance(lang_data, list)

        # Should have obs_horizon elements
        assert len(lang_data) == 3  # obs_horizon=3

        # Each element should be a string
        assert all(isinstance(s, str) for s in lang_data)

        # Check actual values
        assert lang_data[0] == 'pick up the red cube'
        assert lang_data[1] == 'place it on the table'
        assert lang_data[2] == 'return to home position'


    def test_language_with_variable_lengths(self, sample_builder, padded_data):
        """Test language instructions of varying lengths."""
        sample_builder.observation_space.use_language = True
        padded_data[LANGUAGE_KEY] = np.array([
            'short',
            'this is a very long instruction with many many words that describes a complex task in detail',
            'medium instruction',
            'action_embedding',
            'normal length',
            'brief',
            'another long one with lots of descriptive text',
            'ok',
            'standard instruction here',
            'final'
        ], dtype=object)

        sample = {OBSERVATION_KEY: {}}
        sample_builder._add_additional_observation_keys(sample, padded_data)

        lang_data = sample[OBSERVATION_KEY][LANGUAGE_KEY]

        # All should be strings
        assert all(isinstance(s, str) for s in lang_data)

        # Check variable lengths preserved
        assert len(lang_data[0]) < len(lang_data[1])
        assert lang_data[0] == 'short'
        assert 'very long instruction' in lang_data[1]


    def test_language_with_special_characters(self, sample_builder, padded_data):
        """Test language with special characters and punctuation."""
        sample_builder.observation_space.use_language = True
        padded_data[LANGUAGE_KEY] = np.array([
            "pick up the cube & place it (carefully!)",
            "move 5.5cm to the left, then stop",
            "grasp object #3 from the bin",
            "apply 50% force when closing",
            "navigate to position (action_embedding=10, y=20)",
            "warning: fragile item!",
            "task: assembly step 1/5",
            "rotate 90° clockwise",
            "maintain speed ≤ 10cm/s",
            "success criteria: δ < 0.1mm"
        ], dtype=object)

        sample = {OBSERVATION_KEY: {}}
        sample_builder._add_additional_observation_keys(sample, padded_data)

        lang_data = sample[OBSERVATION_KEY][LANGUAGE_KEY]

        # Special characters should be preserved
        assert '&' in lang_data[0]
        assert '!' in lang_data[0]
        assert '5.5cm' in lang_data[1]


    def test_language_not_added_when_disabled(self, sample_builder, padded_data):
        """Test that language is not added when use_language=False."""
        sample_builder.observation_space.use_language = False

        sample = {OBSERVATION_KEY: {}}
        sample_builder._add_additional_observation_keys(sample, padded_data)

        # Language should not be present
        assert LANGUAGE_KEY not in sample[OBSERVATION_KEY]


    def test_build_sample_with_language(self, sample_builder, padded_data, action_dict, sampler_indices):
        """Test complete sample building with language."""
        sample_builder.observation_space.use_language = True
        padded_data[LANGUAGE_KEY] = np.array([
            f'instruction_{i}' for i in range(10)
        ], dtype=object)

        sample = sample_builder.build_sample(
            padded_data=padded_data,
            action_dict=action_dict,
            start_idx=0,
            sampler_indices=sampler_indices,
        )

        # Check complete structure
        assert OBSERVATION_KEY in sample
        assert LANGUAGE_KEY in sample[OBSERVATION_KEY]
        assert isinstance(sample[OBSERVATION_KEY][LANGUAGE_KEY], list)
        assert len(sample[OBSERVATION_KEY][LANGUAGE_KEY]) == 3


    def test_language_with_different_obs_horizons(self, action_config, observation_config,
                                                  mock_augmentation_pipeline, mock_action_processor,
                                                  padded_data, action_dict, sampler_indices):
        """Test language with different observation horizons."""
        observation_config.use_language = True

        # Test with obs_horizon=1
        builder = SampleBuilder(
            action_space=action_config,
            observation_space=observation_config,
            obs_horizon=1,
            pred_horizon=4,
            action_backward_shift=0,
            augmentation_pipeline=mock_augmentation_pipeline,
            action_processor=mock_action_processor,
        )

        padded_data[LANGUAGE_KEY] = np.array([f'inst_{i}' for i in range(10)], dtype=object)

        sample = {OBSERVATION_KEY: {}}
        builder._add_additional_observation_keys(sample, padded_data)

        # Should have only 1 instruction
        assert len(sample[OBSERVATION_KEY][LANGUAGE_KEY]) == 1

        # Test with obs_horizon=5
        builder.obs_horizon = 5
        sample = {OBSERVATION_KEY: {}}
        builder._add_additional_observation_keys(sample, padded_data)

        # Should have 5 instructions
        assert len(sample[OBSERVATION_KEY][LANGUAGE_KEY]) == 5


    def test_language_empty_strings_handled(self, sample_builder, padded_data):
        """Test that empty language strings are handled."""
        sample_builder.observation_space.use_language = True
        padded_data[LANGUAGE_KEY] = np.array([
            'valid instruction',
            '',  # Empty string
            'another valid one',
            '',
            'final instruction',
            'normal',
            '',
            'ok',
            'last',
            ''
        ], dtype=object)

        sample = {OBSERVATION_KEY: {}}
        sample_builder._add_additional_observation_keys(sample, padded_data)

        lang_data = sample[OBSERVATION_KEY][LANGUAGE_KEY]

        # Should handle empty strings gracefully
        assert lang_data[0] == 'valid instruction'
        assert lang_data[1] == ''
        assert lang_data[2] == 'another valid one'


@pytest.mark.integration
class TestNormalizationInSampleBuilder:
    """Test normalization integration in SampleBuilder."""

    def test_build_sample_with_normalizer(
        self, action_config, observation_config, mock_augmentation_pipeline, mock_action_processor,
        padded_data, action_dict, sampler_indices, dummy_normalizer
    ):
        """Test sample building with normalization."""
        # Use dummy normalizer that passes through all data

        builder = SampleBuilder(
            action_space=action_config,
            observation_space=observation_config,
            obs_horizon=3,
            pred_horizon=4,
            action_backward_shift=0,
            augmentation_pipeline=mock_augmentation_pipeline,
            action_processor=mock_action_processor,
            normalizer=dummy_normalizer,
        )

        sample = builder.build_sample(
            padded_data=padded_data,
            action_dict=action_dict,
            start_idx=0,
            sampler_indices=sampler_indices,
        )

        # Verify data is present
        assert OBSERVATION_KEY in sample
        assert ACTION_KEY in sample
        assert isinstance(sample[ACTION_KEY][POSITION_ACTION_KEY], torch.Tensor)
        assert isinstance(sample[OBSERVATION_KEY][PROPRIO_OBS_ROBOT_FRAME_KEY], torch.Tensor)

    def test_normalization_preserves_shapes(
        self, action_config, observation_config, mock_augmentation_pipeline, mock_action_processor,
        padded_data, action_dict, sampler_indices, dummy_normalizer
    ):
        """Test that normalization preserves tensor shapes."""
        # Use dummy normalizer that passes through all data

        builder = SampleBuilder(
            action_space=action_config,
            observation_space=observation_config,
            obs_horizon=3,
            pred_horizon=4,
            action_backward_shift=0,
            augmentation_pipeline=mock_augmentation_pipeline,
            action_processor=mock_action_processor,
            normalizer=dummy_normalizer,
        )

        sample = builder.build_sample(
            padded_data=padded_data,
            action_dict=action_dict,
            start_idx=0,
            sampler_indices=sampler_indices,
        )

        # Shapes should be the same with or without normalization
        assert sample[ACTION_KEY][POSITION_ACTION_KEY].shape == (4, 3)
        assert sample[ACTION_KEY][ORIENTATION_ACTION_KEY].shape == (4, 4)


@pytest.mark.integration
class TestTokenizationInSampleBuilder:
    """Test tokenization integration in SampleBuilder."""

    def test_build_sample_with_observation_tokenizer(
        self, action_config, observation_config, mock_augmentation_pipeline, mock_action_processor,
        padded_data, action_dict, sampler_indices
    ):
        """Test sample building with observation tokenization."""
        observation_config.use_language = True
        padded_data[LANGUAGE_KEY] = np.array([f'instruction_{i}' for i in range(10)], dtype=object)

        obs_tokenizer = ObservationTokenizer(
            tokenizer_model="google/bert_uncased_L-2_H-128_A-2",
            observation_keys=[LANGUAGE_KEY, PROPRIO_OBS_ROBOT_FRAME_KEY],
            bin_continuous_data=False,
            device=torch.device("cpu"),
        )
        obs_tokenizer.fit({})

        tokenizer = Tokenizer(observation_tokenizer=obs_tokenizer)

        builder = SampleBuilder(
            action_space=action_config,
            observation_space=observation_config,
            obs_horizon=3,
            pred_horizon=4,
            action_backward_shift=0,
            augmentation_pipeline=mock_augmentation_pipeline,
            action_processor=mock_action_processor,
            tokenizer=tokenizer,
        )

        sample = builder.build_sample(
            padded_data=padded_data,
            action_dict=action_dict,
            start_idx=0,
            sampler_indices=sampler_indices,
        )

        # Verify tokenized observations are added
        assert TOKENIZED_OBSERVATIONS_KEY in sample[OBSERVATION_KEY]
        assert IS_PAD_OBSERVATION_KEY in sample[OBSERVATION_KEY]
        assert sample[OBSERVATION_KEY][TOKENIZED_OBSERVATIONS_KEY].dtype == torch.long
        assert sample[OBSERVATION_KEY][IS_PAD_OBSERVATION_KEY].dtype == torch.bool

    def test_build_sample_with_action_tokenizer(
        self, action_config, observation_config, mock_augmentation_pipeline, mock_action_processor,
        padded_data, action_dict, sampler_indices
    ):
        """Test sample building with action tokenization."""
        action_tokenizer = ActionTokenizer(
            tokenizer_chain=[TokenizerType.FAST.value],
            use_pretrained_fast=True,
            device=torch.device("cpu"),
        )

        tokenizer = Tokenizer(action_tokenizer=action_tokenizer)

        builder = SampleBuilder(
            action_space=action_config,
            observation_space=observation_config,
            obs_horizon=3,
            pred_horizon=4,
            action_backward_shift=0,
            augmentation_pipeline=mock_augmentation_pipeline,
            action_processor=mock_action_processor,
            tokenizer=tokenizer,
        )

        sample = builder.build_sample(
            padded_data=padded_data,
            action_dict=action_dict,
            start_idx=0,
            sampler_indices=sampler_indices,
        )

        # Verify tokenized actions are added
        assert TOKENIZED_ACTIONS_KEY in sample[ACTION_KEY]
        assert sample[ACTION_KEY][TOKENIZED_ACTIONS_KEY].dtype == torch.long
        # IS_PAD_ACTION_KEY should be replaced with tokenizer's padding mask
        assert IS_PAD_ACTION_KEY in sample[ACTION_KEY]
        assert sample[ACTION_KEY][IS_PAD_ACTION_KEY].dtype == torch.bool

    def test_build_sample_with_both_tokenizers(
        self, action_config, observation_config, mock_augmentation_pipeline, mock_action_processor,
        padded_data, action_dict, sampler_indices
    ):
        """Test sample building with both observation and action tokenization."""
        observation_config.use_language = True
        padded_data[LANGUAGE_KEY] = np.array([f'instruction_{i}' for i in range(10)], dtype=object)

        obs_tokenizer = ObservationTokenizer(
            tokenizer_model="google/bert_uncased_L-2_H-128_A-2",
            observation_keys=[LANGUAGE_KEY, PROPRIO_OBS_ROBOT_FRAME_KEY],
            bin_continuous_data=False,
            device=torch.device("cpu"),
        )
        obs_tokenizer.fit({})

        action_tokenizer = ActionTokenizer(
            tokenizer_chain=[TokenizerType.FAST.value],
            use_pretrained_fast=True,
            device=torch.device("cpu"),
        )

        tokenizer = Tokenizer(
            observation_tokenizer=obs_tokenizer,
            action_tokenizer=action_tokenizer,
        )

        builder = SampleBuilder(
            action_space=action_config,
            observation_space=observation_config,
            obs_horizon=3,
            pred_horizon=4,
            action_backward_shift=0,
            augmentation_pipeline=mock_augmentation_pipeline,
            action_processor=mock_action_processor,
            tokenizer=tokenizer,
        )

        sample = builder.build_sample(
            padded_data=padded_data,
            action_dict=action_dict,
            start_idx=0,
            sampler_indices=sampler_indices,
        )

        # Verify both tokenized observations and actions are added
        assert TOKENIZED_OBSERVATIONS_KEY in sample[OBSERVATION_KEY]
        assert IS_PAD_OBSERVATION_KEY in sample[OBSERVATION_KEY]
        assert TOKENIZED_ACTIONS_KEY in sample[ACTION_KEY]

    def test_observation_tokenization_raises_on_missing_key(
        self, action_config, observation_config, mock_augmentation_pipeline, mock_action_processor,
        padded_data, action_dict, sampler_indices
    ):
        """Test that observation tokenization raises error when required key is missing."""
        # Don't add language to padded_data
        obs_tokenizer = ObservationTokenizer(
            tokenizer_model="google/bert_uncased_L-2_H-128_A-2",
            observation_keys=[LANGUAGE_KEY],  # Require language but don't add it to padded_data
            bin_continuous_data=False,
            device=torch.device("cpu"),
        )
        obs_tokenizer.fit({})

        tokenizer = Tokenizer(observation_tokenizer=obs_tokenizer)

        builder = SampleBuilder(
            action_space=action_config,
            observation_space=observation_config,
            obs_horizon=3,
            pred_horizon=4,
            action_backward_shift=0,
            augmentation_pipeline=mock_augmentation_pipeline,
            action_processor=mock_action_processor,
            tokenizer=tokenizer,
        )

        with pytest.raises(KeyError, match="not found in sample for tokenization"):
            builder.build_sample(
                padded_data=padded_data,
                action_dict=action_dict,
                start_idx=0,
                sampler_indices=sampler_indices,
            )


@pytest.mark.integration
class TestNormalizationAndTokenizationTogether:
    """Test normalization and real tokenization working together."""

    def test_build_sample_with_normalizer_and_tokenizers(
        self, action_config, observation_config, mock_augmentation_pipeline, mock_action_processor,
        padded_data, action_dict, sampler_indices, dummy_normalizer
    ):
        """Test sample building with both normalization and tokenization."""
        observation_config.use_language = True
        padded_data[LANGUAGE_KEY] = np.array([f'instruction_{i}' for i in range(10)], dtype=object)

        # Use dummy normalizer that passes through all data

        # Create tokenizers
        obs_tokenizer = ObservationTokenizer(
            tokenizer_model="google/bert_uncased_L-2_H-128_A-2",
            observation_keys=[LANGUAGE_KEY, PROPRIO_OBS_ROBOT_FRAME_KEY],
            bin_continuous_data=False,
            device=torch.device("cpu"),
        )
        obs_tokenizer.fit({})

        action_tokenizer = ActionTokenizer(
            tokenizer_chain=[TokenizerType.FAST.value],
            use_pretrained_fast=True,
            device=torch.device("cpu"),
        )

        tokenizer = Tokenizer(
            observation_tokenizer=obs_tokenizer,
            action_tokenizer=action_tokenizer,
        )

        builder = SampleBuilder(
            action_space=action_config,
            observation_space=observation_config,
            obs_horizon=3,
            pred_horizon=4,
            action_backward_shift=0,
            augmentation_pipeline=mock_augmentation_pipeline,
            action_processor=mock_action_processor,
            normalizer=dummy_normalizer,
            tokenizer=tokenizer,
        )

        sample = builder.build_sample(
            padded_data=padded_data,
            action_dict=action_dict,
            start_idx=0,
            sampler_indices=sampler_indices,
        )

        # Verify both normalization and tokenization happened
        assert TOKENIZED_OBSERVATIONS_KEY in sample[OBSERVATION_KEY]
        assert TOKENIZED_ACTIONS_KEY in sample[ACTION_KEY]
        # Original data should still be present (normalized)
        assert POSITION_ACTION_KEY in sample[ACTION_KEY]
        assert PROPRIO_OBS_ROBOT_FRAME_KEY in sample[OBSERVATION_KEY]
