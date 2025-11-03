"""Tests for task configuration dataclasses."""
import dataclasses

import pytest

from refactoring.configs.task.task import ActionSpace, ObservationSpace, TaskConfig
from refactoring.configs.task.dataloader import DataloaderConfig
from refactoring.data.constants import GripperType, OrientationRepresentation


@pytest.mark.unit
class TestActionSpace:

    def test_config_has_correct_target(self):
        config = ActionSpace()
        assert config._target_ == "refactoring.configs.task.task.ActionSpace"

    def test_config_can_be_instantiated(self):
        config = ActionSpace()
        assert isinstance(config, ActionSpace)
        assert config.has_position is True
        assert config.position_dim == 3
        assert config.has_gripper is True

    def test_get_total_action_dim(self):
        config = ActionSpace(
            has_position=True,
            position_dim=3,
            has_orientation=True,
            orientation_dim=3,
            has_gripper=True,
            gripper_dim=1,
        )
        assert config.get_total_action_dim() == 7

    def test_get_total_action_dim_with_phases(self):
        config = ActionSpace(
            has_position=True,
            position_dim=3,
            has_gripper=True,
            gripper_dim=1,
            task_has_phases=True,
            number_of_phases=5,
        )
        assert config.get_total_action_dim() == 9

    def test_get_required_zarr_keys_camera_frame(self):
        config = ActionSpace(predict_in_camera_frame=True)
        keys = config.get_required_zarr_keys()
        assert "proprio_camera_frame" in keys
        assert "gripper_state_obs" in keys

    def test_get_required_zarr_keys_robot_frame(self):
        config = ActionSpace(predict_in_camera_frame=False)
        keys = config.get_required_zarr_keys()
        assert "proprio_robot_frame" in keys


@pytest.mark.unit
class TestObservationSpace:

    def test_config_can_be_instantiated(self):
        config = ObservationSpace()
        assert isinstance(config, ObservationSpace)
        assert config.use_proprioceptive_data is False
        assert config.use_language is False

    def test_get_required_zarr_keys_minimal(self):
        config = ObservationSpace(
            camera_keys=["left"],
            use_proprioceptive_data=False,
            use_language=False,
        )
        keys = config.get_required_zarr_keys()
        assert "left" in keys

    def test_get_required_zarr_keys_with_proprio(self):
        config = ObservationSpace(
            camera_keys=["left"],
            use_proprioceptive_data=True,
            use_proprio_camera_frame=True,
        )
        keys = config.get_required_zarr_keys()
        assert "proprio_camera_frame" in keys

    def test_get_required_zarr_keys_with_language(self):
        config = ObservationSpace(
            camera_keys=[],
            use_language=True,
        )
        keys = config.get_required_zarr_keys()
        assert "language_instruction" in keys


@pytest.mark.unit
class TestDataloaderConfig:

    def test_config_can_be_instantiated(self):
        config = DataloaderConfig()
        assert isinstance(config, DataloaderConfig)
        assert config.batch_size == 64
        assert config.num_workers == 16
        assert config.image_height == 270
        assert config.image_width == 480

    def test_validation_catches_invalid_batch_size(self):
        with pytest.raises(ValueError, match="batch_size must be positive"):
            DataloaderConfig(batch_size=0)

    def test_validation_catches_invalid_val_ratio(self):
        with pytest.raises(ValueError, match="val_ratio must be in range"):
            DataloaderConfig(val_ratio=1.5)

    def test_validation_catches_negative_num_workers(self):
        with pytest.raises(ValueError, match="num_workers cannot be negative"):
            DataloaderConfig(num_workers=-1)

    def test_validation_accepts_valid_config(self):
        config = DataloaderConfig(
            batch_size=32,
            num_workers=8,
            val_ratio=0.2,
            total_ratio=0.8,
        )
        assert config.batch_size == 32
        assert config.val_ratio == 0.2
