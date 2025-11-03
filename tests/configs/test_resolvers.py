"""Tests for custom OmegaConf resolvers."""

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError

from refactoring.configs import register_resolvers  # Import to trigger resolver registration
from refactoring.data.constants import Cameras, GripperType, OrientationRepresentation


@pytest.mark.unit
class TestEnumResolvers:
    """Test custom OmegaConf resolvers for enum access in YAML configs."""

    @classmethod
    def setup_class(cls):
        """Register resolvers before running tests."""
        register_resolvers()

    def test_cameras_resolver(self):
        """Test cameras resolver returns correct enum values."""
        # Create a config with camera resolver
        cfg = OmegaConf.create({
            "left": "${cameras:LEFT}",
            "right": "${cameras:RIGHT}",
            "depth": "${cameras:DEPTH}"
        })

        assert cfg.left == Cameras.LEFT.value
        assert cfg.right == Cameras.RIGHT.value
        assert cfg.depth == Cameras.DEPTH.value
        assert cfg.left == "left"
        assert cfg.right == "right"
        assert cfg.depth == "depth"

    def test_gripper_resolver(self):
        """Test gripper resolver returns correct enum values."""
        cfg = OmegaConf.create({
            "binary": "${gripper:BINARY}",
            "continuous": "${gripper:CONTINUOUS}"
        })

        assert cfg.binary == GripperType.BINARY.value
        assert cfg.continuous == GripperType.CONTINUOUS.value
        assert cfg.binary == "binary"
        assert cfg.continuous == "continuous"

    def test_orientation_resolver(self):
        """Test orientation resolver returns correct enum values."""
        cfg = OmegaConf.create({
            "roll": "${orientation:ROLL}",
            "euler": "${orientation:EULER}",
            "quaternion": "${orientation:QUATERNION}"
        })

        assert cfg.roll == OrientationRepresentation.ROLL.value
        assert cfg.euler == OrientationRepresentation.EULER.value
        assert cfg.quaternion == OrientationRepresentation.QUATERNION.value
        assert cfg.roll == "roll"
        assert cfg.euler == "euler"
        assert cfg.quaternion == "quaternion"

    def test_resolver_in_list(self):
        """Test resolvers work in lists (for camera_keys)."""
        cfg = OmegaConf.create({
            "camera_keys": [
                "${cameras:LEFT}",
                "${cameras:RIGHT}",
                "${cameras:DEPTH}"
            ]
        })

        assert cfg.camera_keys == ["left", "right", "depth"]
        assert len(cfg.camera_keys) == 3

    def test_invalid_enum_name_raises_error(self):
        """Test that invalid enum names raise InterpolationResolutionError."""
        with pytest.raises(InterpolationResolutionError):
            cfg = OmegaConf.create({"invalid": "${cameras:INVALID}"})
            _ = cfg.invalid  # Trigger resolution

    def test_resolver_with_nested_config(self):
        """Test resolvers work in nested config structures."""
        cfg = OmegaConf.create({
            "task": {
                "observation_space": {
                    "camera_keys": ["${cameras:LEFT}", "${cameras:RIGHT}"],
                    "gripper_type": "${gripper:BINARY}"
                },
                "action_space": {
                    "gripper_type": "${gripper:CONTINUOUS}",
                    "orientation_repr": "${orientation:QUATERNION}"
                }
            }
        })

        assert cfg.task.observation_space.camera_keys == ["left", "right"]
        assert cfg.task.observation_space.gripper_type == "binary"
        assert cfg.task.action_space.gripper_type == "continuous"
        assert cfg.task.action_space.orientation_repr == "quaternion"

    def test_resolver_mixed_with_interpolation(self):
        """Test resolvers work alongside other OmegaConf interpolations."""
        cfg = OmegaConf.create({
            "default_camera": "${cameras:LEFT}",
            "selected_camera": "${default_camera}",
            "gripper": "${gripper:BINARY}"
        })

        assert cfg.default_camera == "left"
        assert cfg.selected_camera == "left"
        assert cfg.gripper == "binary"