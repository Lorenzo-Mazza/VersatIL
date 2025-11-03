"""Tests for custom OmegaConf resolvers."""

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError

from refactoring.configs import register_resolvers  # Import to trigger resolver registration
from refactoring.data.constants import Cameras, GripperType, OrientationRepresentation
from refactoring.models.encoding.encoders.constants import RGBBackboneType
from refactoring.training.constants import Float32MatmulPrecision, PrecisionType


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

    def test_rgb_backbone_resolver(self):
        """Test rgb_backbone resolver returns correct enum values."""
        cfg = OmegaConf.create({
            "resnet18": "${rgb_backbone:RESNET18}",
            "resnet34": "${rgb_backbone:RESNET34}",
            "resnet50": "${rgb_backbone:RESNET50}",
            "dinov2_vits14": "${rgb_backbone:DINOV2_VITS14}"
        })

        assert cfg.resnet18 == RGBBackboneType.RESNET18.value
        assert cfg.resnet34 == RGBBackboneType.RESNET34.value
        assert cfg.resnet50 == RGBBackboneType.RESNET50.value
        assert cfg.dinov2_vits14 == RGBBackboneType.DINOV2_VITS14.value
        assert cfg.resnet18 == "timm/resnet18.a1_in1k"
        assert cfg.resnet34 == "timm/resnet34.a1_in1k"
        assert cfg.resnet50 == "timm/resnet50.a1_in1k"
        assert cfg.dinov2_vits14 == "timm/vit_small_patch14_dinov2.lvd142m"

    def test_precision_resolver(self):
        """Test precision resolver returns correct enum values."""
        cfg = OmegaConf.create({
            "fp32": "${precision:FP32}",
            "fp16_mixed": "${precision:FP16_MIXED}",
            "bf16_mixed": "${precision:BF16_MIXED}",
            "fp16_true": "${precision:FP16_TRUE}",
            "bf16_true": "${precision:BF16_TRUE}",
            "fp64": "${precision:FP64}",
        })

        assert cfg.fp32 == PrecisionType.FP32.value
        assert cfg.fp16_mixed == PrecisionType.FP16_MIXED.value
        assert cfg.bf16_mixed == PrecisionType.BF16_MIXED.value
        assert cfg.fp16_true == PrecisionType.FP16_TRUE.value
        assert cfg.bf16_true == PrecisionType.BF16_TRUE.value
        assert cfg.fp64 == PrecisionType.FP64.value
        assert cfg.fp32 == "32"
        assert cfg.fp16_mixed == "16-mixed"
        assert cfg.bf16_mixed == "bf16-mixed"

    def test_float32_matmul_resolver(self):
        """Test float32_matmul resolver returns correct enum values."""
        cfg = OmegaConf.create({
            "highest": "${float32_matmul:HIGHEST}",
            "high": "${float32_matmul:HIGH}",
            "medium": "${float32_matmul:MEDIUM}",
        })

        assert cfg.highest == Float32MatmulPrecision.HIGHEST.value
        assert cfg.high == Float32MatmulPrecision.HIGH.value
        assert cfg.medium == Float32MatmulPrecision.MEDIUM.value
        assert cfg.highest == "highest"
        assert cfg.high == "high"
        assert cfg.medium == "medium"

    def test_precision_and_matmul_in_experiment_config(self):
        """Test precision resolvers work in experiment config structure."""
        cfg = OmegaConf.create({
            "experiment": {
                "precision": "${precision:FP16_MIXED}",
                "float32_matmul_precision": "${float32_matmul:MEDIUM}",
                "device": "cuda"
            }
        })

        assert cfg.experiment.precision == "16-mixed"
        assert cfg.experiment.float32_matmul_precision == "medium"
        assert cfg.experiment.device == "cuda"