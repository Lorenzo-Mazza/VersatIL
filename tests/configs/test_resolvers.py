"""Tests for versatil.common.omegaconf_ops module."""
import pytest
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError

from versatil.common.omegaconf_ops import resolve_dict_keys
from versatil.configs import register_resolvers
from versatil.data.constants import (
    Cameras,
    GripperType,
    OrientationRepresentation,
)
from versatil.models.encoding.encoders.constants import RGBBackboneType
from versatil.training.constants import Float32MatmulPrecision, PrecisionType


register_resolvers()


@pytest.mark.unit
class TestResolveDictKeys:

    def test_plain_string_keys_returned_unchanged(self):
        input_dict = {"key_a": 1, "key_b": 2}
        result = resolve_dict_keys(input_dict)
        assert result == {"key_a": 1, "key_b": 2}

    def test_non_string_keys_returned_unchanged(self):
        input_dict = {42: "value", True: "other"}
        result = resolve_dict_keys(input_dict)
        assert result == {42: "value", True: "other"}

    def test_nested_dict_values_resolved_recursively(self):
        input_dict = {"outer": {"inner_key": "inner_value"}}
        result = resolve_dict_keys(input_dict)
        assert result["outer"]["inner_key"] == "inner_value"

    def test_non_dict_values_preserved(self):
        input_dict = {"key": [1, 2, 3], "key2": "string", "key3": 42}
        result = resolve_dict_keys(input_dict)
        assert result["key"] == [1, 2, 3]
        assert result["key2"] == "string"
        assert result["key3"] == 42

    def test_empty_dict_returns_empty(self):
        result = resolve_dict_keys({})
        assert result == {}

    def test_interpolation_key_resolved_via_registered_resolver(self):
        input_dict = {"${cameras:LEFT}": "value"}
        result = resolve_dict_keys(input_dict)
        assert Cameras.LEFT.value in result
        assert result[Cameras.LEFT.value] == "value"

    def test_deeply_nested_dicts_resolved(self):
        input_dict = {
            "level1": {
                "level2": {
                    "level3": "deep_value"
                }
            }
        }
        result = resolve_dict_keys(input_dict)
        assert result["level1"]["level2"]["level3"] == "deep_value"

    def test_mixed_interpolation_and_plain_keys(self):
        input_dict = {
            "plain_key": "value1",
            "${cameras:RIGHT}": "value2",
        }
        result = resolve_dict_keys(input_dict)
        assert result["plain_key"] == "value1"
        assert result[Cameras.RIGHT.value] == "value2"

    def test_returns_new_dict_not_mutating_input(self):
        input_dict = {"key": "value"}
        result = resolve_dict_keys(input_dict)
        assert result is not input_dict
        result["new_key"] = "new_value"
        assert "new_key" not in input_dict


@pytest.mark.unit
class TestEnumResolvers:

    def test_cameras_resolver_returns_enum_values(self):
        cfg = OmegaConf.create({
            "left": "${cameras:LEFT}",
            "right": "${cameras:RIGHT}",
            "depth": "${cameras:DEPTH}",
        })
        assert cfg.left == Cameras.LEFT.value
        assert cfg.right == Cameras.RIGHT.value
        assert cfg.depth == Cameras.DEPTH.value

    def test_gripper_resolver_returns_enum_values(self):
        cfg = OmegaConf.create({
            "binary": "${gripper:BINARY}",
            "continuous": "${gripper:CONTINUOUS}",
        })
        assert cfg.binary == GripperType.BINARY.value
        assert cfg.continuous == GripperType.CONTINUOUS.value

    def test_orientation_resolver_returns_enum_values(self):
        cfg = OmegaConf.create({
            "roll": "${orientation:ROLL}",
            "euler": "${orientation:EULER}",
            "quaternion": "${orientation:QUATERNION}",
        })
        assert cfg.roll == OrientationRepresentation.ROLL.value
        assert cfg.euler == OrientationRepresentation.EULER.value
        assert cfg.quaternion == OrientationRepresentation.QUATERNION.value

    def test_rgb_backbone_resolver_returns_enum_values(self):
        cfg = OmegaConf.create({
            "resnet18": "${rgb_backbone:RESNET18}",
            "dinov2_vits14": "${rgb_backbone:DINOV2_VITS14}",
        })
        assert cfg.resnet18 == RGBBackboneType.RESNET18.value
        assert cfg.dinov2_vits14 == RGBBackboneType.DINOV2_VITS14.value

    def test_precision_resolver_returns_enum_values(self):
        cfg = OmegaConf.create({
            "fp32": "${precision:FP32}",
            "fp16_mixed": "${precision:FP16_MIXED}",
            "bf16_mixed": "${precision:BF16_MIXED}",
        })
        assert cfg.fp32 == PrecisionType.FP32.value
        assert cfg.fp16_mixed == PrecisionType.FP16_MIXED.value
        assert cfg.bf16_mixed == PrecisionType.BF16_MIXED.value

    def test_float32_matmul_resolver_returns_enum_values(self):
        cfg = OmegaConf.create({
            "highest": "${float32_matmul:HIGHEST}",
            "medium": "${float32_matmul:MEDIUM}",
        })
        assert cfg.highest == Float32MatmulPrecision.HIGHEST.value
        assert cfg.medium == Float32MatmulPrecision.MEDIUM.value

    def test_invalid_enum_name_raises_interpolation_error(self):
        cfg = OmegaConf.create({"invalid": "${cameras:NONEXISTENT}"})
        with pytest.raises(InterpolationResolutionError):
            _ = cfg.invalid

    def test_resolver_works_inside_list(self):
        cfg = OmegaConf.create({
            "camera_keys": [
                "${cameras:LEFT}",
                "${cameras:RIGHT}",
                "${cameras:DEPTH}",
            ]
        })
        assert cfg.camera_keys == [
            Cameras.LEFT.value,
            Cameras.RIGHT.value,
            Cameras.DEPTH.value,
        ]

    def test_resolver_works_in_nested_config(self):
        cfg = OmegaConf.create({
            "task": {
                "cameras": ["${cameras:LEFT}"],
                "gripper_type": "${gripper:BINARY}",
            }
        })
        assert cfg.task.cameras == [Cameras.LEFT.value]
        assert cfg.task.gripper_type == GripperType.BINARY.value

    def test_resolver_combined_with_omegaconf_interpolation(self):
        cfg = OmegaConf.create({
            "default_camera": "${cameras:LEFT}",
            "selected_camera": "${default_camera}",
        })
        assert cfg.default_camera == Cameras.LEFT.value
        assert cfg.selected_camera == Cameras.LEFT.value
