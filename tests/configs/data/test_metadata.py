"""Tests for versatil.configs.data.metadata module."""

import importlib

import pytest
from hydra.utils import instantiate
from omegaconf import MISSING

from versatil.configs.data.metadata import (
    ActionMetadataConfig,
    CameraMetadataConfig,
    DepthCameraMetadataConfig,
    GripperActionMetadataConfig,
    GripperObservationMetadataConfig,
    ObservationMetadataConfig,
    OnTheFlyActionMetadataConfig,
    OrientationActionMetadataConfig,
    OrientationObservationMetadataConfig,
    PositionActionMetadataConfig,
    PositionObservationMetadataConfig,
    PrecomputedActionMetadataConfig,
    RGBCameraMetadataConfig,
)
from versatil.data.constants import (
    BinaryGripperRange,
    CoordinateSystem,
    GripperType,
    OrientationRepresentation,
)


@pytest.mark.unit
class TestObservationMetadataConfig:
    def test_target_points_to_observation_metadata(self):
        config = ObservationMetadataConfig()
        assert config._target_ == "versatil.data.metadata.ObservationMetadata"

    def test_required_fields_default_to_missing(self):
        config = ObservationMetadataConfig()
        assert config.raw_data_column_keys == MISSING
        assert config.dimension == MISSING
        assert config.dtype == MISSING
        assert config.is_numerical == MISSING
        assert config.needs_normalization == MISSING


@pytest.mark.unit
class TestPositionObservationMetadataConfig:
    def test_target_points_to_position_observation_metadata(self):
        config = PositionObservationMetadataConfig()
        assert config._target_ == "versatil.data.metadata.PositionObservationMetadata"

    def test_frame_default_is_robot_base_string(self):
        config = PositionObservationMetadataConfig()
        assert config.frame == CoordinateSystem.ROBOT_BASE.value

    def test_inherits_from_observation_metadata_config(self):
        config = PositionObservationMetadataConfig()
        assert isinstance(config, ObservationMetadataConfig)


@pytest.mark.unit
class TestOrientationObservationMetadataConfig:
    def test_target_points_to_orientation_observation_metadata(self):
        config = OrientationObservationMetadataConfig()
        assert (
            config._target_ == "versatil.data.metadata.OrientationObservationMetadata"
        )

    def test_frame_default_is_robot_base_string(self):
        config = OrientationObservationMetadataConfig()
        assert config.frame == CoordinateSystem.ROBOT_BASE.value

    def test_orientation_representation_default_is_roll_string(self):
        config = OrientationObservationMetadataConfig()
        assert config.orientation_representation == OrientationRepresentation.ROLL.value


@pytest.mark.unit
class TestGripperObservationMetadataConfig:
    def test_target_points_to_gripper_observation_metadata(self):
        config = GripperObservationMetadataConfig()
        assert config._target_ == "versatil.data.metadata.GripperObservationMetadata"

    def test_gripper_type_default_is_binary_string(self):
        config = GripperObservationMetadataConfig()
        assert config.gripper_type == GripperType.BINARY.value

    def test_binary_gripper_range_default_is_zero_one_string(self):
        config = GripperObservationMetadataConfig()
        assert config.binary_gripper_range == BinaryGripperRange.ZERO_ONE.value


@pytest.mark.unit
class TestCameraMetadataConfig:
    def test_target_points_to_camera_metadata(self):
        config = CameraMetadataConfig()
        assert config._target_ == "versatil.data.metadata.CameraMetadata"

    def test_required_fields_default_to_missing(self):
        config = CameraMetadataConfig()
        assert config.camera_key == MISSING
        assert config.dtype == MISSING
        assert config.channels == MISSING

    def test_optional_dimensions_default_to_none(self):
        config = CameraMetadataConfig()
        assert config.image_width is None
        assert config.image_height is None
        assert config.max_pixel_value is None


@pytest.mark.unit
class TestDepthCameraMetadataConfig:
    def test_target_points_to_depth_camera_metadata(self):
        config = DepthCameraMetadataConfig()
        assert config._target_ == "versatil.data.metadata.DepthCameraMetadata"

    def test_required_fields_default_to_missing(self):
        config = DepthCameraMetadataConfig()
        assert config.camera_key == MISSING
        assert config.dtype == MISSING

    def test_has_no_channels_field(self):
        config = DepthCameraMetadataConfig()
        assert "channels" not in config.__dataclass_fields__

    def test_optional_dimensions_default_to_none(self):
        config = DepthCameraMetadataConfig()
        assert config.image_width is None
        assert config.image_height is None
        assert config.max_pixel_value is None


@pytest.mark.unit
class TestRGBCameraMetadataConfig:
    def test_target_points_to_rgb_camera_metadata(self):
        config = RGBCameraMetadataConfig()
        assert config._target_ == "versatil.data.metadata.RGBCameraMetadata"

    def test_required_fields_default_to_missing(self):
        config = RGBCameraMetadataConfig()
        assert config.camera_key == MISSING
        assert config.dtype == MISSING

    def test_has_no_channels_field(self):
        config = RGBCameraMetadataConfig()
        assert "channels" not in config.__dataclass_fields__

    def test_optional_dimensions_default_to_none(self):
        config = RGBCameraMetadataConfig()
        assert config.image_width is None
        assert config.image_height is None

    def test_max_pixel_value_defaults_to_255(self):
        config = RGBCameraMetadataConfig()
        assert config.max_pixel_value == 255.0


@pytest.mark.unit
class TestPrecomputedActionMetadataConfig:
    def test_target_points_to_precomputed_action_metadata(self):
        config = PrecomputedActionMetadataConfig()
        assert config._target_ == "versatil.data.metadata.PrecomputedActionMetadata"

    def test_required_fields_default_to_missing(self):
        config = PrecomputedActionMetadataConfig()
        assert config.raw_data_column_keys == MISSING
        assert config.storage_dimension == MISSING
        assert config.prediction_dimension == MISSING


@pytest.mark.unit
class TestPositionActionMetadataConfig:
    def test_target_points_to_position_action_metadata(self):
        config = PositionActionMetadataConfig()
        assert config._target_ == "versatil.data.metadata.PositionActionMetadata"

    def test_frame_required(self):
        config = PositionActionMetadataConfig()
        assert config.frame == MISSING

    def test_computation_method_optional(self):
        config = PositionActionMetadataConfig()
        assert config.computation_method is None

    def test_inherits_from_precomputed_action_metadata_config(self):
        config = PositionActionMetadataConfig()
        assert isinstance(config, PrecomputedActionMetadataConfig)


@pytest.mark.unit
class TestOrientationActionMetadataConfig:
    def test_target_points_to_orientation_action_metadata(self):
        config = OrientationActionMetadataConfig()
        assert config._target_ == "versatil.data.metadata.OrientationActionMetadata"

    def test_frame_and_representation_required(self):
        config = OrientationActionMetadataConfig()
        assert config.frame == MISSING
        assert config.orientation_representation == MISSING


@pytest.mark.unit
class TestGripperActionMetadataConfig:
    def test_target_points_to_gripper_action_metadata(self):
        config = GripperActionMetadataConfig()
        assert config._target_ == "versatil.data.metadata.GripperActionMetadata"

    def test_gripper_type_and_range_required(self):
        config = GripperActionMetadataConfig()
        assert config.gripper_type == MISSING
        assert config.binary_gripper_range == MISSING


@pytest.mark.unit
class TestOnTheFlyActionMetadataConfig:
    def test_target_points_to_on_the_fly_action_metadata(self):
        config = OnTheFlyActionMetadataConfig()
        assert config._target_ == "versatil.data.metadata.OnTheFlyActionMetadata"

    def test_required_fields_default_to_missing(self):
        config = OnTheFlyActionMetadataConfig()
        assert config.source_metadata == MISSING
        assert config.computation_method == MISSING


@pytest.mark.unit
class TestActionMetadataConfig:
    def test_target_points_to_action_metadata(self):
        config = ActionMetadataConfig()
        assert config._target_ == "versatil.data.metadata.ActionMetadata"

    def test_required_fields_default_to_missing(self):
        config = ActionMetadataConfig()
        assert config.prediction_dimension == MISSING
        assert config.is_numerical == MISSING
        assert config.needs_normalization == MISSING
        assert config.dtype == MISSING
        assert config.is_precomputed == MISSING


@pytest.mark.unit
class TestMetadataInstantiation:
    def test_observation_metadata_instantiates(self):
        config = ObservationMetadataConfig(
            raw_data_column_keys=["col_a", "col_b"],
            dimension=2,
            dtype="float32",
            is_numerical=True,
            needs_normalization=True,
        )
        instance = instantiate(config)
        assert type(instance).__name__ == "ObservationMetadata"
        assert instance.dimension == 2
        assert instance.dtype == "float32"

    def test_rgb_camera_metadata_instantiates(self):
        config = RGBCameraMetadataConfig(
            camera_key="left",
            dtype="float32",
        )
        instance = instantiate(config)
        assert type(instance).__name__ == "RGBCameraMetadata"
        assert instance.raw_camera_key == "left"
        assert instance.channels == 3
        assert instance.max_pixel_value == 255.0

    def test_depth_camera_metadata_instantiates(self):
        config = DepthCameraMetadataConfig(
            camera_key="depth",
            dtype="float32",
        )
        instance = instantiate(config)
        assert type(instance).__name__ == "DepthCameraMetadata"
        assert instance.raw_camera_key == "depth"
        assert instance.channels == 1
        assert instance.max_pixel_value is None

    def test_action_metadata_instantiates(self):
        config = ActionMetadataConfig(
            prediction_dimension=3,
            is_numerical=True,
            needs_normalization=True,
            dtype="float32",
            is_precomputed=True,
        )
        instance = instantiate(config)
        assert type(instance).__name__ == "ActionMetadata"
        assert instance.prediction_dimension == 3

    def test_precomputed_action_metadata_instantiates(self):
        config = PrecomputedActionMetadataConfig(
            raw_data_column_keys=["x", "y", "z"],
            storage_dimension=3,
            prediction_dimension=3,
            is_numerical=True,
            needs_normalization=True,
            dtype="float32",
        )
        instance = instantiate(config)
        assert type(instance).__name__ == "PrecomputedActionMetadata"
        assert instance.storage_dimension == 3

    @pytest.mark.parametrize(
        "config_class, expected_class_name",
        [
            (
                lambda: PositionObservationMetadataConfig(),
                "PositionObservationMetadata",
            ),
            (
                lambda: OrientationObservationMetadataConfig(),
                "OrientationObservationMetadata",
            ),
            (lambda: GripperObservationMetadataConfig(), "GripperObservationMetadata"),
            (lambda: RGBCameraMetadataConfig(), "RGBCameraMetadata"),
            (lambda: DepthCameraMetadataConfig(), "DepthCameraMetadata"),
            (lambda: PositionActionMetadataConfig(), "PositionActionMetadata"),
            (lambda: OrientationActionMetadataConfig(), "OrientationActionMetadata"),
            (lambda: GripperActionMetadataConfig(), "GripperActionMetadata"),
            (lambda: OnTheFlyActionMetadataConfig(), "OnTheFlyActionMetadata"),
        ],
    )
    def test_target_resolves_to_importable_class(
        self, config_class, expected_class_name
    ):
        config = config_class()
        target = config._target_
        module_path, class_name = target.rsplit(".", 1)
        module = importlib.import_module(module_path)
        assert hasattr(module, class_name)
        assert class_name == expected_class_name
