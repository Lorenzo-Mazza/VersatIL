"""Tests for versatil.data.metadata module."""

import re
from contextlib import nullcontext as does_not_raise

import pytest

from versatil.data.constants import (
    ActionComputationMethod,
    BinaryGripperRange,
    CameraModality,
    Cameras,
    CoordinateSystem,
    GripperType,
    OrientationRepresentation,
    ProprioceptiveType,
    RawCameraKey,
)
from versatil.data.metadata import (
    ActionMetadata,
    BaseMetadata,
    CameraMetadata,
    DepthCameraMetadata,
    GripperActionMetadata,
    GripperObservationMetadata,
    ObservationMetadata,
    OnTheFlyActionMetadata,
    OrientationActionMetadata,
    OrientationObservationMetadata,
    PositionActionMetadata,
    PositionObservationMetadata,
    PrecomputedActionMetadata,
    RGBCameraMetadata,
)


class TestBaseMetadata:
    def test_non_numerical_with_normalization_raises(self):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Non-numerical observations should not need normalization."
            ),
        ):
            BaseMetadata(dtype="string", is_numerical=False, needs_normalization=True)

    def test_non_numerical_without_normalization_succeeds(self):
        metadata = BaseMetadata(
            dtype="string", is_numerical=False, needs_normalization=False
        )
        assert not metadata.is_numerical
        assert not metadata.needs_normalization

    @pytest.mark.parametrize(
        "dtype, expectation",
        [
            ("float32", does_not_raise()),
            ("float64", does_not_raise()),
            ("int32", does_not_raise()),
            ("int64", does_not_raise()),
            ("string", pytest.raises(ValueError, match="float or int")),
            ("bool", pytest.raises(ValueError, match="float or int")),
            ("datetime64", pytest.raises(ValueError, match="float or int")),
        ],
    )
    def test_numerical_dtype_validation(self, dtype, expectation):
        with expectation:
            BaseMetadata(dtype=dtype, is_numerical=True, needs_normalization=True)

    @pytest.mark.parametrize("dtype", ["string", "bool", "datetime64"])
    def test_numerical_dtype_validation_message(self, dtype):
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"dtype for numerical observations must be float or int type, "
                f"got {dtype}"
            ),
        ):
            BaseMetadata(dtype=dtype, is_numerical=True, needs_normalization=True)

    def test_equality_same_values(self):
        first = BaseMetadata(
            dtype="float32", is_numerical=True, needs_normalization=True
        )
        second = BaseMetadata(
            dtype="float32", is_numerical=True, needs_normalization=True
        )
        assert first == second

    def test_equality_different_dtype(self):
        first = BaseMetadata(
            dtype="float32", is_numerical=True, needs_normalization=True
        )
        second = BaseMetadata(
            dtype="float64", is_numerical=True, needs_normalization=True
        )
        assert first != second

    def test_subclass_never_equals_parent_instance(self):
        # Reflected equality must not fall back to the parent's field-only
        # comparison; different metadata types describe different data.
        parent = BaseMetadata(
            dtype="float32", is_numerical=True, needs_normalization=True
        )
        child = ObservationMetadata(
            raw_data_column_keys=["column"],
            dimension=3,
            dtype="float32",
            is_numerical=True,
            needs_normalization=True,
        )
        assert parent != child
        assert child != parent

    def test_equality_different_type_returns_not_implemented(self):
        metadata = BaseMetadata(
            dtype="float32", is_numerical=True, needs_normalization=True
        )
        assert metadata.__eq__("not_metadata") is NotImplemented


class TestObservationMetadata:
    def test_empty_raw_data_column_keys_raises(self):
        with pytest.raises(
            ValueError, match=re.escape("raw_data_column_keys cannot be empty")
        ):
            ObservationMetadata(
                raw_data_column_keys=[],
                dimension=3,
                dtype="float32",
                is_numerical=True,
                needs_normalization=True,
            )

    @pytest.mark.parametrize("dimension", [0, -1])
    def test_non_positive_dimension_raises(self, dimension):
        with pytest.raises(
            ValueError,
            match=re.escape(f"dimension must be positive, got {dimension}"),
        ):
            ObservationMetadata(
                raw_data_column_keys=["x"],
                dimension=dimension,
                dtype="float32",
                is_numerical=True,
                needs_normalization=True,
            )

    def test_negative_slice_start_raises(self):
        with pytest.raises(
            ValueError,
            match=re.escape("slice_start and slice_end must be non-negative"),
        ):
            ObservationMetadata(
                raw_data_column_keys=["x", "y", "z"],
                dimension=3,
                dtype="float32",
                is_numerical=True,
                needs_normalization=True,
                slice_start=-1,
                slice_end=2,
            )

    def test_slice_start_greater_equal_to_end_raises(self):
        with pytest.raises(
            ValueError,
            match=re.escape("slice_start (5) must be less than slice_end (2)"),
        ):
            ObservationMetadata(
                raw_data_column_keys=["x", "y", "z"],
                dimension=3,
                dtype="float32",
                is_numerical=True,
                needs_normalization=True,
                slice_start=5,
                slice_end=2,
            )

    def test_slice_range_not_matching_dimension_raises(self):
        with pytest.raises(
            ValueError,
            match=re.escape("Slice range (5) must equal dimension (3)"),
        ):
            ObservationMetadata(
                raw_data_column_keys=["x", "y", "z"],
                dimension=3,
                dtype="float32",
                is_numerical=True,
                needs_normalization=True,
                slice_start=0,
                slice_end=5,
            )

    def test_valid_slice_succeeds(self):
        metadata = ObservationMetadata(
            raw_data_column_keys=["x", "y", "z"],
            dimension=3,
            dtype="float32",
            is_numerical=True,
            needs_normalization=True,
            slice_start=0,
            slice_end=3,
        )
        assert metadata.slice_start == 0
        assert metadata.slice_end == 3

    def test_no_slice_defaults_to_none(self):
        metadata = ObservationMetadata(
            raw_data_column_keys=["x"],
            dimension=1,
            dtype="float32",
            is_numerical=True,
            needs_normalization=True,
        )
        assert metadata.slice_start is None
        assert metadata.slice_end is None

    def test_only_slice_start_without_end_skips_validation(self):
        metadata = ObservationMetadata(
            raw_data_column_keys=["x"],
            dimension=1,
            dtype="float32",
            is_numerical=True,
            needs_normalization=True,
            slice_start=5,
            slice_end=None,
        )
        assert metadata.slice_start == 5
        assert metadata.slice_end is None

    def test_equality_includes_slice(self):
        first = ObservationMetadata(
            raw_data_column_keys=["x"],
            dimension=1,
            dtype="float32",
            is_numerical=True,
            needs_normalization=True,
            slice_start=0,
            slice_end=1,
        )
        second = ObservationMetadata(
            raw_data_column_keys=["x"],
            dimension=1,
            dtype="float32",
            is_numerical=True,
            needs_normalization=True,
            slice_start=2,
            slice_end=3,
        )
        assert first != second


class TestPositionObservationMetadata:
    @pytest.mark.parametrize("dtype", ["int32", "int64", "bool", "string"])
    def test_non_float_dtype_raises(self, dtype):
        with pytest.raises(
            ValueError,
            match=re.escape("Position observations dtype must be a float type."),
        ):
            PositionObservationMetadata(
                raw_data_column_keys=["x", "y", "z"],
                dimension=3,
                dtype=dtype,
                needs_normalization=True,
            )

    @pytest.mark.parametrize(
        "frame, expectation",
        [(member.value, does_not_raise()) for member in CoordinateSystem]
        + [
            (
                "invalid_frame",
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        f"frame must be one of {[e.value for e in CoordinateSystem]}, "
                        "got 'invalid_frame'"
                    ),
                ),
            ),
        ],
    )
    def test_frame_validation(self, frame, expectation):
        with expectation:
            PositionObservationMetadata(
                raw_data_column_keys=["x", "y", "z"],
                dimension=3,
                dtype="float32",
                needs_normalization=True,
                frame=frame,
            )

    def test_sets_proprioception_type_to_position(self):
        metadata = PositionObservationMetadata(
            raw_data_column_keys=["x", "y", "z"],
            dimension=3,
            dtype="float32",
            needs_normalization=True,
        )
        assert metadata.proprioception_type == ProprioceptiveType.POSITION.value

    def test_is_always_numerical(self):
        metadata = PositionObservationMetadata(
            raw_data_column_keys=["x", "y", "z"],
            dimension=3,
            dtype="float32",
            needs_normalization=True,
        )
        assert metadata.is_numerical

    def test_equality_includes_frame(self):
        robot_base = PositionObservationMetadata(
            raw_data_column_keys=["x", "y", "z"],
            dimension=3,
            dtype="float32",
            needs_normalization=True,
            frame=CoordinateSystem.ROBOT_BASE.value,
        )
        camera = PositionObservationMetadata(
            raw_data_column_keys=["x", "y", "z"],
            dimension=3,
            dtype="float32",
            needs_normalization=True,
            frame=CoordinateSystem.CAMERA.value,
        )
        assert robot_base != camera

    def test_default_frame_is_robot_base(self):
        metadata = PositionObservationMetadata(
            raw_data_column_keys=["x", "y", "z"],
            dimension=3,
            dtype="float32",
            needs_normalization=True,
        )
        assert metadata.frame == CoordinateSystem.ROBOT_BASE.value


class TestOrientationObservationMetadata:
    @pytest.mark.parametrize("dtype", ["int32", "bool"])
    def test_non_float_dtype_raises(self, dtype):
        with pytest.raises(
            ValueError,
            match=re.escape("Orientation observations dtype must be a float type."),
        ):
            OrientationObservationMetadata(
                raw_data_column_keys=["roll"],
                dimension=1,
                dtype=dtype,
                needs_normalization=True,
            )

    def test_invalid_frame_raises(self):
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"frame must be one of {[e.value for e in CoordinateSystem]}, "
                "got 'invalid'"
            ),
        ):
            OrientationObservationMetadata(
                raw_data_column_keys=["roll"],
                dimension=1,
                dtype="float32",
                needs_normalization=True,
                frame="invalid",
            )

    def test_invalid_orientation_representation_raises(self):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "orientation_representation must be one of "
                f"{[e.value for e in OrientationRepresentation]}, "
                "got 'rotation_matrix'"
            ),
        ):
            OrientationObservationMetadata(
                raw_data_column_keys=["roll"],
                dimension=1,
                dtype="float32",
                needs_normalization=True,
                orientation_representation="rotation_matrix",
            )

    @pytest.mark.parametrize(
        "representation",
        [member.value for member in OrientationRepresentation],
    )
    def test_all_valid_representations_succeed(self, representation):
        dimension = {"roll": 1, "euler": 3, "quaternion": 4}[representation]
        keys = ["roll", "pitch", "yaw", "w"][:dimension]
        metadata = OrientationObservationMetadata(
            raw_data_column_keys=keys,
            dimension=dimension,
            dtype="float32",
            needs_normalization=True,
            orientation_representation=representation,
        )
        assert metadata.orientation_representation == representation

    def test_sets_proprioception_type_to_orientation(self):
        metadata = OrientationObservationMetadata(
            raw_data_column_keys=["roll"],
            dimension=1,
            dtype="float32",
            needs_normalization=True,
        )
        assert metadata.proprioception_type == ProprioceptiveType.ORIENTATION.value

    def test_equality_includes_representation(self):
        roll = OrientationObservationMetadata(
            raw_data_column_keys=["roll"],
            dimension=1,
            dtype="float32",
            needs_normalization=True,
            orientation_representation=OrientationRepresentation.ROLL.value,
        )
        euler = OrientationObservationMetadata(
            raw_data_column_keys=["r"],
            dimension=1,
            dtype="float32",
            needs_normalization=True,
            orientation_representation=OrientationRepresentation.EULER.value,
        )
        assert roll != euler


class TestGripperObservationMetadata:
    def test_invalid_gripper_type_raises(self):
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"gripper_type must be one of {[e.value for e in GripperType]}, "
                "got 'invalid'"
            ),
        ):
            GripperObservationMetadata(
                raw_data_column_keys=["gripper"],
                dimension=1,
                dtype="int32",
                needs_normalization=False,
                gripper_type="invalid",
            )

    def test_invalid_binary_gripper_range_raises(self):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "binary_gripper_range must be one of "
                f"{[e.value for e in BinaryGripperRange]}, got 'zero_two'"
            ),
        ):
            GripperObservationMetadata(
                raw_data_column_keys=["gripper"],
                dimension=1,
                dtype="int32",
                needs_normalization=False,
                gripper_type=GripperType.BINARY.value,
                binary_gripper_range="zero_two",
            )

    def test_binary_dimension_not_one_raises(self):
        with pytest.raises(
            ValueError, match=re.escape("Binary gripper state dimension must be 1.")
        ):
            GripperObservationMetadata(
                raw_data_column_keys=["g1", "g2"],
                dimension=2,
                dtype="int32",
                needs_normalization=False,
                gripper_type=GripperType.BINARY.value,
            )

    def test_binary_with_normalization_raises(self):
        with pytest.raises(
            ValueError,
            match=re.escape("Binary gripper state should not need normalization."),
        ):
            GripperObservationMetadata(
                raw_data_column_keys=["gripper"],
                dimension=1,
                dtype="int32",
                needs_normalization=True,
                gripper_type=GripperType.BINARY.value,
            )

    @pytest.mark.parametrize(
        "dtype, expectation",
        [
            ("int32", does_not_raise()),
            ("int64", does_not_raise()),
            ("float32", pytest.raises(ValueError, match="integer type")),
            ("float64", pytest.raises(ValueError, match="integer type")),
            ("string", pytest.raises(ValueError, match="integer type")),
            ("bool", pytest.raises(ValueError, match="integer type")),
        ],
    )
    def test_binary_gripper_dtype_validation(self, dtype, expectation):
        with expectation:
            GripperObservationMetadata(
                raw_data_column_keys=["gripper"],
                dimension=1,
                dtype=dtype,
                needs_normalization=False,
                gripper_type=GripperType.BINARY.value,
            )

    def test_continuous_with_non_float_dtype_raises(self):
        with pytest.raises(ValueError, match="float type"):
            GripperObservationMetadata(
                raw_data_column_keys=["gripper"],
                dimension=1,
                dtype="int32",
                needs_normalization=True,
                gripper_type=GripperType.CONTINUOUS.value,
            )

    def test_continuous_valid(self):
        metadata = GripperObservationMetadata(
            raw_data_column_keys=["gripper"],
            dimension=1,
            dtype="float32",
            needs_normalization=True,
            gripper_type=GripperType.CONTINUOUS.value,
        )
        assert metadata.gripper_type == GripperType.CONTINUOUS.value

    def test_sets_proprioception_type_to_gripper(self):
        metadata = GripperObservationMetadata(
            raw_data_column_keys=["gripper"],
            dimension=1,
            dtype="int32",
            needs_normalization=False,
            gripper_type=GripperType.BINARY.value,
        )
        assert metadata.proprioception_type == ProprioceptiveType.GRIPPER.value

    def test_equality_includes_gripper_type_and_range(self):
        zero_one = GripperObservationMetadata(
            raw_data_column_keys=["gripper"],
            dimension=1,
            dtype="int32",
            needs_normalization=False,
            gripper_type=GripperType.BINARY.value,
            binary_gripper_range=BinaryGripperRange.ZERO_ONE.value,
        )
        minus_one_one = GripperObservationMetadata(
            raw_data_column_keys=["gripper"],
            dimension=1,
            dtype="int32",
            needs_normalization=False,
            gripper_type=GripperType.BINARY.value,
            binary_gripper_range=BinaryGripperRange.MINUS_ONE_ONE.value,
        )
        assert zero_one != minus_one_one


class TestCameraMetadata:
    @pytest.mark.parametrize(
        "camera_key, expectation",
        [
            (Cameras.LEFT.value, does_not_raise()),
            (Cameras.RIGHT.value, does_not_raise()),
            (Cameras.DEPTH.value, does_not_raise()),
            (
                "nonexistent_camera",
                pytest.raises(ValueError, match="must be a valid raw camera key"),
            ),
        ],
    )
    def test_camera_key_validation(self, camera_key, expectation):
        with expectation:
            CameraMetadata(
                camera_key=camera_key,
                dtype="uint8",
                channels=3,
                image_height=224,
                image_width=224,
            )

    def test_always_numerical_and_needs_normalization(self):
        metadata = CameraMetadata(
            camera_key=Cameras.LEFT.value,
            dtype="uint8",
            channels=3,
            image_height=224,
            image_width=224,
        )
        assert metadata.is_numerical
        assert metadata.needs_normalization

    def test_camera_key_maps_raw_key_to_canonical_key(self):
        metadata = CameraMetadata(
            camera_key=RawCameraKey.IMAGE.value,
            dtype="uint8",
            channels=3,
            image_height=224,
            image_width=224,
        )
        assert metadata.camera_key == Cameras.AGENTVIEW.value

    def test_raw_rgb_camera_is_rgb_after_mapping(self):
        metadata = RGBCameraMetadata(
            camera_key=RawCameraKey.IMAGE.value,
            dtype="uint8",
            image_height=224,
            image_width=224,
        )
        assert metadata.is_rgb

    def test_base_camera_metadata_has_no_modality(self):
        metadata = CameraMetadata(
            camera_key=Cameras.LEFT.value,
            dtype="uint8",
            channels=3,
            image_height=224,
            image_width=224,
        )
        expected_message = (
            "CameraMetadata does not define a modality. Use RGBCameraMetadata "
            "or DepthCameraMetadata."
        )
        with pytest.raises(NotImplementedError, match=re.escape(expected_message)):
            _ = metadata.modality

    @pytest.mark.parametrize(
        "channels, expected_single_channel",
        [
            (1, True),
            (3, False),
            (4, False),
        ],
    )
    def test_channel_type_properties(self, channels, expected_single_channel):
        metadata = CameraMetadata(
            camera_key=Cameras.LEFT.value,
            dtype="uint8",
            channels=channels,
            image_height=224,
            image_width=224,
        )
        assert metadata.is_single_channel == expected_single_channel
        assert not metadata.is_rgb

    def test_rgb_metadata_sets_channels_to_three(self):
        metadata = RGBCameraMetadata(
            camera_key=Cameras.LEFT.value,
            dtype="uint8",
            image_height=224,
            image_width=224,
        )
        assert metadata.channels == 3
        assert metadata.is_rgb
        assert metadata.modality == CameraModality.RGB
        assert metadata.max_pixel_value == 255.0

    def test_optional_image_dimensions(self):
        metadata = CameraMetadata(
            camera_key=Cameras.LEFT.value,
            dtype="uint8",
            channels=3,
            image_width=640,
            image_height=480,
        )
        assert metadata.image_width == 640
        assert metadata.image_height == 480

    def test_base_camera_max_pixel_value_defaults_to_none(self):
        metadata = CameraMetadata(
            camera_key=Cameras.LEFT.value,
            dtype="uint8",
            channels=3,
            image_width=640,
            image_height=480,
        )
        assert metadata.max_pixel_value is None

    def test_rejects_non_positive_max_pixel_value(self):
        expected_message = "max_pixel_value must be positive or None, got 0.0"
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            CameraMetadata(
                camera_key=Cameras.LEFT.value,
                dtype="uint8",
                channels=3,
                image_width=640,
                image_height=480,
                max_pixel_value=0.0,
            )

    def test_equality_ignores_dimensions(self):
        first = CameraMetadata(
            camera_key=Cameras.LEFT.value,
            dtype="uint8",
            channels=3,
            image_width=640,
            image_height=480,
        )
        second = CameraMetadata(
            camera_key=Cameras.LEFT.value,
            dtype="uint8",
            channels=3,
            image_width=320,
            image_height=240,
        )
        assert first == second

    def test_equality_differs_on_max_pixel_value(self):
        unscaled = CameraMetadata(
            camera_key=Cameras.LEFT.value,
            dtype="uint8",
            channels=3,
            image_width=640,
            image_height=480,
        )
        scaled = CameraMetadata(
            camera_key=Cameras.LEFT.value,
            dtype="uint8",
            channels=3,
            image_width=640,
            image_height=480,
            max_pixel_value=255.0,
        )
        assert unscaled != scaled

    def test_equality_differs_on_structural_fields(self):
        rgb = CameraMetadata(
            camera_key=Cameras.LEFT.value,
            dtype="uint8",
            channels=3,
            image_width=224,
            image_height=224,
        )
        depth = CameraMetadata(
            camera_key=Cameras.DEPTH.value,
            dtype="uint8",
            channels=1,
            image_width=224,
            image_height=224,
        )
        assert rgb != depth

    def test_equality_different_type_returns_not_implemented(self):
        metadata = CameraMetadata(
            camera_key=Cameras.LEFT.value,
            dtype="uint8",
            channels=3,
            image_height=224,
            image_width=224,
        )
        assert metadata.__eq__("not_camera") is NotImplemented


class TestDepthCameraMetadata:
    def test_sets_channels_to_one(self):
        metadata = DepthCameraMetadata(
            camera_key=Cameras.DEPTH.value,
            dtype="float32",
            image_height=224,
            image_width=224,
        )
        assert metadata.channels == 1

    def test_marks_camera_as_depth(self):
        metadata = DepthCameraMetadata(
            camera_key=Cameras.DEPTH.value,
            dtype="float32",
            image_height=224,
            image_width=224,
        )
        assert metadata.is_depth
        assert metadata.is_single_channel
        assert not metadata.is_rgb
        assert metadata.modality == CameraModality.DEPTH
        assert metadata.max_pixel_value is None

    def test_rejects_non_float_dtype(self):
        with pytest.raises(
            ValueError,
            match="Depth camera dtype must be a float type, got uint8",
        ):
            DepthCameraMetadata(
                camera_key=Cameras.DEPTH.value,
                dtype="uint8",
                image_height=224,
                image_width=224,
            )


class TestActionMetadata:
    @pytest.mark.parametrize("prediction_dimension", [0, -1])
    def test_non_positive_prediction_dimension_raises(self, prediction_dimension):
        with pytest.raises(ValueError, match="must be positive"):
            ActionMetadata(
                prediction_dimension=prediction_dimension,
                is_numerical=True,
                needs_normalization=True,
                dtype="float32",
                is_precomputed=True,
            )

    def test_default_action_type_is_custom(self):
        metadata = ActionMetadata(
            prediction_dimension=3,
            is_numerical=True,
            needs_normalization=True,
            dtype="float32",
            is_precomputed=True,
        )
        assert metadata.action_type == ProprioceptiveType.CUSTOM.value

    def test_default_requires_prediction_head_is_true(self):
        metadata = ActionMetadata(
            prediction_dimension=3,
            is_numerical=True,
            needs_normalization=True,
            dtype="float32",
            is_precomputed=True,
        )
        assert metadata.requires_prediction_head

    def test_requires_prediction_head_can_be_disabled(self):
        metadata = ActionMetadata(
            prediction_dimension=3,
            is_numerical=True,
            needs_normalization=True,
            dtype="float32",
            is_precomputed=True,
            requires_prediction_head=False,
        )
        assert not metadata.requires_prediction_head


class TestOnTheFlyActionMetadata:
    def test_non_numerical_source_raises(self):
        source = ObservationMetadata(
            raw_data_column_keys=["label"],
            dimension=1,
            dtype="string",
            is_numerical=False,
            needs_normalization=False,
        )
        with pytest.raises(ValueError, match="must be numerical"):
            OnTheFlyActionMetadata(source_metadata=source)

    @pytest.mark.parametrize(
        "method, expectation",
        [(member.value, does_not_raise()) for member in ActionComputationMethod]
        + [
            (
                "velocity",
                pytest.raises(ValueError, match="computation_method must be one of"),
            ),
        ],
    )
    def test_computation_method_validation(self, method, expectation):
        source = PositionObservationMetadata(
            raw_data_column_keys=["x", "y", "z"],
            dimension=3,
            dtype="float32",
            needs_normalization=True,
        )
        with expectation:
            OnTheFlyActionMetadata(
                source_metadata=source,
                computation_method=method,
            )

    def test_inherits_dimension_from_source(self):
        source = PositionObservationMetadata(
            raw_data_column_keys=["x", "y", "z"],
            dimension=3,
            dtype="float32",
            needs_normalization=True,
        )
        metadata = OnTheFlyActionMetadata(source_metadata=source)
        assert metadata.prediction_dimension == 3

    def test_inherits_normalization_from_source(self):
        source = PositionObservationMetadata(
            raw_data_column_keys=["x", "y", "z"],
            dimension=3,
            dtype="float32",
            needs_normalization=False,
        )
        metadata = OnTheFlyActionMetadata(source_metadata=source)
        assert not metadata.needs_normalization

    def test_action_type_set_from_position_source(self):
        source = PositionObservationMetadata(
            raw_data_column_keys=["x", "y", "z"],
            dimension=3,
            dtype="float32",
            needs_normalization=True,
        )
        metadata = OnTheFlyActionMetadata(source_metadata=source)
        assert metadata.action_type == ProprioceptiveType.POSITION.value

    def test_action_type_set_from_orientation_source(self):
        source = OrientationObservationMetadata(
            raw_data_column_keys=["roll"],
            dimension=1,
            dtype="float32",
            needs_normalization=True,
        )
        metadata = OnTheFlyActionMetadata(source_metadata=source)
        assert metadata.action_type == ProprioceptiveType.ORIENTATION.value

    def test_action_type_set_from_gripper_source(self):
        source = GripperObservationMetadata(
            raw_data_column_keys=["gripper"],
            dimension=1,
            dtype="float32",
            needs_normalization=True,
            gripper_type=GripperType.CONTINUOUS.value,
        )
        metadata = OnTheFlyActionMetadata(source_metadata=source)
        assert metadata.action_type == ProprioceptiveType.GRIPPER.value

    def test_is_not_precomputed(self):
        source = PositionObservationMetadata(
            raw_data_column_keys=["x", "y", "z"],
            dimension=3,
            dtype="float32",
            needs_normalization=True,
        )
        metadata = OnTheFlyActionMetadata(source_metadata=source)
        assert not metadata.is_precomputed

    def test_equality_includes_source_and_method(self):
        source = PositionObservationMetadata(
            raw_data_column_keys=["x", "y", "z"],
            dimension=3,
            dtype="float32",
            needs_normalization=True,
        )
        delta = OnTheFlyActionMetadata(
            source_metadata=source,
            computation_method=ActionComputationMethod.DELTA.value,
        )
        next_timestep = OnTheFlyActionMetadata(
            source_metadata=source,
            computation_method=ActionComputationMethod.NEXT_TIMESTEP.value,
        )
        assert delta != next_timestep


class TestPrecomputedActionMetadata:
    def test_empty_raw_data_column_keys_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            PrecomputedActionMetadata(
                raw_data_column_keys=[],
                storage_dimension=3,
                prediction_dimension=3,
                is_numerical=True,
                needs_normalization=True,
                dtype="float32",
            )

    def test_zero_storage_dimension_raises(self):
        with pytest.raises(ValueError, match="storage_dimension must be positive"):
            PrecomputedActionMetadata(
                raw_data_column_keys=["x", "y", "z"],
                storage_dimension=0,
                prediction_dimension=3,
                is_numerical=True,
                needs_normalization=True,
                dtype="float32",
            )

    def test_slice_range_not_matching_prediction_dimension_raises(self):
        with pytest.raises(ValueError, match="must equal prediction_dimension"):
            PrecomputedActionMetadata(
                raw_data_column_keys=["x", "y", "z"],
                storage_dimension=7,
                prediction_dimension=3,
                is_numerical=True,
                needs_normalization=True,
                dtype="float32",
                slice_start=0,
                slice_end=5,
            )

    def test_valid_slice_succeeds(self):
        metadata = PrecomputedActionMetadata(
            raw_data_column_keys=["x", "y", "z"],
            storage_dimension=7,
            prediction_dimension=3,
            is_numerical=True,
            needs_normalization=True,
            dtype="float32",
            slice_start=0,
            slice_end=3,
        )
        assert metadata.slice_start == 0
        assert metadata.slice_end == 3

    def test_is_precomputed_flag_set(self):
        metadata = PrecomputedActionMetadata(
            raw_data_column_keys=["x", "y", "z"],
            storage_dimension=3,
            prediction_dimension=3,
            is_numerical=True,
            needs_normalization=True,
            dtype="float32",
        )
        assert metadata.is_precomputed

    def test_prediction_dimension_can_differ_from_storage(self):
        metadata = PrecomputedActionMetadata(
            raw_data_column_keys=["phase"],
            storage_dimension=1,
            prediction_dimension=5,
            is_numerical=True,
            needs_normalization=False,
            dtype="int32",
        )
        assert metadata.storage_dimension == 1
        assert metadata.prediction_dimension == 5


class TestPositionActionMetadata:
    def test_invalid_frame_raises(self):
        with pytest.raises(ValueError, match="frame must be one of"):
            PositionActionMetadata(
                frame="invalid",
                raw_data_column_keys=["x", "y", "z"],
                storage_dimension=3,
                prediction_dimension=3,
                needs_normalization=True,
                dtype="float32",
            )

    @pytest.mark.parametrize(
        "method, expectation",
        [(member.value, does_not_raise()) for member in ActionComputationMethod]
        + [
            (
                "velocity",
                pytest.raises(ValueError, match="computation_method must be one of"),
            ),
            (None, does_not_raise()),
        ],
    )
    def test_computation_method_validation(self, method, expectation):
        with expectation:
            PositionActionMetadata(
                frame=CoordinateSystem.ROBOT_BASE.value,
                raw_data_column_keys=["x", "y"],
                storage_dimension=2,
                prediction_dimension=2,
                needs_normalization=True,
                dtype="float32",
                computation_method=method,
            )

    def test_sets_action_type_to_position(self):
        metadata = PositionActionMetadata(
            frame=CoordinateSystem.ROBOT_BASE.value,
            raw_data_column_keys=["x", "y", "z"],
            storage_dimension=3,
            prediction_dimension=3,
            needs_normalization=True,
            dtype="float32",
        )
        assert metadata.action_type == ProprioceptiveType.POSITION.value

    def test_equality_includes_frame(self):
        robot = PositionActionMetadata(
            frame=CoordinateSystem.ROBOT_BASE.value,
            raw_data_column_keys=["x", "y", "z"],
            storage_dimension=3,
            prediction_dimension=3,
            needs_normalization=True,
            dtype="float32",
        )
        camera = PositionActionMetadata(
            frame=CoordinateSystem.CAMERA.value,
            raw_data_column_keys=["x", "y", "z"],
            storage_dimension=3,
            prediction_dimension=3,
            needs_normalization=True,
            dtype="float32",
        )
        assert robot != camera

    def test_equality_includes_computation_method(self):
        delta = PositionActionMetadata(
            frame=CoordinateSystem.ROBOT_BASE.value,
            raw_data_column_keys=["x", "y"],
            storage_dimension=2,
            prediction_dimension=2,
            needs_normalization=True,
            dtype="float32",
            computation_method=ActionComputationMethod.DELTA.value,
        )
        next_timestep = PositionActionMetadata(
            frame=CoordinateSystem.ROBOT_BASE.value,
            raw_data_column_keys=["x", "y"],
            storage_dimension=2,
            prediction_dimension=2,
            needs_normalization=True,
            dtype="float32",
            computation_method=ActionComputationMethod.NEXT_TIMESTEP.value,
        )
        assert delta != next_timestep

    def test_equality_treats_missing_computation_method_as_none(self):
        current = PositionActionMetadata(
            frame=CoordinateSystem.ROBOT_BASE.value,
            raw_data_column_keys=["x", "y"],
            storage_dimension=2,
            prediction_dimension=2,
            needs_normalization=True,
            dtype="float32",
        )
        legacy = PositionActionMetadata(
            frame=CoordinateSystem.ROBOT_BASE.value,
            raw_data_column_keys=["x", "y"],
            storage_dimension=2,
            prediction_dimension=2,
            needs_normalization=True,
            dtype="float32",
        )
        del legacy.computation_method
        assert current == legacy


class TestOrientationActionMetadata:
    def test_invalid_frame_raises(self):
        with pytest.raises(ValueError, match="frame must be one of"):
            OrientationActionMetadata(
                frame="invalid",
                orientation_representation=OrientationRepresentation.ROLL.value,
                raw_data_column_keys=["roll"],
                storage_dimension=1,
                prediction_dimension=1,
                needs_normalization=True,
                dtype="float32",
            )

    def test_invalid_orientation_representation_raises(self):
        with pytest.raises(
            ValueError, match="orientation_representation must be one of"
        ):
            OrientationActionMetadata(
                frame=CoordinateSystem.ROBOT_BASE.value,
                orientation_representation="rotation_matrix",
                raw_data_column_keys=["roll"],
                storage_dimension=1,
                prediction_dimension=1,
                needs_normalization=True,
                dtype="float32",
            )

    def test_sets_action_type_to_orientation(self):
        metadata = OrientationActionMetadata(
            frame=CoordinateSystem.ROBOT_BASE.value,
            orientation_representation=OrientationRepresentation.ROLL.value,
            raw_data_column_keys=["roll"],
            storage_dimension=1,
            prediction_dimension=1,
            needs_normalization=True,
            dtype="float32",
        )
        assert metadata.action_type == ProprioceptiveType.ORIENTATION.value

    def test_equality_includes_frame_and_representation(self):
        roll = OrientationActionMetadata(
            frame=CoordinateSystem.ROBOT_BASE.value,
            orientation_representation=OrientationRepresentation.ROLL.value,
            raw_data_column_keys=["roll"],
            storage_dimension=1,
            prediction_dimension=1,
            needs_normalization=True,
            dtype="float32",
        )
        euler = OrientationActionMetadata(
            frame=CoordinateSystem.ROBOT_BASE.value,
            orientation_representation=OrientationRepresentation.EULER.value,
            raw_data_column_keys=["r", "p", "y"],
            storage_dimension=3,
            prediction_dimension=3,
            needs_normalization=True,
            dtype="float32",
        )
        assert roll != euler


class TestGripperActionMetadata:
    def test_invalid_gripper_type_raises(self):
        with pytest.raises(ValueError, match="gripper_type must be one of"):
            GripperActionMetadata(
                gripper_type="invalid",
                raw_data_column_keys=["gripper"],
                storage_dimension=1,
                prediction_dimension=1,
                needs_normalization=False,
                dtype="int32",
            )

    def test_binary_with_normalization_raises(self):
        with pytest.raises(ValueError, match="should not need normalization"):
            GripperActionMetadata(
                gripper_type=GripperType.BINARY.value,
                raw_data_column_keys=["gripper"],
                storage_dimension=1,
                prediction_dimension=1,
                needs_normalization=True,
                dtype="int32",
            )

    @pytest.mark.parametrize("dtype", ["float32", "string", "bool"])
    def test_binary_with_non_integer_dtype_raises(self, dtype):
        with pytest.raises(ValueError, match="integer type"):
            GripperActionMetadata(
                gripper_type=GripperType.BINARY.value,
                raw_data_column_keys=["gripper"],
                storage_dimension=1,
                prediction_dimension=1,
                needs_normalization=False,
                dtype=dtype,
            )

    def test_continuous_with_non_float_dtype_raises(self):
        with pytest.raises(ValueError, match="float type"):
            GripperActionMetadata(
                gripper_type=GripperType.CONTINUOUS.value,
                raw_data_column_keys=["gripper"],
                storage_dimension=1,
                prediction_dimension=1,
                needs_normalization=True,
                dtype="int32",
            )

    def test_sets_action_type_to_gripper(self):
        metadata = GripperActionMetadata(
            gripper_type=GripperType.BINARY.value,
            raw_data_column_keys=["gripper"],
            storage_dimension=1,
            prediction_dimension=1,
            needs_normalization=False,
            dtype="int32",
        )
        assert metadata.action_type == ProprioceptiveType.GRIPPER.value

    def test_equality_includes_gripper_type_and_range(self):
        zero_one = GripperActionMetadata(
            gripper_type=GripperType.BINARY.value,
            raw_data_column_keys=["gripper"],
            storage_dimension=1,
            prediction_dimension=1,
            needs_normalization=False,
            dtype="int32",
            binary_gripper_range=BinaryGripperRange.ZERO_ONE.value,
        )
        minus_one_one = GripperActionMetadata(
            gripper_type=GripperType.BINARY.value,
            raw_data_column_keys=["gripper"],
            storage_dimension=1,
            prediction_dimension=1,
            needs_normalization=False,
            dtype="int32",
            binary_gripper_range=BinaryGripperRange.MINUS_ONE_ONE.value,
        )
        assert zero_one != minus_one_one

    def test_continuous_valid(self):
        metadata = GripperActionMetadata(
            gripper_type=GripperType.CONTINUOUS.value,
            raw_data_column_keys=["gripper"],
            storage_dimension=1,
            prediction_dimension=1,
            needs_normalization=True,
            dtype="float32",
        )
        assert metadata.gripper_type == GripperType.CONTINUOUS.value
        assert metadata.needs_normalization
