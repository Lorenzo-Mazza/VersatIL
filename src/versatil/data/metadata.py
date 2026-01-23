"""Metadata for actions and observations used across the codebase.
 `dtype` across all classes uses the zarr v3 type convention.
 zarr v3 allowed dtypes are defined here https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from typing import Optional

from versatil.data.constants import (
    CoordinateSystem,
    ProprioceptiveType,
    OrientationRepresentation,
    GripperType,
    BinaryGripperRange,
    VALID_CAMERAS,
    ActionComputationMethod,
)


class BaseMetadata:
    """Base metadata class.

    Attributes:
        dtype: Zarr store data type.
        is_numerical: Whether the observation is numerical (float or int) or not (e.g. string or categorical).
        needs_normalization: Whether the data needs normalization at runtime.
    """

    def __init__(
        self,
        dtype: str,
        is_numerical: bool,
        needs_normalization: bool,
    ):
        if not is_numerical:
            if needs_normalization:
                raise ValueError(
                    "Non-numerical observations should not need normalization."
                )
        else:
            if "float" not in dtype and "int" not in dtype:
                raise ValueError(
                    f"dtype for numerical observations must be float or int type, got {dtype}"
                )
        self.dtype = dtype
        self.is_numerical = is_numerical
        self.needs_normalization = needs_normalization

    def __eq__(self, other: object) -> bool:
        """Equality function."""
        if not isinstance(other, BaseMetadata):
            return NotImplemented
        return (
            self.dtype == other.dtype
            and self.is_numerical == other.is_numerical
            and self.needs_normalization == other.needs_normalization
        )


class ObservationMetadata(BaseMetadata):
    """Base observation metadata.

    Attributes:
        raw_data_column_keys: List of keys in the raw dataset corresponding to the observation.
        dimension: Dimension that will be used to store the observation in the zarr store.
        slice_start: Optional starting index for slicing a larger stored observation vector.
        slice_end: Optional ending index (exclusive) for slicing a larger stored observation vector.
    """

    def __init__(
        self,
        raw_data_column_keys: list[str],
        dimension: int,
        dtype: str,
        is_numerical: bool,
        needs_normalization: bool,
        slice_start: Optional[int] = None,
        slice_end: Optional[int] = None,
    ):
        super().__init__(dtype, is_numerical, needs_normalization)
        if not raw_data_column_keys:
            raise ValueError("raw_data_column_keys cannot be empty")
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        if slice_start is not None and slice_end is not None:
            if slice_start < 0 or slice_end < 0:
                raise ValueError("slice_start and slice_end must be non-negative")
            if slice_start >= slice_end:
                raise ValueError(
                    f"slice_start ({slice_start}) must be less than slice_end ({slice_end})"
                )
            if slice_end - slice_start != dimension:
                raise ValueError(
                    f"Slice range ({slice_end - slice_start}) must equal dimension ({dimension})"
                )
        self.raw_data_column_keys = raw_data_column_keys
        self.dimension = dimension
        self.slice_start = slice_start
        self.slice_end = slice_end

    def __eq__(self, other: object) -> bool:
        """Equality function."""
        if not isinstance(other, ObservationMetadata):
            return NotImplemented
        return (
            super().__eq__(other)
            and self.raw_data_column_keys == other.raw_data_column_keys
            and self.dimension == other.dimension
            and self.slice_start == other.slice_start
            and self.slice_end == other.slice_end
        )


class PositionObservationMetadata(ObservationMetadata):
    """Robot position observation metadata.

    Attributes:
        frame: Coordinate frame of the position observation.
    """

    def __init__(
        self,
        raw_data_column_keys: list[str],
        dimension: int,
        dtype: str,
        needs_normalization: bool,
        frame: str = CoordinateSystem.ROBOT_BASE.value,
        slice_start: Optional[int] = None,
        slice_end: Optional[int] = None,
    ):
        if "float" not in dtype:
            raise ValueError("Position observations dtype must be a float type.")
        super().__init__(
            raw_data_column_keys=raw_data_column_keys,
            dimension=dimension,
            dtype=dtype,
            is_numerical=True,
            needs_normalization=needs_normalization,
            slice_start=slice_start,
            slice_end=slice_end,
        )
        valid_frames = [e.value for e in CoordinateSystem]
        if frame not in valid_frames:
            raise ValueError(f"frame must be one of {valid_frames}, got '{frame}'")
        self.frame: str = frame
        self.proprioception_type: str = ProprioceptiveType.POSITION.value

    def __eq__(self, other: object) -> bool:
        """Equality function."""
        if not isinstance(other, PositionObservationMetadata):
            return NotImplemented
        return super().__eq__(other) and self.frame == other.frame


class OrientationObservationMetadata(ObservationMetadata):
    """Robot orientation observation metadata.

    Attributes:
        frame: Coordinate frame of the orientation observation.
        orientation_representation: Representation of the orientation.
    """

    def __init__(
        self,
        raw_data_column_keys: list[str],
        dimension: int,
        dtype: str,
        needs_normalization: bool,
        frame: str = CoordinateSystem.ROBOT_BASE.value,
        orientation_representation: str = OrientationRepresentation.ROLL.value,
        slice_start: Optional[int] = None,
        slice_end: Optional[int] = None,
    ):
        if "float" not in dtype:
            raise ValueError("Orientation observations dtype must be a float type.")
        super().__init__(
            raw_data_column_keys=raw_data_column_keys,
            dimension=dimension,
            dtype=dtype,
            is_numerical=True,
            needs_normalization=needs_normalization,
            slice_start=slice_start,
            slice_end=slice_end,
        )
        valid_frames = [e.value for e in CoordinateSystem]
        if frame not in valid_frames:
            raise ValueError(f"frame must be one of {valid_frames}, got '{frame}'")
        valid_methods = [e.value for e in OrientationRepresentation]
        if orientation_representation not in valid_methods:
            raise ValueError(
                f"orientation_representation must be one of {valid_methods}, got '{orientation_representation}'"
            )
        self.frame = frame
        self.orientation_representation = orientation_representation
        self.proprioception_type: str = ProprioceptiveType.ORIENTATION.value

    def __eq__(self, other: object) -> bool:
        """Equality function."""
        if not isinstance(other, OrientationObservationMetadata):
            return NotImplemented
        return (
            super().__eq__(other)
            and self.frame == other.frame
            and self.orientation_representation == other.orientation_representation
        )


class GripperObservationMetadata(ObservationMetadata):
    """Gripper state observation metadata, representing the clamps state of the gripper.

    Attributes:
        gripper_type: Type of gripper ('binary' or 'continuous').
        binary_gripper_range: Range for binary gripper ('zero_one' or 'minus_one_one').
    """

    def __init__(
        self,
        raw_data_column_keys: list[str],
        dimension: int,
        dtype: str,
        needs_normalization: bool,
        gripper_type: str = GripperType.BINARY.value,
        binary_gripper_range: str = BinaryGripperRange.ZERO_ONE.value,
        slice_start: Optional[int] = None,
        slice_end: Optional[int] = None,
    ):
        super().__init__(
            raw_data_column_keys=raw_data_column_keys,
            dimension=dimension,
            dtype=dtype,
            is_numerical=True,
            needs_normalization=needs_normalization,
            slice_start=slice_start,
            slice_end=slice_end,
        )
        valid_types = [e.value for e in GripperType]
        if gripper_type not in valid_types:
            raise ValueError(
                f"gripper_type must be one of {valid_types}, got '{gripper_type}'"
            )
        valid_ranges = [e.value for e in BinaryGripperRange]
        if binary_gripper_range not in valid_ranges:
            raise ValueError(
                f"binary_gripper_range must be one of {valid_ranges}, got '{binary_gripper_range}'"
            )
        if gripper_type == GripperType.BINARY.value:
            if dimension != 1:
                raise ValueError("Binary gripper state dimension must be 1.")
            if needs_normalization:
                raise ValueError("Binary gripper state should not need normalization.")
            if dtype != "bool" and "int" not in dtype:
                raise ValueError(
                    "Binary gripper state dtype must be 'bool' or an integer type."
                )
        else:
            if "float" not in dtype or not self.is_numerical:
                raise ValueError("Continuous gripper state dtype must be a float type.")

        self.gripper_type: str = gripper_type
        self.binary_gripper_range: str = binary_gripper_range
        self.proprioception_type: str = ProprioceptiveType.GRIPPER.value

    def __eq__(self, other: object) -> bool:
        """Equality function."""
        if not isinstance(other, GripperObservationMetadata):
            return NotImplemented
        return (
            super().__eq__(other)
            and self.gripper_type == other.gripper_type
            and self.binary_gripper_range == other.binary_gripper_range
        )


ProprioceptiveObservationMetadata = (
    PositionObservationMetadata
    | OrientationObservationMetadata
    | GripperObservationMetadata
)


class CameraMetadata(BaseMetadata):
    """Camera observation metadata.

    Attributes:
        camera_key: Key in the raw dataset corresponding to the camera.
            It has to be one of `data.constants.VALID_CAMERAS` values.
        channels: Number of image channels.
        image_width: Optional target image width for resizing when storing images.
        image_height: Optional target image height for resizing when storing images.
    """

    def __init__(
        self,
        camera_key: str,
        dtype: str,
        channels: int,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ):
        super().__init__(dtype, is_numerical=True, needs_normalization=True)
        if camera_key not in VALID_CAMERAS:
            raise ValueError(
                f"camera_key has to be included in {VALID_CAMERAS}. Got {camera_key}"
            )
        self.camera_key = camera_key
        self.channels = channels
        self.image_width = image_width
        self.image_height = image_height

    def __eq__(self, other: object) -> bool:
        """Equality function."""
        if not isinstance(other, CameraMetadata):
            return NotImplemented
        return (
            super().__eq__(other)
            and self.camera_key == other.camera_key
            and self.channels == other.channels
            and self.image_width == other.image_width
            and self.image_height == other.image_height
        )


class ActionMetadata(BaseMetadata):
    """Action metadata.

    Attributes:
        prediction_dimension: Dimension for model prediction. May differ from storage,
            e.g., class labels stored as 1 column but predicted as n_classes logits.
        requires_prediction_head: Whether this action requires a prediction head.
            Set to False for auxiliary/meta data that doesn't need prediction
    """

    def __init__(
        self,
        prediction_dimension: int,
        is_numerical: bool,
        needs_normalization: bool,
        dtype: str,
        is_precomputed: bool,
        requires_prediction_head: bool = True,
    ):
        super().__init__(
            dtype=dtype,
            is_numerical=is_numerical,
            needs_normalization=needs_normalization,
        )
        if prediction_dimension <= 0:
            raise ValueError(
                f"prediction_dimension must be positive, got {prediction_dimension}"
            )
        self.prediction_dimension = prediction_dimension
        self.is_precomputed = is_precomputed
        self.action_type = ProprioceptiveType.CUSTOM.value
        self.requires_prediction_head = requires_prediction_head

    def __eq__(self, other: object) -> bool:
        """Equality function."""
        if not isinstance(other, ActionMetadata):
            return NotImplemented
        return (
            super().__eq__(other)
            and self.prediction_dimension == other.prediction_dimension
            and self.is_precomputed == other.is_precomputed
            and self.action_type == other.action_type
        )


class OnTheFlyActionMetadata(ActionMetadata):
    """Metadata for computing an action on-the-fly from a stored zarr observation group.

    This defines how to derive an action from stored observations at runtime.

    Args:
        source_metadata: Metadata of the source observation used to compute the action.
        computation_method: Method to compute the action, default 'delta' for subtraction between
            consecutive observations.

    """

    def __init__(
        self,
        source_metadata: ProprioceptiveObservationMetadata,
        computation_method: str = ActionComputationMethod.DELTA.value,
        requires_prediction_head: bool = True,
    ):
        if not source_metadata.is_numerical:
            raise ValueError("Source metadata for on-the-fly action must be numerical.")
        super().__init__(
            prediction_dimension=source_metadata.dimension,
            is_numerical=True,
            needs_normalization=source_metadata.needs_normalization,
            dtype=source_metadata.dtype,
            is_precomputed=False,
            requires_prediction_head=requires_prediction_head,
        )
        valid_methods = [e.value for e in ActionComputationMethod]
        if computation_method not in valid_methods:
            raise ValueError(
                f"computation_method must be one of {valid_methods}, got '{computation_method}'"
            )
        self.source_metadata: ProprioceptiveObservationMetadata = source_metadata
        self.computation_method: str = computation_method
        self.action_type: str = source_metadata.proprioception_type

    def __eq__(self, other: object) -> bool:
        """Equality function."""
        if not isinstance(other, OnTheFlyActionMetadata):
            return NotImplemented
        return (
            super().__eq__(other)
            and self.source_metadata == other.source_metadata
            and self.computation_method == other.computation_method
        )


class PrecomputedActionMetadata(ActionMetadata):
    """Precomputed action metadata.

    Attributes:
        raw_data_column_keys: List of keys in the raw dataset corresponding to the action.
        storage_dimension: Dimension that will be used to store the action in the zarr store.
        prediction_dimension: Dimension for model prediction. May differ from storage,
            e.g., class labels stored as 1 column but predicted as n_classes logits.
        slice_start: Optional starting index for slicing a larger stored action vector.
        slice_end: Optional ending index (exclusive) for slicing a larger stored action vector.
    """

    def __init__(
        self,
        raw_data_column_keys: list[str],
        storage_dimension: int,
        prediction_dimension: int,
        is_numerical: bool,
        needs_normalization: bool,
        dtype: str,
        slice_start: Optional[int] = None,
        slice_end: Optional[int] = None,
        requires_prediction_head: bool = True,
    ):
        super().__init__(
            prediction_dimension=prediction_dimension,
            is_numerical=is_numerical,
            needs_normalization=needs_normalization,
            dtype=dtype,
            is_precomputed=True,
            requires_prediction_head=requires_prediction_head,
        )
        if not raw_data_column_keys:
            raise ValueError("raw_data_column_keys cannot be empty")
        if storage_dimension <= 0:
            raise ValueError(
                f"storage_dimension must be positive, got {storage_dimension}"
            )
        if slice_start is not None and slice_end is not None:
            if slice_start < 0 or slice_end < 0:
                raise ValueError("slice_start and slice_end must be non-negative")
            if slice_start >= slice_end:
                raise ValueError(
                    f"slice_start ({slice_start}) must be less than slice_end ({slice_end})"
                )
            if slice_end - slice_start != prediction_dimension:
                raise ValueError(
                    f"Slice range ({slice_end - slice_start}) must equal prediction_dimension ({prediction_dimension})"
                )
        self.raw_data_column_keys = raw_data_column_keys
        self.storage_dimension = storage_dimension
        self.slice_start = slice_start
        self.slice_end = slice_end
        self.action_type = ProprioceptiveType.CUSTOM.value

    def __eq__(self, other: object) -> bool:
        """Equality function."""
        if not isinstance(other, PrecomputedActionMetadata):
            return NotImplemented
        return (
            super().__eq__(other)
            and self.raw_data_column_keys == other.raw_data_column_keys
            and self.storage_dimension == other.storage_dimension
            and self.slice_start == other.slice_start
            and self.slice_end == other.slice_end
        )


class PositionActionMetadata(PrecomputedActionMetadata):
    """Precomputed position action metadata."""

    def __init__(
        self,
        frame: str,
        raw_data_column_keys: list[str],
        storage_dimension: int,
        prediction_dimension: int,
        needs_normalization: bool,
        dtype: str,
        slice_start: Optional[int] = None,
        slice_end: Optional[int] = None,
    ):
        super().__init__(
            raw_data_column_keys=raw_data_column_keys,
            storage_dimension=storage_dimension,
            prediction_dimension=prediction_dimension,
            is_numerical=True,
            needs_normalization=needs_normalization,
            dtype=dtype,
            slice_start=slice_start,
            slice_end=slice_end,
        )
        valid_frames = [e.value for e in CoordinateSystem]
        if frame not in valid_frames:
            raise ValueError(f"frame must be one of {valid_frames}, got '{frame}'")
        self.frame = frame
        self.action_type = ProprioceptiveType.POSITION.value

    def __eq__(self, other: object) -> bool:
        """Equality function."""
        if not isinstance(other, PositionActionMetadata):
            return NotImplemented
        return super().__eq__(other) and self.frame == other.frame


class OrientationActionMetadata(PrecomputedActionMetadata):
    """Precomputed orientation action metadata.

    Attributes:
        orientation_representation: Representation of the orientation.
    """

    def __init__(
        self,
        frame: str,
        orientation_representation: str,
        raw_data_column_keys: list[str],
        storage_dimension: int,
        prediction_dimension: int,
        needs_normalization: bool,
        dtype: str,
        slice_start: Optional[int] = None,
        slice_end: Optional[int] = None,
    ):
        super().__init__(
            raw_data_column_keys=raw_data_column_keys,
            storage_dimension=storage_dimension,
            prediction_dimension=prediction_dimension,
            is_numerical=True,
            needs_normalization=needs_normalization,
            dtype=dtype,
            slice_start=slice_start,
            slice_end=slice_end,
        )
        valid_frames = [e.value for e in CoordinateSystem]
        if frame not in valid_frames:
            raise ValueError(f"frame must be one of {valid_frames}, got '{frame}'")
        valid_methods = [e.value for e in OrientationRepresentation]
        if orientation_representation not in valid_methods:
            raise ValueError(
                f"orientation_representation must be one of {valid_methods}, got '{orientation_representation}'"
            )
        self.raw_data_column_keys = raw_data_column_keys
        self.storage_dimension = storage_dimension
        self.frame = frame
        self.orientation_representation = orientation_representation
        self.action_type = ProprioceptiveType.ORIENTATION.value

    def __eq__(self, other: object) -> bool:
        """Equality function."""
        if not isinstance(other, OrientationActionMetadata):
            return NotImplemented
        return (
            super().__eq__(other)
            and self.frame == other.frame
            and self.orientation_representation == other.orientation_representation
        )


class GripperActionMetadata(PrecomputedActionMetadata):
    """Precomputed gripper action metadata.

    This class represents gripper actions, which can be either binary (open/close) or continuous (partial open).

    Attributes:
        gripper_type: Type of gripper action ('binary' or 'continuous').
        binary_gripper_range: Range for binary gripper action ('zero_one' or 'minus_one_one').
    """

    def __init__(
        self,
        gripper_type: str,
        binary_gripper_range: str,
        raw_data_column_keys: list[str],
        storage_dimension: int,
        prediction_dimension: int,
        needs_normalization: bool,
        dtype: str,
        slice_start: Optional[int] = None,
        slice_end: Optional[int] = None,
    ):
        super().__init__(
            prediction_dimension=prediction_dimension,
            is_numerical=True,
            needs_normalization=needs_normalization,
            dtype=dtype,
            raw_data_column_keys=raw_data_column_keys,
            storage_dimension=storage_dimension,
            slice_start=slice_start,
            slice_end=slice_end,
        )
        valid_types = [e.value for e in GripperType]
        if gripper_type not in valid_types:
            raise ValueError(
                f"gripper_type must be one of {valid_types}, got '{gripper_type}'"
            )
        valid_ranges = [e.value for e in BinaryGripperRange]
        if binary_gripper_range not in valid_ranges:
            raise ValueError(
                f"binary_gripper_range must be one of {valid_ranges}, got '{binary_gripper_range}'"
            )
        if gripper_type == GripperType.BINARY.value:
            if needs_normalization:
                raise ValueError("Binary gripper action should not need normalization.")
            if dtype != "bool" and "int" not in dtype:
                raise ValueError(
                    "Binary gripper action dtype must be 'bool' or an integer type."
                )
        else:
            if "float" not in dtype or not self.is_numerical:
                raise ValueError(
                    "Continuous gripper action dtype must be a float type."
                )

        self.gripper_type: str = gripper_type
        self.binary_gripper_range: str = binary_gripper_range
        self.raw_data_column_keys = raw_data_column_keys
        self.storage_dimension = storage_dimension
        self.action_type = ProprioceptiveType.GRIPPER.value

    def __eq__(self, other: object) -> bool:
        """Equality function."""
        if not isinstance(other, GripperActionMetadata):
            return NotImplemented
        return (
            super().__eq__(other)
            and self.gripper_type == other.gripper_type
            and self.binary_gripper_range == other.binary_gripper_range
        )


ProprioceptiveActionMetadata = (
    PositionActionMetadata | OrientationActionMetadata | GripperActionMetadata
)
