"""Configurations for metadata types.

`dtype` across all configs refers to zarr v3 storage data type.
zarr v3 allowed dtypes are defined here https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING

from versatil.data.constants import (
    BinaryGripperRange,
    CoordinateSystem,
    GripperType,
    OrientationRepresentation,
)


@dataclass
class BaseObservationMetadataConfig:
    """Fields shared by every observation metadata config.

    Attributes:
        _target_: Import path instantiated by Hydra.
        raw_data_column_keys: List of keys in the raw dataset corresponding to the
            observation.
        dimension: Dimension that will be used to store the observation in the zarr
            store.
        dtype: Numpy dtype the values are stored as.
        needs_normalization: Whether the observation is normalized by the fitted
            normalizer.
        slice_start: Optional starting index for slicing a larger stored observation
            vector.
        slice_end: Optional ending index (exclusive) for slicing a larger stored
            observation vector.
    """

    _target_: str = MISSING
    raw_data_column_keys: list[str] = MISSING
    dimension: int = MISSING
    dtype: str = MISSING
    needs_normalization: bool = MISSING
    slice_start: int | None = None
    slice_end: int | None = None


@dataclass
class ObservationMetadataConfig(BaseObservationMetadataConfig):
    """Config for ObservationMetadata.

    Attributes:
        _target_: Import path instantiated by Hydra.
        is_numerical: Whether the observation is numerical rather than text. The
            specialized observation configs omit this field because their runtime
            classes are numerical by definition.
    """

    _target_: str = "versatil.data.metadata.ObservationMetadata"
    is_numerical: bool = MISSING


@dataclass
class PositionObservationMetadataConfig(BaseObservationMetadataConfig):
    """Config for PositionObservationMetadata.

    Attributes:
        _target_: Import path instantiated by Hydra.
        frame: Coordinate frame of the position observation.
    """

    _target_: str = "versatil.data.metadata.PositionObservationMetadata"
    frame: str = CoordinateSystem.ROBOT_BASE.value


@dataclass
class OrientationObservationMetadataConfig(BaseObservationMetadataConfig):
    """Config for OrientationObservationMetadata.

    Attributes:
        _target_: Import path instantiated by Hydra.
        frame: Coordinate frame of the orientation observation.
        orientation_representation: Representation of the orientation.
    """

    _target_: str = "versatil.data.metadata.OrientationObservationMetadata"
    frame: str = CoordinateSystem.ROBOT_BASE.value
    orientation_representation: str = OrientationRepresentation.ROLL.value


@dataclass
class GripperObservationMetadataConfig(BaseObservationMetadataConfig):
    """Config for GripperObservationMetadata.

    Attributes:
        _target_: Import path instantiated by Hydra.
        gripper_type: Type of gripper ('binary' or 'continuous').
        binary_gripper_range: Range for binary gripper ('zero_one' or 'minus_one_one').
    """

    _target_: str = "versatil.data.metadata.GripperObservationMetadata"
    gripper_type: str = GripperType.BINARY.value
    binary_gripper_range: str = BinaryGripperRange.ZERO_ONE.value


@dataclass
class CameraMetadataConfig:
    """Config for CameraMetadata.

    Attributes:
        _target_: Import path instantiated by Hydra.
        camera_key: Camera identifier within the dataset.
        dtype: Numpy dtype the images are stored as.
        channels: Number of image channels.
        image_width: Target image width.
        image_height: Target image height.
        max_pixel_value: Optional value used to scale image tensors after resizing and
            channel reordering.
    """

    _target_: str = "versatil.data.metadata.CameraMetadata"
    camera_key: str = MISSING
    dtype: str = MISSING
    channels: int = MISSING
    image_width: int | None = None
    image_height: int | None = None
    max_pixel_value: float | None = None


@dataclass
class RGBCameraMetadataConfig:
    """Config for RGBCameraMetadata.

    Attributes:
        _target_: Import path instantiated by Hydra.
        camera_key: Key in the raw dataset corresponding to the RGB camera.
        dtype: Zarr storage dtype for RGB values.
        image_width: Target image width.
        image_height: Target image height.
        max_pixel_value: Value used to scale RGB image tensors after resizing and
            channel reordering.
    """

    _target_: str = "versatil.data.metadata.RGBCameraMetadata"
    camera_key: str = MISSING
    dtype: str = MISSING
    image_width: int | None = None
    image_height: int | None = None
    max_pixel_value: float | None = 255.0


@dataclass
class DepthCameraMetadataConfig:
    """Config for DepthCameraMetadata.

    Attributes:
        _target_: Import path instantiated by Hydra.
        camera_key: Key in the raw dataset corresponding to the depth camera.
        dtype: Zarr storage dtype for depth values.
        image_width: Target image width.
        image_height: Target image height.
        max_pixel_value: Optional value used to scale depth image tensors after resizing
            and channel reordering.
    """

    _target_: str = "versatil.data.metadata.DepthCameraMetadata"
    camera_key: str = MISSING
    dtype: str = MISSING
    image_width: int | None = None
    image_height: int | None = None
    max_pixel_value: float | None = None


@dataclass
class ActionMetadataConfig:
    """Config for ActionMetadata.

    Attributes:
        _target_: Import path instantiated by Hydra.
        prediction_dimension: Dimension for model prediction. May differ from storage,
            e.g., class labels stored as 1 column but predicted as n_classes logits.
        is_numerical: Whether the action is numerical rather than text.
        needs_normalization: Whether the action is normalized by the fitted normalizer.
        dtype: Numpy dtype the values are stored as.
        is_precomputed: Whether the action is stored in the dataset instead of computed
            on the fly.
    """

    _target_: str = "versatil.data.metadata.ActionMetadata"
    prediction_dimension: int = MISSING
    is_numerical: bool = MISSING
    needs_normalization: bool = MISSING
    dtype: str = MISSING
    is_precomputed: bool = MISSING


@dataclass
class OnTheFlyActionMetadataConfig:
    """Config for OnTheFlyActionMetadata.

    Attributes:
        _target_: Import path instantiated by Hydra.
        source_metadata: Metadata of the source observation used to compute the action.
        computation_method: Method to compute the action, default 'delta' for
            subtraction between consecutive observations.
    """

    _target_: str = "versatil.data.metadata.OnTheFlyActionMetadata"
    source_metadata: Any = MISSING
    computation_method: str = MISSING


@dataclass
class BasePrecomputedActionMetadataConfig:
    """Fields shared by every precomputed action metadata config.

    Attributes:
        _target_: Import path instantiated by Hydra.
        raw_data_column_keys: List of keys in the raw dataset corresponding to the
            action.
        storage_dimension: Dimension that will be used to store the action in the zarr
            store.
        prediction_dimension: Dimension for model prediction. May differ from storage,
            e.g., class labels stored as 1 column but predicted as n_classes logits.
        needs_normalization: Whether the action is normalized by the fitted normalizer.
        dtype: Numpy dtype the values are stored as.
    """

    _target_: str = MISSING
    raw_data_column_keys: list[str] = MISSING
    storage_dimension: int = MISSING
    prediction_dimension: int = MISSING
    needs_normalization: bool = MISSING
    dtype: str = MISSING


@dataclass
class PrecomputedActionMetadataConfig(BasePrecomputedActionMetadataConfig):
    """Config for PrecomputedActionMetadata.

    Attributes:
        _target_: Import path instantiated by Hydra.
        is_numerical: Whether the action is numerical rather than text. The
            specialized action configs omit this field because their runtime
            classes are numerical by definition.
    """

    _target_: str = "versatil.data.metadata.PrecomputedActionMetadata"
    is_numerical: bool = MISSING


@dataclass
class PositionActionMetadataConfig(BasePrecomputedActionMetadataConfig):
    """Config for PositionActionMetadata.

    Attributes:
        _target_: Import path instantiated by Hydra.
        frame: Coordinate frame of the position, camera or robot base.
        computation_method: Whether actions are deltas or next-timestep poses.
    """

    _target_: str = "versatil.data.metadata.PositionActionMetadata"
    frame: str = MISSING
    computation_method: str | None = None


@dataclass
class OrientationActionMetadataConfig(BasePrecomputedActionMetadataConfig):
    """Config for OrientationActionMetadata.

    Attributes:
        _target_: Import path instantiated by Hydra.
        frame: Coordinate frame of the orientation, camera or robot base.
        orientation_representation: Representation of the orientation.
    """

    _target_: str = "versatil.data.metadata.OrientationActionMetadata"
    frame: str = MISSING
    orientation_representation: str = MISSING


@dataclass
class GripperActionMetadataConfig(BasePrecomputedActionMetadataConfig):
    """Config for GripperActionMetadata.

    Attributes:
        _target_: Import path instantiated by Hydra.
        gripper_type: Type of gripper action ('binary' or 'continuous').
        binary_gripper_range: Range for binary gripper action ('zero_one' or
            'minus_one_one').
    """

    _target_: str = "versatil.data.metadata.GripperActionMetadata"
    gripper_type: str = MISSING
    binary_gripper_range: str = MISSING
