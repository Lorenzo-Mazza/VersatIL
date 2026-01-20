"""Configurations for metadata types.

`dtype` across all configs refers to zarr v3 storage data type.
zarr v3 allowed dtypes are defined here https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""
from dataclasses import dataclass
from typing import Any, Optional

from omegaconf import MISSING

from versatil.data.constants import (
    BinaryGripperRange,
    CoordinateSystem,
    GripperType,
    OrientationRepresentation,
)


@dataclass
class ObservationMetadataConfig:
    """Config for ObservationMetadata."""

    _target_: str = "versatil.data.metadata.ObservationMetadata"
    raw_data_column_keys: list[str] = MISSING
    dimension: int = MISSING
    dtype: str = MISSING
    is_numerical: bool = MISSING
    needs_normalization: bool = MISSING


@dataclass
class PositionObservationMetadataConfig(ObservationMetadataConfig):
    """Config for PositionObservationMetadata."""

    _target_: str = "versatil.data.metadata.PositionObservationMetadata"
    frame: str = CoordinateSystem.ROBOT_BASE.value


@dataclass
class OrientationObservationMetadataConfig(ObservationMetadataConfig):
    """Config for OrientationObservationMetadata."""

    _target_: str = "versatil.data.metadata.OrientationObservationMetadata"
    frame: str = CoordinateSystem.ROBOT_BASE.value
    orientation_representation: str = OrientationRepresentation.ROLL.value


@dataclass
class GripperObservationMetadataConfig(ObservationMetadataConfig):
    """Config for GripperObservationMetadata."""

    _target_: str = "versatil.data.metadata.GripperObservationMetadata"
    gripper_type: str = GripperType.BINARY.value
    binary_gripper_range: str = BinaryGripperRange.ZERO_ONE.value


@dataclass
class CameraMetadataConfig:
    """Config for CameraMetadata."""

    _target_: str = "versatil.data.metadata.CameraMetadata"
    camera_key: str = MISSING
    dtype: str = MISSING
    channels: int = MISSING
    image_width: Optional[int] = None
    image_height: Optional[int] = None


@dataclass
class ActionMetadataConfig:
    """Config for ActionMetadata."""

    _target_: str = "versatil.data.metadata.ActionMetadata"
    prediction_dimension: int = MISSING
    is_numerical: bool = MISSING
    needs_normalization: bool = MISSING
    dtype: str = MISSING
    is_precomputed: bool = MISSING


@dataclass
class OnTheFlyActionMetadataConfig:
    """Config for OnTheFlyActionMetadata."""

    _target_: str = "versatil.data.metadata.OnTheFlyActionMetadata"
    source_metadata: Any = MISSING
    computation_method: str = MISSING


@dataclass
class PrecomputedActionMetadataConfig:
    """Config for PrecomputedActionMetadata."""

    _target_: str = "versatil.data.metadata.PrecomputedActionMetadata"
    raw_data_column_keys: list[str] = MISSING
    storage_dimension: int = MISSING
    prediction_dimension: int = MISSING
    is_numerical: bool = MISSING
    needs_normalization: bool = MISSING
    dtype: str = MISSING


@dataclass
class PositionActionMetadataConfig(PrecomputedActionMetadataConfig):
    """Config for PositionActionMetadata."""

    _target_: str = "versatil.data.metadata.PositionActionMetadata"
    frame: str = MISSING


@dataclass
class OrientationActionMetadataConfig(PrecomputedActionMetadataConfig):
    """Config for OrientationActionMetadata."""

    _target_: str = "versatil.data.metadata.OrientationActionMetadata"
    frame: str = MISSING
    orientation_representation: str = MISSING


@dataclass
class GripperActionMetadataConfig(PrecomputedActionMetadataConfig):
    """Config for GripperActionMetadata."""

    _target_: str = "versatil.data.metadata.GripperActionMetadata"
    gripper_type: str = MISSING
    binary_gripper_range: str = MISSING
