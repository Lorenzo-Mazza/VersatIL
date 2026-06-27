"""Dataset schema path overrides for offline explainability runs."""

from copy import deepcopy
from pathlib import Path

from versatil.data.raw.schemas.base import DatasetSchema
from versatil.data.raw.schemas.csv import CsvDatasetSchema
from versatil.data.raw.schemas.custom.synthetic import SyntheticSchema
from versatil.data.raw.schemas.hdf5 import Hdf5DatasetSchema
from versatil.data.raw.schemas.lerobot import (
    LeRobotDatasetMetadataV30,
    LeRobotDatasetSchemaV30,
)

OFFLINE_DATASET_ZARR_NAME = "offline_dataset.zarr"
ZARR_SUFFIX = ".zarr"


def resolve_dataset_schema_for_explanation(
    schema: DatasetSchema,
    data_path_override: str | list[str] | None,
    zarr_cache_directory: Path,
) -> DatasetSchema:
    """Return the dataset schema used by an offline explanation run.

    Args:
        schema: Checkpoint task schema. Its metadata remains the structure
            contract for offline explanation data.
        data_path_override: Optional offline input location to explain instead
            of the data path stored in the checkpoint task config. ``None``
            returns ``schema`` unchanged. A single path ending in ``.zarr`` is
            an existing replay buffer and is sampled directly. A non-zarr path
            is raw data in the same dataset schema as the checkpoint. A list is
            only for raw schemas that already support multiple inputs, such as
            CSV ``dataset_folders`` or HDF5 ``hdf5_paths``; multiple zarr paths
            are rejected.
        zarr_cache_directory: Directory where raw override data is converted to
            a zarr replay buffer before sampling episodic windows.

    Returns:
        Schema to pass to the existing zarr creation and episodic dataset
        pipeline.

    Raises:
        ValueError: If the override is empty, points to missing input data, uses
            a zarr path that does not exist, or does not match the checkpoint
            schema storage type.
    """
    if data_path_override is None:
        return schema

    data_paths = _normalize_data_paths(data_path_override=data_path_override)
    if _is_zarr_override(data_paths=data_paths):
        return _resolve_zarr_override(schema=schema, zarr_path=data_paths[0])

    for data_path in data_paths:
        if not data_path.exists():
            raise ValueError(f"data_path_override path does not exist: {data_path}")

    override_schema = deepcopy(schema)
    override_schema.zarr_path = str(zarr_cache_directory / OFFLINE_DATASET_ZARR_NAME)

    if isinstance(override_schema, CsvDatasetSchema):
        override_schema.dataset_folders = [str(data_path) for data_path in data_paths]
    elif isinstance(override_schema, Hdf5DatasetSchema):
        override_schema.hdf5_paths = [str(data_path) for data_path in data_paths]
    elif isinstance(override_schema, LeRobotDatasetSchemaV30):
        _replace_lerobot_dataset_path(
            schema=override_schema,
            data_paths=data_paths,
        )
    elif isinstance(override_schema, SyntheticSchema):
        raise ValueError(
            "data_path_override cannot point to raw files for SyntheticSchema. "
            "Pass an existing .zarr path instead."
        )
    else:
        raise ValueError(
            "data_path_override is unsupported for schema type "
            f"{type(schema).__name__}."
        )

    return override_schema


def _normalize_data_paths(data_path_override: str | list[str]) -> list[Path]:
    """Convert configured data paths to concrete ``Path`` instances.

    Args:
        data_path_override: Single path or list of paths from the explain config.

    Returns:
        Non-empty list of expanded filesystem paths.

    Raises:
        ValueError: If no usable path was provided.
    """
    if isinstance(data_path_override, str):
        raw_paths = [data_path_override]
    else:
        raw_paths = data_path_override

    data_paths = [Path(raw_path).expanduser() for raw_path in raw_paths if raw_path]
    if not data_paths:
        raise ValueError("data_path_override must contain at least one path.")
    return data_paths


def _is_zarr_override(data_paths: list[Path]) -> bool:
    """Return whether the override is an existing zarr replay buffer path.

    Args:
        data_paths: Normalized data path override values.

    Returns:
        ``True`` when the override contains one ``.zarr`` path.

    Raises:
        ValueError: If multiple paths are provided and at least one is a zarr
            path.
    """
    zarr_paths = [
        data_path for data_path in data_paths if data_path.suffix == ZARR_SUFFIX
    ]
    if not zarr_paths:
        return False
    if len(data_paths) > 1:
        raise ValueError(
            "data_path_override accepts a single .zarr path, or one or more raw "
            "paths for the checkpoint schema format."
        )
    return True


def _resolve_zarr_override(schema: DatasetSchema, zarr_path: Path) -> DatasetSchema:
    """Clone a checkpoint schema and point it to an existing zarr store.

    Args:
        schema: Checkpoint task schema.
        zarr_path: Existing zarr replay buffer path to explain.

    Returns:
        Cloned schema with ``zarr_path`` replaced.

    Raises:
        ValueError: If the zarr replay buffer path does not exist.
    """
    if not zarr_path.exists():
        raise ValueError(
            "data_path_override points to a .zarr path that does not exist: "
            f"{zarr_path}"
        )

    override_schema = deepcopy(schema)
    override_schema.zarr_path = str(zarr_path)
    return override_schema


def _replace_lerobot_dataset_path(
    schema: LeRobotDatasetSchemaV30,
    data_paths: list[Path],
) -> None:
    """Replace LeRobot raw storage on a cloned schema.

    Args:
        schema: Cloned LeRobot schema to mutate.
        data_paths: Normalized raw dataset path override values.

    Raises:
        ValueError: If more than one LeRobot dataset root is provided.
    """
    if len(data_paths) != 1:
        raise ValueError(
            "data_path_override for LeRobotDatasetSchemaV30 must be a single "
            "dataset root."
        )

    dataset_path = data_paths[0]
    schema.dataset_path = dataset_path
    schema.lerobot_metadata = LeRobotDatasetMetadataV30(dataset_path=dataset_path)
