"""Configuration for the dataset schema.
  The dataset schema defines the content of a raw dataset, including its metadata for zarr storage.
  It does not define which subset of the data is used at runtime (see TaskSpaceConfig for that).
 """
from dataclasses import dataclass
from omegaconf import MISSING
from versatil.configs.data.raw.zarr_meta import DatasetMetadataConfig


@dataclass
class DatasetSchemaConfig:
    """Configuration for the dataset schema."""

    _target_: str = MISSING
    zarr_path: str = MISSING
    metadata: DatasetMetadataConfig = MISSING


@dataclass
class CsvDatasetSchemaConfig(DatasetSchemaConfig):
    """Configuration for CSV dataset schema."""

    dataset_folders: list[str] = MISSING


@dataclass
class Hdf5DatasetSchemaConfig(DatasetSchemaConfig):
    """Configuration for HDF5 dataset schema."""

    hdf5_paths: list[str] = MISSING


@dataclass
class LeRobotDatasetSchemaConfig(DatasetSchemaConfig):
    """Configuration for LeRobot dataset schema (v2.1 and v3.0)."""

    _target_: str = "refactoring.data.raw.schemas.lerobot.create_lerobot_schema"
    lerobot_path: str = MISSING
    has_video_files: bool = True
    tasks_format: str = "jsonl"