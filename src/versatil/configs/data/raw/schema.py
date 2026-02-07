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
    dataset_type: str = MISSING


@dataclass
class CsvDatasetSchemaConfig(DatasetSchemaConfig):
    """Configuration for CSV dataset schema."""

    dataset_folders: list[str] = MISSING


@dataclass
class Hdf5DatasetSchemaConfig(DatasetSchemaConfig):
    """Configuration for HDF5 dataset schema."""

    hdf5_paths: list[str] = MISSING


@dataclass
class SyntheticDatasetSchemaConfig(DatasetSchemaConfig):
    """Configuration for synthetic multimodal benchmark schema."""

    task_name: str = MISSING
    num_episodes: int = 1000
    seed: int = 42
    image_size: int = 64
    num_modes: int = 3
    trajectory_length: int = 60
    noise_std: float = 0.01
    num_styles: int = 4


@dataclass
class LeRobotDatasetSchemaConfig(DatasetSchemaConfig):
    """Configuration for LeRobot dataset schema V30"""

    dataset_path: str = MISSING
