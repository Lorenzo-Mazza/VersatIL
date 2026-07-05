"""Configuration for the dataset schema.
The dataset schema defines the content of a raw dataset, including its metadata for zarr storage.
It does not define which subset of the data is used at runtime (see TaskSpaceConfig for that).
"""

from dataclasses import dataclass

from omegaconf import MISSING

from versatil.configs.data.raw.zarr_meta import DatasetMetadataConfig


@dataclass
class DatasetSchemaConfig:
    """Configuration for the dataset schema.

    Attributes:
        _target_: Import path instantiated by Hydra.
        zarr_path: Path to save/load the zarr file.
        metadata: Metadata of the raw dataset.
        dataset_type: Type of dataset (e.g., 'libero', 'tso', 'metaworld').
    """

    _target_: str = MISSING
    zarr_path: str = MISSING
    metadata: DatasetMetadataConfig = MISSING
    dataset_type: str = MISSING


@dataclass
class CsvDatasetSchemaConfig(DatasetSchemaConfig):
    """Configuration for CSV dataset schema.

    Attributes:
        dataset_folders: List of dataset folder paths.
    """

    dataset_folders: list[str] = MISSING


@dataclass
class Hdf5DatasetSchemaConfig(DatasetSchemaConfig):
    """Configuration for HDF5 dataset schema.

    Attributes:
        hdf5_paths: List of paths to HDF5 files.
    """

    hdf5_paths: list[str] = MISSING


@dataclass
class SyntheticDatasetSchemaConfig(DatasetSchemaConfig):
    """Configuration for synthetic multimodal benchmark schema.

    Attributes:
        task_name: Synthetic task variant to generate.
        num_episodes: Number of generated episodes.
        seed: Random seed for reproducibility.
        image_size: Generated image side length in pixels.
        num_modes: Number of behavior modes in the trajectory distribution.
        trajectory_length: Timesteps per generated trajectory.
        noise_std: Standard deviation of the trajectory noise.
        num_styles: Number of visual styles used for conditional variants.
        mode_weights: Sampling weight per behavior mode.
        num_rollouts: Rollouts sampled per mode when evaluating coverage metrics.
    """

    task_name: str = MISSING
    num_episodes: int = 1000
    seed: int = 42
    image_size: int = 64
    num_modes: int = 2
    trajectory_length: int = 60
    noise_std: float = 0.01
    num_styles: int = 1
    mode_weights: list[float] | None = None
    num_rollouts: int = 200


@dataclass
class LeRobotDatasetSchemaConfig(DatasetSchemaConfig):
    """Configuration for LeRobot dataset schema V30

    Attributes:
        dataset_path: Root directory of the LeRobot dataset.
    """

    dataset_path: str = MISSING
