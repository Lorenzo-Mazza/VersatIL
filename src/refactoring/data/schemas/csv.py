"""Abstract dataset schema for CSV-based datasets."""

import abc

import albumentations as A
import numpy as np
import pandas as pd

from refactoring.configs.data.dataset.image_path import ImagePathConfig
from refactoring.configs.data.dataset.raw_observations import RawObservationsConfig
from refactoring.data.schemas.base import DatasetSchema


class CsvDatasetSchema(DatasetSchema):
    """Abstract schema for CSV-based datasets.

    CSV datasets store observations in tabular format with image paths as columns.
    Each subclass defines its own extraction logic via `extract_episode`.
    """

    def __init__(
        self,
        dataset_folders: list[str],
        zarr_path: str,
        dataset_filename: str,
        raw_observations: RawObservationsConfig,
        image_path_config: ImagePathConfig,
        has_phase_labels: bool = False,
        phase_label_key: str | None = None,
    ):
        """Initialize the CSV dataset schema.

        Args:
            dataset_folders: List of dataset folder paths
            zarr_path: Path to save/load the zarr file
            dataset_filename: Name and format of the dataset file in each folder
            raw_observations: Configuration for raw observations stored in CSV
            image_path_config: Configuration for image paths
            has_phase_labels: Whether dataset has phase labels
            phase_label_key: CSV column name for phase labels
        """
        super().__init__(
            zarr_path=zarr_path,
            raw_observations=raw_observations,
            has_phase_labels=has_phase_labels,
        )
        self.dataset_folders = dataset_folders
        self.dataset_filename = dataset_filename
        self.image_path_config = image_path_config
        self.phase_label_key = phase_label_key

    @abc.abstractmethod
    def extract_episode(
        self,
        episode: pd.DataFrame,
        resizer: A.Resize | A.NoOp,
        depth_resizer: A.Resize | A.NoOp,
    ) -> dict[str, np.ndarray]:
        """Extract all data from an episode DataFrame."""
        ...