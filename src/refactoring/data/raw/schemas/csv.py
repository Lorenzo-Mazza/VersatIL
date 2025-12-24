"""Abstract dataset schema for CSV-based datasets."""

import abc

import albumentations as A
import numpy as np
import pandas as pd

from refactoring.data.raw.zarr_meta import DatasetMetadata
from refactoring.data.raw.schemas.base import DatasetSchema


class CsvDatasetSchema(DatasetSchema):
    """Abstract schema for CSV-based datasets.

    CSV datasets store observations in tabular format. Images are stored as image path values.
    """

    def __init__(
        self,
        dataset_folders: list[str],
        zarr_path: str,
        episode_filename: str,
        metadata: DatasetMetadata,
    ):
        """Initialize the CSV dataset schema.

        Args:
            dataset_folders: List of dataset folder paths
            zarr_path: Path to save/load the zarr file
            episode_filename: Name and format of the episode CSV data file in each folder
            metadata: Metadata to use for creating the zarr store from the raw data.
        """
        super().__init__(
            zarr_path=zarr_path,
            metadata=metadata,
        )
        self.dataset_folders = dataset_folders
        self.dataset_filename = episode_filename

    @abc.abstractmethod
    def extract_episode(
        self,
        episode: pd.DataFrame,
        resizer: A.Resize | A.NoOp,
        depth_resizer: A.Resize | A.NoOp,
    ) -> dict[str, np.ndarray]:
        """Extract all data from an episode DataFrame."""
        ...