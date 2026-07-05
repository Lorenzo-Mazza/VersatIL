"""Abstract dataset schema for HDF5-based datasets."""

import abc

import h5py
import numpy as np

from versatil.data.raw.schemas.base import DatasetSchema
from versatil.data.raw.zarr_meta import DatasetMetadata


class Hdf5DatasetSchema(DatasetSchema):
    """Abstract schema for HDF5-based datasets.

    HDF5 datasets store all data (observations, images, actions) in HDF5 files.
    Subclasses define the HDF5 structure and extraction logic via `extract_episode`.
    """

    def __init__(
        self,
        hdf5_paths: list[str],
        zarr_path: str,
        metadata: DatasetMetadata,
        dataset_type: str,
    ):
        """Initialize the HDF5 dataset schema.

        Args:
            hdf5_paths: List of paths to HDF5 files
            zarr_path: Path to save/load the zarr file
            metadata: Metadata to use for creating the zarr store from the raw data.
            dataset_type: Type of dataset used by the schema (e.g., 'libero', 'tso', 'metaworld')
        """
        super().__init__(
            zarr_path=zarr_path,
            metadata=metadata,
            dataset_type=dataset_type,
        )
        self.hdf5_paths = hdf5_paths

    @abc.abstractmethod
    def get_demo_names(self, hdf5_path: str) -> list[str]:
        """Get list of demonstration/episode names in the specified HDF5 file.

        Args:
            hdf5_path: Path to the HDF5 file

        Returns:
            List of demo identifiers as strings.
        """
        raise NotImplementedError("Subclasses must implement get_demo_names")

    @abc.abstractmethod
    def extract_episode(
        self,
        demo_group: h5py.Group,
    ) -> dict[str, np.ndarray]:
        """Extract episodic data from a single HDF5 group."""
        raise NotImplementedError("Subclasses must implement extract_episode")
