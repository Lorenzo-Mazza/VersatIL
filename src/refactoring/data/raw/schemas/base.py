"""Abstract dataset schema for defining dataset-specific configurations.

This module provides a framework for supporting multiple datasets with different formats and observation modalities.
Schemas define the structure of stored data (what's in the zarr file).
They do not define how data is used at runtime (that's determined by task space config).
"""

import abc
from typing import Any

import albumentations as A
import numpy as np

from refactoring.data.raw.zarr_meta import DatasetMetadata
from refactoring.data.metadata import (
    CameraMetadata,
    ObservationMetadata,
    PrecomputedActionMetadata,
)


class DatasetSchema(abc.ABC):
    """Abstract base class for dataset schemas.

    Defines common interface for zarr array creation, key management, and episode extraction.
    Subclasses (CsvDatasetSchema, Hdf5DatasetSchema) define format-specific data extraction.

    Note:
        A schema defines the raw data structure for a specific dataset, including:
        - What observations are available (position, gripper state, etc.).
        - What action modalities exist (position, orientation, gripper).
        - How to extract episode data via `extract_episode`.

        The schema is used when creating the zarr file from the raw episodic data.
    """

    def __init__(
        self,
        zarr_path: str,
        metadata: DatasetMetadata,
    ):
        """Initialize the dataset schema.

        Args:
            zarr_path: Path to save/load the zarr file
            metadata: Metadata of the raw dataset
        """
        self.zarr_path = zarr_path
        self.metadata = metadata

    @abc.abstractmethod
    def extract_episode(
        self,
        episode_source: Any,
        resizer: A.Resize | A.NoOp,
        depth_resizer: A.Resize | A.NoOp,
    ) -> dict[str, np.ndarray]:
        """Extract all data from an episode source.

        Each schema defines its own extraction logic based on its metadata.
        The episode_source type depends on the schema format:

        Args:
            episode_source: Format-specific episode data source
            resizer: Albumentations resizer for RGB images
            depth_resizer: Albumentations resizer for depth images

        Returns:
            Dictionary mapping zarr keys to numpy arrays
        """
        raise NotImplementedError("Subclasses must implement extract_episode")

    def get_required_zarr_keys(self) -> list[str]:
        """Get all required zarr keys based on the dataset metadata.

        Returns:
            List of all required zarr key names
        """
        return self.metadata.get_all_keys()

    def get_zarr_array_specs(self) -> dict:
        """Get specifications for all zarr arrays to create for the store.

        Returns:
            Dictionary mapping zarr key names to array specifications
                Each spec is a dict with: shape, chunks, dtype, needs_compressor
        """
        specs = {}
        for key, obs in self.metadata.observations.items():
            if isinstance(obs, CameraMetadata):
                specs[key] = {
                    "shape": (0, obs.image_height, obs.image_width, obs.channels),
                    "chunks": (10, obs.image_height, obs.image_width, obs.channels),
                    "dtype": obs.dtype,
                    "needs_compressor": True,
                }
            elif isinstance(obs, ObservationMetadata):
                needs_compression = obs.dtype != "str"
                specs[key] = {
                    "shape": (0, obs.dimension),
                    "chunks": (100, obs.dimension),
                    "dtype": obs.dtype,
                    "needs_compressor": needs_compression,
                }
        for key, action in self.metadata.precomputed_actions.items():
            if action.slice_start is not None and action.slice_end is not None:
                dim = action.slice_end - action.slice_start
            else:
                dim = action.storage_dimension
            specs[key] = {
                "shape": (0, dim),
                "chunks": (100, dim),
                "dtype": action.dtype,
                "needs_compressor": True,
            }
        return specs
