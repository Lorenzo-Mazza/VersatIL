"""Abstract dataset schema for HDF5-based datasets."""

import abc

import albumentations as A
import h5py
import numpy as np

from refactoring.configs.data.dataset.raw_observations import RawObservationsConfig
from refactoring.data.constants import LANGUAGE_KEY
from refactoring.data.schemas.base import DatasetSchema


class Hdf5DatasetSchema(DatasetSchema):
    """Abstract schema for HDF5-based datasets.

    HDF5 datasets store all data (observations, images, actions) in HDF5 files.
    Subclasses define the HDF5 structure and extraction logic via `extract_episode`.
    """

    def __init__(
        self,
        hdf5_paths: list[str],
        zarr_path: str,
        raw_observations: RawObservationsConfig,
        has_phase_labels: bool = False,
        obs_group_path: str = "obs",
        actions_key: str | None = "actions",
        extract_language_from_filename: bool = False,
    ):
        """Initialize the HDF5 dataset schema.

        Args:
            hdf5_paths: List of paths to HDF5 files
            zarr_path: Path to save/load the zarr file
            raw_observations: Configuration for raw observations
            has_phase_labels: Whether dataset has phase labels
            obs_group_path: Path to observations group within each demo (e.g., "obs")
            actions_key: Key for actions within each demo. None if actions not stored.
            extract_language_from_filename: If True, language is extracted from filename
                instead of reading from obs group. Subclasses must override
                get_language_from_filename() to define extraction logic.
        """
        super().__init__(
            zarr_path=zarr_path,
            raw_observations=raw_observations,
            has_phase_labels=has_phase_labels,
        )
        self.hdf5_paths = hdf5_paths
        self.obs_group_path = obs_group_path
        self.actions_key = actions_key
        self.extract_language_from_filename = extract_language_from_filename

    @abc.abstractmethod
    def get_demo_names(self, hdf5_path: str) -> list[str]:
        """Get list of demo names in the specified HDF5 file.

        Args:
            hdf5_path: Path to the HDF5 file

        Returns:
            List of demo identifiers (e.g., ["demo_0", "demo_1", ...])
        """
        raise NotImplementedError("Subclasses must implement get_demo_names")

    @abc.abstractmethod
    def extract_episode(
        self,
        demo_group: h5py.Group,
        resizer: A.Resize | A.NoOp,
        depth_resizer: A.Resize | A.NoOp,
    ) -> dict[str, np.ndarray]:
        """Extract all data from a single HDF5 demo group."""
        ...

    def get_language_from_filename(self, hdf5_path: str) -> str:
        """Extract task language from the HDF5 filename.

        Override in subclasses when extract_language_from_filename=True.

        Args:
            hdf5_path: Path to the HDF5 file to extract language from.

        Returns:
            Task language instruction string
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} has extract_language_from_filename=True "
            "but does not implement get_language_from_filename()"
        )

    def get_required_zarr_keys(self) -> list[str]:
        """Get all required zarr keys based on schema configuration.

        Extends base implementation to include language key when
        extract_language_from_filename is True.
        """
        keys = super().get_required_zarr_keys()
        if self.extract_language_from_filename and LANGUAGE_KEY not in keys:
            keys.append(LANGUAGE_KEY)
        return keys

    def get_zarr_array_specs(self) -> dict:
        """Get specifications for all zarr arrays to create.

        Extends base implementation to include language array when
        extract_language_from_filename is True.
        """
        specs = super().get_zarr_array_specs()
        if self.extract_language_from_filename and LANGUAGE_KEY not in specs:
            specs[LANGUAGE_KEY] = {
                'shape': (0,),
                'chunks': (100,),
                'dtype': 'str',
                'needs_compressor': False
            }
        return specs
