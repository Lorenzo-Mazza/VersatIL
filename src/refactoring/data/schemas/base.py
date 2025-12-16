"""Abstract dataset schema for defining dataset-specific configurations.

This module provides a framework for supporting multiple datasets with
different storage formats (CSV, HDF5) and observation modalities.

Schemas define the structure of stored data (what's in the zarr file).
They do not define how data is used at runtime (that's determined by task config).

Schemas are instantiated via Hydra using the _target_ pattern in config files.
"""

import abc
from typing import Any

import albumentations as A
import numpy as np

from refactoring.configs.data.dataset.raw_observations import RawObservationsConfig
from refactoring.data.constants import (
    GRIPPER_STATE_OBS_KEY,
    PHASE_LABEL_KEY,
    PRECOMPUTED_ACTIONS_KEY,
    PROPRIO_OBS_CAMERA_FRAME_KEY,
    PROPRIO_OBS_ROBOT_FRAME_KEY,
    Cameras, LANGUAGE_KEY,
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
        raw_observations: RawObservationsConfig,
        has_phase_labels: bool = False,
    ):
        """Initialize the dataset schema.

        Args:
            zarr_path: Path to save/load the zarr file
            raw_observations: Configuration for raw observations
            has_phase_labels: Whether dataset has phase labels
        """
        self.zarr_path = zarr_path
        self.raw_observations = raw_observations
        self.has_phase_labels = has_phase_labels

    @abc.abstractmethod
    def extract_episode(
        self,
        episode_source: Any,
        resizer: A.Resize | A.NoOp,
        depth_resizer: A.Resize | A.NoOp,
    ) -> dict[str, np.ndarray]:
        """Extract all data from an episode source.

        Each schema defines its own extraction logic based on raw_observations config.
        The episode_source type depends on the schema format:
        - CsvDatasetSchema: pd.DataFrame
        - Hdf5DatasetSchema: h5py.Group

        Args:
            episode_source: Format-specific episode data source
            resizer: Albumentations resizer for RGB images
            depth_resizer: Albumentations resizer for depth images

        Returns:
            Dictionary mapping zarr keys to numpy arrays
        """
        raise NotImplementedError("Subclasses must implement extract_episode")

    def get_required_zarr_keys(self) -> list[str]:
        """Get all required zarr keys based on schema configuration.

        Returns:
            List of all required zarr key names
        """
        keys = []
        if len(self.raw_observations.camera_keys) > 0:
            keys.extend(self.raw_observations.camera_keys)
        if self.raw_observations.robot_frame_proprio_keys:
            keys.append(PROPRIO_OBS_ROBOT_FRAME_KEY)
        if self.raw_observations.camera_frame_proprio_keys:
            keys.append(PROPRIO_OBS_CAMERA_FRAME_KEY)
        if self.raw_observations.gripper_state_keys:
            keys.append(GRIPPER_STATE_OBS_KEY)
        if self.has_phase_labels:
            keys.append(PHASE_LABEL_KEY)
        if self.raw_observations.language_key:
            keys.append(LANGUAGE_KEY)
        if self.raw_observations.custom_obs_keys:
            keys.extend(self.raw_observations.custom_obs_keys.keys())
        if self.raw_observations.has_precomputed_actions:
            keys.append(PRECOMPUTED_ACTIONS_KEY)
        return keys

    def get_zarr_array_specs(self) -> dict:
        """Get specifications for all zarr arrays to create.

        Returns:
            Dictionary mapping key names to array specifications
                Each spec is a dict with: shape, chunks, dtype, needs_compressor
        """
        specs = {}

        obs = self.raw_observations

        if obs.robot_frame_proprio_keys:
            dim = len(obs.robot_frame_proprio_keys)
            specs[PROPRIO_OBS_ROBOT_FRAME_KEY] = {
                'shape': (0, dim),
                'chunks': (100, dim),
                'dtype': 'float32',
                'needs_compressor': True
            }

        if obs.camera_frame_proprio_keys:
            dim = len(obs.camera_frame_proprio_keys)
            specs[PROPRIO_OBS_CAMERA_FRAME_KEY] = {
                'shape': (0, dim),
                'chunks': (100, dim),
                'dtype': 'float32',
                'needs_compressor': True
            }

        if obs.gripper_state_keys:
            dim = len(obs.gripper_state_keys)
            specs[GRIPPER_STATE_OBS_KEY] = {
                'shape': (0, dim),
                'chunks': (100, dim),
                'dtype': 'float32',
                'needs_compressor': True
            }

        if self.has_phase_labels:
            specs[PHASE_LABEL_KEY] = {
                'shape': (0, 1),
                'chunks': (100, 1),
                'dtype': 'uint8',
                'needs_compressor': True
            }

        if obs.language_key:
            specs[LANGUAGE_KEY] = {
                'shape': (0,),
                'chunks': (100,),
                'dtype': 'str',
                'needs_compressor': False
            }

        for modality_name, keys in obs.custom_obs_keys.items():
            dim = len(keys)
            specs[modality_name] = {
                'shape': (0, dim),
                'chunks': (100, dim),
                'dtype': 'float32',
                'needs_compressor': True
            }

        for cam in obs.camera_keys:
            if cam == Cameras.DEPTH.value:
                specs[cam] = {
                    'shape': (0, obs.image_height, obs.image_width),
                    'chunks': (10, obs.image_height, obs.image_width),
                    'dtype': 'float32',
                    'needs_compressor': True
                }
            else:
                specs[cam] = {
                    'shape': (0, obs.image_height, obs.image_width, 3),
                    'chunks': (10, obs.image_height, obs.image_width, 3),
                    'dtype': 'uint8',
                    'needs_compressor': True
                }

        if obs.has_precomputed_actions:
            specs[PRECOMPUTED_ACTIONS_KEY] = {
                'shape': (0, obs.precomputed_action_dim),
                'chunks': (100, obs.precomputed_action_dim),
                'dtype': 'float32',
                'needs_compressor': True
            }

        return specs
