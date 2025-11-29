"""Abstract dataset schema for defining dataset-specific configurations.

This module provides a flexible framework for supporting multiple datasets with
different CSV structures and observation modalities.

Schemas define the structure of stored data (what's in the zarr file).
They do not define how data is used at runtime (that's determined by task config).

Schemas are instantiated via Hydra using the _target_ pattern in config files.
"""

import abc

import numpy as np
import pandas as pd

from refactoring.configs.data.dataset.image_path import ImagePathConfig
from refactoring.configs.data.dataset.raw_observations import RawObservationsConfig
from refactoring.data.constants import (
    GRIPPER_STATE_OBS_KEY,
    PHASE_LABEL_KEY,
    PROPRIO_OBS_CAMERA_FRAME_KEY,
    PROPRIO_OBS_ROBOT_FRAME_KEY,
    Cameras,
)


class DatasetSchema(abc.ABC):
    """Abstract base class for dataset schemas.

    Note:
        A schema defines the raw data structure for a specific dataset, including:
        - What observations are available (position, gripper state, etc.).
        - What action modalities exist (position, orientation, gripper).
        - How to extract data from CSV files.
        - How to locate images and depth maps.

        The schema is used when creating the zarr file from the raw episodic data.
        Schemas are instantiated via Hydra - subclasses should be specified in config files using the _target_ field.
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
        """Initialize the dataset schema.

        Args:
            dataset_folders: List of dataset folder paths
            zarr_path: Path to save/load the zarr file
            dataset_filename: Name and format of the dataset file in each folder
            raw_observations: Configuration for the raw observations stored in the csv
            image_path_config: Configuration for image paths
            has_phase_labels: Whether dataset has phase labels
            phase_label_key: CSV column name for phase labels
        """
        self.dataset_folders = dataset_folders
        self.zarr_path = zarr_path
        self.dataset_filename = dataset_filename
        self.raw_observations = raw_observations
        self.image_path_config = image_path_config
        self.has_phase_labels = has_phase_labels
        self.phase_label_key = phase_label_key


    def extract_robot_frame_obs(self, df: pd.DataFrame) -> np.ndarray:
        """Extract robot frame observations from CSV.

        These are positions/states in the robot's coordinate frame.
        Used when creating zarr files and computing actions at runtime.

        Args:
            df: DataFrame with episode data

        Returns:
            Array of shape (T, robot_frame_dim)
        """
        if not self.raw_observations.robot_frame_proprio_keys:
            raise ValueError("Robot frame observations requested but no keys defined in schema.")
        else:
            return df[self.raw_observations.robot_frame_proprio_keys].values.astype(np.float32)  # type: ignore[no-any-return]

    def extract_camera_frame_obs(self, df: pd.DataFrame) -> np.ndarray:
        """Extract camera frame observations from CSV.

        These are positions/states in the camera's coordinate frame.
        Used when creating zarr files and computing actions at runtime.

        Args:
            df: DataFrame with episode data

        Returns:
            Array of shape (T, camera_frame_dim)
        """
        if not self.raw_observations.camera_frame_proprio_keys:
            raise ValueError("Camera frame observations requested but no keys defined in schema.")
        else:
            return df[self.raw_observations.camera_frame_proprio_keys].values.astype(np.float32)  # type: ignore[no-any-return]

    def extract_gripper_state(self, df: pd.DataFrame) -> np.ndarray:
        """Extract gripper state from CSV.

        This is the gripper's current state (open/close or continuous value).
        Used when creating zarr files. At runtime, next timestep's gripper state
        becomes the gripper action.

        Args:
            df: DataFrame with episode data

        Returns:
            Array of shape (T, gripper_dim)
        """
        if not self.raw_observations.gripper_state_keys:
            raise ValueError("Gripper state requested but no keys defined in schema.")
        else:
            return df[self.raw_observations.gripper_state_keys].values.astype(np.float32)  # type: ignore[no-any-return]

    def extract_phase_labels(self, df: pd.DataFrame) -> np.ndarray | None:
        """Extract phase labels from CSV.

        These are task phase annotations (if available).

        Args:
            df: DataFrame with episode data

        Returns:
            Array of shape (T,) or None if not available
        """
        if not self.has_phase_labels:
            raise ValueError("Phase labels requested but schema indicates none are available.")
        else:
            return df[self.phase_label_key].values.astype(np.uint8)  # type: ignore[no-any-return]

    def extract_language_instruction(self, df: pd.DataFrame) -> np.ndarray | None:
        """Extract language instruction from CSV.

        This is the language instruction associated with the episode (if available).

        Args:
            df: DataFrame with episode data

        Returns:
            Array of shape (T,) or None if not available
        """
        if not self.raw_observations.language_key:
            raise ValueError("Language instruction requested but no key defined in schema.")
        else:
            return df[self.raw_observations.language_key].astype(str).values  # type: ignore[no-any-return]


    def extract_custom_observations(self, df: pd.DataFrame, modality_name: str) -> np.ndarray:
        """Extract custom observation modality from CSV.

        These are user-defined observation modalities (if available).

        Args:
            df: DataFrame with episode data
            modality_name: Name of the custom observation modality
       """
        if modality_name not in self.raw_observations.custom_obs_keys:
            raise ValueError(f"Custom observation '{modality_name}' requested but no keys defined in schema.")
        else:
            keys = self.raw_observations.custom_obs_keys[modality_name]
            return df[keys].values.astype(np.float32)  # type: ignore[no-any-return]

    @abc.abstractmethod
    def get_image_path_column(self, camera: str) -> str:
        """Get the CSV column name for image paths.

        Args:
            camera: Camera name (from Cameras enum)

        Returns:
            Column name for image paths
        """
        raise NotImplementedError("Subclasses must implement get_image_path_column")


    @abc.abstractmethod
    def compute_depth_path(self, base_image_path: str) -> str:
        """Compute depth file path from a base image path.

        Uses regex replacement to convert base image path to corresponding depth path.

        Args:
            base_image_path: Base path

        Returns:
            Path to corresponding depth file
        """
        raise NotImplementedError("Subclasses must implement compute_depth_path")


    def get_required_zarr_keys(self) -> list[str]:
        """Get all required zarr keys based on schema configuration.

        Returns:
            List of all required zarr key names
        """
        keys = []
        if len(self.raw_observations.camera_keys)>0:
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
            keys.append(self.raw_observations.language_key)
        if self.raw_observations.custom_obs_keys:
            keys.extend(self.raw_observations.custom_obs_keys.keys())
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
            specs[obs.language_key] = {
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

        return specs
