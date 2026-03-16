"""Dataset schema for the bowel retraction surgical dataset."""
import logging
import re

import albumentations as A
import cv2
import numpy as np
import pandas as pd

from versatil.data.constants import (
    Cameras,
    CoordinateSystem,
    DatasetType,
    GripperType,
    ProprioKey,
    ObsKey,
)
from versatil.data.metadata import CameraMetadata, ObservationMetadata
from versatil.data.raw.schemas.csv import CsvDatasetSchema
from versatil.data.raw.zarr_meta import DatasetMetadata


BOWEL_RETRACTION_LEFT_IMAGE_KEY = "frameLeftPath"
BOWEL_RETRACTION_RIGHT_IMAGE_KEY = "frameRightPath"
BOWEL_RETRACTION_RECTIFIED_LEFT_IMAGE_KEY = "frameLeftRectifiedPath"
BOWEL_RETRACTION_RECTIFIED_RIGHT_IMAGE_KEY = "frameRightRectifiedPath"
BOWEL_RETRACTION_EPISODE_FILENAME = "episode.csv"
ALLOWED_CAMERAS = {Cameras.LEFT.value, Cameras.RIGHT.value, Cameras.DEPTH.value}
ALLOWED_POS_OBS_KEYS = {
    ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value,
    ProprioKey.CAMERA_FRAME_CARTESIAN_TIP_POS.value,
}
ALLOWED_ORI_OBS_KEYS = set()
ALLOWED_FRAMES = {CoordinateSystem.ROBOT_BASE.value, CoordinateSystem.CAMERA.value}
BOWEL_RETRACTION_GRIPPER_COL = "open"
BOWEL_RETRACTION_PHASE_COL = "task_phase"
BOWEL_RETRACTION_LANGUAGE_COL = "language"


class BowelRetractionSchema(CsvDatasetSchema):
    """Schema for the bowel retraction zarr dataset.

    This dataset contains:
    - 3D cartesian position in robot and camera frames
    - Binary gripper state (open/close)
    - Optional task phase labels
    - Stereo camera images (left, right) with optional depth
    """

    def __init__(
        self,
        dataset_folders: list[str],
        zarr_path: str,
        metadata: DatasetMetadata,
        dataset_type: str = DatasetType.TSO.value,
    ):
        """Initialize and validate the bowel retraction schema.

        Args:
            dataset_folders: List of folders containing episode CSVs.
            zarr_path: Path to save/load the zarr store.
            metadata: Metadata to use for creating the zarr store from the raw data.
            dataset_type: Type of dataset. Must be 'tso'.
        """
        if dataset_type != DatasetType.TSO.value:
            raise ValueError(
                f"BowelRetractionSchema only supports dataset_type='{DatasetType.TSO.value}', "
                f"got '{dataset_type}'"
            )
        self._validate_metadata(metadata)
        self.left_image_csv_key = BOWEL_RETRACTION_LEFT_IMAGE_KEY
        self.right_image_csv_key = BOWEL_RETRACTION_RIGHT_IMAGE_KEY
        self.rectified_left_image_key = BOWEL_RETRACTION_RECTIFIED_LEFT_IMAGE_KEY
        self.rectified_right_image_key = BOWEL_RETRACTION_RECTIFIED_RIGHT_IMAGE_KEY
        self.use_rectified_images = True
        self.depth_dir_pattern = "depth"
        self.depth_file_pattern = r"depth_\1.npy"
        self.left_dir_pattern = "framesLeft"
        self.rectified_left_dir_pattern = "framesLeftRectified"
        super().__init__(
            dataset_folders=dataset_folders,
            episode_filename=BOWEL_RETRACTION_EPISODE_FILENAME,
            zarr_path=zarr_path,
            metadata=metadata,
            dataset_type=dataset_type,
        )

    @staticmethod
    def _validate_metadata(metadata: DatasetMetadata) -> None:
        """Validate BowelRetraction-specific metadata.

        Args:
            metadata: The metadata to validate against.

        Raises:
            ValueError: If validation fails.
        """
        errors = []
        camera_keys = metadata.get_camera_keys()
        invalid_cameras = set(camera_keys) - ALLOWED_CAMERAS
        if invalid_cameras:
            errors.append(
                f"Invalid cameras for BowelRetraction: {invalid_cameras}. "
                f"Allowed cameras: {ALLOWED_CAMERAS}"
            )
        required_rgb = {Cameras.LEFT.value, Cameras.RIGHT.value}
        rgb_cameras = set(camera_keys) & required_rgb
        if rgb_cameras != required_rgb:
            missing = required_rgb - rgb_cameras
            errors.append(
                f"BowelRetraction requires stereo cameras. Missing: {missing}"
            )
        proprio_keys = set(metadata.position_observations.keys())
        invalid_proprio_keys = proprio_keys - ALLOWED_POS_OBS_KEYS
        if invalid_proprio_keys:
            errors.append(
                f"Invalid proprioceptive observation keys: {invalid_proprio_keys}. "
                f"BowelRetraction requires keys from: {ALLOWED_POS_OBS_KEYS}"
            )
        for key, obs in metadata.position_observations.items():
            if (
                key == ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value
                and obs.frame != CoordinateSystem.ROBOT_BASE.value
            ):
                errors.append(
                    f"'{ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value}' must have frame='{CoordinateSystem.ROBOT_BASE.value}', "
                    f"got: '{obs.frame}'"
                )
            elif (
                key == ProprioKey.CAMERA_FRAME_CARTESIAN_TIP_POS.value
                and obs.frame != CoordinateSystem.CAMERA.value
            ):
                errors.append(
                    f"'{ProprioKey.CAMERA_FRAME_CARTESIAN_TIP_POS.value}' must have frame='{CoordinateSystem.CAMERA.value}', "
                    f"got: '{obs.frame}'"
                )
        if metadata.orientation_observations:
            errors.append(
                "BowelRetraction does not support orientation proprioceptive observations."
            )
        for key, gripper_observation in metadata.gripper_observations.items():
            if gripper_observation.gripper_type != GripperType.BINARY.value:
                errors.append(
                    f"BowelRetraction requires binary gripper, got: "
                    f"{gripper_observation.gripper_type}"
                )
            if gripper_observation.raw_data_column_keys != [
                BOWEL_RETRACTION_GRIPPER_COL
            ]:
                errors.append(
                    f"BowelRetraction requires gripper source column to be {BOWEL_RETRACTION_GRIPPER_COL}, got: "
                    f"{key}"
                )
        if metadata.custom_observations:
            if ObsKey.LANGUAGE.value not in metadata.custom_observations:
                logging.warning(
                    f"Language observation key '{ObsKey.LANGUAGE.value}' not found. Language won't be used."
                )
            else:
                lang_obs = metadata.custom_observations[ObsKey.LANGUAGE.value]
                if lang_obs.raw_data_column_keys != [BOWEL_RETRACTION_LANGUAGE_COL]:
                    errors.append(
                        f"BowelRetraction requires language source column to be {BOWEL_RETRACTION_LANGUAGE_COL}, got: "
                        f"{lang_obs.raw_data_column_keys}"
                    )

        if metadata.custom_actions:
            if ObsKey.PHASE_LABEL.value not in metadata.custom_actions:
                logging.warning(
                    f"Phase action key '{ObsKey.PHASE_LABEL.value}' not found. Phase label won't be used."
                )
            else:
                phase_action = metadata.custom_actions[ObsKey.PHASE_LABEL.value]
                if phase_action.raw_data_column_keys != [BOWEL_RETRACTION_PHASE_COL]:
                    errors.append(
                        f"BowelRetraction requires phase label source column to be {BOWEL_RETRACTION_PHASE_COL}, got: "
                        f"{phase_action.raw_data_column_keys}"
                    )
        if errors:
            raise ValueError(
                f"BowelRetraction schema validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    def extract_episode(
        self,
        episode: pd.DataFrame,
        resizer: A.Resize | A.NoOp,
        depth_resizer: A.Resize | A.NoOp,
    ) -> dict[str, np.ndarray]:
        """Extract all data from a bowel retraction episode, optionally resizing images.

        Args:
            episode: DataFrame with episode data.
            resizer: Albumentations resizer for RGB images.
            depth_resizer: Albumentations resizer for depth images.

        Returns:
            Dictionary mapping zarr keys to numpy arrays.
        """
        data = {}
        for zarr_key, obs in self.metadata.observations.items():
            if isinstance(obs, CameraMetadata):
                continue
            elif isinstance(obs, ObservationMetadata):
                data[zarr_key] = episode[obs.raw_data_column_keys].values.astype(
                    obs.dtype
                )

        for zarr_key, action in self.metadata.precomputed_actions.items():
            data[zarr_key] = episode[action.raw_data_column_keys].values.astype(
                action.dtype
            )

        for zarr_key, cam_metadata in self.metadata.cameras.items():
            cam = cam_metadata.raw_camera_key
            if cam == Cameras.DEPTH.value:
                left_col = self._get_rgb_column(Cameras.LEFT.value)
                paths = [self._compute_depth_path(p) for p in episode[left_col]]
                images = [
                    depth_resizer(image=np.load(p))["image"][..., np.newaxis]
                    for p in paths
                ]  # (H, W, 1)
            else:
                col = self._get_rgb_column(cam)
                images = [
                    resizer(image=cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))[
                        "image"
                    ]
                    for p in episode[col]
                ]
            data[zarr_key] = np.stack(images).astype(cam_metadata.dtype)

        return data

    def _get_rgb_column(self, camera: str) -> str:
        """Get CSV column name for RGB image paths."""
        if camera == Cameras.LEFT.value:
            return (
                self.rectified_left_image_key
                if self.use_rectified_images
                else self.left_image_csv_key
            )
        elif camera == Cameras.RIGHT.value:
            return (
                self.rectified_right_image_key
                if self.use_rectified_images
                else self.right_image_csv_key
            )
        else:
            raise ValueError(
                f"Unknown RGB camera for Bowel Retraction dataset: {camera}"
            )

    def _compute_depth_path(self, base_image_path: str) -> str:
        """Compute depth file path from left RGB image path."""
        dir_to_sub = (
            self.rectified_left_dir_pattern
            if self.use_rectified_images
            else self.left_dir_pattern
        )
        depth_path = base_image_path.replace(dir_to_sub, self.depth_dir_pattern)
        depth_path = re.sub(rf"(\d+)\.png$", self.depth_file_pattern, depth_path)
        return depth_path
