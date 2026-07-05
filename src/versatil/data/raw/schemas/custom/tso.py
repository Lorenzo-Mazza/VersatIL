"""Dataset schema for TSO surgical CSV datasets."""

import logging
import re

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from versatil_constants.tso import TSOObsKey

from versatil.common.omegaconf_ops import resolve_dict_keys
from versatil.data.constants import (
    Cameras,
    CoordinateSystem,
    DatasetType,
    GripperType,
    ObsKey,
    ProprioKey,
)
from versatil.data.metadata import CameraMetadata, ObservationMetadata
from versatil.data.preprocessing.resizers import build_camera_resizer
from versatil.data.raw.schemas.csv import CsvDatasetSchema
from versatil.data.raw.zarr_meta import DatasetMetadata

TSO_LEFT_IMAGE_KEY = "frameLeftPath"
TSO_RIGHT_IMAGE_KEY = "frameRightPath"
TSO_RECTIFIED_LEFT_IMAGE_KEY = "frameLeftRectifiedPath"
TSO_RECTIFIED_RIGHT_IMAGE_KEY = "frameRightRectifiedPath"
TSO_EPISODE_FILENAME = "episode.csv"
ALLOWED_CAMERAS = {Cameras.LEFT.value, Cameras.RIGHT.value, Cameras.DEPTH.value}
ALLOWED_POS_OBS_KEYS = {
    ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value,
    ProprioKey.CAMERA_FRAME_CARTESIAN_TIP_POS.value,
}
TSO_GRIPPER_COL = "open"
TSO_PHASE_COL = "task_phase"
TSO_LANGUAGE_COL = "language"
DEFAULT_TSO_IMAGE_CROPS: dict[str, dict[str, int]] = {}
REQUIRED_TSO_CROP_KEYS = ("x_min", "y_min", "x_max", "y_max")
REQUIRED_TSO_CROP_KEY_SET = set(REQUIRED_TSO_CROP_KEYS)


class TSODatasetSchema(CsvDatasetSchema):
    """Schema for TSO zarr datasets with synchronized CSV and stereo images.

    These datasets can contain:
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
        use_rectified_images: bool = True,
        rgb_image_crops: dict[str, dict[str, int]] | None = None,
    ):
        """Initialize and validate the TSO dataset schema.

        Args:
            dataset_folders: List of folders containing episode CSVs.
            zarr_path: Path to save/load the zarr store.
            metadata: Metadata to use for creating the zarr store from the raw data.
            dataset_type: Type of dataset. Must be 'tso'.
            use_rectified_images: Whether to read rectified image path columns.
            rgb_image_crops: Optional per-camera Albumentations crop params.
        """
        if dataset_type != DatasetType.TSO.value:
            raise ValueError(
                f"TSODatasetSchema only supports dataset_type='{DatasetType.TSO.value}', "
                f"got '{dataset_type}'"
            )
        self._validate_metadata(metadata)
        self.left_image_csv_key = TSO_LEFT_IMAGE_KEY
        self.right_image_csv_key = TSO_RIGHT_IMAGE_KEY
        self.rectified_left_image_key = TSO_RECTIFIED_LEFT_IMAGE_KEY
        self.rectified_right_image_key = TSO_RECTIFIED_RIGHT_IMAGE_KEY
        self.use_rectified_images = use_rectified_images
        self.rgb_image_crops = self._normalize_rgb_image_crops(
            rgb_image_crops or DEFAULT_TSO_IMAGE_CROPS
        )
        self.depth_dir_pattern = "depth"
        self.depth_file_pattern = r"depth_\1.npy"
        self.left_dir_pattern = "framesLeft"
        self.rectified_left_dir_pattern = "framesLeftRectified"
        super().__init__(
            dataset_folders=dataset_folders,
            episode_filename=TSO_EPISODE_FILENAME,
            zarr_path=zarr_path,
            metadata=metadata,
            dataset_type=dataset_type,
        )

    @staticmethod
    def _validate_metadata(metadata: DatasetMetadata) -> None:
        """Validate TSO-specific metadata.

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
                f"Invalid cameras for TSODatasetSchema: {sorted(invalid_cameras)}. "
                f"Allowed cameras: {sorted(ALLOWED_CAMERAS)}"
            )
        required_rgb = {Cameras.LEFT.value, Cameras.RIGHT.value}
        rgb_cameras = set(camera_keys) & required_rgb
        if rgb_cameras != required_rgb:
            missing = required_rgb - rgb_cameras
            errors.append(
                f"TSODatasetSchema requires stereo cameras. Missing: {sorted(missing)}"
            )
        proprio_keys = set(metadata.position_observations.keys())
        invalid_proprio_keys = proprio_keys - ALLOWED_POS_OBS_KEYS
        if invalid_proprio_keys:
            errors.append(
                "Invalid proprioceptive observation keys: "
                f"{sorted(invalid_proprio_keys)}. "
                f"TSODatasetSchema requires keys from: {sorted(ALLOWED_POS_OBS_KEYS)}"
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
                "TSODatasetSchema does not support orientation proprioceptive observations."
            )
        for gripper_observation in metadata.gripper_observations.values():
            if gripper_observation.gripper_type != GripperType.BINARY.value:
                errors.append(
                    f"TSODatasetSchema requires binary gripper, got: "
                    f"{gripper_observation.gripper_type}"
                )
            if gripper_observation.raw_data_column_keys != [TSO_GRIPPER_COL]:
                errors.append(
                    f"TSODatasetSchema requires gripper source column to be {TSO_GRIPPER_COL}, got: "
                    f"{gripper_observation.raw_data_column_keys}"
                )
        if metadata.custom_observations:
            if ObsKey.LANGUAGE.value not in metadata.custom_observations:
                logging.warning(
                    f"Language observation key '{ObsKey.LANGUAGE.value}' not found. Language won't be used."
                )
            else:
                lang_obs = metadata.custom_observations[ObsKey.LANGUAGE.value]
                if lang_obs.raw_data_column_keys != [TSO_LANGUAGE_COL]:
                    errors.append(
                        f"TSODatasetSchema requires language source column to be {TSO_LANGUAGE_COL}, got: "
                        f"{lang_obs.raw_data_column_keys}"
                    )

        if metadata.custom_actions:
            if TSOObsKey.PHASE_LABEL.value not in metadata.custom_actions:
                logging.warning(
                    f"Phase action key '{TSOObsKey.PHASE_LABEL.value}' not found. Phase label won't be used."
                )
            else:
                phase_action = metadata.custom_actions[TSOObsKey.PHASE_LABEL.value]
                if phase_action.raw_data_column_keys != [TSO_PHASE_COL]:
                    errors.append(
                        f"TSODatasetSchema requires phase label source column to be {TSO_PHASE_COL}, got: "
                        f"{phase_action.raw_data_column_keys}"
                    )
        if errors:
            raise ValueError(
                "TSODatasetSchema validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    @staticmethod
    def _normalize_rgb_image_crops(
        rgb_image_crops: dict[str, dict[str, int]],
    ) -> dict[str, dict[str, int]]:
        """Validate and normalize per-camera raw image crop windows."""
        normalized = {}
        for camera, crop_params in resolve_dict_keys(rgb_image_crops).items():
            if camera not in {Cameras.LEFT.value, Cameras.RIGHT.value}:
                raise ValueError(f"Unknown RGB crop camera for TSO dataset: {camera}")
            missing = REQUIRED_TSO_CROP_KEY_SET - set(crop_params)
            extra = set(crop_params) - REQUIRED_TSO_CROP_KEY_SET
            if missing or extra:
                raise ValueError(
                    f"Crop for camera '{camera}' must contain exactly "
                    f"{sorted(REQUIRED_TSO_CROP_KEYS)}. Missing={sorted(missing)}, "
                    f"extra={sorted(extra)}"
                )
            crop = {key: int(crop_params[key]) for key in REQUIRED_TSO_CROP_KEYS}
            if crop["x_min"] < 0 or crop["y_min"] < 0:
                raise ValueError(f"Crop for camera '{camera}' must be non-negative")
            if crop["x_max"] <= crop["x_min"] or crop["y_max"] <= crop["y_min"]:
                raise ValueError(
                    f"Crop for camera '{camera}' must satisfy x_max > x_min and y_max > y_min"
                )
            normalized[camera] = crop
        return normalized

    def extract_episode(
        self,
        episode: pd.DataFrame,
    ) -> dict[str, np.ndarray]:
        """Extract all data from a TSO episode, optionally cropping/resizing images.

        Args:
            episode: DataFrame with episode data.

        Returns:
            Dictionary mapping zarr keys to numpy arrays.
        """
        data = {}
        for zarr_key, observation in self.metadata.observations.items():
            if isinstance(observation, CameraMetadata):
                continue
            elif isinstance(observation, ObservationMetadata):
                data[zarr_key] = episode[
                    observation.raw_data_column_keys
                ].values.astype(observation.dtype)

        for zarr_key, action in self.metadata.precomputed_actions.items():
            data[zarr_key] = episode[action.raw_data_column_keys].values.astype(
                action.dtype
            )

        for zarr_key, cam_metadata in self.metadata.cameras.items():
            camera = cam_metadata.raw_camera_key
            camera_resizer = build_camera_resizer(camera_metadata=cam_metadata)
            if cam_metadata.is_depth:
                left_column = self._get_rgb_column(Cameras.LEFT.value)
                paths = [self._compute_depth_path(p) for p in episode[left_column]]
                images = [
                    camera_resizer(image=np.load(p))["image"][..., np.newaxis]
                    for p in paths
                ]  # (H, W, 1)
            else:
                column = self._get_rgb_column(camera)
                images = [
                    camera_resizer(image=self._read_rgb_image(p, camera))["image"]
                    for p in episode[column]
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
            raise ValueError(f"Unknown RGB camera for TSO dataset: {camera}")

    def get_image_path_column(self, camera: str) -> str:
        """Get CSV column name for a camera image path."""
        return self._get_rgb_column(camera)

    def _read_rgb_image(self, path: str, camera: str) -> np.ndarray:
        """Read one RGB image and apply the optional per-camera crop."""
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Could not read RGB image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        crop_params = self.rgb_image_crops.get(camera)
        if crop_params is not None:
            height, width = image.shape[:2]
            if crop_params["x_max"] > width or crop_params["y_max"] > height:
                raise ValueError(
                    f"Crop for camera '{camera}' exceeds image size {width}x{height}: "
                    f"{crop_params}"
                )
            image = A.Crop(**crop_params, p=1.0)(image=image)["image"]
        return image

    def _compute_depth_path(self, base_image_path: str) -> str:
        """Compute depth file path from left RGB image path."""
        dir_to_sub = (
            self.rectified_left_dir_pattern
            if self.use_rectified_images
            else self.left_dir_pattern
        )
        depth_path = base_image_path.replace(dir_to_sub, self.depth_dir_pattern)
        depth_path = re.sub(r"(\d+)\.png$", self.depth_file_pattern, depth_path)
        return depth_path

    def compute_depth_path(self, base_image_path: str) -> str:
        """Compute depth file path from a left RGB image path."""
        return self._compute_depth_path(base_image_path)
