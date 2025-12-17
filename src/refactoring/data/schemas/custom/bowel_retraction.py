"""Dataset schema for the bowel retraction surgical robotics dataset.

The dataset schema defines the structure and content of the raw bowel retraction dataset, as it was constructed by the tso-sensing repository.

This schema is instantiated via Hydra configuration files.
"""
import re

import albumentations as A
import cv2
import numpy as np
import pandas as pd

from refactoring.configs.data.dataset.image_path import ImagePathConfig
from refactoring.configs.data.dataset.raw_observations import RawObservationsConfig
from refactoring.data.constants import (
    Cameras,
    GripperType,
    GRIPPER_STATE_OBS_KEY,
    LANGUAGE_KEY,
    PHASE_LABEL_KEY,
    PROPRIO_OBS_CAMERA_FRAME_KEY,
    PROPRIO_OBS_ROBOT_FRAME_KEY,
)
from refactoring.data.schemas.csv import CsvDatasetSchema

BOWEL_RETRACTION_ROBOT_FRAME_COLS = [
    "relative_tip_position_x",
    "relative_tip_position_y",
    "relative_tip_position_z"
]
BOWEL_RETRACTION_CAMERA_FRAME_COLS = [
    "camera_frame_tip_position_x",
    "camera_frame_tip_position_y",
    "camera_frame_tip_position_z"
]
BOWEL_RETRACTION_GRIPPER_COL = "open"
BOWEL_RETRACTION_PHASE_COL = "task_phase"
BOWEL_RETRACTION_LEFT_IMAGE_KEY = "frameLeftPath"
BOWEL_RETRACTION_RIGHT_IMAGE_KEY = "frameRightPath"
BOWEL_RETRACTION_RECTIFIED_LEFT_IMAGE_KEY = "frameLeftRectifiedPath"
BOWEL_RETRACTION_RECTIFIED_RIGHT_IMAGE_KEY = "frameRightRectifiedPath"
BOWEL_RETRACTION_EPISODE_FILENAME = "episode.csv"
BOWEL_RETRACTION_LANGUAGE_KEY = "language"


class BowelRetractionSchema(CsvDatasetSchema):
    """Schema for the bowel retraction zarr dataset.

    This dataset contains:
    - 3D cartesian position in robot and camera frames
    - Binary gripper state (open/close)
    - Optional task phase labels
    - Stereo camera images (left, right) with depth

    Instantiated via Hydra.
    """

    def __init__(
            self,
            dataset_folders: list[str],
            zarr_path: str,
            raw_observation_config: RawObservationsConfig | None = None,
            image_path_config: ImagePathConfig | None = None,
            has_phase_labels: bool = False,
    ):
        """Initialize the bowel retraction schema."""
        if raw_observation_config is None:
            raw_observation_config = RawObservationsConfig(
                robot_frame_proprio_keys=BOWEL_RETRACTION_ROBOT_FRAME_COLS,
                camera_frame_proprio_keys=BOWEL_RETRACTION_CAMERA_FRAME_COLS,
                gripper_state_keys=[BOWEL_RETRACTION_GRIPPER_COL],
                camera_keys=[Cameras.LEFT.value, Cameras.RIGHT.value, Cameras.DEPTH.value],
                language_key=BOWEL_RETRACTION_LANGUAGE_KEY,
                has_language=True,
                use_rectified_images=True,
                image_width=480,
                image_height=270,
                has_position=True,
                position_dim=3,
                has_orientation=False,
                orientation_dim=0,
                has_gripper=True,
                gripper_type=GripperType.BINARY.value,
                gripper_dim=1,
            )

        if image_path_config is None:
            image_path_config = ImagePathConfig(
                left_image_key=BOWEL_RETRACTION_LEFT_IMAGE_KEY,
                right_image_key=BOWEL_RETRACTION_RIGHT_IMAGE_KEY,
                rectified_left_image_key=BOWEL_RETRACTION_RECTIFIED_LEFT_IMAGE_KEY,
                rectified_right_image_key=BOWEL_RETRACTION_RECTIFIED_RIGHT_IMAGE_KEY,
                depth_dir_pattern="depth",
                depth_file_pattern=r'depth_\1.npy',
                left_dir_pattern="framesLeft",
                rectified_left_dir_pattern="framesLeftRectified",
            )

        super().__init__(
            dataset_folders=dataset_folders,
            dataset_filename=BOWEL_RETRACTION_EPISODE_FILENAME,
            zarr_path=zarr_path,
            raw_observations=raw_observation_config,
            image_path_config=image_path_config,
            has_phase_labels=has_phase_labels,
            phase_label_key=BOWEL_RETRACTION_PHASE_COL if has_phase_labels else None,
        )

    def extract_episode(
        self,
        episode: pd.DataFrame,
        resizer: A.Resize | A.NoOp,
        depth_resizer: A.Resize | A.NoOp,
    ) -> dict[str, np.ndarray]:
        """Extract all data from a bowel retraction episode.

        Args:
            episode: DataFrame with episode data
            resizer: Albumentations resizer for RGB images
            depth_resizer: Albumentations resizer for depth images

        Returns:
            Dictionary mapping zarr keys to numpy arrays
        """
        data = {}
        obs = self.raw_observations
        if obs.robot_frame_proprio_keys:
            data[PROPRIO_OBS_ROBOT_FRAME_KEY] = episode[obs.robot_frame_proprio_keys].values.astype(np.float32)
        if obs.camera_frame_proprio_keys:
            data[PROPRIO_OBS_CAMERA_FRAME_KEY] = episode[obs.camera_frame_proprio_keys].values.astype(np.float32)
        if obs.gripper_state_keys:
            data[GRIPPER_STATE_OBS_KEY] = episode[obs.gripper_state_keys].values.astype(np.float32)
        if self.has_phase_labels:
            data[PHASE_LABEL_KEY] = episode[self.phase_label_key].values.astype(np.uint8)[:, np.newaxis]
        if obs.language_key:
            data[LANGUAGE_KEY] = episode[obs.language_key].astype(str).values
        for modality_name in obs.custom_obs_keys:
            keys = obs.custom_obs_keys[modality_name]
            data[modality_name] = episode[keys].values.astype(np.float32)
        # Images - BowelRetraction computes depth path from left RGB image path
        for cam in obs.camera_keys:
            if cam == Cameras.DEPTH.value:
                left_col = self._get_rgb_column(Cameras.LEFT.value)
                paths = [self._compute_depth_path(p) for p in episode[left_col]]
                images = [depth_resizer(image=np.load(p))['image'] for p in paths]
            else:
                col = self._get_rgb_column(cam)
                images = [
                    resizer(image=cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))['image']
                    for p in episode[col]
                ]
            data[cam] = np.stack(images)

        return data

    def _get_rgb_column(self, camera: str) -> str:
        """Get CSV column name for RGB image paths."""
        cfg = self.image_path_config
        if camera == Cameras.LEFT.value:
            return cfg.rectified_left_image_key if self.raw_observations.use_rectified_images else cfg.left_image_key
        elif camera == Cameras.RIGHT.value:
            return cfg.rectified_right_image_key if self.raw_observations.use_rectified_images else cfg.right_image_key
        else:
            raise ValueError(f"Unknown RGB camera: {camera}")

    def _compute_depth_path(self, base_image_path: str) -> str:
        """Compute depth file path from left RGB image path using config patterns."""
        cfg = self.image_path_config
        dir_to_sub = (
            cfg.rectified_left_dir_pattern
            if self.raw_observations.use_rectified_images
            else cfg.left_dir_pattern
        )
        depth_path = base_image_path.replace(dir_to_sub, cfg.depth_dir_pattern)
        depth_path = re.sub(
            rf'(\d+){re.escape(cfg.rgb_extension)}$',
            cfg.depth_file_pattern,
            depth_path
        )
        return depth_path


