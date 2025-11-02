"""Dataset schema for the bowel retraction surgical robotics dataset.

This schema is instantiated via Hydra configuration files.
"""
import re

from refactoring.configs.task.dataset.image_path import ImagePathConfig
from refactoring.configs.task.dataset.raw_observations import RawObservationsConfig
from refactoring.data.constants import Cameras, GripperType
from refactoring.data.schemas.base import DatasetSchema

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


class BowelRetractionSchema(DatasetSchema):
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
                language_key=None,
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
            )

        super().__init__(
            dataset_folders=dataset_folders,
            zarr_path=zarr_path,
            raw_observations=raw_observation_config,
            image_path_config=image_path_config,
            has_phase_labels=has_phase_labels,
            phase_label_key=BOWEL_RETRACTION_PHASE_COL if has_phase_labels else None,
        )


    def get_image_path_column(self, camera: str) -> str:
        """Get CSV column name for image paths."""
        cfg = self.image_path_config
        if camera == Cameras.LEFT.value:
            return cfg.rectified_left_image_key if self.raw_observations.use_rectified_images else cfg.left_image_key
        elif camera == Cameras.RIGHT.value:
            return cfg.rectified_right_image_key if self.raw_observations.use_rectified_images else cfg.right_image_key
        else:
            raise ValueError(f"Unknown camera: {camera}")


    def compute_depth_path(self, base_image_path: str) -> str:
        """Compute depth file path from left image path using config patterns."""
        # TODO: we should store depth paths directly in the csv instead of computing them on the fly.
        cfg = self.image_path_config
        key_to_sub = (
            cfg.rectified_left_image_key
            if self.raw_observations.use_rectified_images
            else cfg.left_dir_pattern
        )
        depth_path = base_image_path.replace(key_to_sub, cfg.depth_dir_pattern)
        depth_path = re.sub(
            rf'(\d+){re.escape(cfg.rgb_extension)}$',
            cfg.depth_file_pattern,
            depth_path
        )
        return depth_path
