"""Dataset schema for LIBERO simulation datasets (HDF5 format)."""

import albumentations as A
import h5py
import numpy as np

from refactoring.configs.data.dataset.raw_observations import RawObservationsConfig
from refactoring.data.constants import (
    GRIPPER_STATE_OBS_KEY,
    LANGUAGE_KEY,
    PRECOMPUTED_ACTIONS_KEY,
    PROPRIO_OBS_ROBOT_FRAME_KEY,
    Cameras,
    GripperType,
)
from refactoring.data.schemas.hdf5 import Hdf5DatasetSchema


class LiberoSchema(Hdf5DatasetSchema):
    """Schema for LIBERO HDF5 datasets.

    LIBERO datasets have precomputed actions and store all data in HDF5 format.
    Each HDF5 file represents a single task with multiple demos.

    Structure per demo:
        - actions: (T, 7) - position delta (3) + orientation delta (3) + gripper (1)
        - obs/agentview_rgb: (T, 128, 128, 3)
        - obs/eye_in_hand_rgb: (T, 128, 128, 3)
        - obs/ee_pos: (T, 3)
        - obs/ee_ori: (T, 3)
        - obs/ee_states: (T, 6) - concatenation of ee_pos and ee_ori
        - obs/gripper_states: (T, 2)
        - obs/joint_states: (T, 7)
    """

    def __init__(
            self,
            hdf5_paths: list[str],
            zarr_path: str,
            raw_observation_config: RawObservationsConfig | None = None,
    ):
        """Initialize the LIBERO schema.

        Args:
            hdf5_paths: List of paths to LIBERO HDF5 files. Each file is a separate task.
            zarr_path: Path to save/load the zarr file
            raw_observation_config: Configuration for raw observations. If None, uses defaults.
        """
        if raw_observation_config is None:
            raw_observation_config = RawObservationsConfig(
                robot_frame_proprio_keys=["ee_pos", "ee_ori"],
                camera_frame_proprio_keys=[],
                gripper_state_keys=["gripper_states"],
                camera_keys=[Cameras.AGENTVIEW.value, Cameras.EYE_IN_HAND.value],
                use_rectified_images=False,
                image_width=128,
                image_height=128,
                has_position=True,
                position_dim=3,
                has_orientation=True,
                orientation_dim=3,
                has_gripper=True,
                gripper_type=GripperType.CONTINUOUS.value,
                gripper_dim=2,
                has_precomputed_actions=True,
                precomputed_action_dim=7,
            )

        super().__init__(
            hdf5_paths=hdf5_paths,
            zarr_path=zarr_path,
            raw_observations=raw_observation_config,
            has_phase_labels=False,
            obs_group_path="obs",
            actions_key="actions",
            extract_language_from_filename=True,
        )

    def get_demo_names(self, hdf5_path: str) -> list[str]:
        """Get list of demo names in the specified HDF5 file.

        Args:
            hdf5_path: Path to the HDF5 file.
        """
        with h5py.File(hdf5_path, "r") as f:
            return list(f["data"].keys())

    def extract_episode(
        self,
        demo_group: h5py.Group,
        resizer: A.Resize | A.NoOp,
        depth_resizer: A.Resize | A.NoOp,
    ) -> dict[str, np.ndarray]:
        """Extract all data from a LIBERO demo group.

        Args:
            demo_group: h5py Group for a single demo
            resizer: Albumentations resizer for RGB images
            depth_resizer: Albumentations resizer for depth images

        Returns:
            Dictionary mapping zarr keys to numpy arrays
        """
        data = {}
        obs_config = self.raw_observations
        obs_group = demo_group[self.obs_group_path]

        if obs_config.robot_frame_proprio_keys:
            arrays = [obs_group[key][:] for key in obs_config.robot_frame_proprio_keys if key in obs_group]
            if arrays:
                data[PROPRIO_OBS_ROBOT_FRAME_KEY] = np.concatenate(arrays, axis=-1).astype(np.float32)

        if obs_config.gripper_state_keys:
            arrays = [obs_group[key][:] for key in obs_config.gripper_state_keys if key in obs_group]
            if arrays:
                data[GRIPPER_STATE_OBS_KEY] = np.concatenate(arrays, axis=-1).astype(np.float32)

        if obs_config.has_precomputed_actions and self.actions_key:
            data[PRECOMPUTED_ACTIONS_KEY] = demo_group[self.actions_key][:].astype(np.float32)

        if self.extract_language_from_filename:
            episode_len = self._get_episode_length(demo_group)
            hdf5_path = demo_group.file.filename
            task_language = self.get_language_from_filename(hdf5_path)
            data[LANGUAGE_KEY] = np.array([task_language] * episode_len)
        elif obs_config.language_key and obs_config.language_key in obs_group:
            data[LANGUAGE_KEY] = obs_group[obs_config.language_key][:].astype(str)

        for cam in obs_config.camera_keys:
            if cam not in obs_group:
                raise ValueError(f"Camera key '{cam}' not found in HDF5 obs group")

            raw_images = obs_group[cam][:]
            if cam == Cameras.DEPTH.value:
                images = [depth_resizer(image=img)['image'] for img in raw_images]
                data[cam] = np.stack(images).astype(np.float32)
            else:
                images = [resizer(image=img)['image'] for img in raw_images]
                data[cam] = np.stack(images).astype(np.uint8)

        return data

    def _get_episode_length(self, demo_group: h5py.Group) -> int:
        """Get episode length from demo group."""
        if self.actions_key and self.actions_key in demo_group:
            return demo_group[self.actions_key].shape[0]
        obs_group = demo_group[self.obs_group_path]
        first_key = next(iter(obs_group.keys()))
        return obs_group[first_key].shape[0]

    def get_language_from_filename(self, hdf5_path: str) -> str:
        """Extract task language from LIBERO HDF5 filename.

        LIBERO filenames follow pattern: task_name_demo.hdf5
        E.g., "pick_up_the_black_bowl_demo.hdf5" -> "pick up the black bowl"

        Args:
            hdf5_path: Path to the HDF5 file.
        """
        filename = hdf5_path.rsplit('/', 1)[-1]
        task_name = filename.removesuffix('_demo.hdf5').replace('_', ' ')
        return task_name
