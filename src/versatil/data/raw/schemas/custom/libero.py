"""Dataset schema for LIBERO simulation datasets (HDF5 format)."""

import albumentations as A
import h5py
import numpy as np

from versatil.data.constants import Cameras, DatasetType, ObsKey
from versatil.data.metadata import (
    CameraMetadata,
    ObservationMetadata,
)
from versatil.data.raw.zarr_meta import DatasetMetadata
from versatil.data.raw.schemas.hdf5 import Hdf5DatasetSchema


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
        metadata: DatasetMetadata,
        dataset_type: str = DatasetType.LIBERO.value,
    ):
        """Initialize the LIBERO schema.

        Args:
            hdf5_paths: List of paths to LIBERO HDF5 files. Each file is a separate task.
            zarr_path: Path to save/load the zarr file
            metadata: Metadata to use for creating the zarr store from the raw data.
            dataset_type: Type of dataset. Must be 'libero'.
        """
        if dataset_type != DatasetType.LIBERO.value:
            raise ValueError(
                f"LiberoSchema only supports dataset_type='{DatasetType.LIBERO.value}', "
                f"got '{dataset_type}'"
            )
        self.obs_group_path = "obs"
        self.actions_key = "actions"
        self.extract_language_from_filename = True
        super().__init__(
            hdf5_paths=hdf5_paths,
            zarr_path=zarr_path,
            metadata=metadata,
            dataset_type=dataset_type,
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
            demo_group: h5py Group for a single episode demonstration.
            resizer: Albumentations resizer with interpolation (e.g. RGB images)
            depth_resizer: Albumentations resizer with nearest neighbor (e.g. for depth images)

        Returns:
            Dictionary mapping zarr keys to numpy arrays
        """
        data = {}
        obs_group = demo_group[self.obs_group_path]
        for zarr_key, obs in self.metadata.observations.items():
            if isinstance(obs, CameraMetadata):
                continue
            if (
                zarr_key == ObsKey.LANGUAGE.value
                and self.extract_language_from_filename
            ):
                continue
            elif isinstance(obs, ObservationMetadata):
                if obs.dtype == "str":
                    data[zarr_key] = obs_group[obs.raw_data_column_keys[0]].astype(str)[
                        :
                    ]
                else:
                    values = np.concatenate(
                        [obs_group[key][:] for key in obs.raw_data_column_keys], axis=-1
                    ).astype(obs.dtype)
                    data[zarr_key] = values

        for zarr_key, action in self.metadata.precomputed_actions.items():
            values = np.concatenate(
                [demo_group[key][:] for key in action.raw_data_column_keys], axis=-1
            ).astype(action.dtype)
            values = values[..., action.slice_start : action.slice_end]
            data[zarr_key] = values

        for zarr_key, cam_metadata in self.metadata.cameras.items():
            cam = cam_metadata.raw_camera_key
            if cam not in obs_group:
                raise ValueError(f"Camera key '{cam}' not found in HDF5 obs group")
            raw_images = obs_group[cam][:]
            if cam == Cameras.DEPTH.value:
                images = [depth_resizer(image=img)["image"] for img in raw_images]
                data[zarr_key] = np.stack(images).astype(cam_metadata.dtype)
            else:
                images = [resizer(image=img)["image"] for img in raw_images]
                data[zarr_key] = np.stack(images).astype(cam_metadata.dtype)

        if self.extract_language_from_filename:
            episode_len = self._get_episode_length(demo_group)
            hdf5_path = demo_group.file.filename
            task_language = self.get_language_from_filename(hdf5_path)
            data[ObsKey.LANGUAGE.value] = np.array(
                [[task_language]] * episode_len
            )  # Shape (T, 1)
        return data

    def _get_episode_length(self, demo_group: h5py.Group) -> int:
        """Get episode length from demo group."""
        if self.actions_key and self.actions_key in demo_group:
            return demo_group[self.actions_key].shape[0]
        obs_group = demo_group[self.obs_group_path]
        first_key = next(iter(obs_group.keys()))
        return obs_group[first_key].shape[0]

    @staticmethod
    def get_language_from_filename(hdf5_path: str) -> str:
        """Extract task language from LIBERO HDF5 filename.

        LIBERO filenames follow pattern: task_name_demo.hdf5
        E.g., "pick_up_the_black_bowl_demo.hdf5" -> "pick up the black bowl"

        Args:
            hdf5_path: Path to the HDF5 file.
        """
        filename = hdf5_path.rsplit("/", 1)[-1]
        task_name = filename.removesuffix("_demo.hdf5").replace("_", " ")
        return task_name

    def get_required_zarr_keys(self) -> list[str]:
        """Get all required zarr keys based on schema configuration.

        Extends base implementation to include language key when
        extract_language_from_filename is True.
        """
        keys = super().get_required_zarr_keys()
        if self.extract_language_from_filename and ObsKey.LANGUAGE.value not in keys:
            keys.append(ObsKey.LANGUAGE.value)
        return keys

    def get_zarr_array_specs(self) -> dict:
        """Get specifications for all zarr arrays to create.

        Extends base implementation to include language array when
        extract_language_from_filename is True.
        """
        specs = super().get_zarr_array_specs()
        if self.extract_language_from_filename and ObsKey.LANGUAGE.value not in specs:
            specs[ObsKey.LANGUAGE.value] = {
                "shape": (0, 1),
                "chunks": (100, 1),
                "dtype": "str",
                "needs_compressor": False,
            }
        return specs
