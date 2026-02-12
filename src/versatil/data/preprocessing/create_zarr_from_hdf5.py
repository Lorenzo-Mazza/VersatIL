"""Creates a Zarr-based replay buffer dataset from HDF5 files (e.g., LIBERO)."""

from collections.abc import Generator

import albumentations as A
import cv2
import h5py
import numpy as np

from versatil.data.preprocessing.create_zarr_arrays import create_zarr_replay_buffer
from versatil.data.raw.schemas import Hdf5DatasetSchema
import logging


def _iter_hdf5_episodes(
    schema: Hdf5DatasetSchema,
    resizer: A.Resize | A.NoOp,
    depth_resizer: A.Resize | A.NoOp,
) -> Generator[dict[str, np.ndarray], None, None]:
    """Yield episode data dicts from HDF5 files."""
    for hdf5_path in schema.hdf5_paths:
        logging.info(msg=f"  Processing: {hdf5_path}")
        with h5py.File(hdf5_path, "r") as f:
            demo_names = schema.get_demo_names(hdf5_path)
            demo_names_sorted = sorted(
                demo_names, key=lambda x: int(x.split("_")[1])
            )
            for demo_name in demo_names_sorted:
                demo_group = f[f"data/{demo_name}"]
                yield schema.extract_episode(
                    demo_group=demo_group,
                    resizer=resizer,
                    depth_resizer=depth_resizer,
                )


def _count_hdf5_episodes(schema: Hdf5DatasetSchema) -> int:
    """Count total episodes across all HDF5 files."""
    total = 0
    for hdf5_path in schema.hdf5_paths:
        total += len(schema.get_demo_names(hdf5_path))
    return total


def create_replay_buffer_from_hdf5(schema: Hdf5DatasetSchema) -> None:
    """Creates a Zarr-based replay buffer from multiple HDF5 files.

    Args:
        schema: Hdf5DatasetSchema instance with HDF5 paths and zarr path configured
    """
    cameras = schema.metadata.cameras
    if cameras:
        first_cam = next(iter(cameras.values()))
        resizer = A.Resize(
            height=first_cam.image_height, width=first_cam.image_width
        )
        depth_resizer = A.Resize(
            height=first_cam.image_height,
            width=first_cam.image_width,
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        resizer = A.NoOp()
        depth_resizer = A.NoOp()

    total_episodes = _count_hdf5_episodes(schema=schema)
    episodes = _iter_hdf5_episodes(
        schema=schema, resizer=resizer, depth_resizer=depth_resizer
    )

    create_zarr_replay_buffer(
        schema=schema,
        episodes=episodes,
        total_episodes=total_episodes,
    )
