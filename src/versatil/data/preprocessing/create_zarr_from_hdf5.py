"""Creates a Zarr-based replay buffer dataset from HDF5 files (e.g., LIBERO)."""

import logging
from collections.abc import Generator

import h5py
import numpy as np

from versatil.data.preprocessing.create_zarr_arrays import create_zarr_replay_buffer
from versatil.data.preprocessing.sharding import DEFAULT_IMAGE_FRAMES_PER_SHARD
from versatil.data.raw.schemas import Hdf5DatasetSchema


def _iter_hdf5_episodes(
    schema: Hdf5DatasetSchema,
) -> Generator[dict[str, np.ndarray]]:
    """Yield episode data dicts from HDF5 files."""
    for hdf5_path in schema.hdf5_paths:
        logging.info(msg=f"  Processing: {hdf5_path}")
        with h5py.File(hdf5_path, "r") as f:
            demo_names = schema.get_demo_names(hdf5_path)
            demo_names_sorted = sorted(demo_names, key=lambda x: int(x.split("_")[1]))
            for demo_name in demo_names_sorted:
                demo_group = f[f"data/{demo_name}"]
                yield schema.extract_episode(
                    demo_group=demo_group,
                )


def _count_hdf5_episodes(schema: Hdf5DatasetSchema) -> int:
    """Count total episodes across all HDF5 files."""
    total = 0
    for hdf5_path in schema.hdf5_paths:
        total += len(schema.get_demo_names(hdf5_path))
    return total


def create_replay_buffer_from_hdf5(
    schema: Hdf5DatasetSchema,
    image_frames_per_shard: int | None = DEFAULT_IMAGE_FRAMES_PER_SHARD,
) -> None:
    """Creates a Zarr-based replay buffer from multiple HDF5 files.

    Args:
        schema: Hdf5DatasetSchema instance with HDF5 paths and zarr path configured
        image_frames_per_shard: Number of image frames stored in each shard, or
            None to disable sharding.

    Raises:
        TypeError: If image_frames_per_shard is not an integer.
        ValueError: If image_frames_per_shard is not a positive multiple of the
            image chunk length.
    """
    total_episodes = _count_hdf5_episodes(schema=schema)
    episodes = _iter_hdf5_episodes(schema=schema)

    create_zarr_replay_buffer(
        schema=schema,
        episodes=episodes,
        total_episodes=total_episodes,
        image_frames_per_shard=image_frames_per_shard,
    )
