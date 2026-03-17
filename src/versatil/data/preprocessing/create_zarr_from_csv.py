"""Creates a Zarr-based replay buffer dataset from robot demonstration CSV files and associated images."""

from collections.abc import Generator
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd

from versatil.data.preprocessing.create_zarr_arrays import create_zarr_replay_buffer
from versatil.data.raw.schemas import CsvDatasetSchema


def _iter_csv_episodes(
    schema: CsvDatasetSchema,
    sorted_paths: list[str],
    resizer: A.Resize | A.NoOp,
    depth_resizer: A.Resize | A.NoOp,
) -> Generator[dict[str, np.ndarray]]:
    """Yield episode data dicts from sorted CSV paths."""
    for path in sorted_paths:
        episode_df = pd.read_csv(path)
        yield schema.extract_episode(
            episode=episode_df, resizer=resizer, depth_resizer=depth_resizer
        )


def create_replay_buffer(schema: CsvDatasetSchema, datasets_paths: list[str]) -> None:
    """Creates a Zarr-based replay buffer using a Hydra-instantiated dataset schema.

    Args:
        schema: CsvDatasetSchema instance (instantiated by Hydra)
        datasets_paths: List of paths to episode CSV files
    """
    cameras = schema.metadata.cameras
    # TODO: this assumes all cameras have the same resolution, which may not be true
    if cameras:
        first_cam = next(iter(cameras.values()))
        image_width = first_cam.image_width
        image_height = first_cam.image_height
        if image_width is None or image_height is None:
            resizer = A.NoOp()
            depth_resizer = A.NoOp()
        else:
            resizer = A.Resize(height=image_height, width=image_width)
            depth_resizer = A.Resize(
                height=image_height,
                width=image_width,
                interpolation=cv2.INTER_NEAREST,
            )
    else:
        resizer = A.NoOp()
        depth_resizer = A.NoOp()

    sorted_paths = sorted(datasets_paths, key=lambda x: int(Path(x).parent.name))
    episodes = _iter_csv_episodes(
        schema=schema,
        sorted_paths=sorted_paths,
        resizer=resizer,
        depth_resizer=depth_resizer,
    )

    create_zarr_replay_buffer(
        schema=schema,
        episodes=episodes,
        total_episodes=len(datasets_paths),
    )
