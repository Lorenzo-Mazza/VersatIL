"""Creates a Zarr-based replay buffer dataset from LeRobot datasets."""

from collections.abc import Generator

import albumentations as A
import cv2
import numpy as np

from versatil.data.preprocessing.create_zarr_arrays import create_zarr_replay_buffer
from versatil.data.raw.schemas.lerobot import LeRobotDatasetSchemaV30


def _iter_lerobot_episodes(
    schema: LeRobotDatasetSchemaV30,
    total_episodes: int,
    resizer: A.Resize | A.NoOp,
    depth_resizer: A.Resize | A.NoOp,
) -> Generator[dict[str, np.ndarray], None, None]:
    """Yield episode data dicts from a LeRobot dataset."""
    for episode_id in range(total_episodes):
        yield schema.extract_episode(
            episode_id=episode_id, resizer=resizer, depth_resizer=depth_resizer
        )


def create_replay_buffer_from_lerobot(schema: LeRobotDatasetSchemaV30) -> None:
    """Creates a Zarr-based replay buffer from a LeRobot dataset.

    Args:
        schema: LeRobotDatasetSchemaV30 instance.
    """
    cameras = schema.metadata.cameras
    if cameras:
        first_cam = next(iter(cameras.values()))
        resizer = A.Resize(height=first_cam.image_height, width=first_cam.image_width)
        depth_resizer = A.Resize(
            height=first_cam.image_height,
            width=first_cam.image_width,
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        resizer = A.NoOp()
        depth_resizer = A.NoOp()

    total_episodes = schema.lerobot_metadata.get_total_episodes()
    episodes = _iter_lerobot_episodes(
        schema=schema,
        total_episodes=total_episodes,
        resizer=resizer,
        depth_resizer=depth_resizer,
    )

    create_zarr_replay_buffer(
        schema=schema,
        episodes=episodes,
        total_episodes=total_episodes,
    )
