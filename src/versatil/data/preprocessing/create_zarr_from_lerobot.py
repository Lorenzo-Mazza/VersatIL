"""Creates a Zarr-based replay buffer dataset from LeRobot datasets."""

from collections.abc import Generator

import numpy as np

from versatil.data.preprocessing.create_zarr_arrays import create_zarr_replay_buffer
from versatil.data.raw.schemas.lerobot import LeRobotDatasetSchemaV30


def _iter_lerobot_episodes(
    schema: LeRobotDatasetSchemaV30,
    total_episodes: int,
) -> Generator[dict[str, np.ndarray]]:
    """Yield episode data dicts from a LeRobot dataset."""
    for episode_id in range(total_episodes):
        yield schema.extract_episode(episode_id=episode_id)


def create_replay_buffer_from_lerobot(schema: LeRobotDatasetSchemaV30) -> None:
    """Creates a Zarr-based replay buffer from a LeRobot dataset.

    Args:
        schema: LeRobotDatasetSchemaV30 instance.
    """
    total_episodes = schema.lerobot_metadata.get_total_episodes()
    episodes = _iter_lerobot_episodes(
        schema=schema,
        total_episodes=total_episodes,
    )

    create_zarr_replay_buffer(
        schema=schema,
        episodes=episodes,
        total_episodes=total_episodes,
    )
