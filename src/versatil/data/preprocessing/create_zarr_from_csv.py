"""Creates a Zarr-based replay buffer dataset from robot demonstration CSV files and associated images."""

from collections.abc import Generator
from pathlib import Path

import numpy as np
import pandas as pd

from versatil.data.preprocessing.create_zarr_arrays import create_zarr_replay_buffer
from versatil.data.raw.schemas import CsvDatasetSchema


def _iter_csv_episodes(
    schema: CsvDatasetSchema,
    sorted_paths: list[str],
) -> Generator[dict[str, np.ndarray]]:
    """Yield episode data dicts from sorted CSV paths."""
    for path in sorted_paths:
        episode_df = pd.read_csv(path)
        yield schema.extract_episode(episode=episode_df)


def create_replay_buffer(schema: CsvDatasetSchema, datasets_paths: list[str]) -> None:
    """Creates a Zarr-based replay buffer using a Hydra-instantiated dataset schema.

    Args:
        schema: CsvDatasetSchema instance (instantiated by Hydra)
        datasets_paths: List of paths to episode CSV files
    """
    sorted_paths = sorted(datasets_paths, key=lambda x: int(Path(x).parent.name))
    episodes = _iter_csv_episodes(
        schema=schema,
        sorted_paths=sorted_paths,
    )

    create_zarr_replay_buffer(
        schema=schema,
        episodes=episodes,
        total_episodes=len(datasets_paths),
    )
