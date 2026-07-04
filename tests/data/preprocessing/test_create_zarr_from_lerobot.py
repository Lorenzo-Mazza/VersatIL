"""Tests for versatil.data.preprocessing.create_zarr_from_lerobot module."""

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import numpy as np

from versatil.data.preprocessing.create_zarr_from_lerobot import (
    _iter_lerobot_episodes,
    create_replay_buffer_from_lerobot,
)


class TestIterLerobotEpisodes:
    def test_yields_one_episode_per_id(
        self,
        mock_schema_factory: Callable[..., MagicMock],
    ):
        schema = mock_schema_factory(
            cameras={},
            total_episodes=3,
            extract_return={"state": np.zeros((10, 7), dtype=np.float32)},
        )

        episodes = list(
            _iter_lerobot_episodes(
                schema=schema,
                total_episodes=3,
            )
        )

        assert len(episodes) == 3

    def test_passes_episode_id_to_extract(
        self,
        mock_schema_factory: Callable[..., MagicMock],
    ):
        schema = mock_schema_factory(
            cameras={},
            total_episodes=2,
            extract_return={"state": np.zeros((10, 7), dtype=np.float32)},
        )

        list(
            _iter_lerobot_episodes(
                schema=schema,
                total_episodes=2,
            )
        )

        assert schema.extract_episode.call_count == 2
        schema.extract_episode.assert_any_call(
            episode_id=0,
        )
        schema.extract_episode.assert_any_call(
            episode_id=1,
        )

    def test_zero_episodes_yields_nothing(
        self,
        mock_schema_factory: Callable[..., MagicMock],
    ):
        schema = mock_schema_factory(
            cameras={},
            total_episodes=0,
            extract_return={"state": np.zeros((10, 7), dtype=np.float32)},
        )

        episodes = list(
            _iter_lerobot_episodes(
                schema=schema,
                total_episodes=0,
            )
        )

        assert len(episodes) == 0
        schema.extract_episode.assert_not_called()


class TestCreateReplayBufferFromLerobot:
    @patch(
        "versatil.data.preprocessing.create_zarr_from_lerobot.create_zarr_replay_buffer"
    )
    def test_total_episodes_from_lerobot_metadata(
        self,
        mock_create_zarr,
        mock_schema_factory: Callable[..., MagicMock],
    ):
        schema = mock_schema_factory(
            total_episodes=10,
            cameras={},
            extract_return={"state": np.zeros((10, 7), dtype=np.float32)},
        )

        create_replay_buffer_from_lerobot(schema=schema)

        call_kwargs = mock_create_zarr.call_args
        assert call_kwargs.kwargs["total_episodes"] == 10

    @patch(
        "versatil.data.preprocessing.create_zarr_from_lerobot.create_zarr_replay_buffer"
    )
    def test_schema_passed_through_to_create_zarr(
        self,
        mock_create_zarr,
        mock_schema_factory: Callable[..., MagicMock],
    ):
        schema = mock_schema_factory(
            total_episodes=1,
            cameras={},
            extract_return={"state": np.zeros((10, 7), dtype=np.float32)},
        )

        create_replay_buffer_from_lerobot(schema=schema)

        call_kwargs = mock_create_zarr.call_args
        assert call_kwargs.kwargs["schema"] is schema
