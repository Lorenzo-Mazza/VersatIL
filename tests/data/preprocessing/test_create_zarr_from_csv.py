"""Tests for versatil.data.preprocessing.create_zarr_from_csv module."""

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import numpy as np

from versatil.data.preprocessing.create_zarr_from_csv import (
    _iter_csv_episodes,
    create_replay_buffer,
)


class TestIterCsvEpisodes:
    @patch("versatil.data.preprocessing.create_zarr_from_csv.pd.read_csv")
    def test_yields_one_episode_per_csv_path(
        self,
        mock_read_csv,
        mock_schema_factory: Callable[..., MagicMock],
    ):
        mock_read_csv.return_value = MagicMock()
        schema = mock_schema_factory(
            cameras={},
            extract_return={"position": np.zeros((5, 3), dtype=np.float32)},
        )

        episodes = list(
            _iter_csv_episodes(
                schema=schema,
                sorted_paths=["/path/1/data.csv", "/path/2/data.csv"],
            )
        )

        assert len(episodes) == 2
        assert mock_read_csv.call_count == 2

    @patch("versatil.data.preprocessing.create_zarr_from_csv.pd.read_csv")
    def test_passes_dataframe_and_resizers_to_extract_episode(
        self,
        mock_read_csv,
        mock_schema_factory: Callable[..., MagicMock],
    ):
        mock_dataframe = MagicMock()
        mock_read_csv.return_value = mock_dataframe
        schema = mock_schema_factory(
            cameras={},
            extract_return={"position": np.zeros((5, 3), dtype=np.float32)},
        )

        list(
            _iter_csv_episodes(
                schema=schema,
                sorted_paths=["/path/1/data.csv"],
            )
        )

        schema.extract_episode.assert_called_once_with(
            episode=mock_dataframe,
        )

    @patch("versatil.data.preprocessing.create_zarr_from_csv.pd.read_csv")
    def test_reads_each_csv_path(
        self,
        mock_read_csv,
        mock_schema_factory: Callable[..., MagicMock],
    ):
        mock_read_csv.return_value = MagicMock()
        schema = mock_schema_factory(
            cameras={},
            extract_return={"position": np.zeros((5, 3), dtype=np.float32)},
        )

        list(
            _iter_csv_episodes(
                schema=schema,
                sorted_paths=["/data/1/ep.csv", "/data/2/ep.csv"],
            )
        )

        mock_read_csv.assert_any_call("/data/1/ep.csv")
        mock_read_csv.assert_any_call("/data/2/ep.csv")


class TestCreateReplayBuffer:
    @patch("versatil.data.preprocessing.create_zarr_from_csv.create_zarr_replay_buffer")
    def test_no_cameras_creates_noop_transforms(
        self,
        mock_create_zarr,
        mock_schema_factory: Callable[..., MagicMock],
    ):
        schema = mock_schema_factory(
            cameras={},
            extract_return={"position": np.zeros((5, 3), dtype=np.float32)},
        )

        create_replay_buffer(
            schema=schema,
            datasets_paths=["/data/1/ep.csv"],
        )

        mock_create_zarr.assert_called_once()

    @patch("versatil.data.preprocessing.create_zarr_from_csv.pd.read_csv")
    @patch("versatil.data.preprocessing.create_zarr_from_csv.create_zarr_replay_buffer")
    def test_paths_sorted_by_numeric_parent_directory_name(
        self,
        mock_create_zarr,
        mock_read_csv,
        mock_schema_factory: Callable[..., MagicMock],
    ):
        mock_read_csv.return_value = MagicMock()
        call_order = []
        schema = mock_schema_factory(cameras={})
        schema.extract_episode.side_effect = lambda episode: {
            "position": np.zeros((5, 3), dtype=np.float32)
        }

        def capture_episodes(schema, episodes, total_episodes):
            for episode in episodes:
                call_order.append(episode)

        mock_create_zarr.side_effect = capture_episodes
        unsorted_paths = [
            "/data/10/ep.csv",
            "/data/2/ep.csv",
            "/data/1/ep.csv",
        ]

        create_replay_buffer(schema=schema, datasets_paths=unsorted_paths)

        read_paths = [c.args[0] for c in mock_read_csv.call_args_list]
        assert read_paths == ["/data/1/ep.csv", "/data/2/ep.csv", "/data/10/ep.csv"]

    @patch("versatil.data.preprocessing.create_zarr_from_csv.create_zarr_replay_buffer")
    def test_total_episodes_matches_number_of_paths(
        self,
        mock_create_zarr,
        mock_schema_factory: Callable[..., MagicMock],
    ):
        schema = mock_schema_factory(
            cameras={},
            extract_return={"position": np.zeros((5, 3), dtype=np.float32)},
        )

        create_replay_buffer(
            schema=schema,
            datasets_paths=["/data/1/ep.csv", "/data/2/ep.csv"],
        )

        call_kwargs = mock_create_zarr.call_args
        assert call_kwargs.kwargs["total_episodes"] == 2

    @patch("versatil.data.preprocessing.create_zarr_from_csv.create_zarr_replay_buffer")
    def test_schema_passed_through_to_create_zarr(
        self,
        mock_create_zarr,
        mock_schema_factory: Callable[..., MagicMock],
    ):
        schema = mock_schema_factory(
            cameras={},
            extract_return={"position": np.zeros((5, 3), dtype=np.float32)},
        )

        create_replay_buffer(
            schema=schema,
            datasets_paths=["/data/1/ep.csv"],
        )

        call_kwargs = mock_create_zarr.call_args
        assert call_kwargs.kwargs["schema"] is schema
