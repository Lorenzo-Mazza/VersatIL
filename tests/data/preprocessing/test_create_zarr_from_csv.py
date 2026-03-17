"""Tests for versatil.data.preprocessing.create_zarr_from_csv module."""

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import albumentations as A
import cv2
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
                resizer=A.NoOp(),
                depth_resizer=A.NoOp(),
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
        resizer = A.NoOp()
        depth_resizer = A.NoOp()

        list(
            _iter_csv_episodes(
                schema=schema,
                sorted_paths=["/path/1/data.csv"],
                resizer=resizer,
                depth_resizer=depth_resizer,
            )
        )

        schema.extract_episode.assert_called_once_with(
            episode=mock_dataframe,
            resizer=resizer,
            depth_resizer=depth_resizer,
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
                resizer=A.NoOp(),
                depth_resizer=A.NoOp(),
            )
        )

        mock_read_csv.assert_any_call("/data/1/ep.csv")
        mock_read_csv.assert_any_call("/data/2/ep.csv")


class TestCreateReplayBuffer:
    @patch("versatil.data.preprocessing.create_zarr_from_csv.A.Resize")
    @patch("versatil.data.preprocessing.create_zarr_from_csv.create_zarr_replay_buffer")
    def test_cameras_with_dimensions_creates_resize_transforms(
        self,
        mock_create_zarr,
        mock_resize_class,
        mock_schema_factory: Callable[..., MagicMock],
        mock_camera_metadata_factory: Callable[..., MagicMock],
    ):
        camera = mock_camera_metadata_factory(image_width=128, image_height=96)
        schema = mock_schema_factory(
            cameras={"left": camera},
            extract_return={"position": np.zeros((5, 3), dtype=np.float32)},
        )

        create_replay_buffer(
            schema=schema,
            datasets_paths=["/data/1/ep.csv"],
        )

        # Source always creates both an RGB resizer and a depth resizer
        # (with INTER_NEAREST) whenever cameras have dimensions
        assert mock_resize_class.call_count == 2
        mock_resize_class.assert_any_call(height=96, width=128)
        mock_resize_class.assert_any_call(
            height=96,
            width=128,
            interpolation=cv2.INTER_NEAREST,
        )

    @patch("versatil.data.preprocessing.create_zarr_from_csv.create_zarr_replay_buffer")
    def test_cameras_without_dimensions_creates_noop_transforms(
        self,
        mock_create_zarr,
        mock_schema_factory: Callable[..., MagicMock],
        mock_camera_metadata_factory: Callable[..., MagicMock],
    ):
        camera = mock_camera_metadata_factory(image_width=None, image_height=None)
        schema = mock_schema_factory(
            cameras={"left": camera},
            extract_return={"position": np.zeros((5, 3), dtype=np.float32)},
        )

        create_replay_buffer(
            schema=schema,
            datasets_paths=["/data/1/ep.csv"],
        )

        mock_create_zarr.assert_called_once()

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
        schema.extract_episode.side_effect = lambda episode, resizer, depth_resizer: {
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
