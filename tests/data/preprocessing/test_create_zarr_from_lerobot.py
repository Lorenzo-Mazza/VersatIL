"""Tests for versatil.data.preprocessing.create_zarr_from_lerobot module."""

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import albumentations as A
import cv2
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
                resizer=A.NoOp(),
                depth_resizer=A.NoOp(),
            )
        )

        assert len(episodes) == 3

    def test_passes_episode_id_and_resizers_to_extract(
        self,
        mock_schema_factory: Callable[..., MagicMock],
    ):
        schema = mock_schema_factory(
            cameras={},
            total_episodes=2,
            extract_return={"state": np.zeros((10, 7), dtype=np.float32)},
        )
        resizer = A.NoOp()
        depth_resizer = A.NoOp()

        list(
            _iter_lerobot_episodes(
                schema=schema,
                total_episodes=2,
                resizer=resizer,
                depth_resizer=depth_resizer,
            )
        )

        assert schema.extract_episode.call_count == 2
        schema.extract_episode.assert_any_call(
            episode_id=0,
            resizer=resizer,
            depth_resizer=depth_resizer,
        )
        schema.extract_episode.assert_any_call(
            episode_id=1,
            resizer=resizer,
            depth_resizer=depth_resizer,
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
                resizer=A.NoOp(),
                depth_resizer=A.NoOp(),
            )
        )

        assert len(episodes) == 0
        schema.extract_episode.assert_not_called()


class TestCreateReplayBufferFromLerobot:
    @patch("versatil.data.preprocessing.create_zarr_from_lerobot.A.Resize")
    @patch(
        "versatil.data.preprocessing.create_zarr_from_lerobot.create_zarr_replay_buffer"
    )
    def test_cameras_present_creates_resize_transforms(
        self,
        mock_create_zarr,
        mock_resize_class,
        mock_schema_factory: Callable[..., MagicMock],
        mock_camera_metadata_factory: Callable[..., MagicMock],
    ):
        camera = mock_camera_metadata_factory(image_width=256, image_height=256)
        schema = mock_schema_factory(
            total_episodes=2,
            cameras={"top": camera},
            extract_return={"state": np.zeros((10, 7), dtype=np.float32)},
        )

        create_replay_buffer_from_lerobot(schema=schema)

        # Source always creates both an RGB resizer and a depth resizer
        # (with INTER_NEAREST) whenever cameras are present
        assert mock_resize_class.call_count == 2
        mock_resize_class.assert_any_call(height=256, width=256)
        mock_resize_class.assert_any_call(
            height=256,
            width=256,
            interpolation=cv2.INTER_NEAREST,
        )

    @patch(
        "versatil.data.preprocessing.create_zarr_from_lerobot.create_zarr_replay_buffer"
    )
    def test_no_cameras_creates_noop_transforms(
        self,
        mock_create_zarr,
        mock_schema_factory: Callable[..., MagicMock],
    ):
        schema = mock_schema_factory(
            total_episodes=2,
            cameras={},
            extract_return={"state": np.zeros((10, 7), dtype=np.float32)},
        )

        create_replay_buffer_from_lerobot(schema=schema)

        mock_create_zarr.assert_called_once()

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
