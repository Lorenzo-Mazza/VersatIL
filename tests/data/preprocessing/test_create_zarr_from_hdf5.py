"""Tests for versatil.data.preprocessing.create_zarr_from_hdf5 module."""

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import numpy as np

from versatil.data.preprocessing.create_zarr_from_hdf5 import (
    _count_hdf5_episodes,
    _iter_hdf5_episodes,
    create_replay_buffer_from_hdf5,
)


class TestCountHdf5Episodes:
    def test_single_file_counts_demos(
        self,
        mock_schema_factory: Callable[..., MagicMock],
    ):
        schema = mock_schema_factory(
            hdf5_paths=["/data/file.hdf5"],
            demo_names_per_file={"/data/file.hdf5": ["demo_0", "demo_1", "demo_2"]},
            cameras={},
        )

        result = _count_hdf5_episodes(schema=schema)

        assert result == 3

    def test_multiple_files_sums_demos(
        self,
        mock_schema_factory: Callable[..., MagicMock],
    ):
        schema = mock_schema_factory(
            hdf5_paths=["/data/a.hdf5", "/data/b.hdf5"],
            demo_names_per_file={
                "/data/a.hdf5": ["demo_0", "demo_1"],
                "/data/b.hdf5": ["demo_0"],
            },
            cameras={},
        )

        result = _count_hdf5_episodes(schema=schema)

        assert result == 3

    def test_empty_paths_returns_zero(
        self,
        mock_schema_factory: Callable[..., MagicMock],
    ):
        schema = mock_schema_factory(hdf5_paths=[], cameras={})

        result = _count_hdf5_episodes(schema=schema)

        assert result == 0


class TestIterHdf5Episodes:
    @patch("versatil.data.preprocessing.create_zarr_from_hdf5.h5py.File")
    def test_yields_one_episode_per_demo(
        self,
        mock_h5py_file,
        mock_schema_factory: Callable[..., MagicMock],
    ):
        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__ = MagicMock(return_value=mock_file)
        mock_h5py_file.return_value.__exit__ = MagicMock(return_value=False)
        schema = mock_schema_factory(
            hdf5_paths=["/data/file.hdf5"],
            demo_names_per_file={"/data/file.hdf5": ["demo_0", "demo_1"]},
            cameras={},
            extract_return={"position": np.zeros((5, 3), dtype=np.float32)},
        )

        episodes = list(
            _iter_hdf5_episodes(
                schema=schema,
            )
        )

        assert len(episodes) == 2

    @patch("versatil.data.preprocessing.create_zarr_from_hdf5.h5py.File")
    def test_demos_sorted_by_numeric_suffix(
        self,
        mock_h5py_file,
        mock_schema_factory: Callable[..., MagicMock],
    ):
        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__ = MagicMock(return_value=mock_file)
        mock_h5py_file.return_value.__exit__ = MagicMock(return_value=False)
        schema = mock_schema_factory(
            hdf5_paths=["/data/file.hdf5"],
            demo_names_per_file={
                "/data/file.hdf5": ["demo_10", "demo_2", "demo_1"],
            },
            cameras={},
            extract_return={"position": np.zeros((5, 3), dtype=np.float32)},
        )

        list(
            _iter_hdf5_episodes(
                schema=schema,
            )
        )

        accessed_keys = [c.args[0] for c in mock_file.__getitem__.call_args_list]
        assert accessed_keys == ["data/demo_1", "data/demo_2", "data/demo_10"]

    @patch("versatil.data.preprocessing.create_zarr_from_hdf5.h5py.File")
    def test_passes_demo_group_and_resizers_to_extract_episode(
        self,
        mock_h5py_file,
        mock_schema_factory: Callable[..., MagicMock],
    ):
        mock_file = MagicMock()
        mock_demo_group = MagicMock()
        mock_file.__getitem__.return_value = mock_demo_group
        mock_h5py_file.return_value.__enter__ = MagicMock(return_value=mock_file)
        mock_h5py_file.return_value.__exit__ = MagicMock(return_value=False)
        schema = mock_schema_factory(
            hdf5_paths=["/data/file.hdf5"],
            demo_names_per_file={"/data/file.hdf5": ["demo_0"]},
            cameras={},
            extract_return={"position": np.zeros((5, 3), dtype=np.float32)},
        )

        list(
            _iter_hdf5_episodes(
                schema=schema,
            )
        )

        schema.extract_episode.assert_called_once_with(
            demo_group=mock_demo_group,
        )

    @patch("versatil.data.preprocessing.create_zarr_from_hdf5.h5py.File")
    def test_iterates_over_multiple_hdf5_files(
        self,
        mock_h5py_file,
        mock_schema_factory: Callable[..., MagicMock],
    ):
        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__ = MagicMock(return_value=mock_file)
        mock_h5py_file.return_value.__exit__ = MagicMock(return_value=False)
        schema = mock_schema_factory(
            hdf5_paths=["/data/a.hdf5", "/data/b.hdf5"],
            demo_names_per_file={
                "/data/a.hdf5": ["demo_0"],
                "/data/b.hdf5": ["demo_0", "demo_1"],
            },
            cameras={},
            extract_return={"position": np.zeros((5, 3), dtype=np.float32)},
        )

        episodes = list(
            _iter_hdf5_episodes(
                schema=schema,
            )
        )

        assert len(episodes) == 3

    @patch("versatil.data.preprocessing.create_zarr_from_hdf5.logging")
    @patch("versatil.data.preprocessing.create_zarr_from_hdf5.h5py.File")
    def test_logs_each_hdf5_file_path(
        self,
        mock_h5py_file,
        mock_logging,
        mock_schema_factory: Callable[..., MagicMock],
    ):
        mock_file = MagicMock()
        mock_h5py_file.return_value.__enter__ = MagicMock(return_value=mock_file)
        mock_h5py_file.return_value.__exit__ = MagicMock(return_value=False)
        schema = mock_schema_factory(
            hdf5_paths=["/data/a.hdf5", "/data/b.hdf5"],
            demo_names_per_file={
                "/data/a.hdf5": ["demo_0"],
                "/data/b.hdf5": ["demo_0"],
            },
            cameras={},
            extract_return={"position": np.zeros((5, 3), dtype=np.float32)},
        )

        list(
            _iter_hdf5_episodes(
                schema=schema,
            )
        )

        log_messages = [str(c) for c in mock_logging.info.call_args_list]
        assert any("/data/a.hdf5" in msg for msg in log_messages)
        assert any("/data/b.hdf5" in msg for msg in log_messages)


class TestCreateReplayBufferFromHdf5:
    @patch(
        "versatil.data.preprocessing.create_zarr_from_hdf5.create_zarr_replay_buffer"
    )
    def test_total_episodes_passed_as_sum_of_all_demos(
        self,
        mock_create_zarr,
        mock_schema_factory: Callable[..., MagicMock],
    ):
        schema = mock_schema_factory(
            hdf5_paths=["/data/a.hdf5", "/data/b.hdf5"],
            demo_names_per_file={
                "/data/a.hdf5": ["demo_0", "demo_1"],
                "/data/b.hdf5": ["demo_0"],
            },
            cameras={},
        )

        create_replay_buffer_from_hdf5(schema=schema)

        call_kwargs = mock_create_zarr.call_args
        assert call_kwargs.kwargs["total_episodes"] == 3

    @patch(
        "versatil.data.preprocessing.create_zarr_from_hdf5.create_zarr_replay_buffer"
    )
    def test_schema_passed_through_to_create_zarr(
        self,
        mock_create_zarr,
        mock_schema_factory: Callable[..., MagicMock],
    ):
        schema = mock_schema_factory(
            hdf5_paths=[],
            cameras={},
        )

        create_replay_buffer_from_hdf5(schema=schema)

        call_kwargs = mock_create_zarr.call_args
        assert call_kwargs.kwargs["schema"] is schema
