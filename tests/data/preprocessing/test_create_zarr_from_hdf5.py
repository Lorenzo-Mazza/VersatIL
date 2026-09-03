"""Tests for versatil.data.preprocessing.create_zarr_from_hdf5 module."""

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
import zarr

from versatil.data.constants import Cameras, CoordinateSystem, ObsKey, ProprioKey
from versatil.data.metadata import (
    PositionActionMetadata,
    PositionObservationMetadata,
    RGBCameraMetadata,
)
from versatil.data.preprocessing.create_zarr_from_hdf5 import (
    _count_hdf5_episodes,
    _iter_hdf5_episodes,
    create_replay_buffer_from_hdf5,
)
from versatil.data.raw.schemas.custom.libero import LiberoSchema
from versatil.data.raw.zarr_meta import DatasetMetadata


def _write_toy_libero_hdf5(hdf5_path: Path) -> dict[str, np.ndarray]:
    """Write two small LIBERO-style demonstrations.

    Args:
        hdf5_path: Destination HDF5 path.

    Returns:
        Expected concatenated numerical arrays keyed by their Zarr names.
    """
    expected_positions = []
    expected_actions = []
    with h5py.File(hdf5_path, "w") as hdf5_file:
        data_group = hdf5_file.create_group("data")
        for episode_index in range(2):
            demo_group = data_group.create_group(f"demo_{episode_index}")
            obs_group = demo_group.create_group("obs")
            frame_offset = episode_index * 3
            frame_values = np.arange(
                frame_offset + 10,
                frame_offset + 13,
                dtype=np.uint8,
            )
            agentview_images = np.broadcast_to(
                frame_values[:, None, None, None],
                (3, 8, 8, 3),
            ).copy()
            eye_in_hand_images = 255 - agentview_images
            positions = np.arange(9, dtype=np.float32).reshape(3, 3) + frame_offset
            actions = np.arange(21, dtype=np.float32).reshape(3, 7) + frame_offset

            obs_group.create_dataset(Cameras.AGENTVIEW.value, data=agentview_images)
            obs_group.create_dataset(
                Cameras.EYE_IN_HAND.value,
                data=eye_in_hand_images,
            )
            obs_group.create_dataset("ee_pos", data=positions)
            demo_group.create_dataset("actions", data=actions)
            expected_positions.append(positions)
            expected_actions.append(actions[:, :3])

    return {
        ProprioKey.EE_POS.value: np.concatenate(expected_positions),
        ProprioKey.EE_POS_ACTION.value: np.concatenate(expected_actions),
    }


def _count_zarr_payload_files(zarr_path: Path, array_key: str) -> int:
    """Count physical chunk or shard payload files for one Zarr array."""
    array_path = zarr_path / "data" / array_key
    return sum(
        path.is_file() and path.name != "zarr.json" for path in array_path.rglob("*")
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

        create_replay_buffer_from_hdf5(
            schema=schema,
            image_frames_per_shard=32,
        )

        call_kwargs = mock_create_zarr.call_args
        assert call_kwargs.kwargs["schema"] is schema
        assert call_kwargs.kwargs["image_frames_per_shard"] == 32


@pytest.mark.integration
def test_toy_libero_dataset_uses_image_shards_without_changing_data(
    tmp_path: Path,
) -> None:
    hdf5_path = tmp_path / "pick_up_the_bowl_demo.hdf5"
    expected_arrays = _write_toy_libero_hdf5(hdf5_path=hdf5_path)
    metadata = DatasetMetadata(
        observations={
            Cameras.AGENTVIEW.value: RGBCameraMetadata(
                camera_key=Cameras.AGENTVIEW.value,
                dtype="uint8",
                image_width=8,
                image_height=8,
            ),
            Cameras.EYE_IN_HAND.value: RGBCameraMetadata(
                camera_key=Cameras.EYE_IN_HAND.value,
                dtype="uint8",
                image_width=8,
                image_height=8,
            ),
            ProprioKey.EE_POS.value: PositionObservationMetadata(
                raw_data_column_keys=["ee_pos"],
                dimension=3,
                dtype="float32",
                needs_normalization=True,
                frame=CoordinateSystem.ROBOT_BASE.value,
            ),
        },
        precomputed_actions={
            ProprioKey.EE_POS_ACTION.value: PositionActionMetadata(
                frame=CoordinateSystem.ROBOT_BASE.value,
                raw_data_column_keys=["actions"],
                storage_dimension=7,
                prediction_dimension=3,
                needs_normalization=True,
                dtype="float32",
                slice_start=0,
                slice_end=3,
            ),
        },
    )
    unsharded_path = tmp_path / "libero_unsharded.zarr"
    sharded_path = tmp_path / "libero_sharded.zarr"
    unsharded_schema = LiberoSchema(
        hdf5_paths=[str(hdf5_path)],
        zarr_path=str(unsharded_path),
        metadata=metadata,
    )
    sharded_schema = LiberoSchema(
        hdf5_paths=[str(hdf5_path)],
        zarr_path=str(sharded_path),
        metadata=metadata,
    )

    create_replay_buffer_from_hdf5(
        schema=unsharded_schema,
        image_frames_per_shard=None,
    )
    create_replay_buffer_from_hdf5(
        schema=sharded_schema,
        image_frames_per_shard=4,
    )

    unsharded = zarr.open_group(store=str(unsharded_path), mode="r")
    sharded = zarr.open_group(store=str(sharded_path), mode="r")
    for camera_key in (Cameras.AGENTVIEW.value, Cameras.EYE_IN_HAND.value):
        unsharded_images = unsharded["data"][camera_key]
        sharded_images = sharded["data"][camera_key]
        assert unsharded_images.chunks == (1, 8, 8, 3)
        assert unsharded_images.shards is None
        assert sharded_images.chunks == (1, 8, 8, 3)
        assert sharded_images.shards == (4, 8, 8, 3)
        assert _count_zarr_payload_files(unsharded_path, camera_key) == 6
        assert _count_zarr_payload_files(sharded_path, camera_key) == 2
        np.testing.assert_array_equal(sharded_images[:], unsharded_images[:])

    np.testing.assert_array_equal(
        sharded["data"][Cameras.AGENTVIEW.value][3:5],
        unsharded["data"][Cameras.AGENTVIEW.value][3:5],
    )
    for array_key, expected in expected_arrays.items():
        numerical_array = sharded["data"][array_key]
        assert numerical_array.shards is None
        np.testing.assert_array_equal(numerical_array[:], expected)
    np.testing.assert_array_equal(sharded["meta"]["episode_ends"][:], [3, 6])
    np.testing.assert_array_equal(
        sharded["data"][ObsKey.LANGUAGE.value][:],
        [["pick up the bowl"]] * 6,
    )
