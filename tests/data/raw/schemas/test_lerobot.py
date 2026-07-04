"""Tests for versatil.data.raw.schemas.lerobot module."""

import json
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq_write
import pytest

from versatil.data.constants import (
    Cameras,
    CoordinateSystem,
    LeRobotPathsV30,
    ObsKey,
    RawCameraKey,
)
from versatil.data.metadata import (
    CameraMetadata,
    ObservationMetadata,
    PositionObservationMetadata,
    PrecomputedActionMetadata,
)
from versatil.data.raw.schemas.lerobot import (
    LeRobotDatasetMetadataV30,
    LeRobotDatasetSchemaV30,
    decode_video_frames,
)
from versatil.data.raw.zarr_meta import DatasetMetadata


@pytest.fixture
def encode_png() -> Callable[[np.ndarray], bytes]:
    """Factory that encodes a numpy image as PNG bytes."""

    def factory(image: np.ndarray) -> bytes:
        _, encoded = cv2.imencode(".png", image)
        return encoded.tobytes()

    return factory


class TestDecodeVideoFrames:
    def test_returns_frames_in_original_order(self):
        timestamps = [0.2, 0.0, 0.1]
        mock_frames = []
        for i in range(3):
            frame = MagicMock()
            frame.to_ndarray.return_value = np.zeros((64, 64, 3), dtype=np.uint8) + i
            mock_frames.append(frame)

        mock_stream = MagicMock()
        mock_stream.time_base = 1.0

        mock_container = MagicMock()
        mock_container.streams.video = [mock_stream]

        frame_pts = [0.0, 0.1, 0.2]
        call_count = [0]

        def mock_decode(stream):
            idx = call_count[0]
            call_count[0] += 1
            frame = MagicMock()
            frame.pts = frame_pts[idx]
            frame.to_ndarray.return_value = np.full((64, 64, 3), idx, dtype=np.uint8)
            return [frame]

        mock_container.decode = mock_decode

        with patch(
            "versatil.data.raw.schemas.lerobot.av.open", return_value=mock_container
        ):
            result = decode_video_frames(Path("/video.mp4"), timestamps)

        # Original order: [0.2, 0.0, 0.1] -> sorted [0.0, 0.1, 0.2]
        # sorted_indices = [1, 2, 0]
        # Frame at sorted_pos=0 (ts=0.0) -> original_index=1
        # Frame at sorted_pos=1 (ts=0.1) -> original_index=2
        # Frame at sorted_pos=2 (ts=0.2) -> original_index=0
        assert result[1][0, 0, 0] == 0  # ts=0.0 -> value=0
        assert result[2][0, 0, 0] == 1  # ts=0.1 -> value=1
        assert result[0][0, 0, 0] == 2  # ts=0.2 -> value=2

    def test_frame_before_timestamp_is_skipped(self):
        mock_stream = MagicMock()
        mock_stream.time_base = 1.0

        early_frame = MagicMock()
        early_frame.pts = 0.0
        correct_frame = MagicMock()
        correct_frame.pts = 0.5
        correct_frame.to_ndarray.return_value = np.zeros((64, 64, 3), dtype=np.uint8)

        mock_container = MagicMock()
        mock_container.streams.video = [mock_stream]
        mock_container.decode.return_value = [early_frame, correct_frame]

        with patch(
            "versatil.data.raw.schemas.lerobot.av.open", return_value=mock_container
        ):
            result = decode_video_frames(Path("/video.mp4"), [0.5], tolerance_s=0.01)

        assert len(result) == 1
        assert result[0] is not None

    def test_tolerance_exceeded_raises_runtime_error(self):
        mock_stream = MagicMock()
        mock_stream.time_base = 1.0

        frame = MagicMock()
        frame.pts = 1.0

        mock_container = MagicMock()
        mock_container.streams.video = [mock_stream]
        mock_container.decode.return_value = [frame]

        with (
            patch(
                "versatil.data.raw.schemas.lerobot.av.open", return_value=mock_container
            ),
            pytest.raises(RuntimeError, match="Timestamp tolerance exceeded"),
        ):
            decode_video_frames(Path("/video.mp4"), [0.5], tolerance_s=0.01)

    def test_frame_not_found_raises_runtime_error(self):
        mock_stream = MagicMock()
        mock_stream.time_base = 1.0

        mock_container = MagicMock()
        mock_container.streams.video = [mock_stream]
        mock_container.decode.return_value = []

        with (
            patch(
                "versatil.data.raw.schemas.lerobot.av.open", return_value=mock_container
            ),
            pytest.raises(RuntimeError, match="Failed to read frame"),
        ):
            decode_video_frames(Path("/video.mp4"), [0.5])

    def test_single_frame_at_exact_timestamp(self):
        mock_stream = MagicMock()
        mock_stream.time_base = 1.0

        frame = MagicMock()
        frame.pts = 0.5
        frame.to_ndarray.return_value = np.ones((32, 32, 3), dtype=np.uint8)

        mock_container = MagicMock()
        mock_container.streams.video = [mock_stream]
        mock_container.decode.return_value = [frame]

        with patch(
            "versatil.data.raw.schemas.lerobot.av.open", return_value=mock_container
        ):
            result = decode_video_frames(Path("/video.mp4"), [0.5])

        assert len(result) == 1
        np.testing.assert_array_equal(result[0], np.ones((32, 32, 3), dtype=np.uint8))

    def test_container_close_called(self):
        mock_stream = MagicMock()
        mock_stream.time_base = 1.0

        frame = MagicMock()
        frame.pts = 0.0
        frame.to_ndarray.return_value = np.zeros((32, 32, 3), dtype=np.uint8)

        mock_container = MagicMock()
        mock_container.streams.video = [mock_stream]
        mock_container.decode.return_value = [frame]

        with patch(
            "versatil.data.raw.schemas.lerobot.av.open", return_value=mock_container
        ):
            decode_video_frames(Path("/video.mp4"), [0.0])

        mock_container.close.assert_called_once()


class TestLeRobotDatasetMetadataV30Init:
    def test_stores_dataset_path_as_path(self, tmp_path: Path):
        info = {"codebase_version": "v3.0", "features": {}}
        tasks_table = pa.table({"task_index": [0], "task": ["pick"]})

        info_path = tmp_path / LeRobotPathsV30.INFO_PATH
        info_path.parent.mkdir(parents=True, exist_ok=True)
        with open(info_path, "w") as f:
            json.dump(info, f)

        tasks_path = tmp_path / LeRobotPathsV30.DEFAULT_TASKS_PATH
        tasks_path.parent.mkdir(parents=True, exist_ok=True)
        pq_write.write_table(tasks_table, str(tasks_path))

        episodes_dir = tmp_path / LeRobotPathsV30.EPISODES_DIR
        episodes_dir.mkdir(parents=True, exist_ok=True)
        chunk_dir = episodes_dir / "chunk-000"
        chunk_dir.mkdir()
        ep_table = pa.table({"episode_index": [0]})
        pq_write.write_table(ep_table, str(chunk_dir / "file-000.parquet"))

        meta = LeRobotDatasetMetadataV30(dataset_path=str(tmp_path))

        assert meta.dataset_path == tmp_path
        assert isinstance(meta.dataset_path, Path)


class TestLeRobotDatasetMetadataV30Methods:
    @pytest.fixture
    def lerobot_metadata(self, tmp_path: Path) -> LeRobotDatasetMetadataV30:
        """Create a LeRobotDatasetMetadataV30 with minimal filesystem structure."""
        info = {
            "codebase_version": "v3.0",
            "total_episodes": 2,
            "features": {
                RawCameraKey.FRONT.value: {"dtype": "video", "shape": [128, 128, 3]},
                "observation.images.side": {"dtype": "image", "shape": [64, 64, 3]},
                "observation.state": {"dtype": "float32", "shape": [6]},
                "action": {"dtype": "float32", "shape": [7]},
            },
            "data_path": "data/chunk-{chunk_index:03d}/episode_{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        }
        info_path = tmp_path / LeRobotPathsV30.INFO_PATH
        info_path.parent.mkdir(parents=True, exist_ok=True)
        with open(info_path, "w") as f:
            json.dump(info, f)

        tasks_table = pa.table(
            {"task_index": [0, 1], "task": ["pick bowl", "place cup"]}
        )
        tasks_path = tmp_path / LeRobotPathsV30.DEFAULT_TASKS_PATH
        tasks_path.parent.mkdir(parents=True, exist_ok=True)
        pq_write.write_table(tasks_table, str(tasks_path))

        episodes_dir = tmp_path / LeRobotPathsV30.EPISODES_DIR
        chunk_dir = episodes_dir / "chunk-000"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        ep_table = pa.table(
            {
                "episode_index": [0, 1],
                "data/chunk_index": [0, 0],
                "data/file_index": [0, 1],
                "videos/observation.images.front/chunk_index": [0, 0],
                "videos/observation.images.front/file_index": [0, 1],
                "videos/observation.images.front/from_timestamp": [0.0, 0.0],
                "stats/mean": [0.5, 0.6],
            }
        )
        pq_write.write_table(ep_table, str(chunk_dir / "file-000.parquet"))

        return LeRobotDatasetMetadataV30(dataset_path=str(tmp_path))

    def test_get_version(self, lerobot_metadata: LeRobotDatasetMetadataV30):
        assert lerobot_metadata.get_version() == "v3.0"

    def test_get_total_episodes(self, lerobot_metadata: LeRobotDatasetMetadataV30):
        assert lerobot_metadata.get_total_episodes() == 2

    def test_get_features(self, lerobot_metadata: LeRobotDatasetMetadataV30):
        features = lerobot_metadata.get_features()

        assert RawCameraKey.FRONT.value in features
        assert "action" in features

    def test_get_video_keys_filters_by_dtype_video(
        self, lerobot_metadata: LeRobotDatasetMetadataV30
    ):
        video_keys = lerobot_metadata.get_video_keys()

        assert video_keys == [RawCameraKey.FRONT.value]

    def test_get_image_keys_filters_by_dtype_image(
        self, lerobot_metadata: LeRobotDatasetMetadataV30
    ):
        image_keys = lerobot_metadata.get_image_keys()

        assert image_keys == ["observation.images.side"]

    def test_get_episode_meta_filters_by_index(
        self, lerobot_metadata: LeRobotDatasetMetadataV30
    ):
        episode = lerobot_metadata.get_episode_meta(0)

        assert episode.num_rows == 1
        assert episode["episode_index"][0].as_py() == 0

    def test_stats_columns_filtered_from_episodes(
        self, lerobot_metadata: LeRobotDatasetMetadataV30
    ):
        assert "stats/mean" not in lerobot_metadata.episodes.column_names

    def test_get_data_file_path(
        self, lerobot_metadata: LeRobotDatasetMetadataV30, tmp_path: Path
    ):
        path = lerobot_metadata.get_data_file_path(0)

        assert path == tmp_path / "data/chunk-000/episode_000.parquet"

    def test_get_video_file_path(
        self, lerobot_metadata: LeRobotDatasetMetadataV30, tmp_path: Path
    ):
        path = lerobot_metadata.get_video_file_path(0, RawCameraKey.FRONT.value)

        assert (
            path == tmp_path / "videos/observation.images.front/chunk-000/file-000.mp4"
        )

    def test_get_image_file_path(
        self, lerobot_metadata: LeRobotDatasetMetadataV30, tmp_path: Path
    ):
        path = lerobot_metadata.get_image_file_path(
            episode_index=0, image_key="observation.images.side", frame_index=5
        )

        assert (
            path
            == tmp_path
            / "images/observation.images.side/episode-000000/frame-000005.png"
        )


class TestLeRobotDatasetSchemaV30Init:
    def test_stores_dataset_path_as_path(
        self,
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        metadata = dataset_metadata_factory(observations={}, precomputed_actions={})

        with patch.object(LeRobotDatasetMetadataV30, "__init__", return_value=None):
            schema = LeRobotDatasetSchemaV30(
                dataset_path="/data/lerobot_ds",
                zarr_path="/tmp/test.zarr",
                metadata=metadata,
                dataset_type="metaworld",
            )

        assert schema.dataset_path == Path("/data/lerobot_ds")

    def test_creates_lerobot_metadata(
        self,
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        metadata = dataset_metadata_factory(observations={}, precomputed_actions={})

        with patch.object(
            LeRobotDatasetMetadataV30, "__init__", return_value=None
        ) as mock_init:
            LeRobotDatasetSchemaV30(
                dataset_path="/data/lerobot_ds",
                zarr_path="/tmp/test.zarr",
                metadata=metadata,
                dataset_type="metaworld",
            )

        mock_init.assert_called_once_with(dataset_path="/data/lerobot_ds")

    def test_inherits_base_fields(
        self,
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        metadata = dataset_metadata_factory(observations={}, precomputed_actions={})

        with patch.object(LeRobotDatasetMetadataV30, "__init__", return_value=None):
            schema = LeRobotDatasetSchemaV30(
                dataset_path="/data/lerobot_ds",
                zarr_path="/tmp/test.zarr",
                metadata=metadata,
                dataset_type="metaworld",
            )

        assert schema.zarr_path == "/tmp/test.zarr"
        assert schema.metadata is metadata
        assert schema.dataset_type == "metaworld"


class TestLeRobotDatasetSchemaV30ExtractEpisode:
    @pytest.fixture
    def lerobot_schema(
        self,
        camera_metadata_factory: Callable[..., CameraMetadata],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        precomputed_action_metadata_factory: Callable[..., PrecomputedActionMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ) -> LeRobotDatasetSchemaV30:
        """Create a LeRobotDatasetSchemaV30 with mocked LeRobotDatasetMetadataV30."""
        observations = {
            Cameras.AGENTVIEW.value: camera_metadata_factory(
                camera_key=RawCameraKey.FRONT.value,
                image_height=128,
                image_width=128,
            ),
            "state": position_observation_metadata_factory(
                dimension=6,
                frame=CoordinateSystem.ROBOT_BASE.value,
                raw_data_column_keys=["observation.state"],
            ),
        }
        actions = {
            "action": precomputed_action_metadata_factory(
                raw_data_column_keys=["action"],
                storage_dimension=7,
                prediction_dimension=7,
                slice_start=None,
                slice_end=None,
            ),
        }
        metadata = dataset_metadata_factory(
            observations=observations, precomputed_actions=actions
        )

        with patch.object(LeRobotDatasetMetadataV30, "__init__", return_value=None):
            schema = LeRobotDatasetSchemaV30(
                dataset_path="/data/lerobot_ds",
                zarr_path="/tmp/test.zarr",
                metadata=metadata,
                dataset_type="metaworld",
            )

        return schema

    def test_processes_camera_observation(
        self,
        rng: np.random.Generator,
        lerobot_schema: LeRobotDatasetSchemaV30,
        noop_resizer,
    ):
        mock_frames = [
            rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8) for _ in range(3)
        ]
        state_data = rng.standard_normal((3, 6)).tolist()
        action_data = rng.standard_normal((3, 7)).tolist()

        episode_table = pa.table(
            {
                "observation.state": state_data,
                "action": action_data,
                "timestamp": [0.0, 0.1, 0.2],
                "task_index": [0, 0, 0],
                "frame_index": [0, 1, 2],
                "episode_index": [0, 0, 0],
            }
        )

        with (
            patch.object(
                lerobot_schema, "get_episode_parquet", return_value=episode_table
            ),
            patch.object(
                lerobot_schema,
                "get_episode_language_instructions",
                return_value=[["pick"]] * 3,
            ),
            patch.object(
                lerobot_schema,
                "get_episode_videos_frames",
                return_value={RawCameraKey.FRONT.value: mock_frames},
            ),
            patch.object(lerobot_schema, "get_episode_images", return_value={}),
        ):
            data = lerobot_schema.extract_episode(
                episode_id=0,
            )

        assert Cameras.AGENTVIEW.value in data
        assert data[Cameras.AGENTVIEW.value].shape == (3, 128, 128, 3)
        assert data[Cameras.AGENTVIEW.value].dtype == np.uint8
        np.testing.assert_array_equal(data[Cameras.AGENTVIEW.value][0], mock_frames[0])

    def test_processes_language_observation(
        self,
        rng: np.random.Generator,
        camera_metadata_factory: Callable[..., CameraMetadata],
        precomputed_action_metadata_factory: Callable[..., PrecomputedActionMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
        noop_resizer,
    ):
        language_observation = ObservationMetadata(
            raw_data_column_keys=["language"],
            dimension=1,
            dtype="str",
            is_numerical=False,
            needs_normalization=False,
        )
        observations = {
            ObsKey.LANGUAGE.value: language_observation,
            Cameras.AGENTVIEW.value: camera_metadata_factory(
                camera_key=RawCameraKey.FRONT.value,
                image_height=64,
                image_width=64,
            ),
        }
        actions = {
            "action": precomputed_action_metadata_factory(
                raw_data_column_keys=["action"],
                storage_dimension=7,
                prediction_dimension=7,
            ),
        }
        metadata = dataset_metadata_factory(
            observations=observations, precomputed_actions=actions
        )

        with patch.object(LeRobotDatasetMetadataV30, "__init__", return_value=None):
            schema = LeRobotDatasetSchemaV30(
                dataset_path="/data/ds",
                zarr_path="/tmp/test.zarr",
                metadata=metadata,
                dataset_type="test",
            )

        mock_frames = [np.zeros((64, 64, 3), dtype=np.uint8)] * 2
        action_data = rng.standard_normal((2, 7)).tolist()
        episode_table = pa.table(
            {
                "action": action_data,
                "timestamp": [0.0, 0.1],
                "task_index": [0, 0],
                "frame_index": [0, 1],
                "episode_index": [0, 0],
            }
        )
        language_instructions = [["pick bowl"], ["pick bowl"]]

        with (
            patch.object(schema, "get_episode_parquet", return_value=episode_table),
            patch.object(
                schema,
                "get_episode_language_instructions",
                return_value=language_instructions,
            ),
            patch.object(
                schema,
                "get_episode_videos_frames",
                return_value={RawCameraKey.FRONT.value: mock_frames},
            ),
            patch.object(schema, "get_episode_images", return_value={}),
        ):
            data = schema.extract_episode(
                episode_id=0,
            )

        assert ObsKey.LANGUAGE.value in data
        assert data[ObsKey.LANGUAGE.value].tolist() == [["pick bowl"], ["pick bowl"]]

    def test_missing_camera_key_raises(
        self,
        rng: np.random.Generator,
        lerobot_schema: LeRobotDatasetSchemaV30,
        noop_resizer,
    ):
        state_data = rng.standard_normal((2, 6)).tolist()
        action_data = rng.standard_normal((2, 7)).tolist()
        episode_table = pa.table(
            {
                "observation.state": state_data,
                "action": action_data,
                "timestamp": [0.0, 0.1],
                "task_index": [0, 0],
                "frame_index": [0, 1],
                "episode_index": [0, 0],
            }
        )

        with (
            patch.object(
                lerobot_schema, "get_episode_parquet", return_value=episode_table
            ),
            patch.object(
                lerobot_schema,
                "get_episode_language_instructions",
                return_value=[["pick"]] * 2,
            ),
            patch.object(lerobot_schema, "get_episode_videos_frames", return_value={}),
            patch.object(lerobot_schema, "get_episode_images", return_value={}),
            pytest.raises(ValueError, match="does not exist"),
        ):
            lerobot_schema.extract_episode(
                episode_id=0,
            )

    def test_processes_vector_observation_without_slicing(
        self,
        rng: np.random.Generator,
        lerobot_schema: LeRobotDatasetSchemaV30,
        noop_resizer,
    ):
        mock_frames = [np.zeros((128, 128, 3), dtype=np.uint8)] * 3
        state_data = rng.standard_normal((3, 6)).tolist()
        action_data = rng.standard_normal((3, 7)).tolist()
        episode_table = pa.table(
            {
                "observation.state": state_data,
                "action": action_data,
                "timestamp": [0.0, 0.1, 0.2],
                "task_index": [0, 0, 0],
                "frame_index": [0, 1, 2],
                "episode_index": [0, 0, 0],
            }
        )

        with (
            patch.object(
                lerobot_schema, "get_episode_parquet", return_value=episode_table
            ),
            patch.object(
                lerobot_schema,
                "get_episode_language_instructions",
                return_value=[["pick"]] * 3,
            ),
            patch.object(
                lerobot_schema,
                "get_episode_videos_frames",
                return_value={RawCameraKey.FRONT.value: mock_frames},
            ),
            patch.object(lerobot_schema, "get_episode_images", return_value={}),
        ):
            data = lerobot_schema.extract_episode(
                episode_id=0,
            )

        assert "state" in data
        assert data["state"].shape == (3, 6)
        expected_state = np.stack(state_data).astype(np.float32)
        np.testing.assert_array_almost_equal(data["state"], expected_state)

    def test_processes_vector_observation_with_slicing(
        self,
        rng: np.random.Generator,
        camera_metadata_factory: Callable[..., CameraMetadata],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        precomputed_action_metadata_factory: Callable[..., PrecomputedActionMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
        noop_resizer,
    ):
        observations = {
            "state": position_observation_metadata_factory(
                dimension=3,
                frame=CoordinateSystem.ROBOT_BASE.value,
                raw_data_column_keys=["observation.state"],
                slice_start=0,
                slice_end=3,
            ),
        }
        actions = {
            "action": precomputed_action_metadata_factory(
                raw_data_column_keys=["action"],
                storage_dimension=7,
                prediction_dimension=7,
            ),
        }
        metadata = dataset_metadata_factory(
            observations=observations, precomputed_actions=actions
        )

        with patch.object(LeRobotDatasetMetadataV30, "__init__", return_value=None):
            schema = LeRobotDatasetSchemaV30(
                dataset_path="/data/ds",
                zarr_path="/tmp/test.zarr",
                metadata=metadata,
                dataset_type="test",
            )

        state_data = rng.standard_normal((3, 6)).tolist()
        action_data = rng.standard_normal((3, 7)).tolist()
        episode_table = pa.table(
            {
                "observation.state": state_data,
                "action": action_data,
                "timestamp": [0.0, 0.1, 0.2],
                "task_index": [0, 0, 0],
                "frame_index": [0, 1, 2],
                "episode_index": [0, 0, 0],
            }
        )

        with (
            patch.object(schema, "get_episode_parquet", return_value=episode_table),
            patch.object(
                schema, "get_episode_language_instructions", return_value=[["pick"]] * 3
            ),
            patch.object(schema, "get_episode_videos_frames", return_value={}),
            patch.object(schema, "get_episode_images", return_value={}),
        ):
            data = schema.extract_episode(
                episode_id=0,
            )

        assert data["state"].shape == (3, 3)

    def test_missing_vector_column_raises(
        self,
        rng: np.random.Generator,
        lerobot_schema: LeRobotDatasetSchemaV30,
        noop_resizer,
    ):
        mock_frames = [np.zeros((128, 128, 3), dtype=np.uint8)] * 2
        action_data = rng.standard_normal((2, 7)).tolist()
        episode_table = pa.table(
            {
                "action": action_data,
                "timestamp": [0.0, 0.1],
                "task_index": [0, 0],
                "frame_index": [0, 1],
                "episode_index": [0, 0],
            }
        )

        with (
            patch.object(
                lerobot_schema, "get_episode_parquet", return_value=episode_table
            ),
            patch.object(
                lerobot_schema,
                "get_episode_language_instructions",
                return_value=[["pick"]] * 2,
            ),
            patch.object(
                lerobot_schema,
                "get_episode_videos_frames",
                return_value={RawCameraKey.FRONT.value: mock_frames},
            ),
            patch.object(lerobot_schema, "get_episode_images", return_value={}),
            pytest.raises(ValueError, match="does not exist"),
        ):
            lerobot_schema.extract_episode(
                episode_id=0,
            )

    def test_processes_precomputed_actions_without_slicing(
        self,
        rng: np.random.Generator,
        lerobot_schema: LeRobotDatasetSchemaV30,
        noop_resizer,
    ):
        mock_frames = [np.zeros((128, 128, 3), dtype=np.uint8)] * 3
        state_data = rng.standard_normal((3, 6)).tolist()
        action_data = rng.standard_normal((3, 7)).tolist()
        episode_table = pa.table(
            {
                "observation.state": state_data,
                "action": action_data,
                "timestamp": [0.0, 0.1, 0.2],
                "task_index": [0, 0, 0],
                "frame_index": [0, 1, 2],
                "episode_index": [0, 0, 0],
            }
        )

        with (
            patch.object(
                lerobot_schema, "get_episode_parquet", return_value=episode_table
            ),
            patch.object(
                lerobot_schema,
                "get_episode_language_instructions",
                return_value=[["pick"]] * 3,
            ),
            patch.object(
                lerobot_schema,
                "get_episode_videos_frames",
                return_value={RawCameraKey.FRONT.value: mock_frames},
            ),
            patch.object(lerobot_schema, "get_episode_images", return_value={}),
        ):
            data = lerobot_schema.extract_episode(
                episode_id=0,
            )

        assert data["action"].shape == (3, 7)
        expected_action = np.stack(action_data).astype(np.float32)
        np.testing.assert_array_almost_equal(data["action"], expected_action)

    def test_processes_precomputed_actions_with_slicing(
        self,
        rng: np.random.Generator,
        camera_metadata_factory: Callable[..., CameraMetadata],
        precomputed_action_metadata_factory: Callable[..., PrecomputedActionMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
        noop_resizer,
    ):
        actions = {
            "action": precomputed_action_metadata_factory(
                raw_data_column_keys=["action"],
                storage_dimension=7,
                prediction_dimension=3,
                slice_start=0,
                slice_end=3,
            ),
        }
        metadata = dataset_metadata_factory(
            observations={}, precomputed_actions=actions
        )

        with patch.object(LeRobotDatasetMetadataV30, "__init__", return_value=None):
            schema = LeRobotDatasetSchemaV30(
                dataset_path="/data/ds",
                zarr_path="/tmp/test.zarr",
                metadata=metadata,
                dataset_type="test",
            )

        action_data = rng.standard_normal((3, 7)).tolist()
        episode_table = pa.table(
            {
                "action": action_data,
                "timestamp": [0.0, 0.1, 0.2],
                "task_index": [0, 0, 0],
                "frame_index": [0, 1, 2],
                "episode_index": [0, 0, 0],
            }
        )

        with (
            patch.object(schema, "get_episode_parquet", return_value=episode_table),
            patch.object(
                schema, "get_episode_language_instructions", return_value=[["pick"]] * 3
            ),
            patch.object(schema, "get_episode_videos_frames", return_value={}),
            patch.object(schema, "get_episode_images", return_value={}),
        ):
            data = schema.extract_episode(
                episode_id=0,
            )

        assert data["action"].shape == (3, 3)
        expected = np.stack(action_data)[:, 0:3].astype(np.float32)
        np.testing.assert_array_almost_equal(data["action"], expected)

    def test_missing_action_column_raises(
        self,
        rng: np.random.Generator,
        precomputed_action_metadata_factory: Callable[..., PrecomputedActionMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
        noop_resizer,
    ):
        actions = {
            "action": precomputed_action_metadata_factory(
                raw_data_column_keys=["nonexistent_action"],
                storage_dimension=7,
                prediction_dimension=7,
            ),
        }
        metadata = dataset_metadata_factory(
            observations={}, precomputed_actions=actions
        )

        with patch.object(LeRobotDatasetMetadataV30, "__init__", return_value=None):
            schema = LeRobotDatasetSchemaV30(
                dataset_path="/data/ds",
                zarr_path="/tmp/test.zarr",
                metadata=metadata,
                dataset_type="test",
            )

        episode_table = pa.table(
            {
                "timestamp": [0.0],
                "task_index": [0],
                "frame_index": [0],
                "episode_index": [0],
            }
        )

        with (
            patch.object(schema, "get_episode_parquet", return_value=episode_table),
            patch.object(
                schema, "get_episode_language_instructions", return_value=[["pick"]]
            ),
            patch.object(schema, "get_episode_videos_frames", return_value={}),
            patch.object(schema, "get_episode_images", return_value={}),
            pytest.raises(ValueError, match="does not exist"),
        ):
            schema.extract_episode(
                episode_id=0,
            )


class TestLeRobotSchemaHelperMethods:
    @pytest.fixture
    def minimal_schema(
        self,
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ) -> LeRobotDatasetSchemaV30:
        metadata = dataset_metadata_factory(observations={}, precomputed_actions={})
        with patch.object(LeRobotDatasetMetadataV30, "__init__", return_value=None):
            schema = LeRobotDatasetSchemaV30(
                dataset_path="/data/ds",
                zarr_path="/tmp/test.zarr",
                metadata=metadata,
                dataset_type="test",
            )
        schema.lerobot_metadata = MagicMock()
        return schema

    def test_get_episode_videos_frames_with_video_keys(
        self,
        minimal_schema: LeRobotDatasetSchemaV30,
    ):
        minimal_schema.lerobot_metadata.get_video_keys.return_value = ["obs.images.top"]

        episode_table = pa.table(
            {
                "timestamp": [0.0, 0.1, 0.2],
            }
        )

        mock_frames = [np.zeros((64, 64, 3), dtype=np.uint8)] * 3
        with patch.object(
            minimal_schema,
            "_get_frames_from_videos",
            return_value={"obs.images.top": mock_frames},
        ):
            result = minimal_schema.get_episode_videos_frames(
                episode_id=0, preloaded_episode_table=episode_table
            )

        assert "obs.images.top" in result
        assert len(result["obs.images.top"]) == 3

    def test_get_episode_videos_frames_without_video_keys_returns_empty(
        self,
        minimal_schema: LeRobotDatasetSchemaV30,
    ):
        minimal_schema.lerobot_metadata.get_video_keys.return_value = []

        episode_table = pa.table({"timestamp": [0.0]})

        result = minimal_schema.get_episode_videos_frames(
            episode_id=0, preloaded_episode_table=episode_table
        )

        assert result == {}

    def test_get_episode_images_embedded_bytes(
        self,
        minimal_schema: LeRobotDatasetSchemaV30,
        encode_png: Callable[[np.ndarray], bytes],
    ):
        minimal_schema.lerobot_metadata.get_image_keys.return_value = [
            "obs.images.side"
        ]

        mock_img = np.zeros((64, 64, 3), dtype=np.uint8)
        encoded_bytes = encode_png(mock_img)
        episode_table = pa.table(
            {
                "obs.images.side": [{"bytes": encoded_bytes}],
                "frame_index": [0],
            }
        )

        with patch("versatil.data.raw.schemas.lerobot.cv2") as mock_cv2:
            mock_cv2.imdecode.return_value = mock_img
            mock_cv2.cvtColor.return_value = mock_img
            mock_cv2.IMREAD_COLOR = 1
            mock_cv2.COLOR_BGR2RGB = 4

            result = minimal_schema.get_episode_images(
                episode_id=0, preloaded_episode_table=episode_table
            )

        assert "obs.images.side" in result
        assert len(result["obs.images.side"]) == 1

    def test_get_episode_images_embedded_path(
        self,
        minimal_schema: LeRobotDatasetSchemaV30,
    ):
        minimal_schema.lerobot_metadata.get_image_keys.return_value = [
            "obs.images.side"
        ]
        minimal_schema.dataset_path = Path("/data/ds")

        episode_table = pa.table(
            {
                "obs.images.side": [{"path": "images/frame_000.png"}],
                "frame_index": [0],
            }
        )

        mock_img = np.zeros((64, 64, 3), dtype=np.uint8)
        with patch("versatil.data.raw.schemas.lerobot.cv2") as mock_cv2:
            mock_cv2.imread.return_value = mock_img
            mock_cv2.cvtColor.return_value = mock_img
            mock_cv2.COLOR_BGR2RGB = 4

            result = minimal_schema.get_episode_images(
                episode_id=0, preloaded_episode_table=episode_table
            )

        assert "obs.images.side" in result
        mock_cv2.imread.assert_called_once_with(
            str(Path("/data/ds/images/frame_000.png"))
        )

    def test_get_episode_images_from_filesystem(
        self,
        minimal_schema: LeRobotDatasetSchemaV30,
    ):
        minimal_schema.lerobot_metadata.get_image_keys.return_value = [
            "obs.images.side"
        ]

        episode_table = pa.table(
            {
                "frame_index": [0, 1],
            }
        )

        mock_frames = [np.zeros((64, 64, 3), dtype=np.uint8)] * 2
        with patch.object(
            minimal_schema, "_get_images_from_filesystem", return_value=mock_frames
        ):
            result = minimal_schema.get_episode_images(
                episode_id=0, preloaded_episode_table=episode_table
            )

        assert "obs.images.side" in result
        assert len(result["obs.images.side"]) == 2

    def test_get_episode_images_without_image_keys_returns_empty(
        self,
        minimal_schema: LeRobotDatasetSchemaV30,
    ):
        minimal_schema.lerobot_metadata.get_image_keys.return_value = []

        episode_table = pa.table({"frame_index": [0]})

        result = minimal_schema.get_episode_images(
            episode_id=0, preloaded_episode_table=episode_table
        )

        assert result == {}

    def test_get_episode_language_instructions_maps_task_index(
        self,
        minimal_schema: LeRobotDatasetSchemaV30,
    ):
        task_names = ["pick bowl", "place cup"]
        minimal_schema.lerobot_metadata.tasks = pa.table(
            {
                "task_index": [0, 1],
                "task": task_names,
            }
        )

        episode_table = pa.table(
            {
                "task_index": [0, 0, 1],
            }
        )

        result = minimal_schema.get_episode_language_instructions(
            episode_id=0, preloaded_episode_table=episode_table
        )

        assert result == [["pick bowl"], ["pick bowl"], ["place cup"]]

    def test_get_episode_language_instructions_falls_back_to_get_episode_parquet(
        self,
        minimal_schema: LeRobotDatasetSchemaV30,
    ):
        task_names = ["pick bowl", "place cup"]
        minimal_schema.lerobot_metadata.tasks = pa.table(
            {
                "task_index": [0, 1],
                "task": task_names,
            }
        )

        episode_table = pa.table(
            {
                "task_index": [1, 0],
            }
        )

        with patch.object(
            minimal_schema, "get_episode_parquet", return_value=episode_table
        ) as mock_get_parquet:
            result = minimal_schema.get_episode_language_instructions(
                episode_id=0, preloaded_episode_table=None
            )

        mock_get_parquet.assert_called_once_with(0)
        assert result == [["place cup"], ["pick bowl"]]

    def test_get_episode_parquet_filters_by_episode_id(
        self,
        minimal_schema: LeRobotDatasetSchemaV30,
    ):
        full_table = pa.table(
            {
                "episode_index": [0, 0, 1, 1],
                "value": [10, 20, 30, 40],
            }
        )
        data_file_path = Path("/tmp/data.parquet")
        minimal_schema.lerobot_metadata.get_data_file_path.return_value = data_file_path

        with patch(
            "versatil.data.raw.schemas.lerobot.pq.read_table", return_value=full_table
        ):
            result = minimal_schema.get_episode_parquet(0)

        assert result.num_rows == 2
        assert result["value"].to_pylist() == [10, 20]

    def test_get_frames_from_videos_applies_timestamp_offset(
        self,
        minimal_schema: LeRobotDatasetSchemaV30,
    ):
        from_timestamp = 5.0
        episode_meta = pa.table(
            {
                "episode_index": [0],
                "videos/obs.images.top/from_timestamp": [from_timestamp],
                "videos/obs.images.top/chunk_index": [0],
                "videos/obs.images.top/file_index": [0],
            }
        )
        minimal_schema.lerobot_metadata.get_episode_meta.return_value = episode_meta
        minimal_schema.lerobot_metadata.get_video_file_path.return_value = Path(
            "/data/video.mp4"
        )

        mock_frames = [np.zeros((64, 64, 3), dtype=np.uint8)] * 2

        with patch(
            "versatil.data.raw.schemas.lerobot.decode_video_frames",
            return_value=mock_frames,
        ) as mock_decode:
            result = minimal_schema._get_frames_from_videos(
                query_timestamps={"obs.images.top": [0.0, 0.1]},
                episode_index=0,
            )

        called_timestamps = mock_decode.call_args[0][1]
        assert called_timestamps == [from_timestamp + 0.0, from_timestamp + 0.1]
        assert "obs.images.top" in result

    def test_get_images_from_filesystem_missing_image_raises(
        self,
        tmp_path: Path,
        minimal_schema: LeRobotDatasetSchemaV30,
    ):
        minimal_schema.lerobot_metadata.get_image_file_path.return_value = (
            tmp_path / "missing" / "image.png"
        )

        with pytest.raises(ValueError, match="Image was not found"):
            minimal_schema._get_images_from_filesystem(
                episode_index=0,
                image_key="obs.images.side",
                frame_indexes=[0],
            )

    def test_get_images_from_filesystem_loads_and_converts_images(
        self,
        tmp_path: Path,
        minimal_schema: LeRobotDatasetSchemaV30,
    ):
        image_path = tmp_path / "frame.png"
        image_path.touch()
        mock_bgr_image = np.zeros((64, 64, 3), dtype=np.uint8)
        mock_rgb_image = np.ones((64, 64, 3), dtype=np.uint8)

        minimal_schema.lerobot_metadata.get_image_file_path.return_value = image_path

        with patch("versatil.data.raw.schemas.lerobot.cv2") as mock_cv2:
            mock_cv2.imread.return_value = mock_bgr_image
            mock_cv2.cvtColor.return_value = mock_rgb_image
            mock_cv2.COLOR_BGR2RGB = 4

            result = minimal_schema._get_images_from_filesystem(
                episode_index=0,
                image_key="obs.images.side",
                frame_indexes=[0, 1],
            )

        assert len(result) == 2
        mock_cv2.imread.assert_called_with(str(image_path))
        np.testing.assert_array_equal(result[0], mock_rgb_image)

    def test_get_episode_videos_frames_loads_from_parquet_when_no_preloaded_table(
        self,
        minimal_schema: LeRobotDatasetSchemaV30,
    ):
        minimal_schema.lerobot_metadata.get_video_keys.return_value = []

        episode_table = pa.table({"timestamp": [0.0]})
        with patch.object(
            minimal_schema, "get_episode_parquet", return_value=episode_table
        ) as mock_parquet:
            result = minimal_schema.get_episode_videos_frames(
                episode_id=0, preloaded_episode_table=None
            )

        mock_parquet.assert_called_once_with(0)
        assert result == {}

    def test_get_episode_images_loads_from_parquet_when_no_preloaded_table(
        self,
        minimal_schema: LeRobotDatasetSchemaV30,
    ):
        minimal_schema.lerobot_metadata.get_image_keys.return_value = []

        episode_table = pa.table({"frame_index": [0]})
        with patch.object(
            minimal_schema, "get_episode_parquet", return_value=episode_table
        ) as mock_parquet:
            result = minimal_schema.get_episode_images(
                episode_id=0, preloaded_episode_table=None
            )

        mock_parquet.assert_called_once_with(0)
        assert result == {}
