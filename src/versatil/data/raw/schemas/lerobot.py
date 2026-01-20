"""Dataset schema for LeRobot datasets (v2.1 and v3.0 formats).

LeRobot is a HuggingFace robotics dataset format using:
- Parquet files for tabular data (observation.state, action, timestamps)
- MP4 videos OR embedded PNG bytes for camera observations
- JSON metadata files (info.json, stats.json, tasks.jsonl/tasks.parquet)

Reference: https://github.com/huggingface/lerobot
"""

import abc
import io
import json
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from PIL import Image

from versatil.data.constants import ObsKey
from versatil.data.metadata import CameraMetadata
from versatil.data.raw.schemas.base import DatasetSchema
from versatil.data.raw.zarr_meta import DatasetMetadata


class LeRobotDatasetSchema(DatasetSchema):
    """Abstract base class for LeRobot dataset schemas.

    LeRobot datasets use Parquet files for tabular data, MP4 videos or
    embedded PNG images for camera observations, and JSON/Parquet metadata.
    Requires explicit DatasetMetadata to specify observation/action mappings.
    """

    def __init__(
        self,
        lerobot_path: str,
        zarr_path: str,
        metadata: DatasetMetadata,
        has_video_files: bool = True,
        tasks_format: str = "jsonl",
    ):
        self.lerobot_path = Path(lerobot_path)
        self.has_video_files = has_video_files
        self.tasks_format = tasks_format
        self.info = self._load_info_json()
        self.version = self._detect_version()
        super().__init__(zarr_path=zarr_path, metadata=metadata)
        self.tasks = self._load_tasks()

    def _load_info_json(self) -> dict:
        info_path = self.lerobot_path / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"info.json not found at {info_path}")
        with open(info_path, "r") as f:
            return json.load(f)

    def _detect_version(self) -> str:
        """Detect v2.1 or v3.0 from file naming patterns.

        v2.1: chunk*/episode_*.parquet (one episode per file)
        v3.0: chunk*/file*.parquet (multiple episodes per file)
        """
        data_dir = self.lerobot_path / "data"
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found at {data_dir}")
        if list(data_dir.glob("chunk*/episode_*.parquet")):
            return "2.1"
        if list(data_dir.glob("chunk*/file*.parquet")):
            return "3.0"
        raise ValueError(
            f"Could not detect LeRobot version from {data_dir}. "
            "Expected chunk*/episode_*.parquet (v2.1) or chunk*/file*.parquet (v3.0)"
        )

    def _load_tasks(self) -> dict[int, str]:
        tasks: dict[int, str] = {}
        if self.tasks_format == "parquet":
            tasks_path = self.lerobot_path / "meta" / "tasks.parquet"
            if tasks_path.exists():
                df = pd.read_parquet(tasks_path)
                if "task_index" in df.columns:
                    for _, row in df.iterrows():
                        tasks[int(row["task_index"])] = str(row.iloc[0])
                else:
                    for idx, row in df.iterrows():
                        tasks[int(idx)] = str(row.iloc[0])
        else:
            tasks_path = self.lerobot_path / "meta" / "tasks.jsonl"
            if tasks_path.exists():
                with open(tasks_path, "r") as f:
                    for line in f:
                        if line.strip():
                            task = json.loads(line)
                            tasks[task["task_index"]] = task["task"]
        return tasks

    @staticmethod
    def _decode_png_bytes(img_data: dict | bytes) -> np.ndarray:
        if isinstance(img_data, dict):
            img_bytes = img_data.get("bytes", img_data.get("path", b""))
        else:
            img_bytes = img_data
        img = Image.open(io.BytesIO(img_bytes))
        return np.array(img)

    @abc.abstractmethod
    def get_episode_identifiers(self) -> list[Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def load_episode_parquet(self, episode_id: Any) -> pd.DataFrame:
        raise NotImplementedError

    @abc.abstractmethod
    def load_episode_video_frames(
        self,
        episode_id: Any,
        camera_key: str,
        frame_indices: list[int],
    ) -> np.ndarray:
        raise NotImplementedError


class LeRobotV21Schema(LeRobotDatasetSchema):
    """Schema for LeRobot v2.1 datasets with per-episode files.

    Structure:
        data/chunk-000/episode_000000.parquet (one episode per file)
        videos/chunk-000/<camera>/episode_000000.mp4
    """

    def get_episode_identifiers(self) -> list[int]:
        data_dir = self.lerobot_path / "data"
        episode_files = sorted(data_dir.glob("chunk*/episode_*.parquet"))
        return [int(f.stem.replace("episode_", "")) for f in episode_files]

    def _find_episode_parquet(self, episode_id: int) -> Path:
        data_dir = self.lerobot_path / "data"
        pattern = f"chunk*/episode_{episode_id:06d}.parquet"
        matches = list(data_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"Episode {episode_id} parquet not found")
        return matches[0]

    def _find_episode_video(self, episode_id: int, camera_key: str) -> Path:
        videos_dir = self.lerobot_path / "videos"
        pattern = f"chunk*/{camera_key}/episode_{episode_id:06d}.mp4"
        matches = list(videos_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"Episode {episode_id} video for {camera_key} not found")
        return matches[0]

    def load_episode_parquet(self, episode_id: int) -> pd.DataFrame:
        parquet_path = self._find_episode_parquet(episode_id)
        return pd.read_parquet(parquet_path)

    def load_episode_video_frames(
        self,
        episode_id: int,
        camera_key: str,
        frame_indices: list[int],
    ) -> np.ndarray:
        video_path = self._find_episode_video(episode_id, camera_key)
        return self._decode_video_frames(video_path, frame_indices)

    def _decode_video_frames(
        self,
        video_path: Path,
        frame_indices: list[int],
    ) -> np.ndarray:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        frames = []
        frame_set = set(frame_indices)
        current_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if current_idx in frame_set:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            current_idx += 1
            if len(frames) == len(frame_indices):
                break
        cap.release()
        idx_to_frame = {idx: f for idx, f in zip(sorted(frame_set), frames)}
        return np.stack([idx_to_frame[i] for i in frame_indices])

    def extract_episode(
        self,
        episode_id: int,
        resizer: A.Resize | A.NoOp,
        depth_resizer: A.Resize | A.NoOp,
    ) -> dict[str, np.ndarray]:
        data: dict[str, np.ndarray] = {}
        df = self.load_episode_parquet(episode_id)
        episode_length = len(df)
        for zarr_key, obs in self.metadata.observations.items():
            if isinstance(obs, CameraMetadata):
                continue
            col_key = obs.raw_data_column_keys[0]
            if col_key in df.columns:
                data[zarr_key] = np.stack(df[col_key].values).astype(obs.dtype)
        for zarr_key, action in self.metadata.precomputed_actions.items():
            col_key = action.raw_data_column_keys[0]
            if col_key in df.columns:
                data[zarr_key] = np.stack(df[col_key].values).astype(action.dtype)
        frame_indices = list(range(episode_length))
        for zarr_key, cam_meta in self.metadata.cameras.items():
            frames = self.load_episode_video_frames(
                episode_id, cam_meta.camera_key, frame_indices
            )
            resized = [resizer(image=f)["image"] for f in frames]
            data[zarr_key] = np.stack(resized).astype(cam_meta.dtype)
        if "task_index" in df.columns and self.tasks:
            task_texts = [
                [self.tasks.get(idx, "")] for idx in df["task_index"].values
            ]
            data[ObsKey.LANGUAGE.value] = np.array(task_texts)
        return data


class LeRobotV30Schema(LeRobotDatasetSchema):
    """Schema for LeRobot v3.0 datasets with multi-episode files.

    Structure:
        data/chunk_000/file_000.parquet (contains multiple episodes)
        videos/<camera>/chunk_000/file_000.mp4 (multi-episode)
        meta/episodes/chunk_000.parquet (episode metadata with bounds)
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._episode_metadata = self._load_episode_metadata()
        self._data_cache: dict[str, pd.DataFrame] = {}
        self._video_cache: dict[str, cv2.VideoCapture] = {}

    def _load_episode_metadata(self) -> pd.DataFrame:
        episodes_dir = self.lerobot_path / "meta" / "episodes"
        if not episodes_dir.exists():
            raise FileNotFoundError(f"Episodes metadata not found: {episodes_dir}")
        parquet_files = sorted(episodes_dir.glob("*.parquet"))
        return pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

    def get_episode_identifiers(self) -> list[int]:
        return list(self._episode_metadata["episode_index"].unique())

    def _get_episode_data_range(self, episode_id: int) -> tuple[int, int]:
        row = self._episode_metadata[
            self._episode_metadata["episode_index"] == episode_id
        ].iloc[0]
        if "data_index_from" in row:
            return int(row["data_index_from"]), int(row["data_index_to"])
        episode_df = self._episode_metadata[
            self._episode_metadata["episode_index"] <= episode_id
        ]
        if "length" in episode_df.columns:
            end = episode_df["length"].sum()
            start = end - row["length"]
            return int(start), int(end)
        return 0, 1

    def _get_data_file_for_index(self, global_idx: int) -> tuple[Path, int]:
        data_dir = self.lerobot_path / "data"
        chunks = sorted(data_dir.glob("chunk_*"))
        cumulative = 0
        for chunk_dir in chunks:
            files = sorted(chunk_dir.glob("*.parquet"))
            for file_path in files:
                cache_key = str(file_path)
                if cache_key not in self._data_cache:
                    self._data_cache[cache_key] = pd.read_parquet(file_path)
                file_len = len(self._data_cache[cache_key])
                if cumulative + file_len > global_idx:
                    return file_path, global_idx - cumulative
                cumulative += file_len
        raise IndexError(f"Global index {global_idx} out of range")

    def load_episode_parquet(self, episode_id: int) -> pd.DataFrame:
        start_idx, end_idx = self._get_episode_data_range(episode_id)
        all_rows = []
        for global_idx in range(start_idx, end_idx):
            file_path, local_idx = self._get_data_file_for_index(global_idx)
            df = self._data_cache[str(file_path)]
            all_rows.append(df.iloc[local_idx : local_idx + 1])
        return pd.concat(all_rows, ignore_index=True)

    def load_episode_video_frames(
        self,
        episode_id: int,
        camera_key: str,
        frame_indices: list[int],
    ) -> np.ndarray:
        start_idx, _ = self._get_episode_data_range(episode_id)
        frames = []
        for local_frame_idx in frame_indices:
            global_idx = start_idx + local_frame_idx
            frames.append(self._decode_single_frame(camera_key, global_idx))
        return np.stack(frames)

    def _decode_single_frame(self, camera_key: str, global_idx: int) -> np.ndarray:
        videos_dir = self.lerobot_path / "videos" / camera_key
        chunks = sorted(videos_dir.glob("chunk_*"))
        cumulative = 0
        for chunk_dir in chunks:
            files = sorted(chunk_dir.glob("*.mp4"))
            for video_path in files:
                cache_key = str(video_path)
                if cache_key not in self._video_cache:
                    self._video_cache[cache_key] = cv2.VideoCapture(str(video_path))
                cap = self._video_cache[cache_key]
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if cumulative + frame_count > global_idx:
                    local_frame_idx = global_idx - cumulative
                    cap.set(cv2.CAP_PROP_POS_FRAMES, local_frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        raise IOError(
                            f"Failed to read frame {local_frame_idx} from {video_path}"
                        )
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cumulative += frame_count
        raise IndexError(f"Global frame index {global_idx} out of range")

    def extract_episode(
        self,
        episode_id: int,
        resizer: A.Resize | A.NoOp,
        depth_resizer: A.Resize | A.NoOp,
    ) -> dict[str, np.ndarray]:
        data: dict[str, np.ndarray] = {}
        df = self.load_episode_parquet(episode_id)
        episode_length = len(df)
        for zarr_key, obs in self.metadata.observations.items():
            if isinstance(obs, CameraMetadata):
                continue
            col_key = obs.raw_data_column_keys[0]
            if col_key in df.columns:
                data[zarr_key] = np.stack(df[col_key].values).astype(obs.dtype)
        for zarr_key, action in self.metadata.precomputed_actions.items():
            col_key = action.raw_data_column_keys[0]
            if col_key in df.columns:
                data[zarr_key] = np.stack(df[col_key].values).astype(action.dtype)
        frame_indices = list(range(episode_length))
        for zarr_key, cam_meta in self.metadata.cameras.items():
            frames = self.load_episode_video_frames(
                episode_id, cam_meta.camera_key, frame_indices
            )
            resized = [resizer(image=f)["image"] for f in frames]
            data[zarr_key] = np.stack(resized).astype(cam_meta.dtype)
        if "task_index" in df.columns and self.tasks:
            task_texts = [
                [self.tasks.get(idx, "")] for idx in df["task_index"].values
            ]
            data[ObsKey.LANGUAGE.value] = np.array(task_texts)
        return data

    def __del__(self) -> None:
        for cap in self._video_cache.values():
            if cap.isOpened():
                cap.release()


def create_lerobot_schema(
    lerobot_path: str,
    zarr_path: str,
    metadata: DatasetMetadata,
) -> LeRobotDatasetSchema:
    """Factory function to create appropriate LeRobot schema based on version.

    Auto-detects v2.1 (per-episode) or v3.0 (multi-episode) from directory structure.

    Args:
        lerobot_path: Path to LeRobot dataset root directory.
        zarr_path: Output path for Zarr store.
        metadata: DatasetMetadata specifying observations and actions.

    Returns:
        LeRobotV21Schema or LeRobotV30Schema instance.
    """
    lerobot_path_obj = Path(lerobot_path)
    data_dir = lerobot_path_obj / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found at {data_dir}")
    if list(data_dir.glob("chunk*/episode_*.parquet")):
        return LeRobotV21Schema(
            lerobot_path=lerobot_path,
            zarr_path=zarr_path,
            metadata=metadata,
        )
    if list(data_dir.glob("chunk_*/file_*.parquet")):
        return LeRobotV30Schema(
            lerobot_path=lerobot_path,
            zarr_path=zarr_path,
            metadata=metadata,
        )
    raise ValueError(
        f"Could not detect LeRobot version from {data_dir}. "
        "Expected chunk*/episode_*.parquet (v2.1) or chunk_*/file_*.parquet (v3.0)"
    )