"""Dataset schema for LeRobot dataset V3.0.

LeRobot is a HuggingFace robotics dataset format using:
- Parquet files for tabular data (observation.state, action, timestamps)
- MP4 videos OR embedded PNG bytes for camera observations
- JSON metadata files (info.json, stats.json, tasks.jsonl/tasks.parquet)

Reference: https://github.com/huggingface/lerobot
"""

import io
import json
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from PIL import Image

from versatil.data.constants import ObsKey, LeRobotPathsV30
from versatil.data.metadata import CameraMetadata
from versatil.data.raw.schemas.base import DatasetSchema
from versatil.data.raw.zarr_meta import DatasetMetadata

def decode_video_frames(
    video_path: Path,
    timestamps: list[float],
    tolerance_s: float = 0.01,
) -> list:
    """
    Loads frames from a video at given timestamps using OpenCV.

    Args:
        video_path (str or Path): path to video file
        timestamps (list[float]): timestamps in seconds
        tolerance_s (float): maximum allowed deviation in seconds

    Returns:
        list of PIL.Image
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError(f"Failed to read FPS from video: {video_path}")

    loaded_frames = []

    for timestamp in timestamps:
        frame_idx = round(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame at timestamp={timestamp:.4f}s from video: {video_path}")

        actual_timestamp_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        if abs(actual_timestamp_s - timestamp) > tolerance_s:
            raise RuntimeError(
                f"Timestamp tolerance exceeded: requested={timestamp:.4f}, got={actual_timestamp_s:.4f}, video={video_path}"
            )
        
        # Convert BGR (OpenCV) -> RGB (PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        loaded_frames.append(pil_image)
    cap.release()
    
    return loaded_frames


class LeRobotDatasetMetadataV30:
    
    def __init__(
        self,
        dataset_path: str | Path
    ):
        self.dataset_path = Path(dataset_path)
        self.info = self._load_info()
        self.tasks = pd.read_parquet(self.dataset_path / LeRobotPathsV30.DEFAULT_TASKS_PATH)
        self.episodes = self._load_episodes()

    def _load_info(self):
        info_path = self.dataset_path / LeRobotPathsV30.INFO_PATH
        with open(info_path) as f:
            return json.load(f)

    def _load_episodes(self) ->  pd.DataFrame:
        episodes_dir = self.dataset_path / LeRobotPathsV30.EPISODES_DIR
        paths = sorted(episodes_dir.glob("*/*.parquet"))
        dfs = [pd.read_parquet(str(path)) for path in paths]
        df = pd.concat(dfs, ignore_index=True)
        df = df.loc[:, ~df.columns.str.startswith("stats/")]
        
        return df

    def get_version(self) -> str:
        return self.info["codebase_version"]

    def get_data_file_path(self, episode_index: int) -> Path:
        episode = self.get_episode_meta(episode_index)
        chunk_idx = episode["data/chunk_index"].item()
        file_idx = episode["data/file_index"].item()
        fpath = self.info["data_path"].format(chunk_index=chunk_idx, file_index=file_idx)
        return self.dataset_path / Path(fpath)

    def get_image_file_path(self, episode_index: int, image_key: str, frame_index: int) -> Path:
        fpath = LeRobotPathsV30.DEFAULT_IMAGE_PATH.format(
            image_key=image_key, episode_index=episode_index, frame_index=frame_index
        )
        return self.dataset_path / Path(fpath)

    def get_video_file_path(self, episode_index: int, video_key: str) -> Path:
        episode = self.get_episode_meta(episode_index)
        chunk_idx = episode[f"videos/{video_key}/chunk_index"].item()
        file_idx = episode[f"videos/{video_key}/file_index"].item()
        fpath = self.info["video_path"].format(video_key=video_key, chunk_index=chunk_idx, file_index=file_idx)
        return self.dataset_path / Path(fpath)

    def get_episode_meta(self, episode_index: int) -> pd.DataFrame:
        return self.episodes[self.episodes['episode_index'] == episode_index]

    def get_features(self) -> dict:
        return self.info['features']

    def get_image_keys(self) -> list[str]:
        return [key for key, ft in self.get_features().items() if ft["dtype"] == "image"]

    def get_video_keys(self) -> list[str]:
        return [key for key, ft in self.get_features().items() if ft["dtype"] == "video"]

    def get_total_episodes(self) -> int:
        return self.info["total_episodes"]

class LeRobotDatasetSchemaV30(DatasetSchema):

    def __init__(self, dataset_path: str, zarr_path: str, metadata: DatasetMetadata):
        self.dataset_path = Path(dataset_path)
        super().__init__(zarr_path=zarr_path, metadata=metadata)

        self.lerobot_metadata = LeRobotDatasetMetadataV30(dataset_path=dataset_path)

    def _get_frames_from_videos(self, query_timestamps: dict[str, list[float]], episode_index: int) -> dict[str, list]:
        episode = self.lerobot_metadata.get_episode_meta(episode_index)
        videos = {}
        for video_key, query_timestamp in query_timestamps.items():
            from_timestamp = episode[f"videos/{video_key}/from_timestamp"].item()
            shifted_query_timestamp = [from_timestamp + timestamp for timestamp in query_timestamp]
            video_path = self.lerobot_metadata.get_video_file_path(episode_index, video_key)
            frames = decode_video_frames(video_path, shifted_query_timestamp)
            videos[video_key] = frames
            
        return videos

    def _get_images_from_filesystem(self, episode_index: int, image_key: str, frame_indexes: list[int]) -> list:
        frames = []
        for frame_index in frame_indexes:
            image_path = self.lerobot_metadata.get_image_file_path(episode_index, image_key, frame_index)
            assert image_path.exists(), f"Image was not found: {image_path}"
            frames.append(Image.open(image_path))
        return frames

    def get_episode_parquet(self, episode_id: int) -> pd.DataFrame:
        episode_df = pd.read_parquet(self.lerobot_metadata.get_data_file_path(episode_id))
        episode_df = episode_df[episode_df['episode_index'] == episode_id].reset_index(drop=True)
        return episode_df

    def get_episode_videos_frames(self, episode_id: int, preloaded_episode_df: pd.DataFrame = None) -> dict[str, list]:
        episode_df = preloaded_episode_df if preloaded_episode_df is not None else self.get_episode_parquet(episode_id)
        
        video_keys = self.lerobot_metadata.get_video_keys()        
        
        frames = {}
        if video_keys:
            timestamps = episode_df['timestamp'].tolist()
            query_timestamps = {
                k: timestamps for k in video_keys
            }
            frames = self._get_frames_from_videos(query_timestamps, episode_id)
            
        return frames

    def get_episode_images(self, episode_id: int, preloaded_episode_df: pd.DataFrame = None) -> dict[str, list]:
        episode_df = preloaded_episode_df if preloaded_episode_df is not None else self.get_episode_parquet(episode_id)
        
        image_keys = self.lerobot_metadata.get_image_keys()
        images = {}
        
        if image_keys:
            frame_indexes = episode_df['frame_index'].tolist()
            episode_cols = episode_df.columns.tolist()
            
            for image_key in image_keys:   
                frames = []
                
                if image_key in episode_cols:
                    encoded_images = episode_df[image_key].tolist()
                    for encoded_image in encoded_images:
                        if 'bytes' in encoded_image:
                            frames.append(Image.open(io.BytesIO(encoded_image['bytes'])))
                        else:
                            path = self.dataset_path / Path(encoded_image['path'])
                            frames.append(Image.open(path))
                else:
                    frames = self._get_images_from_filesystem(episode_id, image_key, frame_indexes)
                
                images[image_key] = frames
        
        return images

    def get_episode_language_instruction(self, episode_id: int, preloaded_episode_df: pd.DataFrame = None) -> str:
        episode_df = preloaded_episode_df if preloaded_episode_df is not None else self.get_episode_parquet(episode_id)
        task_index = episode_df.iloc[0]['task_index'].item()
        return self.lerobot_metadata.tasks.iloc[task_index].name
        
    
    def extract_episode(self, episode_id: int, resizer: A.Resize | A.NoOp, depth_resizer: A.Resize | A.NoOp,) -> dict:
        
        episode_df = self.get_episode_parquet(episode_id)
        language_instruction = self.get_episode_language_instruction(episode_id, episode_df)       
        videos = self.get_episode_videos_frames(episode_id, episode_df)
        images = self.get_episode_images(episode_id, episode_df)
        frames = videos | images

        data = {}

        # Process observations
        for zarr_key, obs in self.metadata.observations.items():
            if isinstance(obs, CameraMetadata):
                camera_key = obs.camera_key
                assert camera_key in frames, f'The camera key: {camera_key} does not exist in the dataset.'
                resized_frames = [resizer(image=np.array(f))["image"] for f in frames[camera_key]]
                data[zarr_key] = np.stack(resized_frames).astype(obs.dtype)

            else:
                col_key = obs.raw_data_column_keys[0]
                assert col_key in episode_df.columns, f'The column {col_key} does not exist in the dataset.'
                
                obs_array = np.stack(episode_df[col_key].values)
                if obs.slice_start is not None and obs.slice_end is not None:
                    obs_array = obs_array[:, obs.slice_start : obs.slice_end]
                data[zarr_key] = obs_array.astype(obs.dtype)

        # Process precomputed actions
        for zarr_key, action in self.metadata.precomputed_actions.items():
            col_key = action.raw_data_column_keys[0]
            assert col_key in episode_df.columns, f'The column {col_key} does not exist in the dataset.'
            
            action_array = np.stack(episode_df[col_key].values)
            if action.slice_start is not None and action.slice_end is not None:
                action_array = action_array[:, action.slice_start : action.slice_end]
            data[zarr_key] = action_array.astype(action.dtype)

        data[ObsKey.LANGUAGE.value] = language_instruction
            
        return data