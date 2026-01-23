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
    """Load frames from a video file at specified timestamps using OpenCV.

    This function seeks to specific timestamps in a video file and extracts
    the corresponding frames. It validates that the actual frame timestamps
    are within a tolerance of the requested timestamps.

    Args:
        video_path: Path to the video file (typically MP4).
        timestamps: List of timestamps in seconds to extract frames at.
        tolerance_s: Maximum allowed deviation in seconds between requested
            and actual frame timestamps. Defaults to 0.01 (10ms).

    Returns:
        List of PIL Images corresponding to each requested timestamp.

    Raises:
        ValueError: If the video FPS cannot be read (invalid or corrupted video).
        RuntimeError: If a frame cannot be read at a given timestamp, or if
            the actual timestamp deviates from the requested by more than
            the tolerance.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError(f"Failed to read FPS from video: {video_path}")

    loaded_frames = []

    for timestamp in timestamps:
        # Convert timestamp to frame index and seek
        frame_idx = round(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(
                f"Failed to read frame at timestamp={timestamp:.4f}s "
                f"from video: {video_path}"
            )

        # Validate the actual timestamp is within tolerance
        actual_timestamp_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        if abs(actual_timestamp_s - timestamp) > tolerance_s:
            raise RuntimeError(
                f"Timestamp tolerance exceeded: requested={timestamp:.4f}, "
                f"got={actual_timestamp_s:.4f}, video={video_path}"
            )

        # Convert BGR (OpenCV) -> RGB (PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        loaded_frames.append(pil_image)

    cap.release()

    return loaded_frames


class LeRobotDatasetMetadataV30:
    """Metadata handler for LeRobot dataset format V3.0.

    This class parses and provides access to LeRobot dataset metadata, including:
    - info.json: Dataset configuration (version, paths, features, etc.)
    - tasks.parquet: Task descriptions indexed by task_index
    - episodes/*.parquet: Per-episode metadata (chunk/file indices, video timestamps)

    The LeRobot V3.0 format organizes data in chunks for efficient storage:
    - Data files: data/chunk-{chunk_index}/episode_{file_index}.parquet
    - Videos: videos/chunk-{chunk_index}/{video_key}_episode_{file_index}.mp4

    Attributes:
        dataset_path: Root path to the LeRobot dataset directory.
        info: Parsed info.json containing dataset configuration.
        tasks: DataFrame of task descriptions from tasks.parquet.
        episodes: Concatenated DataFrame of all episode metadata.
    """

    def __init__(self, dataset_path: str | Path):
        self.dataset_path = Path(dataset_path)
        self.info = self._load_info()
        self.tasks = pd.read_parquet(
            self.dataset_path / LeRobotPathsV30.DEFAULT_TASKS_PATH
        )
        self.episodes = self._load_episodes()

    def _load_info(self) -> dict:
        """Load and parse the info.json metadata file.

        Returns:
            Dictionary containing dataset configuration from info.json.
        """
        info_path = self.dataset_path / LeRobotPathsV30.INFO_PATH
        with open(info_path) as f:
            return json.load(f)

    def _load_episodes(self) -> pd.DataFrame:
        """Load and concatenate all episode metadata parquet files.

        Episode metadata is stored in chunks under meta/episodes/.
        This method loads all chunks and concatenates them into a single DataFrame,
        excluding statistics columns (prefixed with "stats/").

        Returns:
            DataFrame containing metadata for all episodes.
        """
        episodes_dir = self.dataset_path / LeRobotPathsV30.EPISODES_DIR
        paths = sorted(episodes_dir.glob("*/*.parquet"))
        dfs = [pd.read_parquet(str(path)) for path in paths]
        df = pd.concat(dfs, ignore_index=True)
        # Remove statistics columns as they're not needed for data loading
        df = df.loc[:, ~df.columns.str.startswith("stats/")]

        return df

    def get_version(self) -> str:
        return self.info["codebase_version"]

    def get_data_file_path(self, episode_index: int) -> Path:
        """Get the path to the parquet data file for a specific episode.

        LeRobot stores episode data in chunked parquet files. This method
        resolves the chunk and file indices from episode metadata.

        Args:
            episode_index: Index of the episode to locate.

        Returns:
            Absolute path to the episode's parquet data file.
        """
        episode = self.get_episode_meta(episode_index)
        chunk_idx = episode["data/chunk_index"].item()
        file_idx = episode["data/file_index"].item()
        fpath = self.info["data_path"].format(
            chunk_index=chunk_idx, file_index=file_idx
        )
        return self.dataset_path / Path(fpath)

    def get_image_file_path(
        self, episode_index: int, image_key: str, frame_index: int
    ) -> Path:
        """Get the path to a specific image file.

        Used when images are stored as individual files on disk rather than
        embedded in parquet or encoded as video.

        Args:
            episode_index: Index of the episode.
            image_key: Camera/image stream identifier (e.g., "observation.images.top").
            frame_index: Frame number within the episode.

        Returns:
            Absolute path to the image file.
        """
        fpath = LeRobotPathsV30.DEFAULT_IMAGE_PATH.format(
            image_key=image_key, episode_index=episode_index, frame_index=frame_index
        )
        return self.dataset_path / Path(fpath)

    def get_video_file_path(self, episode_index: int, video_key: str) -> Path:
        """Get the path to a video file for a specific episode and camera.

        Videos are stored in chunks similar to data files. This method
        resolves the chunk and file indices from episode metadata.

        Args:
            episode_index: Index of the episode.
            video_key: Camera/video stream identifier (e.g., "observation.images.top").

        Returns:
            Absolute path to the video file (typically MP4).
        """
        episode = self.get_episode_meta(episode_index)
        chunk_idx = episode[f"videos/{video_key}/chunk_index"].item()
        file_idx = episode[f"videos/{video_key}/file_index"].item()
        fpath = self.info["video_path"].format(
            video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
        )
        return self.dataset_path / Path(fpath)

    def get_episode_meta(self, episode_index: int) -> pd.DataFrame:
        """Get metadata for a specific episode.

        Args:
            episode_index: Index of the episode to retrieve.

        Returns:
            DataFrame row(s) containing the episode's metadata.
        """
        return self.episodes[self.episodes["episode_index"] == episode_index]

    def get_features(self) -> dict:
        """Get the feature schema definition from info.json.

        Features describe the data columns and their types (e.g., state, action,
        image, video).

        Returns:
            Dictionary mapping feature names to their specifications.
        """
        return self.info["features"]

    def get_image_keys(self) -> list[str]:
        return [
            key for key, ft in self.get_features().items() if ft["dtype"] == "image"
        ]

    def get_video_keys(self) -> list[str]:
        return [
            key for key, ft in self.get_features().items() if ft["dtype"] == "video"
        ]

    def get_total_episodes(self) -> int:
        return self.info["total_episodes"]

class LeRobotDatasetSchemaV30(DatasetSchema):
    """Dataset schema for converting LeRobot V3.0 datasets to Zarr format.

    This schema handles the conversion of LeRobot datasets (stored as parquet files
    with MP4 videos or PNG images) into the Zarr format. It supports:
    - Video-based observations (MP4 files decoded at specific timestamps)
    - Image-based observations (PNG files or embedded bytes in parquet)
    - State/action vectors from parquet columns
    - Language instructions from task descriptions

    The conversion flow:
    1. Load episode data from parquet files
    2. Extract frames from videos or load images
    3. Map LeRobot features to observation/action keys
    4. Apply resizing transforms to images
    5. Return data dictionary ready for Zarr storage

    Attributes:
        dataset_path: Root path to the LeRobot dataset directory.
        lerobot_metadata: Metadata handler for accessing dataset structure.
    """

    def __init__(self, dataset_path: str, zarr_path: str, metadata: DatasetMetadata):
        self.dataset_path = Path(dataset_path)
        super().__init__(zarr_path=zarr_path, metadata=metadata)

        self.lerobot_metadata = LeRobotDatasetMetadataV30(dataset_path=dataset_path)

    def _get_frames_from_videos(
        self, query_timestamps: dict[str, list[float]], episode_index: int
    ) -> dict[str, list]:
        """Extract frames from video files at specified timestamps.

        LeRobot videos may have an offset (from_timestamp) that needs to be
        added to the query timestamps to get the correct frame positions.

        Args:
            query_timestamps: Dictionary mapping video keys to lists of timestamps
                (in seconds, relative to episode start).
            episode_index: Index of the episode to extract frames from.

        Returns:
            Dictionary mapping video keys to lists of PIL Images.
        """
        episode = self.lerobot_metadata.get_episode_meta(episode_index)
        videos = {}
        for video_key, query_timestamp in query_timestamps.items():
            # Videos may start at an offset within the file
            from_timestamp = episode[f"videos/{video_key}/from_timestamp"].item()
            shifted_query_timestamp = [
                from_timestamp + timestamp for timestamp in query_timestamp
            ]
            video_path = self.lerobot_metadata.get_video_file_path(
                episode_index, video_key
            )
            frames = decode_video_frames(video_path, shifted_query_timestamp)
            videos[video_key] = frames

        return videos

    def _get_images_from_filesystem(
        self, episode_index: int, image_key: str, frame_indexes: list[int]
    ) -> list:
        """Load images from individual files on the filesystem.

        Used when images are stored as separate files rather than embedded
        in parquet or encoded as video.

        Args:
            episode_index: Index of the episode.
            image_key: Camera/image stream identifier.
            frame_indexes: List of frame indices to load.

        Returns:
            List of PIL Images in frame order.

        Raises:
            AssertionError: If an expected image file does not exist.
        """
        frames = []
        for frame_index in frame_indexes:
            image_path = self.lerobot_metadata.get_image_file_path(
                episode_index, image_key, frame_index
            )
            assert image_path.exists(), f"Image was not found: {image_path}"
            frames.append(Image.open(image_path))
        return frames

    def get_episode_parquet(self, episode_id: int) -> pd.DataFrame:
        episode_df = pd.read_parquet(
            self.lerobot_metadata.get_data_file_path(episode_id)
        )
        episode_df = episode_df[episode_df["episode_index"] == episode_id].reset_index(
            drop=True
        )
        return episode_df

    def get_episode_videos_frames(
        self, episode_id: int, preloaded_episode_df: pd.DataFrame = None
    ) -> dict[str, list]:
        """Extract all video frames for an episode.

        Args:
            episode_id: Index of the episode.
            preloaded_episode_df: Optional pre-loaded episode DataFrame to avoid
                redundant parquet reads.

        Returns:
            Dictionary mapping video keys to lists of PIL Images, one per timestep.
            Returns empty dict if dataset has no video features.
        """
        episode_df = (
            preloaded_episode_df
            if preloaded_episode_df is not None
            else self.get_episode_parquet(episode_id)
        )

        video_keys = self.lerobot_metadata.get_video_keys()

        frames = {}
        if video_keys:
            # Use episode timestamps to query video frames
            timestamps = episode_df["timestamp"].tolist()
            query_timestamps = {k: timestamps for k in video_keys}
            frames = self._get_frames_from_videos(query_timestamps, episode_id)

        return frames

    def get_episode_images(
        self, episode_id: int, preloaded_episode_df: pd.DataFrame = None
    ) -> dict[str, list]:
        """Extract all images for an episode.

        Images in LeRobot can be stored in three ways:
        1. Embedded as bytes in the parquet file (encoded_image['bytes'])
        2. As paths in the parquet file pointing to external files (encoded_image['path'])
        3. As separate files on disk (loaded via _get_images_from_filesystem)

        Args:
            episode_id: Index of the episode.
            preloaded_episode_df: Optional pre-loaded episode DataFrame to avoid
                redundant parquet reads.

        Returns:
            Dictionary mapping image keys to lists of PIL Images, one per timestep.
            Returns empty dict if dataset has no image features.
        """
        episode_df = (
            preloaded_episode_df
            if preloaded_episode_df is not None
            else self.get_episode_parquet(episode_id)
        )

        image_keys = self.lerobot_metadata.get_image_keys()
        images = {}

        if image_keys:
            frame_indexes = episode_df["frame_index"].tolist()
            episode_cols = episode_df.columns.tolist()

            for image_key in image_keys:
                frames = []

                if image_key in episode_cols:
                    # Images are embedded in parquet (as bytes or paths)
                    encoded_images = episode_df[image_key].tolist()
                    for encoded_image in encoded_images:
                        if "bytes" in encoded_image:
                            # Decode from embedded bytes
                            frames.append(
                                Image.open(io.BytesIO(encoded_image["bytes"]))
                            )
                        else:
                            # Load from path reference
                            path = self.dataset_path / Path(encoded_image["path"])
                            frames.append(Image.open(path))
                else:
                    # Images stored as separate files on disk
                    frames = self._get_images_from_filesystem(
                        episode_id, image_key, frame_indexes
                    )

                images[image_key] = frames

        return images

    def get_episode_language_instructions(
        self, episode_id: int, preloaded_episode_df: pd.DataFrame = None
    ) -> list:
        episode_df = (
            preloaded_episode_df
            if preloaded_episode_df is not None
            else self.get_episode_parquet(episode_id)
        )

        # Map task indices to task names (language instructions)
        language_instructions = [
            [self.lerobot_metadata.tasks.iloc[i].name]
            for i in episode_df["task_index"].tolist()
        ]

        return language_instructions

    def extract_episode(
        self,
        episode_id: int,
        resizer: A.Resize | A.NoOp,
        depth_resizer: A.Resize | A.NoOp,
    ) -> dict:
        """Extract and convert a complete episode to zarr format.

        This is the main conversion method that:
        1. Loads all raw data (parquet, videos, images)
        2. Maps LeRobot features to zarr observation/action keys
        3. Applies image resizing transforms
        4. Handles optional slicing of vector observations/actions

        Args:
            episode_id: Index of the episode to extract.
            resizer: Albumentations transform for resizing RGB images.
            depth_resizer: Albumentations transform for resizing depth images
                (currently unused but kept for API consistency with base class).

        Returns:
            Dictionary mapping Zarr keys to numpy arrays ready for storage.
            Keys correspond to observations and precomputed_actions defined
            in the DatasetMetadata.

        Raises:
            AssertionError: If a required camera_key or column is not found
                in the dataset.
        """
        # Load all raw data for this episode
        episode_df = self.get_episode_parquet(episode_id)
        language_instructions = self.get_episode_language_instructions(
            episode_id, episode_df
        )
        videos = self.get_episode_videos_frames(episode_id, episode_df)
        images = self.get_episode_images(episode_id, episode_df)
        # Merge video and image frames (video keys take precedence if overlap)
        frames = videos | images

        data = {}

        # Process observations based on their type
        for zarr_key, obs in self.metadata.observations.items():

            if zarr_key == ObsKey.LANGUAGE.value:
                # Language instructions are stored as string arrays
                data[zarr_key] = np.array(language_instructions)

            elif isinstance(obs, CameraMetadata):
                # Camera observations: extract frames and resize
                camera_key = obs.camera_key
                assert camera_key in frames, (
                    f"The camera key: {camera_key} does not exist in the dataset."
                )
                resized_frames = [
                    resizer(image=np.array(f))["image"] for f in frames[camera_key]
                ]
                data[zarr_key] = np.stack(resized_frames).astype(obs.dtype)

            else:
                # Vector observations (proprioceptive state, etc.)
                col_key = obs.raw_data_column_keys[0]
                assert col_key in episode_df.columns, (
                    f"The column {col_key} does not exist in the dataset."
                )

                obs_array = np.stack(episode_df[col_key].values)
                # Apply optional slicing to extract specific dimensions
                if obs.slice_start is not None and obs.slice_end is not None:
                    obs_array = obs_array[:, obs.slice_start : obs.slice_end]
                data[zarr_key] = obs_array.astype(obs.dtype)

        # Process precomputed actions (pre-computed in LeRobot format)
        for zarr_key, action in self.metadata.precomputed_actions.items():
            col_key = action.raw_data_column_keys[0]
            assert col_key in episode_df.columns, (
                f"The column {col_key} does not exist in the dataset."
            )

            action_array = np.stack(episode_df[col_key].values)
            # Apply optional slicing to extract specific action dimensions
            if action.slice_start is not None and action.slice_end is not None:
                action_array = action_array[:, action.slice_start : action.slice_end]
            data[zarr_key] = action_array.astype(action.dtype)

        return data