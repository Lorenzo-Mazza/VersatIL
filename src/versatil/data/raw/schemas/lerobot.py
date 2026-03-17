"""Dataset schema for LeRobot dataset V3.0.

LeRobot is a HuggingFace robotics dataset format using:
- Parquet files for tabular data (observation.state, action, timestamps)
- MP4 videos OR embedded PNG bytes for camera observations
- JSON metadata files (info.json, stats.json, tasks.jsonl/tasks.parquet)

Reference: https://github.com/huggingface/lerobot
"""

import json
from pathlib import Path
from typing import Any

import albumentations as A
import av
import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from versatil.data.constants import LeRobotPathsV30, ObsKey
from versatil.data.metadata import CameraMetadata
from versatil.data.raw.schemas.base import DatasetSchema
from versatil.data.raw.zarr_meta import DatasetMetadata


def decode_video_frames(
    video_path: Path,
    timestamps: list[float],
    tolerance_s: float = 0.01,
) -> list[np.ndarray]:
    """Load frames from a video file at specified timestamps using PyAV.

    Returns:
        List of RGB numpy arrays (uint8) corresponding to each requested timestamp.
    """
    container = av.open(str(video_path))
    stream = container.streams.video[0]

    # Ensure timestamps are sorted for efficient seeking
    sorted_indices = np.argsort(timestamps)
    sorted_timestamps = [timestamps[i] for i in sorted_indices]

    loaded_frames = [None] * len(timestamps)

    for sorted_pos, timestamp in enumerate(sorted_timestamps):
        # Seek close to requested timestamp (in stream time_base units)
        seek_ts = int(timestamp / stream.time_base)
        container.seek(seek_ts, any_frame=False, backward=True, stream=stream)

        frame_found = False

        for frame in container.decode(stream):
            actual_timestamp_s = float(frame.pts * stream.time_base)

            if actual_timestamp_s + tolerance_s < timestamp:
                continue

            if abs(actual_timestamp_s - timestamp) > tolerance_s:
                raise RuntimeError(
                    f"Timestamp tolerance exceeded: requested={timestamp:.4f}, "
                    f"got={actual_timestamp_s:.4f}, video={video_path}"
                )

            img_rgb = frame.to_ndarray(format="rgb24")
            original_index = sorted_indices[sorted_pos]
            loaded_frames[original_index] = img_rgb
            frame_found = True
            break

        if not frame_found:
            raise RuntimeError(
                f"Failed to read frame at timestamp={timestamp:.4f}s "
                f"from video: {video_path}"
            )

    container.close()

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
        self.tasks = pq.read_table(
            self.dataset_path / LeRobotPathsV30.DEFAULT_TASKS_PATH
        )
        self.episodes = self._load_episodes()

    def _load_info(self) -> dict[str, Any]:
        """Load and parse the info.json metadata file.

        Returns:
            Dictionary containing dataset configuration from info.json.
        """
        info_path = self.dataset_path / LeRobotPathsV30.INFO_PATH
        with open(info_path) as f:
            return json.load(f)

    def _load_episodes(self) -> pa.Table:
        """Load and concatenate all episode metadata parquet files.

        Episode metadata is stored in chunks under meta/episodes/.
        This method loads all chunks and concatenates them into a single Table,
        excluding statistics columns (prefixed with "stats/").

        Returns:
            PyArrow Table containing metadata for all episodes.
        """
        episodes_dir = self.dataset_path / LeRobotPathsV30.EPISODES_DIR
        paths = sorted(episodes_dir.glob("*/*.parquet"))
        tables = [pq.read_table(str(path)) for path in paths]
        table = pa.concat_tables(tables)
        # Remove statistics columns as they're not needed for data loading
        cols_to_keep = [
            col for col in table.column_names if not col.startswith("stats/")
        ]
        table = table.select(cols_to_keep)

        return table

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
        chunk_idx = episode["data/chunk_index"][0].as_py()
        file_idx = episode["data/file_index"][0].as_py()
        file_path = self.info["data_path"].format(
            chunk_index=chunk_idx, file_index=file_idx
        )
        return self.dataset_path / Path(file_path)

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
        file_path = LeRobotPathsV30.DEFAULT_IMAGE_PATH.format(
            image_key=image_key, episode_index=episode_index, frame_index=frame_index
        )
        return self.dataset_path / Path(file_path)

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
        chunk_idx = episode[f"videos/{video_key}/chunk_index"][0].as_py()
        file_idx = episode[f"videos/{video_key}/file_index"][0].as_py()
        file_path = self.info["video_path"].format(
            video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
        )
        return self.dataset_path / Path(file_path)

    def get_episode_meta(self, episode_index: int) -> pa.Table:
        """Get metadata for a specific episode.

        Args:
            episode_index: Index of the episode to retrieve.

        Returns:
            PyArrow Table row(s) containing the episode's metadata.
        """
        mask = pa.compute.equal(self.episodes["episode_index"], episode_index)
        return self.episodes.filter(mask)

    def get_features(self) -> dict[str, Any]:
        """Get the feature schema definition from info.json.

        Features describe the data columns and their types (e.g., state, action,
        image, video).

        Returns:
            Dictionary mapping feature names to their specifications.
        """
        return self.info["features"]

    def get_image_keys(self) -> list[str]:
        """Get feature keys for image-type observations.

        Returns:
            List of image key names (e.g., ["observation.images.top"]).
        """
        return [
            key for key, ft in self.get_features().items() if ft["dtype"] == "image"
        ]

    def get_video_keys(self) -> list[str]:
        """Get feature keys for video-type observations.

        Returns:
            List of video key names (e.g., ["observation.images.top"]).
        """
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
        dataset_type: String with the dataset type used by the schema (e.g. libero, metaworld, etc.)
    """

    def __init__(
        self,
        dataset_path: str,
        zarr_path: str,
        metadata: DatasetMetadata,
        dataset_type: str,
    ):
        self.dataset_path = Path(dataset_path)
        super().__init__(
            zarr_path=zarr_path, metadata=metadata, dataset_type=dataset_type
        )

        self.lerobot_metadata = LeRobotDatasetMetadataV30(dataset_path=dataset_path)

    def _get_frames_from_videos(
        self, query_timestamps: dict[str, list[float]], episode_index: int
    ) -> dict[str, list[np.ndarray]]:
        """Extract frames from video files at specified timestamps.

        LeRobot videos may have an offset (from_timestamp) that needs to be
        added to the query timestamps to get the correct frame positions.

        Args:
            query_timestamps: Dictionary mapping video keys to lists of timestamps
                (in seconds, relative to episode start).
            episode_index: Index of the episode to extract frames from.

        Returns:
            Dictionary mapping video keys to lists of cv2 images (numpy arrays).
        """
        episode = self.lerobot_metadata.get_episode_meta(episode_index)
        videos = {}
        for video_key, query_timestamp in query_timestamps.items():
            # Videos may start at an offset within the file
            from_timestamp = episode[f"videos/{video_key}/from_timestamp"][0].as_py()
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
    ) -> list[np.ndarray]:
        """Load images from individual files on the filesystem.

        Used when images are stored as separate files rather than embedded
        in parquet or encoded as video.

        Args:
            episode_index: Index of the episode.
            image_key: Camera/image stream identifier.
            frame_indexes: List of frame indices to load.

        Returns:
            List of cv2 images (numpy arrays) in frame order.
        """
        frames = []
        for frame_index in frame_indexes:
            image_path = self.lerobot_metadata.get_image_file_path(
                episode_index, image_key, frame_index
            )
            if not image_path.exists():
                raise ValueError(f"Image was not found: {image_path}")

            img = cv2.imread(str(image_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
        return frames

    def get_episode_parquet(self, episode_id: int) -> pa.Table:
        table = pq.read_table(self.lerobot_metadata.get_data_file_path(episode_id))
        mask = pa.compute.equal(table["episode_index"], episode_id)
        return table.filter(mask)

    def get_episode_videos_frames(
        self, episode_id: int, preloaded_episode_table: pa.Table = None
    ) -> dict[str, list[np.ndarray]]:
        """Extract all video frames for an episode.

        Args:
            episode_id: Index of the episode.
            preloaded_episode_table: Optional pre-loaded episode Table to avoid
                redundant parquet reads.

        Returns:
            Dictionary mapping video keys to lists of cv2 images (numpy arrays), one per timestep.
            Returns empty dict if dataset has no video features.
        """
        episode_table = (
            preloaded_episode_table
            if preloaded_episode_table is not None
            else self.get_episode_parquet(episode_id)
        )

        video_keys = self.lerobot_metadata.get_video_keys()

        frames = {}
        if video_keys:
            # Use episode timestamps to query video frames
            timestamps = episode_table["timestamp"].to_pylist()
            query_timestamps = dict.fromkeys(video_keys, timestamps)
            frames = self._get_frames_from_videos(query_timestamps, episode_id)

        return frames

    def get_episode_images(
        self, episode_id: int, preloaded_episode_table: pa.Table = None
    ) -> dict[str, list[np.ndarray]]:
        """Extract all images for an episode.

        Images in LeRobot can be stored in three ways:
        1. Embedded as bytes in the parquet file (encoded_image['bytes'])
        2. As paths in the parquet file pointing to external files (encoded_image['path'])
        3. As separate files on disk (loaded via _get_images_from_filesystem)

        Args:
            episode_id: Index of the episode.
            preloaded_episode_table: Optional pre-loaded episode Table to avoid
                redundant parquet reads.

        Returns:
            Dictionary mapping image keys to lists of cv2 images (numpy arrays), one per timestep.
            Returns empty dict if dataset has no image features.
        """
        episode_table = (
            preloaded_episode_table
            if preloaded_episode_table is not None
            else self.get_episode_parquet(episode_id)
        )

        image_keys = self.lerobot_metadata.get_image_keys()
        images = {}

        if image_keys:
            frame_indexes = episode_table["frame_index"].to_pylist()
            episode_cols = episode_table.column_names

            for image_key in image_keys:
                frames = []

                if image_key in episode_cols:
                    # Images are embedded in parquet (as bytes or paths)
                    encoded_images = episode_table[image_key].to_pylist()
                    for encoded_image in encoded_images:
                        if "bytes" in encoded_image:
                            img = cv2.imdecode(
                                np.frombuffer(encoded_image["bytes"], np.uint8),
                                cv2.IMREAD_COLOR,
                            )
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            frames.append(img)

                        else:
                            img_path = self.dataset_path / Path(encoded_image["path"])
                            img = cv2.imread(str(img_path))
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            frames.append(img)
                else:
                    # Images stored as separate files on disk
                    frames = self._get_images_from_filesystem(
                        episode_id, image_key, frame_indexes
                    )

                images[image_key] = frames

        return images

    def get_episode_language_instructions(
        self, episode_id: int, preloaded_episode_table: pa.Table = None
    ) -> list[list[str]]:
        episode_table = (
            preloaded_episode_table
            if preloaded_episode_table is not None
            else self.get_episode_parquet(episode_id)
        )

        # Map task indices to task names (language instructions)
        task_names = self.lerobot_metadata.tasks[1].to_pylist()
        language_instructions = [
            [task_names[i]] for i in episode_table["task_index"].to_pylist()
        ]

        return language_instructions

    def extract_episode(
        self,
        episode_id: int,
        resizer: A.Resize | A.NoOp,
        depth_resizer: A.Resize | A.NoOp,
    ) -> dict[str, np.ndarray]:
        """Extract and convert a complete episode to zarr format.

        Args:
            episode_id: Index of the episode to extract.
            resizer: Albumentations transform for resizing RGB images.
            depth_resizer: Albumentations transform for resizing depth images
                (currently unused but kept for API consistency with base class).

        Returns:
            Dictionary mapping Zarr keys to numpy arrays ready for storage.
            Keys correspond to observations and precomputed_actions defined
            in the DatasetMetadata.
        """
        # Load all raw data for this episode
        episode_table = self.get_episode_parquet(episode_id)
        language_instructions = self.get_episode_language_instructions(
            episode_id, episode_table
        )
        videos = self.get_episode_videos_frames(episode_id, episode_table)
        images = self.get_episode_images(episode_id, episode_table)
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
                camera_key = obs.raw_camera_key

                if camera_key not in frames:
                    raise ValueError(
                        f"The camera key: {camera_key} does not exist in the dataset."
                    )

                resized_frames = [
                    resizer(image=np.array(f))["image"] for f in frames[camera_key]
                ]
                data[zarr_key] = np.stack(resized_frames).astype(obs.dtype)

            else:
                # Vector observations (proprioceptive state, etc.)
                col_key = obs.raw_data_column_keys[0]

                if col_key not in episode_table.column_names:
                    raise ValueError(
                        f"The column {col_key} does not exist in the dataset."
                    )

                obs_array = np.stack(episode_table[col_key].to_pylist())
                # Apply optional slicing to extract specific dimensions
                if obs.slice_start is not None and obs.slice_end is not None:
                    obs_array = obs_array[:, obs.slice_start : obs.slice_end]
                data[zarr_key] = obs_array.astype(obs.dtype)

        # Process precomputed actions (pre-computed in LeRobot format)
        for zarr_key, action in self.metadata.precomputed_actions.items():
            col_key = action.raw_data_column_keys[0]

            if col_key not in episode_table.column_names:
                raise ValueError(f"The column {col_key} does not exist in the dataset.")

            action_array = np.stack(episode_table[col_key].to_pylist())
            # Apply optional slicing to extract specific action dimensions
            if action.slice_start is not None and action.slice_end is not None:
                action_array = action_array[:, action.slice_start : action.slice_end]
            data[zarr_key] = action_array.astype(action.dtype)

        return data
