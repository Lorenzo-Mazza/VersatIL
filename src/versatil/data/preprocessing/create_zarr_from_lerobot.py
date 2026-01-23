"""Creates a Zarr-based replay buffer dataset from LeRobot datasets."""

import albumentations as A
import cv2
import numpy as np
import zarr
import zarr.storage
from threadpoolctl import threadpool_limits
from zarr.codecs import BloscCodec, BloscShuffle

from versatil.data.raw.schemas.lerobot import LeRobotDatasetSchemaV30


def create_replay_buffer_from_lerobot(schema: LeRobotDatasetSchemaV30) -> None:
    """Creates a Zarr-based replay buffer from a LeRobot dataset.

    Args:
        schema: LeRobotDatasetSchema instance (v2.1 or v3.0).
    """
    print(
        f"Creating Zarr dataset at {schema.zarr_path} "
        f"from LeRobot dataset at {schema.dataset_path}"
    )
    print(f"Using schema: {schema.__class__.__name__}")

    store = zarr.storage.LocalStore(schema.zarr_path)
    root = zarr.open_group(store=store, mode="w")
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")

    episode_ends = []
    cumulative_len = 0
    compressor = BloscCodec(cname="lz4", clevel=5, shuffle=BloscShuffle.noshuffle)

    cameras = schema.metadata.cameras
    if cameras:
        first_cam = next(iter(cameras.values()))
        image_width = first_cam.image_width
        image_height = first_cam.image_height
        resizer = A.Resize(height=image_height, width=image_width)
        depth_resizer = A.Resize(
            height=image_height, width=image_width, interpolation=cv2.INTER_NEAREST
        )
    else:
        resizer = A.NoOp()
        depth_resizer = A.NoOp()

    _create_zarr_arrays(data_group, schema, compressor)

    total_episodes = schema.lerobot_metadata.get_total_episodes()
    print(f"Processing {total_episodes} episodes...")

    with threadpool_limits(1):
        for i, episode_id in enumerate(range(total_episodes)):
            if i % 50 == 0:
                print(f"  Processing episode {i+1}/{total_episodes}...")
            episode_data = schema.extract_episode(episode_id, resizer, depth_resizer)
            for key, array in episode_data.items():
                data_group[key].append(array)
            cumulative_len += len(next(iter(episode_data.values())))
            episode_ends.append(cumulative_len)

    meta_group.create_array(
        "episode_ends",
        data=np.array(episode_ends),
        chunks=(len(episode_ends),),
        compressors=None,
    )

    print(
        f"Created Zarr dataset with {len(episode_ends)} episodes, "
        f"{cumulative_len} total steps."
    )


def _create_zarr_arrays(
    data_group: zarr.Group,
    schema: LeRobotDatasetSchemaV30,
    compressor: BloscCodec,
) -> None:
    specs = schema.get_zarr_array_specs()
    for key, spec in specs.items():
        dtype = str if spec["dtype"] == "str" else getattr(np, spec["dtype"])
        data_group.create_array(
            key,
            shape=spec["shape"],
            chunks=spec["chunks"],
            dtype=dtype,
            compressors=[compressor] if spec["needs_compressor"] else None,
        )