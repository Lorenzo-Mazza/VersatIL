"""Creates a Zarr-based replay buffer dataset from robot demonstration CSV files and associated images."""
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import zarr
import zarr.storage
from threadpoolctl import threadpool_limits
from zarr.codecs import BloscCodec, BloscShuffle

from refactoring.data.constants import (
    GRIPPER_STATE_OBS_KEY,
    PHASE_LABEL_KEY,
    PROPRIO_OBS_CAMERA_FRAME_KEY,
    PROPRIO_OBS_ROBOT_FRAME_KEY,
    Cameras, LANGUAGE_KEY,
)
from refactoring.data.schemas.base import DatasetSchema


def create_replay_buffer(
        schema: DatasetSchema,
        datasets_paths: list[str]
) -> None:
    """Creates a Zarr-based replay buffer using a Hydra-instantiated dataset schema.

    Args:
        schema: DatasetSchema instance (instantiated by Hydra)
        datasets_paths: List of paths to episode CSV files
    """

    print(f"Creating Zarr dataset at {schema.zarr_path} with {len(datasets_paths)} episodes...")
    print(f"Using dataset schema: {schema.__class__.__name__}")

    store = zarr.storage.LocalStore(schema.zarr_path)
    root = zarr.open_group(store=store, mode='w')
    data_group = root.create_group('data')
    meta_group = root.create_group('meta')

    episode_ends = []
    cumulative_len = 0
    compressor = BloscCodec(cname='lz4', clevel=5, shuffle=BloscShuffle.noshuffle)

    if schema.raw_observations.image_width is None or schema.raw_observations.image_height is None:
        # Don't resize , use albumentations no-op
        resizer = A.NoOp()
        depth_resizer = A.NoOp()
    else:
        resizer = A.Resize(height=schema.raw_observations.image_height, width=schema.raw_observations.image_width)
        depth_resizer = A.Resize(
            height=schema.raw_observations.image_height,
            width=schema.raw_observations.image_width,
            interpolation=cv2.INTER_NEAREST # For depth, use nearest neighbor to avoid artifacts
        )
    # Create empty zarr arrays based on schema
    _create_zarr_arrays(data_group=data_group, schema=schema, compressor=compressor)

    # Insert each episode into the zarr dataset in-place
    with threadpool_limits(1):
        for path in sorted(datasets_paths, key=lambda x: int(Path(x).parent.name)):
            episode = pd.read_csv(path)
            # Append observations
            _append_observations(episode=episode, data_group=data_group, schema=schema)
            # Process and append images
            _append_images(episode=episode, data_group=data_group, schema=schema, resizer=resizer, depth_resizer=depth_resizer)
            cumulative_len += len(episode)
            episode_ends.append(cumulative_len)

    # Save metadata
    meta_group.create_array(
        'episode_ends',
        data=np.array(episode_ends),
        chunks=(len(episode_ends),),
        compressors=None,
    )

    print(f"Created Zarr dataset with {len(episode_ends)} episodes.")
    return


def _create_zarr_arrays(
        data_group: zarr.Group,
        schema: DatasetSchema,
        compressor: BloscCodec,
) -> None:
    """Create zarr arrays based on schema configuration and append to `data_group` in-place."""
    specs = schema.get_zarr_array_specs()
    for key, spec in specs.items():
        dtype = str if spec['dtype'] == 'str' else getattr(np, spec['dtype'])
        data_group.create_array(
            key,
            shape=spec['shape'],
            chunks=spec['chunks'],
            dtype=dtype,
            compressors=[compressor] if spec['needs_compressor'] else None,
        )


def _append_observations(
        episode: pd.DataFrame,
        data_group: zarr.Group,
        schema: DatasetSchema,
) -> None:
    """Append observations to zarr `data_group` in-place."""
    obs = schema.raw_observations

    if obs.robot_frame_proprio_keys:
        data_group[PROPRIO_OBS_ROBOT_FRAME_KEY].append(schema.extract_robot_frame_obs(episode))  # type: ignore[union-attr]

    if obs.camera_frame_proprio_keys:
        data_group[PROPRIO_OBS_CAMERA_FRAME_KEY].append(schema.extract_camera_frame_obs(episode))  # type: ignore[union-attr]

    if obs.gripper_state_keys:
        data_group[GRIPPER_STATE_OBS_KEY].append(schema.extract_gripper_state(episode))  # type: ignore[union-attr]

    if schema.has_phase_labels:
        phase_labels = schema.extract_phase_labels(episode)
        data_group[PHASE_LABEL_KEY].append(phase_labels[:, np.newaxis])  # type: ignore[union-attr, index]

    if obs.language_key:
        data_group[LANGUAGE_KEY].append(schema.extract_language_instruction(episode))  # type: ignore[union-attr]

    for modality_name, keys in obs.custom_obs_keys.items():
        data_group[modality_name].append(schema.extract_custom_observations(
            df=episode, modality_name=modality_name))  # type: ignore[union-attr]


def _append_images(
        episode: pd.DataFrame,
        data_group: zarr.Group,
        schema: DatasetSchema,
        resizer: A.Resize | A.NoOp,
        depth_resizer: A.Resize | A.NoOp,
) -> None:
    """Append images to zarr `data_group` in-place."""
    for cam in schema.raw_observations.camera_keys:
        if cam == Cameras.DEPTH.value:
            # Get depth paths from left image paths
            # TODO: we should store depth paths directly in the csv instead of computing them on the fly.
            base_col = schema.get_image_path_column(camera=Cameras.LEFT.value)
            image_paths = episode[base_col].apply(
                lambda x: schema.compute_depth_path(base_image_path=x)
            )
        elif cam in [Cameras.LEFT.value, Cameras.RIGHT.value]:
            col = schema.get_image_path_column(camera=cam)
            image_paths = episode[col]
        else:
            raise ValueError(f"Unknown camera: {cam}")

        images = []
        for img_path in image_paths:
            if cam == Cameras.DEPTH.value:
                depth = np.load(img_path)
                resized = depth_resizer(image=depth)['image']
            else:
                rgb = cv2.cvtColor(
                    cv2.imread(img_path, cv2.IMREAD_COLOR),  # type: ignore[arg-type]
                    cv2.COLOR_BGR2RGB
                )
                resized = resizer(image=rgb)['image']
            images.append(resized)

        data_group[cam].append(np.stack(images))  # type: ignore[union-attr]
