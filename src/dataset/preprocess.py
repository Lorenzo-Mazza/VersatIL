import os.path

import zarr
import zarr.storage
from zarr.codecs import BloscCodec, BloscShuffle
from threadpoolctl import threadpool_limits
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from albumentations import Resize, CenterCrop
from legacy_constants import (
    Cameras,
    ROBOT_FRAME_OBS_KEY,
    CAMERA_FRAME_OBS_KEY,
    GRIPPER_STATE_OBS_KEY,
    PHASE_LABEL_KEY,
)

import re

ROBOT_FRAME_KINEMATICS_COLS = [
    "relative_tip_position_x",
    "relative_tip_position_y",
    "relative_tip_position_z",
]
CAMERA_FRAME_KINEMATICS_COLS = [
    "camera_frame_tip_position_x",
    "camera_frame_tip_position_y",
    "camera_frame_tip_position_z",
]
GRIPPER_STATE_COL = "open"
ACTION_COL_KEYS: list[str] = ["action_x", "action_y", "action_z"]
RECTIFIED_LEFT_IMAGE_PATH_KEY = "frameLeftRectifiedPath"
LEFT_IMAGE_PATH_KEY = "frameLeftPath"
RECTIFIED_RIGHT_IMAGE_PATH_KEY = "frameRightRectifiedPath"
RIGHT_IMAGE_PATH_KEY = "frameRightPath"
PHASE_LABEL_COL = "task_phase"


def create_replay_buffer(
    dataset_paths: list[str],
    image_height: int,
    image_width: int,
    center_crop: bool,
    center_crop_size: int,
    camera_names: list[str],
    center_initial_position: bool,
    use_rectified_images: bool,
    downsample_factor: int,
    zarr_path: str = "dataset.zarr",
    predict_gripper_action: bool = False,
    task_has_phases: bool = False,
):
    """Creates a Zarr-based replay buffer dataset from robot demonstration CSV files and associated images.

    This function processes a list of CSV files containing robot kinematics and image paths. It downsamples the data,
    optionally centers positions relative to the initial frame, loads and transforms images (crop and resize if specified),
    and stores everything in a Zarr hierarchy. The structure includes 'data' group with arrays for raw robot_pos, camera_pos,
    and per-camera images (RGB or depth). The 'meta' group stores episode end indices. Depth images are
    computed from disparity maps by taking reciprocal (1/disp where disp > 0, else 0). Images are loaded using OpenCV,
    transformed with Albumentations, and compressed with Blosc in Zarr. Processing is done under threadpool limit of 1
    for stability. Episodes are sorted by directory name (assumed numeric). Actions and observation kinematics are not
    precomputed here; they are derived on-the-fly in the dataset based on configuration.

    Args:
        dataset_paths: List of paths to CSV files, each representing an episode with columns for kinematics and image paths.
        image_height: Target height for resized images.
        image_width: Target width for resized images.
        center_crop: If True, apply center crop to images before resizing.
        center_crop_size: Size (square) for center crop if enabled.
        camera_names: List of camera names (from Cameras enum) to include, e.g., ['left', 'right', 'depth'].
        center_initial_position: If True, subtract initial episode position from all kinematics.
        use_rectified_images: If True, use rectified image paths; else, original.
        downsample_factor: Step size for downsampling CSV rows (e.g., 2 takes every other row, always includes last).
        zarr_path: Path to save the Zarr dataset (default 'dataset.zarr').
        predict_gripper_action: If True, include gripper state in the dataset.
        task_has_phases: If True, include task phases in the dataset.
    Returns:
        The path to the created Zarr file.
    """
    print(f"Creating Zarr dataset at {zarr_path} with {len(dataset_paths)} episodes...")
    store = zarr.storage.LocalStore(zarr_path)
    root = zarr.open_group(store=store, mode="w")
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")
    episode_ends = []
    cumulative_len = 0

    compressor = BloscCodec(cname="lz4", clevel=5, shuffle=BloscShuffle.noshuffle)

    crop_transform = (
        CenterCrop(height=center_crop_size, width=center_crop_size)
        if center_crop
        else None
    )
    resizer = Resize(height=image_height, width=image_width)
    depth_resizer = Resize(
        height=image_height, width=image_width, interpolation=cv2.INTER_NEAREST
    )

    data_group.create_array(
        ROBOT_FRAME_OBS_KEY,
        shape=(0, 3),
        chunks=(100, 3),
        dtype=np.float32,
        compressors=[compressor],
    )
    data_group.create_array(
        CAMERA_FRAME_OBS_KEY,
        shape=(0, 3),
        chunks=(100, 3),
        dtype=np.float32,
        compressors=[compressor],
    )
    if predict_gripper_action:
        data_group.create_array(
            GRIPPER_STATE_OBS_KEY,
            shape=(0, 1),
            chunks=(100, 1),
            dtype=np.float32,
            compressors=[compressor],
        )

    if task_has_phases:
        data_group.create_array(
            PHASE_LABEL_KEY,
            shape=(0, 1),
            chunks=(100, 1),
            dtype=np.uint8,
            compressors=[compressor],
        )

    for cam in camera_names:
        if cam == Cameras.DEPTH.value:
            shape_suffix = (image_height, image_width)
            dtype = np.float32
        else:
            shape_suffix = (image_height, image_width, 3)
            dtype = np.uint8
        data_group.create_array(
            cam,
            shape=(0, *shape_suffix),
            chunks=(10, *shape_suffix),
            dtype=dtype,
            compressors=[compressor],
        )

    with threadpool_limits(1):
        for path in sorted(dataset_paths, key=lambda x: int(Path(x).parent.name)):
            df = pd.read_csv(path)

            indices = list(range(0, len(df), downsample_factor))
            if len(df) - 1 not in indices:
                indices.append(len(df) - 1)
            df = df.iloc[indices]

            col_list = ROBOT_FRAME_KINEMATICS_COLS + CAMERA_FRAME_KINEMATICS_COLS
            if center_initial_position:
                initial_pos = df.loc[0, col_list].astype(float)
                df[col_list] = df[col_list].astype(float).sub(initial_pos, axis=1)

            # Append positions
            data_group[ROBOT_FRAME_OBS_KEY].append(
                df[ROBOT_FRAME_KINEMATICS_COLS].values.astype(np.float32)
            )
            data_group[CAMERA_FRAME_OBS_KEY].append(
                df[CAMERA_FRAME_KINEMATICS_COLS].values.astype(np.float32)
            )

            if predict_gripper_action:
                data_group[GRIPPER_STATE_OBS_KEY].append(
                    df[GRIPPER_STATE_COL].values[:, np.newaxis].astype(np.float32)
                )
            if task_has_phases:
                data_group[PHASE_LABEL_KEY].append(
                    df[PHASE_LABEL_COL].values[:, np.newaxis].astype(np.uint8)
                )

            left_image_col = (
                LEFT_IMAGE_PATH_KEY
                if not use_rectified_images
                else RECTIFIED_LEFT_IMAGE_PATH_KEY
            )
            right_image_col = (
                RIGHT_IMAGE_PATH_KEY
                if not use_rectified_images
                else RECTIFIED_RIGHT_IMAGE_PATH_KEY
            )
            key_to_sub = (
                "framesLeft" if not use_rectified_images else "framesLeftRectified"
            )
            # Append images for selected cameras
            for cam in camera_names:
                if cam == Cameras.DEPTH.value:
                    col = Cameras.DEPTH.value
                    df[col] = df[left_image_col].apply(
                        lambda x: re.sub(
                            r"(\d+)\.png$",
                            r"depth_\1.npy",
                            x.replace(key_to_sub, "depth"),
                        )
                    )
                else:
                    col = {
                        Cameras.LEFT.value: left_image_col,
                        Cameras.RIGHT.value: right_image_col,
                    }[cam]

                images = []
                for img_path in df[col]:
                    if cam == Cameras.DEPTH.value:
                        depth = np.load(img_path)
                        if crop_transform:
                            depth = crop_transform(image=depth)["image"]
                        resized = depth_resizer(image=depth)["image"]
                    else:
                        rgb = cv2.cvtColor(
                            cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
                        )
                        if crop_transform:
                            rgb = crop_transform(image=rgb)["image"]
                        resized = resizer(image=rgb)["image"]
                    images.append(resized)
                data_group[cam].append(np.stack(images))

            cumulative_len += len(df)
            episode_ends.append(cumulative_len)

    meta_group.create_array(
        "episode_ends",
        data=np.array(episode_ends),
        chunks=(len(episode_ends),),
        compressors=None,
    )
    print(f"Created Zarr dataset at {zarr_path} with {len(episode_ends)} episodes.")
    return zarr_path
