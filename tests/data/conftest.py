"""Data package test fixtures: synthetic data generators and metadata factories."""

import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import zarr

from versatil.configs.data.dataloader import DataLoaderConfig
from versatil.data.constants import (
    ActionComputationMethod,
    BinaryGripperRange,
    Cameras,
    CoordinateSystem,
    GripperType,
    ProprioKey,
)
from versatil.data.episodic_dataset import EpisodicDataset
from versatil.data.metadata import (
    CameraMetadata,
    GripperObservationMetadata,
    OnTheFlyActionMetadata,
    PositionObservationMetadata,
)
from versatil.data.task import ActionSpace, ObservationSpace


def generate_synthetic_positions(
    rng: np.random.Generator,
    num_samples: int = 10,
    num_dimensions: int = 3,
    position_range: tuple = (-1.0, 1.0),
) -> np.ndarray:
    """Generate synthetic position data.

    Returns:
        Array of shape (num_samples, num_dimensions).
    """
    return rng.uniform(
        position_range[0], position_range[1], (num_samples, num_dimensions)
    ).astype(np.float32)


def generate_synthetic_quaternions(
    rng: np.random.Generator,
    num_samples: int = 10,
) -> np.ndarray:
    """Generate synthetic unit quaternions (w, x, y, z).

    Returns:
        Array of shape (num_samples, 4) with normalized quaternions.
    """
    quaternions = rng.standard_normal((num_samples, 4)).astype(np.float32)
    quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
    return quaternions


def generate_synthetic_euler_angles(
    rng: np.random.Generator,
    num_samples: int = 10,
    angle_range: tuple = (-np.pi, np.pi),
) -> np.ndarray:
    """Generate synthetic Euler angles (xyz convention).

    Returns:
        Array of shape (num_samples, 3).
    """
    return rng.uniform(angle_range[0], angle_range[1], (num_samples, 3)).astype(
        np.float32
    )


def generate_synthetic_gripper_states(
    rng: np.random.Generator,
    num_samples: int = 10,
    gripper_type: str = GripperType.BINARY.value,
) -> np.ndarray:
    """Generate synthetic gripper states.

    Returns:
        Array of shape (num_samples, 1).
    """
    if gripper_type == GripperType.BINARY.value:
        return rng.integers(0, 2, (num_samples, 1)).astype(np.float32)
    elif gripper_type == GripperType.CONTINUOUS.value:
        return rng.uniform(0.0, 1.0, (num_samples, 1)).astype(np.float32)
    else:
        raise ValueError(f"Unknown gripper_type: {gripper_type}")


def generate_synthetic_rgb_images(
    rng: np.random.Generator,
    num_timesteps: int = 5,
    height: int = 64,
    width: int = 64,
) -> np.ndarray:
    """Generate synthetic RGB images.

    Returns:
        Array of shape (num_timesteps, height, width, 3) with values in [0, 1].
    """
    return rng.random((num_timesteps, height, width, 3)).astype(np.float32)


def generate_synthetic_depth_images(
    rng: np.random.Generator,
    num_timesteps: int = 5,
    height: int = 64,
    width: int = 64,
    depth_range: tuple = (0.5, 5.0),
) -> np.ndarray:
    """Generate synthetic depth images.

    Returns:
        Array of shape (num_timesteps, height, width) with depth values.
    """
    return rng.uniform(
        depth_range[0], depth_range[1], (num_timesteps, height, width)
    ).astype(np.float32)


def generate_synthetic_episode(
    rng: np.random.Generator,
    num_timesteps: int = 10,
    position_dim: int = 3,
    orientation_dim: int = 4,
    has_gripper: bool = True,
    cameras: list = None,
    image_height: int = 64,
    image_width: int = 64,
) -> dict:
    """Generate a complete synthetic episode.

    Returns:
        Dictionary with episode data matching replay buffer structure.
    """
    if cameras is None:
        cameras = [Cameras.LEFT.value, Cameras.RIGHT.value]

    episode = {}

    positions = generate_synthetic_positions(rng, num_timesteps, position_dim)
    if orientation_dim == 4:
        orientations = generate_synthetic_quaternions(rng, num_timesteps)
    elif orientation_dim == 3:
        orientations = generate_synthetic_euler_angles(rng, num_timesteps)
    elif orientation_dim == 1:
        orientations = generate_synthetic_euler_angles(rng, num_timesteps)[:, :1]
    else:
        orientations = np.zeros((num_timesteps, 0), dtype=np.float32)

    proprio = np.concatenate([positions, orientations], axis=1)
    episode[ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value] = proprio
    episode[ProprioKey.CAMERA_FRAME_CARTESIAN_TIP_POS.value] = proprio.copy()

    if has_gripper:
        episode[ProprioKey.GRIPPER_STATE.value] = generate_synthetic_gripper_states(
            rng, num_timesteps, gripper_type=GripperType.BINARY.value
        )

    for camera in cameras:
        if camera == Cameras.DEPTH.value:
            episode[camera] = generate_synthetic_depth_images(
                rng, num_timesteps, image_height, image_width
            )
        else:
            rgb = generate_synthetic_rgb_images(
                rng, num_timesteps, image_height, image_width
            )
            episode[camera] = (rgb * 255).astype(np.uint8)

    return episode


def create_synthetic_replay_buffer(
    rng: np.random.Generator,
    num_episodes: int = 5,
    num_timesteps_per_episode: int = 10,
    position_dim: int = 3,
    orientation_dim: int = 4,
    has_gripper: bool = True,
    cameras: list = None,
    image_height: int = 64,
    image_width: int = 64,
) -> tuple:
    """Create a synthetic replay buffer in Zarr format.

    Returns:
        Tuple of (zarr_path, episode_ends).
    """
    if cameras is None:
        cameras = [Cameras.LEFT.value, Cameras.RIGHT.value]

    temp_dir = tempfile.mkdtemp()
    zarr_path = Path(temp_dir) / "test_replay_buffer.zarr"

    store = zarr.storage.LocalStore(str(zarr_path))
    root = zarr.open_group(store=store, mode="w")
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")

    total_timesteps = num_episodes * num_timesteps_per_episode
    proprio_dim = position_dim + orientation_dim

    data_group.create_array(
        ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value,
        shape=(total_timesteps, proprio_dim),
        chunks=(100, proprio_dim),
        dtype=np.float32,
    )
    data_group.create_array(
        ProprioKey.CAMERA_FRAME_CARTESIAN_TIP_POS.value,
        shape=(total_timesteps, proprio_dim),
        chunks=(100, proprio_dim),
        dtype=np.float32,
    )

    if has_gripper:
        data_group.create_array(
            ProprioKey.GRIPPER_STATE.value,
            shape=(total_timesteps, 1),
            chunks=(100, 1),
            dtype=np.float32,
        )

    for camera in cameras:
        if camera == Cameras.DEPTH.value:
            data_group.create_array(
                camera,
                shape=(total_timesteps, image_height, image_width),
                chunks=(1, image_height, image_width),
                dtype=np.float32,
            )
        else:
            data_group.create_array(
                camera,
                shape=(total_timesteps, image_height, image_width, 3),
                chunks=(1, image_height, image_width, 3),
                dtype=np.uint8,
            )

    episode_ends = []
    current_index = 0

    for _episode_index in range(num_episodes):
        episode = generate_synthetic_episode(
            rng=rng,
            num_timesteps=num_timesteps_per_episode,
            position_dim=position_dim,
            orientation_dim=orientation_dim,
            has_gripper=has_gripper,
            cameras=cameras,
            image_height=image_height,
            image_width=image_width,
        )

        end_index = current_index + num_timesteps_per_episode
        data_group[ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value][
            current_index:end_index
        ] = episode[ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value]
        data_group[ProprioKey.CAMERA_FRAME_CARTESIAN_TIP_POS.value][
            current_index:end_index
        ] = episode[ProprioKey.CAMERA_FRAME_CARTESIAN_TIP_POS.value]

        if has_gripper:
            data_group[ProprioKey.GRIPPER_STATE.value][current_index:end_index] = (
                episode[ProprioKey.GRIPPER_STATE.value]
            )

        for camera in cameras:
            data_group[camera][current_index:end_index] = episode[camera]

        episode_ends.append(end_index)
        current_index = end_index

    meta_group.create_array(
        "episode_ends",
        data=np.array(episode_ends),
        chunks=(len(episode_ends),),
    )

    return str(zarr_path), episode_ends


@pytest.fixture
def synthetic_rgb_images(rng: np.random.Generator) -> Callable[..., np.ndarray]:
    """Factory for generating synthetic RGB images."""

    def factory(
        num_timesteps: int = 5,
        height: int = 64,
        width: int = 64,
    ) -> np.ndarray:
        return generate_synthetic_rgb_images(rng, num_timesteps, height, width)

    return factory


@pytest.fixture
def synthetic_depth_images(rng: np.random.Generator) -> Callable[..., np.ndarray]:
    """Factory for generating synthetic depth images."""

    def factory(
        num_timesteps: int = 5,
        height: int = 64,
        width: int = 64,
        depth_range: tuple = (0.5, 5.0),
    ) -> np.ndarray:
        return generate_synthetic_depth_images(
            rng, num_timesteps, height, width, depth_range
        )

    return factory


@pytest.fixture
def synthetic_replay_buffer(rng: np.random.Generator) -> Callable[..., tuple]:
    """Factory for creating a synthetic replay buffer in Zarr format."""

    def factory(
        num_episodes: int = 5,
        num_timesteps_per_episode: int = 10,
        position_dim: int = 3,
        orientation_dim: int = 4,
        has_gripper: bool = True,
        cameras: list = None,
        image_height: int = 64,
        image_width: int = 64,
    ) -> tuple:
        return create_synthetic_replay_buffer(
            rng,
            num_episodes,
            num_timesteps_per_episode,
            position_dim,
            orientation_dim,
            has_gripper,
            cameras,
            image_height,
            image_width,
        )

    return factory


@pytest.fixture
def real_dataset_factory(
    synthetic_replay_buffer: Callable[..., tuple],
    position_observation_metadata_factory: Callable[..., PositionObservationMetadata],
    gripper_observation_metadata_factory: Callable[..., GripperObservationMetadata],
    camera_metadata_factory: Callable[..., CameraMetadata],
    on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
) -> Callable[..., EpisodicDataset]:
    """Factory for EpisodicDataset backed by a real zarr replay buffer."""
    created_paths = []

    def factory(
        num_episodes: int = 5,
        timesteps_per_episode: int = 20,
        position_dim: int = 3,
        orientation_dim: int = 4,
        has_gripper: bool = True,
        cameras: list[str] | None = None,
        image_height: int = 16,
        image_width: int = 16,
        pred_horizon: int = 4,
        obs_horizon: int = 2,
        train: bool = True,
        val_ratio: float = 0.1,
        trailing_padded_actions: int | None = None,
        action_backward_shift: int = 1,
    ) -> EpisodicDataset:
        if cameras is None:
            cameras = []

        zarr_path, episode_ends = synthetic_replay_buffer(
            num_episodes=num_episodes,
            num_timesteps_per_episode=timesteps_per_episode,
            position_dim=position_dim,
            orientation_dim=orientation_dim,
            has_gripper=has_gripper,
            cameras=cameras,
            image_height=image_height,
            image_width=image_width,
        )
        created_paths.append(zarr_path)

        position_meta = position_observation_metadata_factory(
            dimension=position_dim,
            frame=CoordinateSystem.ROBOT_BASE.value,
            raw_data_column_keys=[f"pos_{i}" for i in range(position_dim)],
        )

        observations_metadata = {
            ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value: position_meta,
        }
        for camera_key in cameras:
            channels = 1 if camera_key == Cameras.DEPTH.value else 3
            observations_metadata[camera_key] = camera_metadata_factory(
                camera_key=camera_key,
                channels=channels,
                image_height=image_height,
                image_width=image_width,
            )

        actions_metadata = {
            ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value: on_the_fly_action_metadata_factory(
                source_metadata=position_meta,
            ),
        }
        if has_gripper:
            gripper_meta = gripper_observation_metadata_factory(
                gripper_type=GripperType.BINARY.value,
                binary_gripper_range=BinaryGripperRange.ZERO_ONE.value,
            )
            observations_metadata[ProprioKey.GRIPPER_STATE.value] = gripper_meta
            actions_metadata[ProprioKey.GRIPPER_STATE.value] = (
                on_the_fly_action_metadata_factory(
                    source_metadata=gripper_meta,
                    computation_method=ActionComputationMethod.NEXT_TIMESTEP.value,
                )
            )

        action_space = ActionSpace(
            actions_metadata=actions_metadata,
            denoise_actions=False,
        )
        observation_space = ObservationSpace(
            observations_metadata=observations_metadata,
        )

        config = DataLoaderConfig()
        config.val_ratio = val_ratio
        config.total_ratio = 1.0
        config.downsample_factor = 1
        config.skip_initial_episode_steps = 0
        config.action_backward_shift = action_backward_shift
        config.preload_data_in_memory = False
        config.image_height = image_height
        config.image_width = image_width
        config.color_augmentation = None
        config.spatial_augmentation = None
        config.trailing_padded_actions = trailing_padded_actions

        dataset = EpisodicDataset(
            zarr_path=zarr_path,
            action_space=action_space,
            observation_space=observation_space,
            dataloader_config=config,
            pred_horizon=pred_horizon,
            obs_horizon=obs_horizon,
            train=train,
            seed=42,
        )
        return dataset

    yield factory

    for path in created_paths:
        shutil.rmtree(path, ignore_errors=True)
