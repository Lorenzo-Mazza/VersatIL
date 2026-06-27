"""Explainability package test fixtures."""

import json
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from versatil.configs.data.dataloader import DataLoaderConfig
from versatil.data.constants import (
    Cameras,
    DatasetType,
    LeRobotPathsV30,
    ProprioKey,
    RawCameraKey,
)
from versatil.data.metadata import CameraMetadata, PrecomputedActionMetadata
from versatil.data.normalization.normalizer import LinearNormalizer
from versatil.data.raw.schemas.base import DatasetSchema
from versatil.data.raw.schemas.csv import CsvDatasetSchema
from versatil.data.raw.schemas.custom.libero import LiberoSchema
from versatil.data.raw.schemas.custom.synthetic import SyntheticSchema
from versatil.data.raw.schemas.custom.tso import (
    TSO_EPISODE_FILENAME,
    TSO_RECTIFIED_LEFT_IMAGE_KEY,
    TSO_RECTIFIED_RIGHT_IMAGE_KEY,
    TSODatasetSchema,
)
from versatil.data.raw.schemas.hdf5 import Hdf5DatasetSchema
from versatil.data.raw.schemas.lerobot import LeRobotDatasetSchemaV30
from versatil.data.raw.zarr_meta import DatasetMetadata
from versatil.data.synthetic.constants import SyntheticTaskName
from versatil.data.task import ActionSpace, ObservationSpace

EXPLANATION_ACTION_KEY = "action"
TSO_ACTION_COLUMNS = ["action_x", "action_y", "action_z"]
IMAGE_HEIGHT = 8
IMAGE_WIDTH = 8
EPISODE_LENGTH = 5


@pytest.fixture
def csv_schema_factory() -> Callable[..., MagicMock]:
    def factory(
        dataset_folders: list[str] | None = None,
        zarr_path: str = "/tmp/training.zarr",
    ) -> MagicMock:
        schema = MagicMock(spec=CsvDatasetSchema)
        schema.dataset_folders = (
            dataset_folders if dataset_folders is not None else ["/tmp/training_raw"]
        )
        schema.zarr_path = zarr_path
        schema.dataset_filename = "episode.csv"
        return schema

    return factory


@pytest.fixture
def hdf5_schema_factory() -> Callable[..., MagicMock]:
    def factory(
        hdf5_paths: list[str] | None = None,
        zarr_path: str = "/tmp/training.zarr",
    ) -> MagicMock:
        schema = MagicMock(spec=Hdf5DatasetSchema)
        schema.hdf5_paths = hdf5_paths if hdf5_paths is not None else ["/tmp/raw.hdf5"]
        schema.zarr_path = zarr_path
        return schema

    return factory


@pytest.fixture
def lerobot_schema_factory() -> Callable[..., MagicMock]:
    def factory(
        dataset_path: Path | None = None,
        zarr_path: str = "/tmp/training.zarr",
    ) -> MagicMock:
        schema = MagicMock(spec=LeRobotDatasetSchemaV30)
        schema.dataset_path = dataset_path if dataset_path is not None else Path("/tmp")
        schema.zarr_path = zarr_path
        schema.lerobot_metadata = MagicMock()
        return schema

    return factory


@pytest.fixture
def synthetic_schema_factory() -> Callable[..., MagicMock]:
    def factory(zarr_path: str = "/tmp/training.zarr") -> MagicMock:
        schema = MagicMock(spec=SyntheticSchema)
        schema.zarr_path = zarr_path
        return schema

    return factory


@pytest.fixture
def explanation_policy_mock() -> MagicMock:
    policy = MagicMock()
    policy.normalizer = LinearNormalizer()
    policy.tokenizer = None
    return policy


@pytest.fixture
def explanation_source_config_factory() -> Callable[..., MagicMock]:
    def factory(
        schema: DatasetSchema,
        action_space: ActionSpace,
        observation_space: ObservationSpace,
        prediction_horizon: int = 2,
        observation_horizon: int = 1,
        val_ratio: float = 0.0,
        seed: int = 13,
    ) -> MagicMock:
        dataloader_config = DataLoaderConfig()
        dataloader_config.batch_size = 1
        dataloader_config.num_workers = 0
        dataloader_config.shuffle = False
        dataloader_config.val_ratio = val_ratio
        dataloader_config.total_ratio = 1.0
        dataloader_config.preload_data_in_memory = False
        dataloader_config.color_augmentation = None
        dataloader_config.spatial_augmentation = None
        dataloader_config.action_backward_shift = 0
        dataloader_config.trailing_padded_actions = 0

        task = MagicMock()
        task.dataset_schema = schema
        task.dataloader = dataloader_config
        task.action_space = action_space
        task.observation_space = observation_space
        task.prediction_horizon = prediction_horizon
        task.observation_horizon = observation_horizon

        experiment = MagicMock()
        experiment.seed = seed

        config = MagicMock()
        config.task = task
        config.experiment = experiment
        return config

    return factory


@pytest.fixture
def explanation_schema_case_factory(
    tmp_path: Path,
    rng: np.random.Generator,
    camera_metadata_factory: Callable[..., CameraMetadata],
    precomputed_action_metadata_factory: Callable[..., PrecomputedActionMetadata],
    dataset_metadata_factory: Callable[..., DatasetMetadata],
    action_space_factory: Callable[..., ActionSpace],
    observation_space_factory: Callable[..., ObservationSpace],
) -> Callable[
    ...,
    tuple[
        DatasetSchema,
        str | list[str],
        ActionSpace,
        ObservationSpace,
        list[str],
        str,
    ],
]:
    def factory(
        schema_name: str,
        raw_path_count: int = 1,
    ) -> tuple[
        DatasetSchema,
        str | list[str],
        ActionSpace,
        ObservationSpace,
        list[str],
        str,
    ]:
        if schema_name == "tso_csv":
            return _make_tso_case(
                root=tmp_path / "tso",
                rng=rng,
                raw_path_count=raw_path_count,
                camera_metadata_factory=camera_metadata_factory,
                precomputed_action_metadata_factory=precomputed_action_metadata_factory,
                dataset_metadata_factory=dataset_metadata_factory,
                action_space_factory=action_space_factory,
                observation_space_factory=observation_space_factory,
            )
        if schema_name == "libero_hdf5":
            return _make_libero_case(
                root=tmp_path / "libero",
                rng=rng,
                raw_path_count=raw_path_count,
                camera_metadata_factory=camera_metadata_factory,
                precomputed_action_metadata_factory=precomputed_action_metadata_factory,
                dataset_metadata_factory=dataset_metadata_factory,
                action_space_factory=action_space_factory,
                observation_space_factory=observation_space_factory,
            )
        if schema_name == "lerobot":
            return _make_lerobot_case(
                root=tmp_path / "lerobot",
                rng=rng,
                camera_metadata_factory=camera_metadata_factory,
                precomputed_action_metadata_factory=precomputed_action_metadata_factory,
                dataset_metadata_factory=dataset_metadata_factory,
                action_space_factory=action_space_factory,
                observation_space_factory=observation_space_factory,
            )
        if schema_name == "synthetic":
            return _make_synthetic_case(
                root=tmp_path / "synthetic",
                camera_metadata_factory=camera_metadata_factory,
                precomputed_action_metadata_factory=precomputed_action_metadata_factory,
                dataset_metadata_factory=dataset_metadata_factory,
                action_space_factory=action_space_factory,
                observation_space_factory=observation_space_factory,
            )
        raise ValueError(f"Unknown explainability schema case: {schema_name}")

    return factory


def _make_tso_case(
    root: Path,
    rng: np.random.Generator,
    raw_path_count: int,
    camera_metadata_factory: Callable[..., CameraMetadata],
    precomputed_action_metadata_factory: Callable[..., PrecomputedActionMetadata],
    dataset_metadata_factory: Callable[..., DatasetMetadata],
    action_space_factory: Callable[..., ActionSpace],
    observation_space_factory: Callable[..., ObservationSpace],
) -> tuple[
    DatasetSchema, str | list[str], ActionSpace, ObservationSpace, list[str], str
]:
    action_metadata = precomputed_action_metadata_factory(
        raw_data_column_keys=TSO_ACTION_COLUMNS,
        storage_dimension=3,
        prediction_dimension=3,
        slice_start=None,
        slice_end=None,
    )
    camera_metadata = {
        Cameras.LEFT.value: camera_metadata_factory(
            camera_key=Cameras.LEFT.value,
            image_height=IMAGE_HEIGHT,
            image_width=IMAGE_WIDTH,
        ),
        Cameras.RIGHT.value: camera_metadata_factory(
            camera_key=Cameras.RIGHT.value,
            image_height=IMAGE_HEIGHT,
            image_width=IMAGE_WIDTH,
        ),
    }
    metadata = dataset_metadata_factory(
        observations=camera_metadata,
        precomputed_actions={EXPLANATION_ACTION_KEY: action_metadata},
    )
    training_root = root / "training_raw"
    _write_tso_raw_root(root=training_root, rng=rng, episode_offset=0)
    override_paths = []
    for path_index in range(raw_path_count):
        override_root = root / f"override_raw_{path_index}"
        _write_tso_raw_root(
            root=override_root,
            rng=rng,
            episode_offset=path_index + 1,
        )
        override_paths.append(str(override_root))
    schema = TSODatasetSchema(
        dataset_folders=[str(training_root)],
        zarr_path=str(root / "training.zarr"),
        metadata=metadata,
        dataset_type=DatasetType.TSO.value,
    )
    action_space, observation_space = _build_task_spaces(
        action_key=EXPLANATION_ACTION_KEY,
        action_metadata=action_metadata,
        camera_metadata=camera_metadata,
        action_space_factory=action_space_factory,
        observation_space_factory=observation_space_factory,
    )
    return (
        schema,
        override_paths[0] if raw_path_count == 1 else override_paths,
        action_space,
        observation_space,
        list(camera_metadata),
        EXPLANATION_ACTION_KEY,
    )


def _make_libero_case(
    root: Path,
    rng: np.random.Generator,
    raw_path_count: int,
    camera_metadata_factory: Callable[..., CameraMetadata],
    precomputed_action_metadata_factory: Callable[..., PrecomputedActionMetadata],
    dataset_metadata_factory: Callable[..., DatasetMetadata],
    action_space_factory: Callable[..., ActionSpace],
    observation_space_factory: Callable[..., ObservationSpace],
) -> tuple[
    DatasetSchema, str | list[str], ActionSpace, ObservationSpace, list[str], str
]:
    action_metadata = precomputed_action_metadata_factory(
        raw_data_column_keys=[EXPLANATION_ACTION_KEY],
        storage_dimension=3,
        prediction_dimension=3,
        slice_start=None,
        slice_end=None,
    )
    camera_metadata = {
        Cameras.AGENTVIEW.value: camera_metadata_factory(
            camera_key=Cameras.AGENTVIEW.value,
            image_height=IMAGE_HEIGHT,
            image_width=IMAGE_WIDTH,
        ),
    }
    metadata = dataset_metadata_factory(
        observations=camera_metadata,
        precomputed_actions={EXPLANATION_ACTION_KEY: action_metadata},
    )
    training_hdf5_path = root / "training_task_demo.hdf5"
    _write_libero_hdf5(path=training_hdf5_path, rng=rng)
    override_paths = []
    for path_index in range(raw_path_count):
        hdf5_path = root / f"override_task_{path_index}_demo.hdf5"
        _write_libero_hdf5(path=hdf5_path, rng=rng)
        override_paths.append(str(hdf5_path))
    schema = LiberoSchema(
        hdf5_paths=[str(training_hdf5_path)],
        zarr_path=str(root / "training.zarr"),
        metadata=metadata,
        dataset_type=DatasetType.LIBERO.value,
    )
    action_space, observation_space = _build_task_spaces(
        action_key=EXPLANATION_ACTION_KEY,
        action_metadata=action_metadata,
        camera_metadata=camera_metadata,
        action_space_factory=action_space_factory,
        observation_space_factory=observation_space_factory,
    )
    return (
        schema,
        override_paths[0] if raw_path_count == 1 else override_paths,
        action_space,
        observation_space,
        list(camera_metadata),
        EXPLANATION_ACTION_KEY,
    )


def _make_lerobot_case(
    root: Path,
    rng: np.random.Generator,
    camera_metadata_factory: Callable[..., CameraMetadata],
    precomputed_action_metadata_factory: Callable[..., PrecomputedActionMetadata],
    dataset_metadata_factory: Callable[..., DatasetMetadata],
    action_space_factory: Callable[..., ActionSpace],
    observation_space_factory: Callable[..., ObservationSpace],
) -> tuple[
    DatasetSchema, str | list[str], ActionSpace, ObservationSpace, list[str], str
]:
    action_metadata = precomputed_action_metadata_factory(
        raw_data_column_keys=[EXPLANATION_ACTION_KEY],
        storage_dimension=3,
        prediction_dimension=3,
        slice_start=None,
        slice_end=None,
    )
    camera_metadata = {
        Cameras.AGENTVIEW.value: camera_metadata_factory(
            camera_key=RawCameraKey.FRONT.value,
            image_height=IMAGE_HEIGHT,
            image_width=IMAGE_WIDTH,
        ),
    }
    metadata = dataset_metadata_factory(
        observations=camera_metadata,
        precomputed_actions={EXPLANATION_ACTION_KEY: action_metadata},
    )
    training_root = root / "training_raw"
    override_root = root / "override_raw"
    _write_lerobot_raw_root(root=training_root, rng=rng)
    _write_lerobot_raw_root(root=override_root, rng=rng)
    schema = LeRobotDatasetSchemaV30(
        dataset_path=str(training_root),
        zarr_path=str(root / "training.zarr"),
        metadata=metadata,
        dataset_type=DatasetType.METAWORLD.value,
    )
    action_space, observation_space = _build_task_spaces(
        action_key=EXPLANATION_ACTION_KEY,
        action_metadata=action_metadata,
        camera_metadata=camera_metadata,
        action_space_factory=action_space_factory,
        observation_space_factory=observation_space_factory,
    )
    return (
        schema,
        str(override_root),
        action_space,
        observation_space,
        list(camera_metadata),
        EXPLANATION_ACTION_KEY,
    )


def _make_synthetic_case(
    root: Path,
    camera_metadata_factory: Callable[..., CameraMetadata],
    precomputed_action_metadata_factory: Callable[..., PrecomputedActionMetadata],
    dataset_metadata_factory: Callable[..., DatasetMetadata],
    action_space_factory: Callable[..., ActionSpace],
    observation_space_factory: Callable[..., ObservationSpace],
) -> tuple[
    DatasetSchema, str | list[str], ActionSpace, ObservationSpace, list[str], str
]:
    action_key = ProprioKey.SYNTHETIC_POSITION_ACTION.value
    action_metadata = precomputed_action_metadata_factory(
        raw_data_column_keys=[EXPLANATION_ACTION_KEY],
        storage_dimension=2,
        prediction_dimension=2,
        slice_start=None,
        slice_end=None,
    )
    camera_metadata = {
        Cameras.AGENTVIEW.value: camera_metadata_factory(
            camera_key=Cameras.AGENTVIEW.value,
            image_height=IMAGE_HEIGHT,
            image_width=IMAGE_WIDTH,
        ),
    }
    metadata = dataset_metadata_factory(
        observations=camera_metadata,
        precomputed_actions={action_key: action_metadata},
    )
    raw_override = root / "unsupported_raw"
    raw_override.mkdir(parents=True, exist_ok=True)
    schema = SyntheticSchema(
        zarr_path=str(root / "training.zarr"),
        metadata=metadata,
        dataset_type=DatasetType.SYNTHETIC.value,
        task_name=SyntheticTaskName.CIRCLE.value,
        num_episodes=2,
        seed=7,
        image_size=IMAGE_HEIGHT,
        num_modes=2,
        trajectory_length=EPISODE_LENGTH,
        noise_std=0.01,
        num_styles=1,
        num_rollouts=2,
    )
    action_space, observation_space = _build_task_spaces(
        action_key=action_key,
        action_metadata=action_metadata,
        camera_metadata=camera_metadata,
        action_space_factory=action_space_factory,
        observation_space_factory=observation_space_factory,
    )
    return (
        schema,
        str(raw_override),
        action_space,
        observation_space,
        list(camera_metadata),
        action_key,
    )


def _build_task_spaces(
    action_key: str,
    action_metadata: PrecomputedActionMetadata,
    camera_metadata: dict[str, CameraMetadata],
    action_space_factory: Callable[..., ActionSpace],
    observation_space_factory: Callable[..., ObservationSpace],
) -> tuple[ActionSpace, ObservationSpace]:
    action_space = action_space_factory(
        actions_metadata={action_key: action_metadata},
        denoise_actions=False,
    )
    observation_space = observation_space_factory(
        observations_metadata=camera_metadata,
    )
    return action_space, observation_space


def _write_tso_raw_root(
    root: Path, rng: np.random.Generator, episode_offset: int
) -> None:
    episode_directory = root / str(episode_offset)
    left_directory = episode_directory / "framesLeftRectified"
    right_directory = episode_directory / "framesRightRectified"
    left_directory.mkdir(parents=True, exist_ok=True)
    right_directory.mkdir(parents=True, exist_ok=True)

    rows = []
    actions = _action_array(rng=rng)
    for frame_index in range(EPISODE_LENGTH):
        left_path = left_directory / f"{frame_index}.png"
        right_path = right_directory / f"{frame_index}.png"
        _write_rgb_image(path=left_path, image=_rgb_image(rng=rng))
        _write_rgb_image(path=right_path, image=_rgb_image(rng=rng))
        rows.append(
            {
                TSO_RECTIFIED_LEFT_IMAGE_KEY: str(left_path),
                TSO_RECTIFIED_RIGHT_IMAGE_KEY: str(right_path),
                TSO_ACTION_COLUMNS[0]: actions[frame_index, 0],
                TSO_ACTION_COLUMNS[1]: actions[frame_index, 1],
                TSO_ACTION_COLUMNS[2]: actions[frame_index, 2],
            }
        )
    pd.DataFrame(rows).to_csv(
        episode_directory / TSO_EPISODE_FILENAME,
        index=False,
    )


def _write_libero_hdf5(path: Path, rng: np.random.Generator) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as hdf5_file:
        demo_group = hdf5_file.create_group("data/demo_0")
        observation_group = demo_group.create_group("obs")
        observation_group.create_dataset(
            Cameras.AGENTVIEW.value,
            data=_rgb_image_sequence(rng=rng),
        )
        demo_group.create_dataset(
            EXPLANATION_ACTION_KEY,
            data=_action_array(rng=rng),
        )


def _write_lerobot_raw_root(root: Path, rng: np.random.Generator) -> None:
    info_path = root / LeRobotPathsV30.INFO_PATH
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info = {
        "codebase_version": "v3.0",
        "total_episodes": 1,
        "features": {
            RawCameraKey.FRONT.value: {
                "dtype": "image",
                "shape": [IMAGE_HEIGHT, IMAGE_WIDTH, 3],
            },
            EXPLANATION_ACTION_KEY: {"dtype": "float32", "shape": [3]},
        },
        "data_path": str(LeRobotPathsV30.DEFAULT_DATA_PATH),
        "video_path": str(LeRobotPathsV30.DEFAULT_VIDEO_PATH),
    }
    with open(info_path, "w") as info_file:
        json.dump(info, info_file)

    tasks_path = root / LeRobotPathsV30.DEFAULT_TASKS_PATH
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"task_index": [0], "task": ["explain the sample"]}),
        str(tasks_path),
    )

    episode_metadata_path = root / str(LeRobotPathsV30.DEFAULT_EPISODES_PATH).format(
        chunk_index=0,
        file_index=0,
    )
    episode_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": [0],
                "data/chunk_index": [0],
                "data/file_index": [0],
            }
        ),
        str(episode_metadata_path),
    )

    data_path = root / str(LeRobotPathsV30.DEFAULT_DATA_PATH).format(
        chunk_index=0,
        file_index=0,
    )
    data_path.parent.mkdir(parents=True, exist_ok=True)
    image_column = pa.array(
        [
            {"bytes": _encode_png(image=_rgb_image(rng=rng)), "path": None}
            for _ in range(EPISODE_LENGTH)
        ],
        type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
    )
    actions = _action_array(rng=rng).tolist()
    data_table = pa.table(
        {
            "episode_index": [0] * EPISODE_LENGTH,
            "timestamp": [0.1 * frame_index for frame_index in range(EPISODE_LENGTH)],
            "frame_index": list(range(EPISODE_LENGTH)),
            "task_index": [0] * EPISODE_LENGTH,
            EXPLANATION_ACTION_KEY: pa.array(
                actions,
                type=pa.list_(pa.float32()),
            ),
            RawCameraKey.FRONT.value: image_column,
        }
    )
    pq.write_table(data_table, str(data_path))


def _action_array(rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((EPISODE_LENGTH, 3)).astype(np.float32)


def _rgb_image_sequence(rng: np.random.Generator) -> np.ndarray:
    return np.stack([_rgb_image(rng=rng) for _ in range(EPISODE_LENGTH)])


def _rgb_image(rng: np.random.Generator) -> np.ndarray:
    return rng.integers(
        low=0,
        high=256,
        size=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        dtype=np.uint8,
    )


def _write_rgb_image(path: Path, image: np.ndarray) -> None:
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(str(path), image_bgr)
    if not success:
        raise RuntimeError(f"Failed to write RGB test image: {path}")


def _encode_png(image: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError("Failed to encode LeRobot test image as PNG.")
    return encoded.tobytes()
