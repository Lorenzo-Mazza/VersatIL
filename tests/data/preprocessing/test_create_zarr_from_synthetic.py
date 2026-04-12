"""Tests for versatil.data.preprocessing.create_zarr_from_synthetic module."""

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import zarr

from versatil.data.constants import Cameras, ProprioKey, SyntheticObsKey
from versatil.data.preprocessing.create_zarr_from_synthetic import (
    GENERATOR_KEY_TO_ZARR_KEY,
    create_replay_buffer_from_synthetic,
)
from versatil.data.raw.schemas.custom.synthetic import SyntheticSchema
from versatil.data.synthetic.constants import SyntheticTaskName


def _default_zarr_array_specs(image_size: int) -> dict[str, dict]:
    return {
        Cameras.AGENTVIEW.value: {
            "shape": (0, image_size, image_size, 3),
            "chunks": (16, image_size, image_size, 3),
            "dtype": "uint8",
            "needs_compressor": True,
        },
        ProprioKey.SYNTHETIC_POSITION.value: {
            "shape": (0, 2),
            "chunks": (256, 2),
            "dtype": "float32",
            "needs_compressor": True,
        },
        ProprioKey.SYNTHETIC_POSITION_ACTION.value: {
            "shape": (0, 2),
            "chunks": (256, 2),
            "dtype": "float32",
            "needs_compressor": True,
        },
        SyntheticObsKey.CONTEXT.value: {
            "shape": (0, 1),
            "chunks": (256, 1),
            "dtype": "int32",
            "needs_compressor": True,
        },
        SyntheticObsKey.MODE_ID.value: {
            "shape": (0, 1),
            "chunks": (256, 1),
            "dtype": "int32",
            "needs_compressor": True,
        },
    }


@pytest.fixture
def fake_episode_factory(
    rng: np.random.Generator,
) -> Callable[..., list[dict[str, np.ndarray]]]:
    def factory(
        num_episodes: int = 3,
        trajectory_length: int = 10,
        image_size: int = 16,
    ) -> list[dict[str, np.ndarray]]:
        episodes = []
        for _ in range(num_episodes):
            episode = {
                "image": rng.integers(
                    0,
                    255,
                    size=(trajectory_length, image_size, image_size, 3),
                    dtype=np.uint8,
                ),
                "position": rng.uniform(0.0, 1.0, size=(trajectory_length, 2)).astype(
                    np.float32
                ),
                "action": rng.uniform(-0.1, 0.1, size=(trajectory_length, 2)).astype(
                    np.float32
                ),
                "context": rng.integers(
                    0, 3, size=(trajectory_length, 1), dtype=np.int32
                ),
                "mode_id": rng.integers(
                    0, 3, size=(trajectory_length, 1), dtype=np.int32
                ),
            }
            episodes.append(episode)
        return episodes

    return factory


@pytest.fixture
def mock_schema_factory(tmp_path: Path) -> Callable[..., MagicMock]:
    def factory(
        image_size: int = 16,
        task_name: str = SyntheticTaskName.MULTI_PATH_NAVIGATION.value,
        num_episodes: int = 3,
        seed: int = 42,
        num_modes: int = 3,
        trajectory_length: int = 10,
        noise_std: float = 0.01,
        num_styles: int = 4,
        zarr_array_specs: dict[str, dict] | None = None,
    ) -> MagicMock:
        schema = MagicMock(spec=SyntheticSchema)
        schema.zarr_path = str(tmp_path / "test.zarr")
        schema.task_name = task_name
        schema.num_episodes = num_episodes
        schema.seed = seed
        schema.image_size = image_size
        schema.num_modes = num_modes
        schema.trajectory_length = trajectory_length
        schema.noise_std = noise_std
        schema.num_styles = num_styles
        schema.get_zarr_array_specs.return_value = (
            zarr_array_specs
            if zarr_array_specs is not None
            else _default_zarr_array_specs(image_size=image_size)
        )
        return schema

    return factory


def _run_with_patched_generators(
    schema: MagicMock,
    episodes: list[dict[str, np.ndarray]],
    extra_patches: dict[str, object] | None = None,
) -> MagicMock | None:
    generator_patch = patch(
        "versatil.data.preprocessing.create_zarr_from_synthetic.generate_task_episodes",
        return_value=episodes,
    )
    plot_patch = patch(
        "versatil.data.preprocessing.create_zarr_from_synthetic.plot_trajectories_2d"
    )
    with generator_patch, plot_patch as mock_plot:
        if extra_patches:
            for target, value in extra_patches.items():
                patcher = patch(target, value)
                patcher.start()
        create_replay_buffer_from_synthetic(schema=schema)
    return mock_plot


@pytest.mark.unit
def test_creates_data_and_meta_groups(
    mock_schema_factory: Callable[..., MagicMock],
    fake_episode_factory: Callable[..., list[dict[str, np.ndarray]]],
):
    schema = mock_schema_factory()
    episodes = fake_episode_factory(num_episodes=2, trajectory_length=5)

    _run_with_patched_generators(schema=schema, episodes=episodes)

    root = zarr.open_group(schema.zarr_path, mode="r")
    assert "data" in root
    assert "meta" in root


@pytest.mark.unit
def test_episode_ends_accumulate_trajectory_lengths(
    mock_schema_factory: Callable[..., MagicMock],
    fake_episode_factory: Callable[..., list[dict[str, np.ndarray]]],
):
    trajectory_length = 10
    num_episodes = 3
    schema = mock_schema_factory(
        num_episodes=num_episodes,
        trajectory_length=trajectory_length,
    )
    episodes = fake_episode_factory(
        num_episodes=num_episodes,
        trajectory_length=trajectory_length,
    )

    _run_with_patched_generators(schema=schema, episodes=episodes)

    root = zarr.open_group(schema.zarr_path, mode="r")
    episode_ends = root["meta"]["episode_ends"][:]
    expected_ends = np.array(
        [trajectory_length * (index + 1) for index in range(num_episodes)]
    )
    np.testing.assert_array_equal(episode_ends, expected_ends)


@pytest.mark.unit
def test_zarr_array_shapes_match_total_timesteps(
    mock_schema_factory: Callable[..., MagicMock],
    fake_episode_factory: Callable[..., list[dict[str, np.ndarray]]],
):
    trajectory_length = 8
    num_episodes = 2
    image_size = 16
    total_timesteps = num_episodes * trajectory_length
    schema = mock_schema_factory(
        num_episodes=num_episodes,
        trajectory_length=trajectory_length,
        image_size=image_size,
    )
    episodes = fake_episode_factory(
        num_episodes=num_episodes,
        trajectory_length=trajectory_length,
        image_size=image_size,
    )

    _run_with_patched_generators(schema=schema, episodes=episodes)

    data = zarr.open_group(schema.zarr_path, mode="r")["data"]
    assert data[Cameras.AGENTVIEW.value].shape == (
        total_timesteps,
        image_size,
        image_size,
        3,
    )
    assert data[ProprioKey.SYNTHETIC_POSITION.value].shape == (total_timesteps, 2)
    assert data[ProprioKey.SYNTHETIC_POSITION_ACTION.value].shape == (
        total_timesteps,
        2,
    )


@pytest.mark.unit
def test_str_dtype_array_created_without_compressor(
    mock_schema_factory: Callable[..., MagicMock],
    fake_episode_factory: Callable[..., list[dict[str, np.ndarray]]],
):
    image_size = 16
    specs = _default_zarr_array_specs(image_size=image_size)
    specs["language"] = {
        "shape": (0, 1),
        "chunks": (100, 1),
        "dtype": "str",
        "needs_compressor": False,
    }
    schema = mock_schema_factory(image_size=image_size, zarr_array_specs=specs)
    episodes = fake_episode_factory(num_episodes=2, trajectory_length=5)
    for episode in episodes:
        episode["language"] = np.array([["pick up the box"]] * 5, dtype=object)
    generator_mapping = dict(GENERATOR_KEY_TO_ZARR_KEY)
    generator_mapping["language"] = "language"

    _run_with_patched_generators(
        schema=schema,
        episodes=episodes,
        extra_patches={
            "versatil.data.preprocessing.create_zarr_from_synthetic.GENERATOR_KEY_TO_ZARR_KEY": generator_mapping,
        },
    )

    data = zarr.open_group(schema.zarr_path, mode="r")["data"]
    assert "language" in data
    assert len(data["language"].compressors) == 0


@pytest.mark.unit
def test_needs_compressor_false_skips_compression(
    mock_schema_factory: Callable[..., MagicMock],
    fake_episode_factory: Callable[..., list[dict[str, np.ndarray]]],
):
    image_size = 16
    specs = _default_zarr_array_specs(image_size=image_size)
    specs[SyntheticObsKey.MODE_ID.value]["needs_compressor"] = False
    schema = mock_schema_factory(image_size=image_size, zarr_array_specs=specs)
    episodes = fake_episode_factory(num_episodes=2, trajectory_length=5)

    _run_with_patched_generators(schema=schema, episodes=episodes)

    data = zarr.open_group(schema.zarr_path, mode="r")["data"]
    assert len(data[SyntheticObsKey.MODE_ID.value].compressors) == 0


@pytest.mark.unit
def test_saves_training_trajectory_png_alongside_zarr(
    mock_schema_factory: Callable[..., MagicMock],
    fake_episode_factory: Callable[..., list[dict[str, np.ndarray]]],
    tmp_path: Path,
):
    schema = mock_schema_factory()
    episodes = fake_episode_factory(num_episodes=3, trajectory_length=5)

    # Do not patch plot_trajectories_2d here — we want to verify the
    # real PNG is written to the expected path.
    with patch(
        "versatil.data.preprocessing.create_zarr_from_synthetic.generate_task_episodes",
        return_value=episodes,
    ):
        create_replay_buffer_from_synthetic(schema=schema)

    zarr_path_obj = Path(schema.zarr_path)
    expected_png = zarr_path_obj.parent / f"{zarr_path_obj.stem}_trajectories.png"
    assert expected_png.exists()
    assert expected_png.stat().st_size > 0


@pytest.mark.unit
def test_plot_trajectories_receives_episode_positions_and_mode_ids(
    mock_schema_factory: Callable[..., MagicMock],
    fake_episode_factory: Callable[..., list[dict[str, np.ndarray]]],
):
    num_episodes = 2
    trajectory_length = 5
    schema = mock_schema_factory(
        num_episodes=num_episodes, trajectory_length=trajectory_length
    )
    episodes = fake_episode_factory(
        num_episodes=num_episodes, trajectory_length=trajectory_length
    )

    mock_plot = _run_with_patched_generators(schema=schema, episodes=episodes)

    mock_plot.assert_called_once()
    call_kwargs = mock_plot.call_args.kwargs
    assert call_kwargs["task_name"] == schema.task_name
    assert call_kwargs["trajectories"].shape == (num_episodes, trajectory_length, 2)
    assert call_kwargs["mode_ids"].shape == (num_episodes,)
    expected_trajectories = np.array([episode["position"] for episode in episodes])
    np.testing.assert_array_equal(call_kwargs["trajectories"], expected_trajectories)
    expected_mode_ids = np.array(
        [int(episode["mode_id"][0, 0]) for episode in episodes]
    )
    np.testing.assert_array_equal(call_kwargs["mode_ids"], expected_mode_ids)
