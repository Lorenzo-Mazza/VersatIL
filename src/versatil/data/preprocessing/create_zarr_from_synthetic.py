"""Creates a Zarr-based replay buffer dataset from synthetic episode generators."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr
import zarr.storage
from zarr.codecs import BloscCodec, BloscShuffle

from versatil.data.constants import Cameras, ProprioKey, SyntheticObsKey
from versatil.data.raw.schemas.custom.synthetic import SyntheticSchema
from versatil.data.synthetic.generators import generate_task_episodes
from versatil.data.synthetic.visualization import plot_trajectories_2d

GENERATOR_KEY_TO_ZARR_KEY = {
    "image": Cameras.AGENTVIEW.value,
    "position": ProprioKey.SYNTHETIC_POSITION.value,
    "action": ProprioKey.SYNTHETIC_POSITION_ACTION.value,
    "context": SyntheticObsKey.CONTEXT.value,
    "mode_id": SyntheticObsKey.MODE_ID.value,
}


def create_replay_buffer_from_synthetic(schema: SyntheticSchema) -> None:
    """Create a Zarr-based replay buffer from procedurally generated synthetic episodes.

    Args:
        schema: SyntheticSchema instance with generation parameters and zarr path.
    """
    print(
        f"Creating synthetic Zarr at {schema.zarr_path} "
        f"(task={schema.task_name}, episodes={schema.num_episodes})"
    )

    store = zarr.storage.LocalStore(schema.zarr_path)
    root = zarr.open_group(store=store, mode="w")
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")
    compressor = BloscCodec(cname="lz4", clevel=5, shuffle=BloscShuffle.noshuffle)
    _create_zarr_arrays(data_group=data_group, schema=schema, compressor=compressor)
    episodes = generate_task_episodes(
        task_name=schema.task_name,
        num_episodes=schema.num_episodes,
        seed=schema.seed,
        image_size=schema.image_size,
        num_modes=schema.num_modes,
        trajectory_length=schema.trajectory_length,
        noise_std=schema.noise_std,
        num_styles=schema.num_styles,
        mode_weights=schema.mode_weights,
    )
    episode_ends = []
    cumulative_length = 0
    for episode in episodes:
        for generator_key, zarr_key in GENERATOR_KEY_TO_ZARR_KEY.items():
            if zarr_key in data_group:
                data_group[zarr_key].append(episode[generator_key])
        episode_length = len(episode["position"])
        cumulative_length += episode_length
        episode_ends.append(cumulative_length)
    meta_group.create_array(
        "episode_ends",
        data=np.array(episode_ends),
        chunks=(len(episode_ends),),
        compressors=None,
    )
    _save_training_visualization(
        episodes=episodes,
        task_name=schema.task_name,
        zarr_path=schema.zarr_path,
    )
    print(
        f"Created Zarr dataset with {len(episode_ends)} episodes, "
        f"{cumulative_length} total steps."
    )


def _save_training_visualization(
    episodes: list[dict[str, np.ndarray]],
    task_name: str,
    zarr_path: str,
) -> None:
    """Save a 2D trajectory PNG alongside the zarr store.

    The PNG is written to ``<zarr_parent>/<zarr_stem>_trajectories.png`` and
    shows all training trajectories color-coded by mode. Used as a quick
    visual sanity check of the generated dataset.

    Args:
        episodes: List of episode dicts from ``generate_task_episodes``.
        task_name: SyntheticTaskName.value string for layout lookup.
        zarr_path: Path to the zarr store (used to derive the PNG path).
    """
    trajectories = np.array([episode["position"] for episode in episodes])
    mode_ids = np.array([int(episode["mode_id"][0, 0]) for episode in episodes])
    zarr_path_obj = Path(zarr_path)
    output_path = zarr_path_obj.parent / f"{zarr_path_obj.stem}_trajectories.png"
    figure = plot_trajectories_2d(
        trajectories=trajectories,
        task_name=task_name,
        output_path=str(output_path),
        mode_ids=mode_ids,
    )
    plt.close(figure)
    print(f"Saved trajectory visualization to {output_path}")


def _create_zarr_arrays(
    data_group: zarr.Group,
    schema: SyntheticSchema,
    compressor: BloscCodec,
) -> None:
    """Create zarr arrays based on schema configuration."""
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
