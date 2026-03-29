"""Tests for end-to-end training pipeline."""

import gc
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path
from unittest.mock import MagicMock, patch

import hydra
import numpy as np
import pytest
import torch
import zarr
import zarr.storage
from hydra import compose, initialize_config_dir
from tso_robotics_sockets import CompressionType

import versatil.configs  # noqa: F401
from tests.endpoints.conftest import (
    DATASET_SPECS,
    HYDRA_CONFIG_DIR,
    _generate_array_for_key,
    build_tiny_overrides,
    discover_e2e_configs,
    get_free_port,
    resolve_dataset_type,
    start_mock_observation_server,
)
from versatil.inference.inference_client import InferenceClient
from versatil.inference.policy_loading.float_loader import PolicyLoader
from versatil.inference.socket_transport import (
    SocketActionTransport,
    SocketObservationTransport,
)
from versatil.workspace import Workspace

COMMON_OVERRIDES = [
    "task.dataloader.batch_size=2",
    "task.dataloader.num_workers=1",
    "task.dataloader.val_ratio=0.0",
    "training.num_epochs=1",
    "experiment.use_wandb=false",
    "experiment.name=e2e_test",
    "experiment.device=cpu",
]

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
NUM_EPISODES = 3
TIMESTEPS_PER_EPISODE = 15

E2E_CONFIGS = discover_e2e_configs()


def _create_synthetic_zarr(
    zarr_path: str,
    dataset_type: str,
    rng: np.random.Generator,
) -> None:
    spec = DATASET_SPECS[dataset_type]
    total_timesteps = NUM_EPISODES * TIMESTEPS_PER_EPISODE

    store = zarr.storage.LocalStore(zarr_path)
    root = zarr.open_group(store=store, mode="w")
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")

    for key, key_spec in spec.items():
        array_data = _generate_array_for_key(
            rng=rng,
            spec=key_spec,
            total_timesteps=total_timesteps,
            image_height=IMAGE_HEIGHT,
            image_width=IMAGE_WIDTH,
        )
        if key_spec["kind"] == "language":
            language_array = data_group.create_array(
                key,
                shape=(total_timesteps,),
                dtype=str,
            )
            language_array[:] = array_data
        else:
            data_group.create_array(
                key,
                data=array_data,
                chunks=array_data.shape,
            )

    episode_ends = np.array(
        [(i + 1) * TIMESTEPS_PER_EPISODE for i in range(NUM_EPISODES)],
        dtype=np.int64,
    )
    meta_group.create_array(
        "episode_ends",
        data=episode_ends,
        chunks=(NUM_EPISODES,),
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "config_name",
    E2E_CONFIGS,
    ids=[c.split("/")[-1] for c in E2E_CONFIGS],
)
def test_train_one_epoch_reload_checkpoint_and_infer(config_name, tmp_path):
    if "flow_unet" in config_name and "libero_hdf5" in config_name:
        pytest.skip("libero_hdf5/flow_unet has broken dropout_rate interpolation")

    dataset_type = resolve_dataset_type(config_name)
    rng = np.random.default_rng(42)
    zarr_path = str(tmp_path / "data.zarr")
    checkpoint_dir = str(tmp_path / "checkpoints")

    _create_synthetic_zarr(
        zarr_path=zarr_path,
        dataset_type=dataset_type,
        rng=rng,
    )

    decoder_overrides = build_tiny_overrides(config_name)
    all_overrides = (
        COMMON_OVERRIDES
        + decoder_overrides
        + [
            f"experiment.checkpoint_folder={checkpoint_dir}",
            f"task.dataset_schema.zarr_path={zarr_path}",
        ]
    )

    with initialize_config_dir(config_dir=HYDRA_CONFIG_DIR, version_base=None):
        yaml_config = compose(
            config_name=config_name,
            overrides=all_overrides,
        )
        with patch(
            "versatil.data.raw.schemas.lerobot.LeRobotDatasetMetadataV30.__init__",
            lambda self, dataset_path: setattr(self, "dataset_path", dataset_path),
        ):
            config = hydra.utils.instantiate(yaml_config)

    config.policy.to(torch.device("cpu"))

    with patch("versatil.workspace.HydraConfig") as mock_hydra:
        mock_hydra.get.return_value = MagicMock()
        mock_hydra.get.return_value.job.config_name = "test_e2e"
        workspace = Workspace(config, original_yaml_config=yaml_config)
        workspace.run()

    output_dir = Path(checkpoint_dir) / "test_e2e" / "e2e_test"
    assert (output_dir / "last.ckpt").exists()
    del workspace
    gc.collect()
    torch.cuda.empty_cache()
    # --- Reload checkpoint and run inference against mock server ---
    with patch(
        "versatil.data.raw.schemas.lerobot.LeRobotDatasetMetadataV30.__init__",
        lambda self, dataset_path: setattr(self, "dataset_path", dataset_path),
    ):
        policy_loader = PolicyLoader(
            device=torch.device("cpu"),
            checkpoint_path=str(output_dir),
            checkpoint_name="last.ckpt",
            precision="32",
        )

    port = get_free_port()
    server = start_mock_observation_server(
        observation_space=policy_loader.observation_space,
        port=port,
    )
    try:
        observation_transport = SocketObservationTransport(
            server_address="127.0.0.1",
            server_port=port,
        )
        action_transport = SocketActionTransport(
            server_address="127.0.0.1",
            server_port=port,
        )
        client = InferenceClient(
            policy_loader=policy_loader,
            observation_transport=observation_transport,
            action_transport=action_transport,
            compression_type=CompressionType.RAW.value,
        )
        status = client.step()
        assert status == "continue"

        action_metadata = client.action_postprocessor.build_action_metadata()
        assert len(action_metadata) > 0
    finally:
        server.stop()

    del policy_loader, client
    gc.collect()
