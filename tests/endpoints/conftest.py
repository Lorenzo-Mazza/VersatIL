"""Endpoint test fixtures: synthetic zarr factories and e2e config helpers."""

import socket as socket_module
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import zarr
import zarr.storage
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tso_robotics_sockets import (
    CompressionType,
    InferenceRequestKey,
    InferenceResponseKey,
    ServerRoute,
    ServerStatus,
    SocketServer,
    TransportKey,
    compress_array,
)
from versatil_constants.tso import TSOObsKey

import versatil.configs  # noqa: F401 — registers ConfigStore entries
from versatil.data.constants import (
    Cameras,
    ObsKey,
    ProprioKey,
    SyntheticObsKey,
)
from versatil.data.task import ObservationSpace

BOWEL_RETRACTION_ZARR_SPEC: dict[str, dict] = {
    ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value: {
        "dimension": 3,
        "dtype": np.float32,
        "kind": "proprio",
    },
    ProprioKey.CAMERA_FRAME_CARTESIAN_TIP_POS.value: {
        "dimension": 3,
        "dtype": np.float32,
        "kind": "proprio",
    },
    ProprioKey.GRIPPER_STATE.value: {
        "dimension": 1,
        "dtype": np.int8,
        "kind": "gripper_binary",
    },
    Cameras.LEFT.value: {
        "channels": 3,
        "dtype": np.uint8,
        "kind": "rgb",
    },
    Cameras.RIGHT.value: {
        "channels": 3,
        "dtype": np.uint8,
        "kind": "rgb",
    },
    Cameras.DEPTH.value: {
        "channels": 1,
        "dtype": np.float32,
        "kind": "depth",
    },
    ObsKey.LANGUAGE.value: {
        "dtype": str,
        "kind": "language",
    },
    TSOObsKey.PHASE_LABEL.value: {
        "dimension": 1,
        "dtype": np.uint8,
        "kind": "label",
    },
}

LIBERO_LEROBOT_ZARR_SPEC: dict[str, dict] = {
    ProprioKey.EE_POS.value: {
        "dimension": 3,
        "dtype": np.float32,
        "kind": "proprio",
    },
    ProprioKey.EE_ORI.value: {
        "dimension": 3,
        "dtype": np.float32,
        "kind": "proprio",
    },
    ProprioKey.GRIPPER_STATE.value: {
        "dimension": 2,
        "dtype": np.float32,
        "kind": "gripper_continuous",
    },
    Cameras.AGENTVIEW.value: {
        "channels": 3,
        "dtype": np.uint8,
        "kind": "rgb",
    },
    Cameras.EYE_IN_HAND.value: {
        "channels": 3,
        "dtype": np.uint8,
        "kind": "rgb",
    },
    ObsKey.LANGUAGE.value: {
        "dtype": str,
        "kind": "language",
    },
    ProprioKey.EE_POS_ACTION.value: {
        "dimension": 3,
        "dtype": np.float32,
        "kind": "action",
    },
    ProprioKey.EE_ORI_ACTION.value: {
        "dimension": 3,
        "dtype": np.float32,
        "kind": "action",
    },
    ProprioKey.GRIPPER_STATE_ACTION.value: {
        "dimension": 1,
        "dtype": np.float32,
        "kind": "action",
    },
}

METAWORLD_ZARR_SPEC: dict[str, dict] = {
    ProprioKey.EE_POS.value: {
        "dimension": 3,
        "dtype": np.float32,
        "kind": "proprio",
    },
    ProprioKey.GRIPPER_STATE.value: {
        "dimension": 1,
        "dtype": np.float32,
        "kind": "gripper_continuous",
    },
    Cameras.AGENTVIEW.value: {
        "channels": 3,
        "dtype": np.uint8,
        "kind": "rgb",
    },
    ObsKey.LANGUAGE.value: {
        "dtype": str,
        "kind": "language",
    },
    ProprioKey.EE_POS_ACTION.value: {
        "dimension": 3,
        "dtype": np.float32,
        "kind": "action",
    },
    ProprioKey.GRIPPER_STATE_ACTION.value: {
        "dimension": 1,
        "dtype": np.float32,
        "kind": "action",
    },
}

ANT_ZARR_SPEC: dict[str, dict] = {
    ProprioKey.ANT_QPOS.value: {
        "dimension": 15,
        "dtype": np.float32,
        "kind": "proprio",
    },
    ProprioKey.ANT_QVEL.value: {
        "dimension": 14,
        "dtype": np.float32,
        "kind": "proprio",
    },
    ProprioKey.ANT_GOAL_COORDS.value: {
        "dimension": 8,
        "dtype": np.float32,
        "kind": "proprio",
    },
    ProprioKey.ANT_ACHIEVED.value: {
        "dimension": 4,
        "dtype": np.float32,
        "kind": "proprio",
    },
    ProprioKey.ANT_TORQUE_ACTION.value: {
        "dimension": 8,
        "dtype": np.float32,
        "kind": "action",
    },
}

BLOCK_PUSHING_ZARR_SPEC: dict[str, dict] = {
    ProprioKey.BLOCK_PUSH_BLOCK1_POS.value: {
        "dimension": 2,
        "dtype": np.float32,
        "kind": "proprio",
    },
    ProprioKey.BLOCK_PUSH_BLOCK1_ANGLE.value: {
        "dimension": 1,
        "dtype": np.float32,
        "kind": "proprio",
    },
    ProprioKey.BLOCK_PUSH_BLOCK2_POS.value: {
        "dimension": 2,
        "dtype": np.float32,
        "kind": "proprio",
    },
    ProprioKey.BLOCK_PUSH_BLOCK2_ANGLE.value: {
        "dimension": 1,
        "dtype": np.float32,
        "kind": "proprio",
    },
    ProprioKey.EE_POS.value: {
        "dimension": 2,
        "dtype": np.float32,
        "kind": "proprio",
    },
    ProprioKey.BLOCK_PUSH_EE_COMMANDED.value: {
        "dimension": 2,
        "dtype": np.float32,
        "kind": "proprio",
    },
    ProprioKey.BLOCK_PUSH_TARGET1_POS.value: {
        "dimension": 2,
        "dtype": np.float32,
        "kind": "proprio",
    },
    ProprioKey.BLOCK_PUSH_TARGET1_ANGLE.value: {
        "dimension": 1,
        "dtype": np.float32,
        "kind": "proprio",
    },
    ProprioKey.BLOCK_PUSH_TARGET2_POS.value: {
        "dimension": 2,
        "dtype": np.float32,
        "kind": "proprio",
    },
    ProprioKey.BLOCK_PUSH_TARGET2_ANGLE.value: {
        "dimension": 1,
        "dtype": np.float32,
        "kind": "proprio",
    },
    ProprioKey.EE_POS_ACTION.value: {
        "dimension": 2,
        "dtype": np.float32,
        "kind": "action",
    },
}

KITCHEN_ZARR_SPEC: dict[str, dict] = {
    ProprioKey.KITCHEN_ARM_QPOS.value: {
        "dimension": 9,
        "dtype": np.float32,
        "kind": "proprio",
    },
    ProprioKey.KITCHEN_OBJECT_QPOS.value: {
        "dimension": 21,
        "dtype": np.float32,
        "kind": "proprio",
    },
    ProprioKey.KITCHEN_TASK_GOAL.value: {
        "dimension": 7,
        "dtype": np.float32,
        "kind": "proprio",
    },
    Cameras.AGENTVIEW.value: {
        "channels": 3,
        "dtype": np.uint8,
        "kind": "rgb",
    },
    ProprioKey.KITCHEN_ARM_ACTION.value: {
        "dimension": 9,
        "dtype": np.float32,
        "kind": "action",
    },
}

PUSHT_ZARR_SPEC: dict[str, dict] = {
    ProprioKey.EE_POS.value: {
        "dimension": 2,
        "dtype": np.float32,
        "kind": "proprio",
    },
    ProprioKey.PUSHT_BLOCK_POS.value: {
        "dimension": 2,
        "dtype": np.float32,
        "kind": "proprio",
    },
    ProprioKey.PUSHT_BLOCK_ANGLE.value: {
        "dimension": 1,
        "dtype": np.float32,
        "kind": "proprio",
    },
    ProprioKey.PUSHT_KEYPOINTS.value: {
        "dimension": 18,
        "dtype": np.float32,
        "kind": "proprio",
    },
    ProprioKey.PUSHT_CONTACTS.value: {
        "dimension": 1,
        "dtype": np.float32,
        "kind": "proprio",
    },
    Cameras.AGENTVIEW.value: {
        "channels": 3,
        "dtype": np.uint8,
        "kind": "rgb",
    },
    ProprioKey.EE_POS_ACTION.value: {
        "dimension": 2,
        "dtype": np.float32,
        "kind": "action",
    },
}

UR3_ZARR_SPEC: dict[str, dict] = {
    ProprioKey.UR3_EE_POS.value: {
        "dimension": 2,
        "dtype": np.float32,
        "kind": "proprio",
    },
    ProprioKey.UR3_BLOCK1_POS.value: {
        "dimension": 2,
        "dtype": np.float32,
        "kind": "proprio",
    },
    ProprioKey.UR3_BLOCK2_POS.value: {
        "dimension": 2,
        "dtype": np.float32,
        "kind": "proprio",
    },
    ProprioKey.UR3_EE_TARGET_ACTION.value: {
        "dimension": 2,
        "dtype": np.float32,
        "kind": "action",
    },
}

SYNTHETIC_ZARR_SPEC: dict[str, dict] = {
    Cameras.AGENTVIEW.value: {
        "channels": 3,
        "dtype": np.uint8,
        "kind": "rgb",
    },
    ProprioKey.SYNTHETIC_POSITION.value: {
        "dimension": 2,
        "dtype": np.float32,
        "kind": "proprio",
    },
    SyntheticObsKey.CONTEXT.value: {
        "dimension": 3,
        "dtype": np.float32,
        "kind": "proprio",
    },
    SyntheticObsKey.MODE_ID.value: {
        "dimension": 1,
        "dtype": np.uint8,
        "kind": "label",
    },
    ProprioKey.SYNTHETIC_POSITION_ACTION.value: {
        "dimension": 2,
        "dtype": np.float32,
        "kind": "action",
    },
}

DATASET_SPECS: dict[str, dict[str, dict]] = {
    "ant": ANT_ZARR_SPEC,
    "block_pushing": BLOCK_PUSHING_ZARR_SPEC,
    "bowel_retraction": BOWEL_RETRACTION_ZARR_SPEC,
    "kitchen": KITCHEN_ZARR_SPEC,
    "libero_lerobot": LIBERO_LEROBOT_ZARR_SPEC,
    "metaworld": METAWORLD_ZARR_SPEC,
    "pusht": PUSHT_ZARR_SPEC,
    "synthetic": SYNTHETIC_ZARR_SPEC,
    "ur3": UR3_ZARR_SPEC,
}


def _generate_array_for_key(
    rng: np.random.Generator,
    spec: dict,
    total_timesteps: int,
    image_height: int,
    image_width: int,
) -> np.ndarray:
    """Generate a synthetic array for a single zarr key based on its spec."""
    kind = spec["kind"]
    dtype = spec["dtype"]

    match kind:
        case "rgb":
            return rng.integers(
                0, 255, (total_timesteps, image_height, image_width, spec["channels"])
            ).astype(dtype)
        case "depth":
            return rng.uniform(
                0.5, 5.0, (total_timesteps, image_height, image_width, spec["channels"])
            ).astype(dtype)
        case "proprio" | "action":
            return rng.standard_normal((total_timesteps, spec["dimension"])).astype(
                dtype
            )
        case "gripper_binary":
            return rng.integers(0, 2, (total_timesteps, spec["dimension"])).astype(
                dtype
            )
        case "gripper_continuous":
            return rng.uniform(0.0, 1.0, (total_timesteps, spec["dimension"])).astype(
                dtype
            )
        case "label":
            return rng.integers(0, 5, (total_timesteps, spec["dimension"])).astype(
                dtype
            )
        case "language":
            return ["pick up object"] * total_timesteps
        case _:
            raise ValueError(f"Unknown spec kind: {kind}")


@pytest.fixture
def hydra_config_dir() -> str:
    """Path to the hydra_configs directory."""
    return str(Path(__file__).parents[2] / "hydra_configs")


@pytest.fixture
def synthetic_zarr_factory(rng: np.random.Generator) -> Callable[..., str]:
    """Factory that creates a zarr store matching a dataset's zarr_meta schema.

    Returns:
        Callable that creates and populates a zarr store, returning its path.
    """

    def factory(
        dataset_type: str,
        zarr_path: str,
        image_height: int = 32,
        image_width: int = 32,
        num_episodes: int = 3,
        timesteps_per_episode: int = 15,
    ) -> str:
        if dataset_type not in DATASET_SPECS:
            raise ValueError(
                f"Unknown dataset_type '{dataset_type}'. "
                f"Expected one of {list(DATASET_SPECS.keys())}"
            )

        spec = DATASET_SPECS[dataset_type]
        total_timesteps = num_episodes * timesteps_per_episode

        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode="w")
        data_group = root.create_group("data")
        meta_group = root.create_group("meta")

        for key, key_spec in spec.items():
            array_data = _generate_array_for_key(
                rng=rng,
                spec=key_spec,
                total_timesteps=total_timesteps,
                image_height=image_height,
                image_width=image_width,
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
            [(i + 1) * timesteps_per_episode for i in range(num_episodes)],
            dtype=np.int64,
        )
        meta_group.create_array(
            "episode_ends",
            data=episode_ends,
            chunks=(num_episodes,),
        )

        return zarr_path

    return factory


HYDRA_CONFIG_DIR = str(Path(__file__).parents[2] / "hydra_configs")

DATASET_TYPE_TO_ZARR_SPEC: dict[str, str] = {
    "ant": "ant",
    "block_pushing": "block_pushing",
    "kitchen": "kitchen",
    "tso": "bowel_retraction",
    "libero": "libero_lerobot",
    "metaworld": "metaworld",
    "pusht": "pusht",
    "synthetic": "synthetic",
    "ur3": "ur3",
}

E2E_UNSUPPORTED_CONFIG_SUBSTRINGS = (
    "pi0",
    "smolvla",
)

E2E_FULL_COVERAGE_CONFIGS = (
    "end_to_end_training_runs/pusht/act_flow_rgb",
    "end_to_end_training_runs/pusht/act_flow_state",
    "end_to_end_training_runs/pusht/conditional_mmd_cwae_learned_prior_rgb",
    "end_to_end_training_runs/pusht/conditional_mmd_cwae_learned_prior_state",
    "end_to_end_training_runs/pusht/kl_cvae_learned_prior_rgb",
    "end_to_end_training_runs/pusht/kl_cvae_learned_prior_rgb_learned_var",
    "end_to_end_training_runs/pusht/kl_cvae_learned_prior_state",
    "end_to_end_training_runs/pusht/kl_cvae_learned_prior_state_learned_var",
    "end_to_end_training_runs/pusht/kl_cvae_rgb",
    "end_to_end_training_runs/pusht/kl_cvae_state",
    "end_to_end_training_runs/pusht/lat_flow_rgb",
    "end_to_end_training_runs/pusht/lat_flow_state",
    "end_to_end_training_runs/pusht/relaxed_conditional_sinkhorn_cwae_learned_prior_rgb",
    "end_to_end_training_runs/pusht/relaxed_conditional_sinkhorn_cwae_learned_prior_state",
    "end_to_end_training_runs/pusht/sinkhorn_cwae_fixed_gaussian_rgb",
    "end_to_end_training_runs/pusht/sinkhorn_cwae_fixed_gaussian_state",
    "end_to_end_training_runs/pusht/sinkhorn_cwae_learned_prior_rgb",
    "end_to_end_training_runs/pusht/sinkhorn_cwae_learned_prior_state",
    "end_to_end_training_runs/pusht/vq_vae_prior_rgb_800_200",
    "end_to_end_training_runs/pusht/vq_vae_prior_state_800_200",
    "end_to_end_training_runs/pusht/vq_vae_rgb",
    "end_to_end_training_runs/pusht/vq_vae_state",
    "end_to_end_training_runs/pusht/vq_vae_state_codes16",
)

E2E_REPRESENTATIVE_CONFIGS = (
    "end_to_end_training_runs/ant/act_flow_state",
    "end_to_end_training_runs/block_pushing/act_flow_state",
    "end_to_end_training_runs/bowel_retraction/action_transformer",
    "end_to_end_training_runs/kitchen/act_flow_rgb",
    "end_to_end_training_runs/libero_hdf5/action_transformer",
    "end_to_end_training_runs/libero_lerobot/action_transformer",
    "end_to_end_training_runs/libero_plus/vision_sweep/siglip2_base",
    "end_to_end_training_runs/metaworld/action_transformer",
    "end_to_end_training_runs/multimodal_peg_transfer/action_transformer",
    "end_to_end_training_runs/synthetic/bcat",
    "end_to_end_training_runs/ur3/act_flow_state",
)

E2E_EXTRA_ARCHITECTURE_CONFIGS = (
    "end_to_end_training_runs/libero_lerobot/gpt_transformer",
    "end_to_end_training_runs/libero_lerobot/flow_dit_cross_attention",
    "end_to_end_training_runs/libero_lerobot/flow_dit_multimodal",
    "end_to_end_training_runs/libero_lerobot/mode_act",
    "end_to_end_training_runs/bowel_retraction/discrete_detr",
    "end_to_end_training_runs/bowel_retraction/free_transformer",
    "end_to_end_training_runs/bowel_retraction/mixture_act",
    "end_to_end_training_runs/bowel_retraction/phase_act",
)

TINY_SCALAR_FIELDS: dict[str, int] = {
    "embedding_dimension": 16,
    "feedforward_dimension": 32,
    "number_of_heads": 2,
    "number_of_layers": 1,
    "number_of_encoder_layers": 1,
    "number_of_decoder_layers": 2,
    "diffusion_step_embed_dim": 16,
}

TINY_LIST_FIELDS: dict[str, str] = {
    "down_dimensions": "[16,32]",
}


def discover_e2e_configs() -> list[str]:
    """Return a bounded smoke matrix for expensive endpoint e2e tests."""
    selected_configs = (
        E2E_FULL_COVERAGE_CONFIGS
        + E2E_REPRESENTATIVE_CONFIGS
        + E2E_EXTRA_ARCHITECTURE_CONFIGS
    )
    runnable_configs = []
    seen_configs = set()
    for config_name in selected_configs:
        if config_name in seen_configs:
            continue
        seen_configs.add(config_name)
        if any(
            substring in config_name.lower()
            for substring in E2E_UNSUPPORTED_CONFIG_SUBSTRINGS
        ):
            continue
        if not (Path(HYDRA_CONFIG_DIR) / f"{config_name}.yaml").exists():
            continue
        runnable_configs.append(config_name)
    return runnable_configs


def resolve_dataset_type(config_name: str) -> str:
    """Compose config and read dataset_type from the resolved schema."""
    with initialize_config_dir(config_dir=HYDRA_CONFIG_DIR, version_base=None):
        cfg = compose(config_name=config_name)
        raw_type = cfg.task.dataset_schema.dataset_type
        return DATASET_TYPE_TO_ZARR_SPEC[raw_type]


def build_tiny_overrides(config_name: str) -> list[str]:
    """Compose config and auto-discover decoder + encoder fields to override."""
    overrides = []
    with initialize_config_dir(config_dir=HYDRA_CONFIG_DIR, version_base=None):
        cfg = compose(config_name=config_name)
        decoder_dict = OmegaConf.to_container(cfg.policy.decoder, resolve=False)
        encoders_dict = OmegaConf.to_container(
            cfg.policy.encoding_pipeline.encoders, resolve=False
        )

    for field, value in TINY_SCALAR_FIELDS.items():
        if field in decoder_dict:
            overrides.append(f"policy.decoder.{field}={value}")

    for field, value in TINY_LIST_FIELDS.items():
        if field in decoder_dict:
            overrides.append(f"++policy.decoder.{field}={value}")

    for encoder_name, encoder_cfg in encoders_dict.items():
        if "backbone" in encoder_cfg:
            target = encoder_cfg.get("_target_", "")
            if "conditional_cnn" in target:
                overrides.append(
                    f"policy.encoding_pipeline.encoders.{encoder_name}"
                    f".backbone=${{rgb_backbone:RESNET18}}"
                )
            elif "flat" in target.lower():
                overrides.append(
                    f"policy.encoding_pipeline.encoders.{encoder_name}"
                    f".backbone=${{rgb_backbone:DEIT_TINY}}"
                )
            else:
                overrides.append(
                    f"policy.encoding_pipeline.encoders.{encoder_name}"
                    f".backbone=${{rgb_backbone:MOBILENETV4_SMALL_050}}"
                )
        if "model_name" in encoder_cfg:
            target = encoder_cfg.get("_target_", "")
            if "two_tower_vlm" in target.lower():
                overrides.append(
                    f"policy.encoding_pipeline.encoders.{encoder_name}"
                    f".model_name=${{vlm_model:CLIP_VITB32}}"
                )
            elif "paligemma" not in target.lower() and "smolvlm" not in target.lower():
                overrides.append(
                    f"policy.encoding_pipeline.encoders.{encoder_name}"
                    f".model_name=${{language_model:ALBERT_BASE}}"
                )
        target = encoder_cfg.get("_target_", "")
        if "geometric_rgbd" not in target and "proprioceptive" not in target:
            overrides.append(
                f"++policy.encoding_pipeline.encoders.{encoder_name}.frozen=true"
            )

    # Match observation tokenizer model to the encoder type
    tokenization = OmegaConf.to_container(
        cfg.task.dataloader.get("tokenization", OmegaConf.create({})),
        resolve=False,
    )
    has_two_tower_vlm = any(
        "two_tower_vlm" in enc.get("_target_", "").lower()
        for enc in encoders_dict.values()
    )
    has_generative_vlm = any(
        any(
            keyword in enc.get("_target_", "").lower()
            for keyword in ("paligemma", "smolvlm")
        )
        for enc in encoders_dict.values()
    )
    if tokenization:
        obs_tok = tokenization.get("observation_tokenizer", {})
        if obs_tok and "tokenizer_model" in obs_tok:
            if has_two_tower_vlm:
                overrides.append(
                    "task.dataloader.image_norm_type=${image_norm_type:CLIP}"
                )
                overrides.append(
                    "task.dataloader.tokenization.observation_tokenizer"
                    ".tokenizer_model=${vlm_model:CLIP_VITB32}"
                )
            elif not has_generative_vlm:
                overrides.append(
                    "task.dataloader.tokenization.observation_tokenizer"
                    ".tokenizer_model=${language_model:ALBERT_BASE}"
                )

    return overrides


def get_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket_module.socket(socket_module.AF_INET, socket_module.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def start_mock_observation_server(
    observation_space: ObservationSpace,
    port: int,
) -> SocketServer:
    """Create and start a ZMQ SocketServer that returns observations matching the policy's space.

    Args:
        observation_space: The policy's observation space (cameras, proprio, language).
        port: TCP port for the server.

    Returns:
        Running SocketServer instance (call .stop() when done).
    """
    camera_keys = list(observation_space.cameras.keys())
    state_observations = observation_space.numerical_observations
    has_language = ObsKey.LANGUAGE.value in observation_space.observations_metadata
    server_rng = np.random.default_rng(seed=777)

    def handle_get_observation(request_data: dict) -> tuple[bool, dict]:
        requested = request_data.get(InferenceRequestKey.REQUESTED_KEYS.value, [])
        compression = request_data.get(
            InferenceRequestKey.COMPRESSION_TYPE.value,
            CompressionType.RAW.value,
        )
        response: dict = {
            TransportKey.STATUS.value: ServerStatus.WAITING_ACTION.value,
            InferenceResponseKey.COMPRESSION_TYPE.value: compression,
        }
        for key in camera_keys:
            if key in requested:
                channels = observation_space.cameras[key].channels
                image = server_rng.integers(0, 256, (64, 64, channels), dtype=np.uint8)
                response[key] = compress_array(
                    image, method=compression, as_base64=True
                )
        for key, metadata in state_observations.items():
            if key in requested:
                response[key] = (
                    server_rng.standard_normal(metadata.dimension)
                    .astype(np.float32)
                    .tolist()
                )
        if has_language and ObsKey.LANGUAGE.value in requested:
            response[ObsKey.LANGUAGE.value] = "pick up the object"
        return True, response

    def handle_send_action(request_data: dict) -> tuple[bool, dict]:
        return True, {}

    def handle_register(request_data: dict) -> tuple[bool, dict]:
        return True, {}

    server = SocketServer(
        ip_address="127.0.0.1",
        port=port,
        max_workers=1,
    )
    server.add_route(
        ServerRoute.GET_OBSERVATION.value,
        handle_get_observation,
        blocking=True,
    )
    server.add_route(
        ServerRoute.SEND_ACTION.value,
        handle_send_action,
        blocking=True,
    )
    server.add_route(
        ServerRoute.REGISTER_CLIENT.value,
        handle_register,
        blocking=True,
    )

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server
