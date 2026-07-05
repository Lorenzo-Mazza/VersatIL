"""Hydra-based endpoint for real-time model deployment."""

import logging
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from versatil.common.logging import override_log_format
from versatil.configs.paths import get_hydra_configs_dir
from versatil.inference.inference_client import InferenceClient
from versatil.inference.policy_runtime.base import PolicyRuntime
from versatil.inference.policy_runtime.compressed_runtime import (
    CompressedPolicyRuntime,
)
from versatil.inference.policy_runtime.float_runtime import FloatPolicyRuntime
from versatil.inference.socket_transport import (
    SocketActionTransport,
    SocketObservationTransport,
)
from versatil.post_training_compression.constants import CompressionFilename
from versatil.training.constants import CheckpointFilename

EXPERIMENTS_DIR = get_hydra_configs_dir()


def load_policy(
    checkpoint_path: str,
    device: torch.device,
    checkpoint_name: str = CheckpointFilename.DEFAULT_CHECKPOINT.value,
    compile_model: bool = True,
) -> PolicyRuntime:
    """Load a policy for inference, auto-detecting compressed checkpoints.

    Inspects the checkpoint directory for compression/quantization metadata.
    If found, returns a compressed policy runtime. Otherwise returns a
    floating-point policy runtime.

    Args:
        checkpoint_path: Path to the checkpoint directory.
        device: Device to load the model onto.
        checkpoint_name: Name of the checkpoint file (for float policies).
        compile_model: Whether to compile the model with torch.compile.

    Returns:
        Runtime capable of policy inference.
    """
    compression_metadata = os.path.join(
        checkpoint_path, CompressionFilename.COMPRESSION_METADATA.value
    )
    if os.path.exists(compression_metadata):
        return CompressedPolicyRuntime(
            device=device,
            checkpoint_path=checkpoint_path,
            compile_model=compile_model,
        )
    else:
        return FloatPolicyRuntime(
            device=device,
            checkpoint_path=checkpoint_path,
            checkpoint_name=checkpoint_name,
            compile_model=compile_model,
        )


@hydra.main(
    version_base=None,
    config_path=str(EXPERIMENTS_DIR),
    config_name="end_to_end_deploy/default.yaml",
)
def main(config: DictConfig) -> None:
    """Run the deployment endpoint.

    Args:
        config: Hydra configuration for the deployment client.
    """
    override_log_format()
    logging.info("Deployment endpoint")
    logging.info(OmegaConf.to_yaml(config))

    if config.device is not None:
        device = torch.device(config.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cpu"):
        logging.warning("Running on CPU. Consider using a GPU for better performance.")

    policy_runtime = load_policy(
        checkpoint_path=config.checkpoint_path,
        device=device,
        checkpoint_name=config.checkpoint_name,
        compile_model=config.compile_model,
    )

    observation_transport = SocketObservationTransport(
        server_address=config.client.model_server_address,
        server_port=config.client.model_server_port,
        request_timeout_seconds=config.client.request_timeout_seconds,
    )
    action_transport = SocketActionTransport(
        server_address=config.client.model_server_address,
        server_port=config.client.model_server_port,
        request_timeout_seconds=config.client.request_timeout_seconds,
    )

    client = InferenceClient(
        policy_runtime=policy_runtime,
        observation_transport=observation_transport,
        action_transport=action_transport,
        temporal_aggregation=config.client.temporal_aggregation,
        action_execution_horizon=config.client.action_execution_horizon,
        compression_type=config.client.compression_type,
        max_timesteps=config.client.temporal_max_timesteps,
        timing_log=config.client.timing_log,
        update_rate_hz=config.client.update_rate_hz,
    )

    try:
        client.run_episode(max_steps=config.max_steps)
    except KeyboardInterrupt:
        logging.info("Shutting down client...")
    finally:
        client.shutdown()


if __name__ == "__main__":
    main()
