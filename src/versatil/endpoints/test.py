"""Inference endpoint for real-time model deployment."""

import argparse
import logging
import os

import torch

from versatil.inference.inference_client import InferenceClient
from versatil.inference.policy_loading.compressed_loader import CompressedPolicyLoader
from versatil.inference.policy_loading.float_loader import PolicyLoader
from versatil.inference.protocol import PolicyInference
from versatil.inference.socket_transport import (
    SocketActionTransport,
    SocketObservationTransport,
)
from versatil.post_training_compression.constants import CompressionFilename
from versatil.training.constants import CheckpointFilename


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="VersatIL Inference Client")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the checkpoint directory.",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default=CheckpointFilename.DEFAULT_CHECKPOINT.value,
        help="Name of the checkpoint file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on (e.g., cuda, cpu).",
    )
    parser.add_argument(
        "--model_server_address",
        type=str,
        default="127.0.0.1",
        help="Address of the environment server.",
    )
    parser.add_argument(
        "--model_server_port",
        type=int,
        default=5555,
        help="Port of the environment server.",
    )
    parser.add_argument(
        "--temporal_aggregation",
        action="store_true",
        help="Enable temporal ensemble (query every step, average overlapping chunks).",
    )
    parser.add_argument(
        "--action_execution_horizon",
        type=int,
        default=None,
        help="Actions to execute per chunk when temporal aggregation is off. Default: prediction_horizon.",
    )
    parser.add_argument(
        "--update_frequency",
        type=float,
        default=None,
        help="Update rate in Hz (for robot deployment). None for simulation.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000000,
        help="Maximum steps per episode.",
    )
    parser.add_argument(
        "--temporal_max_timesteps",
        type=int,
        default=800,
        help="Maximum number of steps tracked by temporal aggregation.",
    )
    parser.add_argument(
        "--timing_log",
        action="store_true",
        help="Log per-step timing breakdown.",
    )
    parser.add_argument(
        "--no_compile",
        action="store_true",
        help="Disable torch.compile model optimization.",
    )
    return parser.parse_args()


def load_policy(
    checkpoint_path: str,
    device: torch.device,
    checkpoint_name: str = CheckpointFilename.DEFAULT_CHECKPOINT.value,
    compile_model: bool = True,
) -> PolicyInference:
    """Load a policy for inference, auto-detecting compressed checkpoints.

    Inspects the checkpoint directory for compression/quantization metadata.
    If found, returns a CompressedPolicyLoader. Otherwise returns a
    standard PolicyLoader.

    Args:
        checkpoint_path: Path to the checkpoint directory.
        device: Device to load the model onto.
        checkpoint_name: Name of the checkpoint file (for float policies).
        compile_model: Whether to compile the model with torch.compile.

    Returns:
        A PolicyInference-compatible loader.
    """
    compression_metadata = os.path.join(
        checkpoint_path, CompressionFilename.COMPRESSION_METADATA.value
    )
    if os.path.exists(compression_metadata):
        return CompressedPolicyLoader(
            device=device,
            checkpoint_path=checkpoint_path,
            compile_model=compile_model,
        )
    else:
        return PolicyLoader(
            device=device,
            checkpoint_path=checkpoint_path,
            checkpoint_name=checkpoint_name,
            compile_model=compile_model,
        )


def main() -> None:
    """Main entry point for inference endpoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(module)s %(levelname)s %(message)s",
        force=True,
    )
    for handler in logging.root.handlers:
        handler.flush = handler.stream.flush
    args = parse_args()
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device == torch.device("cpu"):
        logging.warning("Running on CPU. Consider using a GPU for better performance.")

    policy_loader = load_policy(
        checkpoint_path=args.checkpoint_path,
        device=device,
        checkpoint_name=args.checkpoint_name,
        compile_model=not args.no_compile,
    )

    observation_transport = SocketObservationTransport(
        server_address=args.model_server_address,
        server_port=args.model_server_port,
    )
    action_transport = SocketActionTransport(
        server_address=args.model_server_address,
        server_port=args.model_server_port,
    )

    client = InferenceClient(
        policy_loader=policy_loader,
        observation_transport=observation_transport,
        action_transport=action_transport,
        temporal_aggregation=args.temporal_aggregation,
        action_execution_horizon=args.action_execution_horizon,
        max_timesteps=args.temporal_max_timesteps,
        timing_log=args.timing_log,
        update_rate_hz=args.update_frequency,
    )

    try:
        client.run_episode(max_steps=args.max_steps)
    except KeyboardInterrupt:
        logging.info("Shutting down client...")
    finally:
        client.shutdown()


if __name__ == "__main__":
    main()
