"""Inference endpoint for real-time model deployment."""
import argparse
import logging
import os

import torch
from omegaconf import OmegaConf

from versatil.data.constants import DatasetType
from versatil.inference.inference_client import InferenceClient
from versatil.inference.policy_loader import PolicyLoader
from versatil.inference.socket_transport import (
    SocketActionTransport,
    SocketObservationTransport,
)


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
        default="last.ckpt",
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
        help="Enable temporal aggregation.",
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
        default=1000,
        help="Maximum steps per episode.",
    )
    parser.add_argument(
        "--timing_log",
        action="store_true",
        help="Log per-step timing breakdown.",
    )
    return parser.parse_args()


def main():
    """Main entry point for inference endpoint."""
    args = parse_args()
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device == torch.device("cpu"):
        logging.warning(
            "Running on CPU. Consider using a GPU for better performance."
        )

    policy_loader = PolicyLoader(
        device=device,
        checkpoint_path=args.checkpoint_path,
        checkpoint_name=args.checkpoint_name,
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
