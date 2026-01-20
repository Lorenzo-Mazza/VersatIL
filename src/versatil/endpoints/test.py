"""Inference endpoint for real-time model deployment.

This module provides the main entry point for running the inference client
with a trained policy model for real-time robot control.
"""
import argparse
import enum
import os

import torch
from omegaconf import OmegaConf

from versatil.inference.tso_client import TSOPolicyClient
from versatil.inference.libero_client import LiberoClient


class ClientType(enum.Enum):
    """Enum for policy client types."""

    TSO = "tso"
    LIBERO = "libero"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run inference client for real-time robot control"
    )
    parser.add_argument(
        "--model-server-address",
        type=str,
        default="127.0.0.1",
        help="Address of the model server",
    )
    parser.add_argument(
        "--model-server-port",
        type=int,
        default=5555,
        help="Port of the model server",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="last.ckpt",
        help="Name of checkpoint torch file",
    )
    parser.add_argument(
        "--temporal-agg",
        type=int,
        default=1,
        choices=(0, 1),
        help="1 = use temporal aggregation for actions, 0 = no temporal aggregation",
    )
    parser.add_argument(
        "--update-frequency",
        type=float,
        default=None,
        help="Update frequency in Hz (overrides config file)",
    )
    parser.add_argument(
        "--enable-logging",
        type=int,
        default=0,
        choices=(0, 1),
        help="1 = enable debug logging, 0 = disable",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (e.g., cuda:0). Auto-detected if not specified.",
    )
    args = parser.parse_args()
    args.temporal_agg = bool(args.temporal_agg)
    args.enable_logging = bool(args.enable_logging)
    return args


def detect_client_type(checkpoint_path: str) -> str:
    """Detect client type from dataset schema in config.

    Args:
        checkpoint_path: Path to checkpoint directory containing config.yaml

    Returns:
        string with the client type.

    Raises:
        ValueError: If dataset schema is unknown
    """
    config_path = os.path.join(checkpoint_path, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    config = OmegaConf.load(config_path)
    target = config.task.dataset_schema._target_
    if "libero" in target.lower():
        return ClientType.LIBERO.value
    elif "bowel_retraction" in target.lower():
        return ClientType.TSO.value
    else:
        raise ValueError(
            f"Unknown dataset schema: {target}. Cannot determine client type."
        )


def main():
    """Main entry point for inference endpoint."""
    args = parse_args()
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device == torch.device("cpu"):
        print(
            "Warning: Running on CPU, this may be slow or go OOM. "
            "Consider using a GPU for better performance."
        )

    client_type = detect_client_type(args.checkpoint_path)
    print(f"Detected client type: {client_type}")

    if client_type == ClientType.LIBERO.value:
        client = LiberoClient(
            device=device,
            checkpoint_path=args.checkpoint_path,
            checkpoint_name=args.checkpoint_name,
            model_server_address=args.model_server_address,
            model_server_port=args.model_server_port,
            temporal_agg=args.temporal_agg,
            enable_logging=args.enable_logging,
        )
    elif client_type == ClientType.TSO.value:
        client = TSOPolicyClient(
            device=device,
            checkpoint_path=args.checkpoint_path,
            checkpoint_name=args.checkpoint_name,
            model_server_address=args.model_server_address,
            model_server_port=args.model_server_port,
            temporal_agg=args.temporal_agg,
            update_rate_hz=args.update_frequency,
        )
    else:
        raise ValueError(f"Unrecognized client type : {client_type}")

    try:
        client.update_loop()
    except KeyboardInterrupt:
        print("Shutting down client...")
        client.shutdown()
    except Exception as e:
        print(f"Error: {e}")
        client.shutdown()
        raise


if __name__ == "__main__":
    main()
