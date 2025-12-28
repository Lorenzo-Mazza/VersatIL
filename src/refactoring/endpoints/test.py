"""Inference endpoint for real-time model deployment.

This module provides the main entry point for running the inference client
with a trained policy model for real-time robot control.
"""
import argparse

import torch

from refactoring.inference.client import TSOPolicyClient


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Run inference client for real-time robot control")
    parser.add_argument(
        "--model-server-address",
        type=str,
        default="localhost",
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
        required=True,
        help="Path to checkpoint torch model name",
    )

    # latest-39.ckpt
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
    args = parser.parse_args()
    args.temporal_agg = bool(args.temporal_agg)
    return args


def main():
    """Main entry point for inference endpoint."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cpu"):
        print(
            "Warning: Running on CPU, this may be slow or go OOM. "
            "Consider using a GPU for better performance."
        )

    client = TSOPolicyClient(
        model_server_address=args.model_server_address,
        model_server_port=args.model_server_port,
        checkpoint_path=args.checkpoint_path,
        checkpoint_name=args.checkpoint_name, 
        temporal_agg=args.temporal_agg,
        device=device,
        update_rate_hz=args.update_frequency,
    )

    try:
        client.update_loop()
    except KeyboardInterrupt:
        print("Shutting down client...")
        client.shutdown()
    except Exception as e:
        print(f"Error: {e}")
        client.shutdown()


if __name__ == "__main__":
    main()
