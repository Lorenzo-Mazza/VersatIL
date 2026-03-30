"""Download Pi0/Pi0.5/SmolVLA LIBERO checkpoints from HuggingFace.

Downloads the safetensors weights and config files for each model.
Does NOT convert to VersatIL format — that requires mapping the state dict
keys which depends on the exact VersatIL decoder configuration.

Usage:
    python scripts/download_vla_checkpoints.py [--output-dir ./checkpoints/vla]
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

CHECKPOINTS = {
    "pi0_libero": "lerobot/pi0_libero_base",
    "pi05_libero": "lerobot/pi05_libero_base",
    "smolvla_libero": "HuggingFaceVLA/smolvla_libero",
}

FILES_TO_DOWNLOAD = [
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
]


def download_checkpoint(
    model_id: str,
    output_directory: Path,
    checkpoint_name: str,
) -> None:
    """Download a single checkpoint from HuggingFace Hub.

    Args:
        model_id: HuggingFace model ID.
        output_directory: Base output directory.
        checkpoint_name: Subdirectory name for this checkpoint.
    """
    checkpoint_directory = output_directory / checkpoint_name
    checkpoint_directory.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {model_id} → {checkpoint_directory}")
    for filename in FILES_TO_DOWNLOAD:
        try:
            downloaded_path = hf_hub_download(
                repo_id=model_id,
                filename=filename,
                local_dir=checkpoint_directory,
            )
            print(f"  {filename} → {downloaded_path}")
        except Exception as error:
            print(f"  {filename} — skipped ({error})")

    config_path = checkpoint_directory / "config.json"
    if config_path.exists():
        with open(config_path) as config_file:
            config = json.load(config_file)
        print(f"  Model type: {config.get('model_type', 'unknown')}")
        policy_config = config.get("policy", config)
        if "action_feature" in policy_config:
            print(f"  Action dim: {policy_config['action_feature'].get('shape', '?')}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Pi0/Pi0.5/SmolVLA LIBERO checkpoints"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/vla"),
        help="Output directory for downloaded checkpoints",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(CHECKPOINTS.keys()) + ["all"],
        default=["all"],
        help="Which models to download",
    )
    args = parser.parse_args()

    models_to_download = (
        CHECKPOINTS
        if "all" in args.models
        else {name: CHECKPOINTS[name] for name in args.models}
    )

    for checkpoint_name, model_id in models_to_download.items():
        download_checkpoint(
            model_id=model_id,
            output_directory=args.output_dir,
            checkpoint_name=checkpoint_name,
        )

    print(f"\nDone. Checkpoints saved to {args.output_dir}")


if __name__ == "__main__":
    main()
