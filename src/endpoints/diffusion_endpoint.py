import sys
import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from legacy_config import DiffusionConfig
from workspace import DiffusionWorkspace


def main(config=None):
    if config is None:
        # Create default config
        config = DiffusionConfig()

    config.distributed_training = os.getenv("CONFIG_DISTRIBUTED_TRAINING", "false").lower() == "true"

    # Print configuration
    print("\nConfiguration:")
    for key, value in vars(config).items():
        if not key.startswith('_'):  # Skip internal attributes
            print(f"{key}: {value}")

    # Create workspace and run
    workspace = DiffusionWorkspace(config)

    if config.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {config.resume_from_checkpoint}")
        workspace.load_checkpoint(config.resume_from_checkpoint)

    workspace.run()


if __name__ == "__main__":
    main()