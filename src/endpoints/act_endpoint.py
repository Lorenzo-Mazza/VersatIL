import sys
import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)


from legacy_config import ACTConfig
from workspace import ACTWorkspace

def main(config=None):
    if config is None:
        # Create default config
        config = ACTConfig()

    # Print configuration
    print("\nConfiguration:")
    for key, value in vars(config).items():
        if not key.startswith('_'):  # Skip internal attributes
            print(f"{key}: {value}")
    print()
    # Create workspace and run
    workspace = ACTWorkspace(config)

    if config.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {config.resume_from_checkpoint}")
        workspace.load_checkpoint(config.resume_from_checkpoint)

    workspace.run()


if __name__ == "__main__":
    main()
