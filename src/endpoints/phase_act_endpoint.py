from legacy_config import PhaseACTConfig
from workspace import PhaseACTWorkspace


def main(config=None):
    if config is None:
        # Create default config
        config = PhaseACTConfig()

    # Print configuration
    print("\nConfiguration:")
    for key, value in vars(config).items():
        if not key.startswith("_"):  # Skip internal attributes
            print(f"{key}: {value}")
    print()
    # Create workspace and run
    workspace = PhaseACTWorkspace(config)

    if config.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {config.resume_from_checkpoint}")
        workspace.load_checkpoint(config.resume_from_checkpoint)

    workspace.run()


if __name__ == "__main__":
    main()
