"""Configuration for experiment tracking and setup."""
from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class ExperimentConfig:
    """Experiment tracking and setup configuration."""
    name: str = MISSING  # Required - will be set per experiment
    seed: int = 42
    #: Folder to save checkpoints and logs
    checkpoint_folder: str = MISSING  # Required - user must specify
    resume_from: str | None = None
    use_wandb: bool = True
    wandb_project: str | None = None
    wandb_entity: str | None = None
    device: str = "cuda"
    distributed: bool = False

    # Checkpointing and validation
    checkpoint_every: int = 100
    val_every: int = 1
    plot_every: int = 200
