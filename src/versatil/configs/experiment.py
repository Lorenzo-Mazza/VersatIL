"""Configuration for experiment tracking and setup."""
from dataclasses import dataclass

from omegaconf import MISSING

from versatil.training.constants import Float32MatmulPrecision, PrecisionType


@dataclass
class ExperimentConfig:
    """Experiment tracking and setup configuration."""

    name: str = MISSING
    seed: int = 42
    checkpoint_folder: str = MISSING
    resume_from: str | None = None
    use_wandb: bool = True
    wandb_project: str | None = None
    wandb_entity: str | None = None
    device: str = "cuda"
    distributed: bool = False
    precision: str = PrecisionType.FP16_MIXED.value
    float32_matmul_precision: str | None = (
        Float32MatmulPrecision.MEDIUM.value
    )  #: Float32 matmul precision for Tensor Cores (None to disable)
    checkpoint_every: int = 100
    val_every: int = 1
    plot_every: int = 200
