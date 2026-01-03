"""Configuration for experiment tracking and setup."""
from dataclasses import dataclass

from omegaconf import MISSING

from refactoring.training.constants import Float32MatmulPrecision, PrecisionType


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
    #: PyTorch Lightning precision setting
    precision: str = PrecisionType.FP16_MIXED.value
    #: Float32 matmul precision for Tensor Cores (None to disable)
    #: "medium" enables TF32 on Ampere+ GPUs for ~8x speedup with minimal precision loss
    float32_matmul_precision: str | None = Float32MatmulPrecision.MEDIUM.value
    # Checkpointing and validation
    checkpoint_every: int = 100
    val_every: int = 1
    plot_every: int = 200
