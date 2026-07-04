"""Configuration for experiment tracking and setup."""

from dataclasses import dataclass

from omegaconf import MISSING

from versatil.training.constants import Float32MatmulPrecision, PrecisionType


@dataclass
class ExperimentConfig:
    """Experiment tracking and setup configuration.

    Attributes:
        name: Human-readable name.
        seed: Random seed for reproducibility.
        checkpoint_folder: Root directory receiving run checkpoints.
        resume_from: Checkpoint path training resumes from, or null.
        use_wandb: Whether metrics are logged to Weights & Biases.
        wandb_project: Weights & Biases project name.
        wandb_entity: Weights & Biases entity, or null for the default.
        device: Torch device for the module.
        distributed: Whether distributed training is enabled.
        precision: Lightning precision setting for training.
        float32_matmul_precision: torch.set_float32_matmul_precision mode.
        checkpoint_every: Epoch interval between periodic checkpoints.
        save_checkpoints: Whether checkpoints are written at all.
        val_every: Epoch interval between validation runs.
        plot_every: Epoch interval between figure-logging callbacks.
        validate_loss_keys: Whether loss modules are validated against decoder output
            keys.
    """

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
    save_checkpoints: bool = (
        True  #: When False, skip ModelCheckpoint callbacks entirely.
    )
    val_every: int = 1
    plot_every: int = 200
    validate_loss_keys: bool = True
