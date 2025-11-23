from dataclasses import dataclass, field, MISSING


@dataclass
class ParameterGroupConfig:
    """Configuration for a parameter group with specific learning rate."""
    name: str  # e.g., "backbone", "encoder", "decoder", "router"
    lr: float
    weight_decay: float | None = None  # Override global weight decay
    params_pattern: str | None = None  # Pattern to match parameter names


@dataclass
class OptimizerConfig:
    """Base optimizer configuration.
    """
    target_class: str = MISSING
    # Base learning rate (required by all optimizers)
    lr: float = 1e-4

    # Parameter groups with different learning rates
    param_groups: list[ParameterGroupConfig] = field(default_factory=list)


@dataclass
class AdamWConfig(OptimizerConfig):
    """Configuration for torch.optim.AdamW optimizer."""
    target_class: str = "torch.optim.AdamW"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False


@dataclass
class AdamConfig(OptimizerConfig):
    """Configuration for torch.optim.Adam optimizer."""
    target_class: str  = "torch.optim.Adam"
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    amsgrad: bool = False


@dataclass
class SGDConfig(OptimizerConfig):
    """Configuration for torch.optim.SGD optimizer."""
    target_class: str = "torch.optim.SGD"
    lr: float = 1e-2
    momentum: float = 0.0
    weight_decay: float = 0.0
    dampening: float = 0.0
    nesterov: bool = False


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    num_epochs: int = 100
    gradient_accumulate_every: int = 1

    # Optimizer (defaults to AdamW)
    optimizer: OptimizerConfig = field(default_factory=AdamWConfig)

    # Gradient clipping
    clip_gradient_norm: bool = False
    clip_max_norm: float = 0.1

    # Learning rate schedule
    lr_schedule: str | None = None  # "cosine", "linear", None
    lr_warmup_steps: int = 5000

    # EMA
    use_ema: bool = True
    ema_power: float = 0.75

    # Stochastic Weight Averaging (SWA)
    swa_lrs: float | None = None  # If not None, enables SWA with this learning rate
    swa_epoch_start: float = 0.5  # Start SWA at this fraction of total epochs (default: 80% through training)
    swa_annealing_epochs: int = 10  # Number of epochs to anneal learning rate to swa_lrs

    tune_lr: bool = False  # If True, automatically find optimal learning rate before training
    early_stopping_patience: int = 10  # Number of validation checks with no improvement to stop training

    # ReduceLROnPlateau - reduce learning rate when validation loss plateaus
    reduce_lr_on_plateau: bool = False  # If True, reduce LR when val_loss plateaus
    reduce_lr_patience: int = 10  # Number of epochs with no improvement before reducing LR
    reduce_lr_cooldown: int = 10 # Number of epochs to wait after LR reduction before resuming normal operation
