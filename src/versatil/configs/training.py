from dataclasses import dataclass, field

from omegaconf import MISSING

from versatil.training.constants import CompileMode


@dataclass
class ParameterGroupConfig:
    """Configuration for a parameter group with specific learning rate."""

    name: str  # e.g., "backbone", "encoder", "decoder", "router"
    lr: float
    weight_decay: float | None = None  # Override global weight decay
    params_pattern: str | None = None  # Pattern to match parameter names


@dataclass
class ProgressiveFreezingConfig:
    """Epoch-triggered parameter freezing rule.

    Patterns are regular expressions matched against ``policy.named_parameters()``
    names, for example ``^algorithm\\.prior\\.``.

    Args:
        epoch: Zero-based epoch where this rule becomes active.
        trainable_patterns: If non-empty, only matching parameters remain trainable.
        frozen_patterns: Matching parameters are frozen after trainable patterns are
            applied. Use this for additive freezing rules.
        eval_frozen_modules: Put modules whose parameters are all frozen in eval mode.
        log: Log trainable/frozen parameter counts when the rule is active.
    """

    epoch: int
    trainable_patterns: list[str] = field(default_factory=list)
    frozen_patterns: list[str] = field(default_factory=list)
    eval_frozen_modules: bool = True
    log: bool = True


@dataclass
class OptimizerConfig:
    """Base optimizer configuration."""

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

    target_class: str = "torch.optim.Adam"
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

    # Learning rate schedule (uses transformers.get_scheduler)
    # https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.get_scheduler
    lr_schedule: str | None = (
        None  # One of https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.SchedulerType
    )
    lr_warmup_steps: int = 5000
    lr_scheduler_kwargs: dict[str, float] = field(default_factory=dict)

    # Exponential Moving Average (EMA) of model parameters
    use_ema: bool = True
    ema_power: float = 0.75

    # Stochastic Weight Averaging (SWA)
    swa_lrs: float | None = None  # If not None, enables SWA with this learning rate
    swa_epoch_start: float = 0.5  # Start SWA at this fraction of total epochs (default: 80% through training)
    swa_annealing_epochs: int = (
        10  # Number of epochs to anneal learning rate to swa_lrs
    )

    compile: bool = False
    compile_mode: str = CompileMode.DEFAULT.value

    tune_lr: bool = (
        False  # If True, automatically find optimal learning rate before training
    )
    early_stopping_patience: int | None = (
        10  # Validation checks with no improvement before stopping. None disables early stopping.
    )

    # ReduceLROnPlateau - reduce learning rate when validation loss plateaus
    reduce_lr_on_plateau: bool = False  # If True, reduce LR when val_loss plateaus
    reduce_lr_patience: int = (
        10  # Number of epochs with no improvement before reducing LR
    )
    reduce_lr_cooldown: int = 10  # Number of epochs to wait after LR reduction before resuming normal operation

    # Progressive parameter freezing/unfreezing schedule. Each entry is applied
    # at the start of the configured epoch and remains active until another
    # entry becomes active.
    progressive_freezing: list[ProgressiveFreezingConfig] = field(default_factory=list)
