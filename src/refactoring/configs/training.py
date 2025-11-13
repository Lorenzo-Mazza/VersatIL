from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class ParameterGroupConfig:
    """Configuration for a parameter group with specific learning rate."""
    name: str  # e.g., "backbone", "encoder", "decoder", "router"
    lr: float
    weight_decay: float | None = None  # Override global weight decay
    params_pattern: str | None = None  # Pattern to match parameter names


@dataclass
class OptimizerConfig:
    """Base optimizer configuration using Hydra instantiation.

    Uses _target_ to directly instantiate torch.optim optimizers.
    All optimizer-specific parameters (lr, weight_decay, betas, etc.) are passed
    directly to the torch optimizer via Hydra.

    Example:
        optimizer:
            _target_: torch.optim.AdamW
            lr: 1e-4
            weight_decay: 1e-4
            betas: [0.9, 0.999]
            eps: 1e-8
            param_groups: []  # Optional parameter groups with different LRs
    """
    _target_: str = MISSING  # e.g., "torch.optim.AdamW"

    # Base learning rate (required by all optimizers)
    lr: float = 1e-4

    # Parameter groups with different learning rates
    param_groups: list[ParameterGroupConfig] = field(default_factory=list)


@dataclass
class AdamWConfig(OptimizerConfig):
    """Configuration for torch.optim.AdamW optimizer."""
    _target_: str = "torch.optim.AdamW"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False


@dataclass
class AdamConfig(OptimizerConfig):
    """Configuration for torch.optim.Adam optimizer."""
    _target_: str = "torch.optim.Adam"
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    amsgrad: bool = False


@dataclass
class SGDConfig(OptimizerConfig):
    """Configuration for torch.optim.SGD optimizer."""
    _target_: str = "torch.optim.SGD"
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

    early_stopping_patience: int = 10  # Number of validation checks with no improvement to stop training
