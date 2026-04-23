from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

from versatil.training.constants import (
    OPTIMIZER_UNMATCHED_GROUPS_NAME,
    CompileMode,
)


@dataclass
class ParameterGroupConfig:
    """Configuration for a parameter group with specific learning rate."""

    name: str  # e.g., "backbone", "encoder", "decoder", "router"
    lr: float
    weight_decay: float | None = None  # Override global weight decay
    params_pattern: str | None = None  # Pattern to match parameter names

    def __post_init__(self) -> None:
        """Reject names owned by the optimizer grouping runtime."""
        if self.name == OPTIMIZER_UNMATCHED_GROUPS_NAME:
            raise ValueError(
                f"'{OPTIMIZER_UNMATCHED_GROUPS_NAME}' is reserved for unmatched "
                "parameters."
            )


@dataclass
class TrainingStageConfig:
    """Hydra schema for one declarative multi-stage training snapshot.

    ``training.stages`` is ordered by ``start_epoch`` and interpreted as a
    sequence of deltas layered on top of the base training config. A stage may
    independently override parameter trainability, optimizer hyperparameters,
    and loss weights, while any omitted field falls back to the cached base
    regime.

    ``loss_weights`` is a nested patch that must match the structure exposed by
    ``policy.loss_module.weights``. Cross-object checks such as stage ordering,
    optimizer-group existence, and loss-weight path validation run later in
    ``versatil.validation.validate_experiment`` once the full policy and
    optimizer layout exist. The instantiated runtime object validates only
    self-contained invariants.
    """

    _target_: str = "versatil.training.stage.TrainingStage"
    name: str = MISSING
    start_epoch: int = MISSING
    end_epoch: int | None = None
    trainable_groups: list[str] = field(default_factory=list)
    frozen_groups: list[str] = field(default_factory=list)
    group_lrs: dict[str, float] = field(default_factory=dict)
    group_weight_decays: dict[str, float] = field(default_factory=dict)
    loss_weights: dict[str, Any] = field(default_factory=dict)
    eval_frozen_modules: bool = True


@dataclass
class OptimizerConfig:
    """Base optimizer configuration."""

    target_class: str = MISSING
    # Base learning rate (required by all optimizers)
    lr: float = 1e-4
    # Parameter groups with different learning rates
    param_groups: list[ParameterGroupConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate named optimizer groups used by training stages."""
        group_names = [group.name for group in self.param_groups]
        duplicates = _duplicates(group_names)
        if duplicates:
            raise ValueError(
                f"Optimizer parameter group names must be unique: {duplicates}."
            )
        if OPTIMIZER_UNMATCHED_GROUPS_NAME in group_names:
            raise ValueError(
                f"'{OPTIMIZER_UNMATCHED_GROUPS_NAME}' is reserved for unmatched "
                "parameters."
            )


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
    """Training hyperparameters.

    The optional ``stages`` list enables declarative multi-stage training.
    Each stage is applied as a delta over the init-time base regime cached by
    ``TrainingStageCallback``. Epochs that belong to no stage explicitly fall
    back to that base regime.
    """

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

    # Ordered training stage "delta" regimes applied on top of the base training regime.
    stages: list[TrainingStageConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate training knobs that are incompatible with staged control."""
        if self.stages and self.reduce_lr_on_plateau:
            raise ValueError("training.stages does not support reduce_lr_on_plateau.")


def _duplicates(values: Sequence[Any]) -> list[Any]:
    """Return duplicate values while preserving their first repeated order."""
    seen: set[Any] = set()
    duplicates: list[Any] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates
