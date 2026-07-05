"""Callback for epoch-based training-stage transitions."""

import copy
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

from versatil.metrics.base import WeightsDictionary
from versatil.models.policy import Policy
from versatil.training.constants import OPTIMIZER_UNMATCHED_GROUPS_NAME
from versatil.training.lightning_policy import LightningPolicy
from versatil.training.stage import TrainingStage


class TrainingStageCallback(Callback):
    """Apply ordered training stages on top of the cached base regime.

    The callback assumes stage ordering, optimizer-group references, and
    ``loss_weights`` paths were already validated in ``versatil.validation``.
    At runtime it only:

    - caches the base optimizer / trainability / loss state once
    - picks the stage active for ``trainer.current_epoch``
    - applies that stage as a delta over the cached base regime
    - restores the base regime during gaps between stages

    Loss overrides are routed purely through the generic
    ``loss_module.weights`` / ``set_weights`` / ``update_weights`` API. The
    callback does not know about concrete loss classes or composite internals.

    When ``group_lrs`` is used together with a scheduler, staged learning rates
    are interpreted as new scheduler base rates. The current scheduler factor
    is preserved; the callback does not reset scheduler progress.
    """

    _BASE_STAGE_KEY = "base"

    def __init__(
        self,
        stages: list[TrainingStage],
        *,
        learning_rate_schedule_active: bool = False,
    ) -> None:
        """Initialize the training stage callback.

        Args:
            stages: Ordered runtime stages from ``training.stages``.
            learning_rate_schedule_active: Whether the Lightning module configured
                a scheduler. Used to fail fast when staged learning rates cannot
                update scheduler base rates.
        """
        super().__init__()
        if not stages:
            raise ValueError("TrainingStageCallback requires a non-empty stage list.")
        self.stages = stages
        self.learning_rate_schedule_active = learning_rate_schedule_active
        self._uses_group_learning_rates = any(stage.group_lrs for stage in self.stages)
        self._last_applied_stage_key: str | None = None
        self._validated = False
        self._base_group_learning_rates: dict[str, float] = {}
        self._base_group_weight_decays: dict[str, float] = {}
        self._base_loss_weights: WeightsDictionary | None = None
        self._parameter_group_by_id: dict[int, str] = {}
        self._base_parameter_trainability: dict[int, bool] = {}
        self._frozen_modules: tuple[torch.nn.Module, ...] = ()
        self._trainable_modules: tuple[torch.nn.Module, ...] = ()

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Cache the base regime and apply the stage active at resume epoch."""
        lightning_policy = self._require_lightning_policy(pl_module)
        self._ensure_initialized(trainer=trainer, pl_module=lightning_policy)
        self._apply_active_stage(trainer=trainer, pl_module=lightning_policy)

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Apply a newly active stage, or restore modes for the current stage."""
        lightning_policy = self._require_lightning_policy(pl_module)
        self._ensure_initialized(trainer=trainer, pl_module=lightning_policy)
        self._apply_active_stage(trainer=trainer, pl_module=lightning_policy)

    @staticmethod
    def _require_lightning_policy(pl_module: pl.LightningModule) -> LightningPolicy:
        """Narrow Lightning's generic ``pl_module`` to the expected ``LightningPolicy``."""
        if not isinstance(pl_module, LightningPolicy):
            raise TypeError(
                "TrainingStageCallback requires a LightningPolicy module, got "
                f"{type(pl_module).__name__}."
            )
        return pl_module

    def _ensure_initialized(
        self, trainer: pl.Trainer, pl_module: LightningPolicy
    ) -> None:
        """Cache all base snapshots once before any stage mutates runtime state."""
        if self._validated:
            return
        optimizer = self._get_optimizer(trainer=trainer)
        self._cache_optimizer_groups(trainer=trainer, optimizer=optimizer)
        self._cache_base_parameter_trainability(policy=pl_module.policy)
        self._cache_stage_loss_modules(policy=pl_module.policy)
        self._validated = True

    @staticmethod
    def _get_optimizer(trainer: pl.Trainer) -> torch.optim.Optimizer:
        """Return the primary training optimizer."""
        return trainer.optimizers[0]

    def _cache_optimizer_groups(
        self, trainer: pl.Trainer, optimizer: torch.optim.Optimizer
    ) -> None:
        """Cache base optimizer values and the parameter-to-group mapping.

        Stage trainability and optimizer overrides operate by optimizer-group
        name, so the callback records both the base optimizer values and the
        owning group for each policy parameter tensor.
        """
        group_names: list[str] = []
        for group in optimizer.param_groups:
            group_name = group.get("name")
            if not isinstance(group_name, str):
                raise ValueError(
                    "training.stages requires every optimizer parameter group to "
                    "have a string 'name'."
                )
            group_names.append(group_name)
        duplicate_names = sorted(
            {
                group_name
                for group_name in group_names
                if group_names.count(group_name) > 1
            }
        )
        if duplicate_names:
            raise ValueError(
                f"Optimizer parameter group names must be unique: {duplicate_names}."
            )
        if OPTIMIZER_UNMATCHED_GROUPS_NAME not in group_names:
            raise ValueError(
                "training.stages requires an optimizer parameter group named "
                f"'{OPTIMIZER_UNMATCHED_GROUPS_NAME}'."
            )

        scheduler_base_learning_rates = (
            self._scheduler_base_learning_rates_for_optimizer(
                trainer=trainer,
                optimizer=optimizer,
            )
        )
        for group_index, group in enumerate(optimizer.param_groups):
            group_name = group["name"]
            group_learning_rate = (
                scheduler_base_learning_rates[group_index]
                if scheduler_base_learning_rates is not None
                else float(group["lr"])
            )
            self._base_group_learning_rates[group_name] = group_learning_rate
            self._base_group_weight_decays[group_name] = float(
                group.get("weight_decay", 0.0)
            )
            for parameter in group["params"]:
                self._parameter_group_by_id[id(parameter)] = group_name

    def _scheduler_base_learning_rates_for_optimizer(
        self,
        trainer: pl.Trainer,
        optimizer: torch.optim.Optimizer,
    ) -> list[float] | None:
        """Return scheduler raw base learning rates when stages need them.

        Step-based schedulers may already have written scaled values into the
        optimizer before ``on_train_start``. When stages use ``group_lrs``, the
        cached base snapshot must therefore come from ``scheduler.base_lrs`` so
        omitted stage keys restore the YAML-configured scheduler anchors.
        """
        if not self._uses_group_learning_rates:
            return None
        schedulers = self._get_learning_rate_schedulers(trainer)
        if not schedulers:
            return None
        return self._validate_scheduler_base_learning_rates(
            scheduler=schedulers[0],
            optimizer=optimizer,
        )

    def _cache_base_parameter_trainability(self, policy: Policy) -> None:
        """Cache base ``requires_grad`` values before any stage mutates them."""
        for _, parameter in policy.named_parameters():
            self._base_parameter_trainability[id(parameter)] = parameter.requires_grad

    def _cache_stage_loss_modules(self, policy: Policy) -> None:
        """Cache the base loss-weight tree for later restore.

        Note:
            Stages drive loss weighting purely through the generic
            ``loss_module.weights`` / ``set_weights`` / ``update_weights`` API.
        """
        if not any(stage.loss_weights for stage in self.stages):
            return
        self._base_loss_weights = copy.deepcopy(policy.loss_module.weights)

    def _apply_active_stage(
        self, trainer: pl.Trainer, pl_module: LightningPolicy
    ) -> None:
        """Apply the regime active for the current epoch.

        Re-entering the same stage only reasserts cached module modes. A full
        runtime rewrite happens only when the active stage key changes.
        """
        active_index = self._active_stage_index(int(trainer.current_epoch))
        active_key = (
            str(active_index) if active_index is not None else self._BASE_STAGE_KEY
        )
        if active_key == self._last_applied_stage_key:
            self._restore_cached_module_modes()
            return
        if active_index is None:
            self._apply_base_snapshot(trainer=trainer, pl_module=pl_module)
            self._last_applied_stage_key = active_key
            return
        self._apply_stage(
            trainer=trainer,
            pl_module=pl_module,
            stage=self.stages[active_index],
            stage_index=active_index,
        )
        self._last_applied_stage_key = active_key

    def _active_stage_index(self, current_epoch: int) -> int | None:
        """Return the active stage index, or ``None`` when the base regime applies."""
        active_index: int | None = None
        for candidate_index, stage in enumerate(self.stages):
            next_stage = (
                self.stages[candidate_index + 1]
                if candidate_index + 1 < len(self.stages)
                else None
            )
            if current_epoch < stage.start_epoch:
                break
            if stage.is_active_at(current_epoch, next_stage=next_stage):
                active_index = candidate_index
        return active_index

    def _apply_base_snapshot(
        self, trainer: pl.Trainer, pl_module: LightningPolicy
    ) -> None:
        """Restore the init-time base regime outside explicit stages."""
        policy = pl_module.policy
        optimizer = self._get_optimizer(trainer=trainer)
        counts = self._apply_base_trainability(policy=policy)
        policy.train()
        self._frozen_modules = ()
        self._trainable_modules = ()
        effective_learning_rates = self._apply_base_optimizer_snapshot(
            optimizer=optimizer
        )
        if self._uses_group_learning_rates:
            self._sync_scheduler_learning_rates(
                trainer=trainer,
                optimizer=optimizer,
                effective_learning_rates=effective_learning_rates,
            )
        self._apply_base_loss_snapshot(pl_module=pl_module)
        self._log_regime(
            pl_module=pl_module,
            stage_index=-1,
            start_epoch=-1,
            trainable_tensors=counts["trainable_tensors"],
            frozen_tensors=counts["frozen_tensors"],
            trainable_parameters=counts["trainable_parameters"],
            frozen_parameters=counts["frozen_parameters"],
            optimizer=optimizer,
        )
        logging.info("Applied base training regime outside configured stages")

    def _apply_base_trainability(self, policy: Policy) -> dict[str, int]:
        """Restore cached base ``requires_grad`` values and return counts."""
        counts = {
            "trainable_tensors": 0,
            "frozen_tensors": 0,
            "trainable_parameters": 0,
            "frozen_parameters": 0,
        }
        for _, parameter in policy.named_parameters():
            should_train = self._base_parameter_trainability.get(id(parameter), True)
            was_trainable = parameter.requires_grad
            parameter.requires_grad_(should_train)
            if was_trainable and not should_train:
                parameter.grad = None
            parameter_count = parameter.numel()
            if should_train:
                counts["trainable_tensors"] += 1
                counts["trainable_parameters"] += parameter_count
            else:
                counts["frozen_tensors"] += 1
                counts["frozen_parameters"] += parameter_count
        return counts

    def _apply_stage(
        self,
        trainer: pl.Trainer,
        pl_module: LightningPolicy,
        stage: TrainingStage,
        stage_index: int,
    ) -> None:
        """Apply trainability, optimizer, scheduler, and loss snapshots.

        Note:
            Trainability uses delta-on-init semantics: the init config is the
            ground truth, cached once at ``on_train_start``. Each stage applies a
            delta on top of it — ``frozen_groups`` forces freeze, ``trainable_groups``
            forces unfreezing, and any group not mentioned retains its init-time state.
        """
        policy = pl_module.policy
        optimizer = self._get_optimizer(trainer=trainer)
        stage_trainable_groups = set(stage.trainable_groups)
        stage_frozen_groups = set(stage.frozen_groups)
        trainable_tensors = 0
        frozen_tensors = 0
        trainable_parameters = 0
        frozen_parameters = 0
        for _, parameter in policy.named_parameters():
            group_name = self._parameter_group_by_id.get(id(parameter))
            if group_name is None:
                raise ValueError(
                    "training.stages found a policy parameter that is not present "
                    "in any optimizer parameter group."
                )
            if group_name in stage_frozen_groups:
                should_train = False
            elif group_name in stage_trainable_groups:
                should_train = True
            else:
                should_train = self._base_parameter_trainability.get(
                    id(parameter), True
                )
            was_trainable = parameter.requires_grad
            parameter.requires_grad_(should_train)
            if was_trainable and not should_train:
                parameter.grad = None
            parameter_count = parameter.numel()
            if should_train:
                trainable_tensors += 1
                trainable_parameters += parameter_count
            else:
                frozen_tensors += 1
                frozen_parameters += parameter_count

        if stage.eval_frozen_modules:
            self._frozen_modules, self._trainable_modules = self._sync_module_modes(
                policy
            )
        else:
            policy.train()
            self._frozen_modules = ()
            self._trainable_modules = ()

        effective_learning_rates = self._apply_optimizer_snapshot(
            optimizer=optimizer,
            stage=stage,
        )
        if self._uses_group_learning_rates:
            self._sync_scheduler_learning_rates(
                trainer=trainer,
                optimizer=optimizer,
                effective_learning_rates=effective_learning_rates,
            )
        self._apply_loss_snapshot(pl_module=pl_module, stage=stage)
        self._log_regime(
            pl_module=pl_module,
            stage_index=stage_index,
            start_epoch=stage.start_epoch,
            trainable_tensors=trainable_tensors,
            frozen_tensors=frozen_tensors,
            trainable_parameters=trainable_parameters,
            frozen_parameters=frozen_parameters,
            optimizer=optimizer,
        )
        logging.info(
            f"Applied training stage '{stage.name}' at epoch "
            f"{stage.start_epoch}: {trainable_tensors} trainable tensors, "
            f"{frozen_tensors} frozen tensors"
        )

    @staticmethod
    def _sync_module_modes(
        policy: Policy,
    ) -> tuple[tuple[torch.nn.Module, ...], tuple[torch.nn.Module, ...]]:
        """Put fully frozen modules in eval mode and fully trainable ones in train mode."""
        frozen_modules: list[torch.nn.Module] = []
        trainable_modules: list[torch.nn.Module] = []
        for name, module in policy.named_modules():
            if name == "":
                continue
            parameters = list(module.parameters(recurse=True))
            if not parameters:
                continue
            if all(not parameter.requires_grad for parameter in parameters):
                module.eval()
                frozen_modules.append(module)
            elif all(parameter.requires_grad for parameter in parameters):
                module.train()
                trainable_modules.append(module)
        return tuple(frozen_modules), tuple(trainable_modules)

    def _restore_cached_module_modes(self) -> None:
        """Restore stage-managed module modes without rewalking parameters."""
        for module in self._trainable_modules:
            module.train()
        for module in self._frozen_modules:
            module.eval()

    def _apply_optimizer_snapshot(
        self,
        optimizer: torch.optim.Optimizer,
        stage: TrainingStage,
    ) -> list[float]:
        """Apply per-group optimizer overrides, falling back to cached base values."""
        effective_learning_rates: list[float] = []
        for group in optimizer.param_groups:
            group_name = group["name"]
            group_learning_rate = float(
                stage.group_lrs.get(
                    group_name, self._base_group_learning_rates[group_name]
                )
            )
            group_weight_decay = float(
                stage.group_weight_decays.get(
                    group_name,
                    self._base_group_weight_decays[group_name],
                )
            )
            group["lr"] = group_learning_rate
            group["weight_decay"] = group_weight_decay
            effective_learning_rates.append(group_learning_rate)
        return effective_learning_rates

    def _apply_base_optimizer_snapshot(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> list[float]:
        """Restore cached base learning rate and weight decay for every group."""
        effective_learning_rates: list[float] = []
        for group in optimizer.param_groups:
            group_name = group["name"]
            group_learning_rate = self._base_group_learning_rates[group_name]
            group["lr"] = group_learning_rate
            group["weight_decay"] = self._base_group_weight_decays[group_name]
            effective_learning_rates.append(group_learning_rate)
        return effective_learning_rates

    def _sync_scheduler_learning_rates(
        self,
        trainer: pl.Trainer,
        optimizer: torch.optim.Optimizer,
        effective_learning_rates: list[float],
    ) -> None:
        """Re-anchor scheduler base rates without resetting schedule progress.

        Staged learning rates are interpreted as scheduler base rates. The
        current scheduler multiplier is preserved, so the optimizer immediately
        uses ``new_base_learning_rate * current_scheduler_factor`` instead of
        one raw learning-rate batch before the next scheduler step.
        """
        schedulers = self._get_learning_rate_schedulers(trainer)
        if self.learning_rate_schedule_active and not schedulers:
            raise ValueError(
                "training.stages uses group_lrs with an active lr_schedule, but "
                "no Lightning scheduler was found."
            )
        for scheduler in schedulers:
            scheduler_base_learning_rates = (
                self._validate_scheduler_base_learning_rates(
                    scheduler=scheduler,
                    optimizer=optimizer,
                )
            )
            current_learning_rates = self._scheduler_current_learning_rates(
                scheduler=scheduler,
                optimizer=optimizer,
            )
            scaled_learning_rates = self._scale_learning_rates(
                base_learning_rates=scheduler_base_learning_rates,
                current_learning_rates=current_learning_rates,
                new_base_learning_rates=effective_learning_rates,
            )
            scheduler.base_lrs = list(effective_learning_rates)
            if isinstance(getattr(scheduler, "_last_lr", None), list):
                scheduler._last_lr = list(scaled_learning_rates)
            for group, learning_rate in zip(
                optimizer.param_groups,
                scaled_learning_rates,
                strict=True,
            ):
                group["lr"] = learning_rate

    @staticmethod
    def _validate_scheduler_base_learning_rates(
        scheduler: Any,
        optimizer: torch.optim.Optimizer,
    ) -> list[float]:
        """Return validated scheduler base learning rates for every group."""
        scheduler_base_learning_rates = getattr(scheduler, "base_lrs", None)
        if not isinstance(scheduler_base_learning_rates, list) or len(
            scheduler_base_learning_rates
        ) != len(optimizer.param_groups):
            raise ValueError(
                "training.stages with group_lrs requires the scheduler to expose "
                "base_lrs with one entry per optimizer parameter group."
            )
        return [float(learning_rate) for learning_rate in scheduler_base_learning_rates]

    @staticmethod
    def _scheduler_current_learning_rates(
        scheduler: Any,
        optimizer: torch.optim.Optimizer,
    ) -> list[float]:
        """Return current scheduler-managed learning rates without stepping it."""
        last_learning_rates = getattr(scheduler, "_last_lr", None)
        if isinstance(last_learning_rates, list) and len(last_learning_rates) == len(
            optimizer.param_groups
        ):
            return [float(learning_rate) for learning_rate in last_learning_rates]
        get_last_lr = getattr(scheduler, "get_last_lr", None)
        if callable(get_last_lr):
            resolved_learning_rates = list(get_last_lr())
            if len(resolved_learning_rates) == len(optimizer.param_groups):
                return [
                    float(learning_rate) for learning_rate in resolved_learning_rates
                ]
        raise ValueError(
            "training.stages with group_lrs requires the scheduler to expose "
            "current learning rates via _last_lr or get_last_lr()."
        )

    @staticmethod
    def _scale_learning_rates(
        base_learning_rates: Sequence[float],
        current_learning_rates: Sequence[float],
        new_base_learning_rates: Sequence[float],
    ) -> list[float]:
        """Apply current scheduler multipliers to new base learning rates.

        For each group, the effective rate is:

        ``new_base_learning_rate * (current_learning_rate / old_base_learning_rate)``

        with a fallback of ``0.0`` when the previous base rate is zero.
        """
        scaled_learning_rates: list[float] = []
        for base_learning_rate, current_learning_rate, new_base_learning_rate in zip(
            base_learning_rates,
            current_learning_rates,
            new_base_learning_rates,
            strict=True,
        ):
            if base_learning_rate == 0.0:
                scaled_learning_rates.append(0.0)
                continue
            scaled_learning_rates.append(
                new_base_learning_rate * (current_learning_rate / base_learning_rate)
            )
        return scaled_learning_rates

    @staticmethod
    def _get_learning_rate_schedulers(trainer: pl.Trainer) -> list[Any]:
        """Return scheduler objects from Lightning's scheduler config list."""
        scheduler_configs = getattr(trainer, "lr_scheduler_configs", None)
        if not isinstance(scheduler_configs, (list, tuple)):
            return []
        schedulers: list[Any] = []
        for scheduler_config in scheduler_configs:
            scheduler = getattr(scheduler_config, "scheduler", None)
            if scheduler is None and isinstance(scheduler_config, Mapping):
                scheduler = scheduler_config.get("scheduler")
            if scheduler is not None:
                schedulers.append(scheduler)
        return schedulers

    def _apply_loss_snapshot(
        self, pl_module: LightningPolicy, stage: TrainingStage
    ) -> None:
        """Restore the base loss tree, then apply the stage's nested patch."""
        if self._base_loss_weights is None:
            return
        loss_module = pl_module.policy.loss_module
        loss_module.set_weights(copy.deepcopy(self._base_loss_weights))
        if stage.loss_weights:
            loss_module.update_weights(stage.loss_weights)

    def _apply_base_loss_snapshot(self, pl_module: LightningPolicy) -> None:
        """Restore the cached base loss tree without any stage override."""
        if self._base_loss_weights is None:
            return
        pl_module.policy.loss_module.set_weights(copy.deepcopy(self._base_loss_weights))

    @staticmethod
    def _log_regime(
        pl_module: pl.LightningModule,
        stage_index: int,
        start_epoch: int,
        trainable_tensors: int,
        frozen_tensors: int,
        trainable_parameters: int,
        frozen_parameters: int,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Log the active stage, trainability counts, and per-group rates."""
        log_kwargs = {
            "on_step": False,
            "on_epoch": True,
            "prog_bar": False,
            "logger": True,
        }
        pl_module.log("training_stage/index", float(stage_index), **log_kwargs)
        pl_module.log("training_stage/start_epoch", float(start_epoch), **log_kwargs)
        pl_module.log(
            "training_stage/trainable_tensors",
            float(trainable_tensors),
            **log_kwargs,
        )
        pl_module.log(
            "training_stage/frozen_tensors",
            float(frozen_tensors),
            **log_kwargs,
        )
        pl_module.log(
            "training_stage/trainable_parameters",
            float(trainable_parameters),
            **log_kwargs,
        )
        pl_module.log(
            "training_stage/frozen_parameters",
            float(frozen_parameters),
            **log_kwargs,
        )
        for group in optimizer.param_groups:
            pl_module.log(
                f"training_stage/learning_rate/{group['name']}",
                float(group["lr"]),
                **log_kwargs,
            )
