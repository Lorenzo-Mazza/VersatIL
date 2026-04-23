"""PyTorch Lightning wrapper for Policy."""

import re
import time
from typing import Any

import pytorch_lightning as pl
import torch
from hydra.utils import get_class
from omegaconf import OmegaConf
from transformers import get_scheduler

from versatil.configs import OptimizerConfig
from versatil.configs.training import TrainingConfig
from versatil.metrics.accumulators import MetricsAccumulator
from versatil.metrics.base import LossOutput
from versatil.models.policy import Policy
from versatil.training.constants import OPTIMIZER_UNMATCHED_GROUPS_NAME


class LightningPolicy(pl.LightningModule):
    """PyTorch Lightning wrapper around Policy.

    This wrapper handles:
    - Training and validation steps
    - Optimizer configuration with parameter groups
    - Learning rate scheduling
    - Metric accumulation and logging
    - Gradient clipping (via Lightning Trainer)
    """

    def __init__(
        self,
        policy: Policy,
        training_config: TrainingConfig,
        total_training_steps: int | None = None,
    ):
        """Initialize LightningPolicy.

        Args:
            policy: The policy to train
            training_config: Training configuration
            total_training_steps: Total number of training steps for LR scheduling.
                Calculated as: (len(train_loader) * num_epochs) // gradient_accumulate_every
                If None, will use trainer.estimated_stepping_batches as fallback.
        """
        super().__init__()
        self.policy = policy
        self.training_config = training_config
        self.total_training_steps = total_training_steps
        self.train_metrics = MetricsAccumulator()
        self.val_metrics = MetricsAccumulator()
        self.save_hyperparameters(ignore=["policy"])
        self._train_dataloader = None
        self._val_dataloader = None
        self.lr = None

    def on_train_epoch_start(self) -> None:
        """Record epoch start time for duration tracking."""
        self._epoch_start_time = time.monotonic()

    def training_step(
        self, batch: dict[str, dict[str, torch.Tensor]], batch_idx: int
    ) -> torch.Tensor:
        """Training step.

        Args:
            batch: Batch dictionary with observations and actions
            batch_idx: Batch index

        Returns:
            Total loss tensor
        """
        loss_output: LossOutput = self.policy.compute_loss(batch)
        self.train_metrics.add_loss_output(loss_output)
        # Log only on epoch to avoid batch size dependency in plots
        self.log(
            "train_loss",
            loss_output.total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss_output.total_loss

    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch to log accumulated metrics."""
        metrics = self.train_metrics.to_dict()
        self.log_dict(
            {f"train/{k}": v for k, v in metrics.items()},
            on_epoch=True,
            sync_dist=True,
        )
        self.train_metrics.reset()

        # Log epoch duration in seconds
        if hasattr(self, "_epoch_start_time"):
            epoch_duration = time.monotonic() - self._epoch_start_time
            self.log("train/epoch_time_seconds", epoch_duration, on_epoch=True)

        # Log peak GPU memory usage in GB, then reset for next epoch
        if torch.cuda.is_available() and self.device.type == "cuda":
            peak_memory_gb = torch.cuda.max_memory_allocated(device=self.device) / (
                1024**3
            )
            self.log("train/gpu_memory_peak_gb", peak_memory_gb, on_epoch=True)
            torch.cuda.reset_peak_memory_stats(device=self.device)

    def validation_step(
        self, batch: dict[str, dict[str, torch.Tensor]], batch_idx: int
    ) -> torch.Tensor:
        """Validation step.

        Args:
            batch: Batch dictionary with observations and actions
            batch_idx: Batch index

        Returns:
            Total loss tensor
        """
        loss_output: LossOutput = self.policy.compute_loss(batch)
        self.val_metrics.add_loss_output(loss_output)
        self.log(
            "val_loss",
            loss_output.total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss_output.total_loss

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch to log accumulated metrics."""
        metrics = self.val_metrics.to_dict()
        self.log_dict(
            {f"val/{k}": v for k, v in metrics.items()},
            on_epoch=True,
            sync_dist=True,
        )
        self.val_metrics.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizers and learning rate schedulers.

        Uses Hydra's get_class() to resolve optimizer from config.target_class.
        Supports parameter groups with different learning rates.

        Returns:
            Dictionary with optimizer and optional lr_scheduler
        """
        optimizer_config = self.training_config.optimizer
        param_groups = self._create_parameter_groups(optimizer_config)

        optimizer_config_omega = OmegaConf.structured(optimizer_config)
        optimizer_config_dict = OmegaConf.to_container(
            optimizer_config_omega, resolve=True
        )
        target = optimizer_config_dict.pop("target_class")
        optimizer_cls = get_class(target)
        # Remove custom field that's not passed to torch.optim
        optimizer_config_dict.pop("param_groups", None)
        optimizer = optimizer_cls(param_groups, **optimizer_config_dict)
        if self.training_config.lr_schedule is None:
            return {"optimizer": optimizer}
        scheduler_config = self._create_scheduler_config(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }

    def _create_parameter_groups(
        self, optimizer_config: OptimizerConfig
    ) -> list[dict[str, Any]]:
        """Create parameter groups with different learning rates.

        Args:
            optimizer_config: Optimizer configuration

        Returns:
            List of parameter group dictionaries
        """
        base_weight_decay = float(getattr(optimizer_config, "weight_decay", 0.0))
        if not optimizer_config.param_groups:
            return [
                {
                    "name": OPTIMIZER_UNMATCHED_GROUPS_NAME,
                    "params": list(self.policy.parameters()),
                    "lr": optimizer_config.lr,
                    "weight_decay": base_weight_decay,
                }
            ]

        group_names = [group.name for group in optimizer_config.param_groups]
        duplicate_names = {
            group_name
            for group_name in group_names
            if group_names.count(group_name) > 1
        }
        if duplicate_names:
            raise ValueError(
                "Optimizer parameter group names must be unique: "
                f"{sorted(duplicate_names)}."
            )
        if OPTIMIZER_UNMATCHED_GROUPS_NAME in group_names:
            raise ValueError(
                f"'{OPTIMIZER_UNMATCHED_GROUPS_NAME}' is reserved for unmatched "
                "parameters."
            )

        param_groups_dict: dict[str, list[torch.nn.Parameter]] = {
            group_config.name: [] for group_config in optimizer_config.param_groups
        }
        default_params: list[torch.nn.Parameter] = []
        for parameter_name, parameter in list(self.policy.named_parameters()):
            best_name: str | None = None
            best_length = -1
            best_order = -1
            for yaml_order, group_config in enumerate(optimizer_config.param_groups):
                pattern = group_config.params_pattern
                if not pattern:
                    continue
                match = re.search(pattern, parameter_name)
                if match is None:
                    continue
                match_length = match.end() - match.start()
                if match_length > best_length or (
                    match_length == best_length and yaml_order < best_order
                ):
                    best_length = match_length
                    best_order = yaml_order
                    best_name = group_config.name
            if best_name is None:
                default_params.append(parameter)
            else:
                param_groups_dict[best_name].append(parameter)

        empty_groups = [
            group_name
            for group_name, parameters in param_groups_dict.items()
            if not parameters
        ]
        if empty_groups:
            raise ValueError(
                "Configured optimizer parameter groups matched zero parameters: "
                f"{empty_groups}."
            )

        param_groups = [
            {
                "name": OPTIMIZER_UNMATCHED_GROUPS_NAME,
                "params": default_params,
                "lr": optimizer_config.lr,
                "weight_decay": base_weight_decay,
            }
        ]

        for group_config in optimizer_config.param_groups:
            group_dict = {
                "name": group_config.name,
                "params": param_groups_dict[group_config.name],
                "lr": group_config.lr,
                "weight_decay": (
                    group_config.weight_decay
                    if group_config.weight_decay is not None
                    else base_weight_decay
                ),
            }
            param_groups.append(group_dict)

        return param_groups

    def _create_scheduler_config(
        self, optimizer: torch.optim.Optimizer
    ) -> dict[str, Any]:
        """Create learning rate scheduler configuration.

        Args:
            optimizer: The optimizer

        Returns:
            Scheduler configuration dictionary for Lightning
        """

        if self.total_training_steps is not None:
            total_steps = self.total_training_steps
        else:
            total_steps = self.trainer.estimated_stepping_batches

        scheduler = get_scheduler(
            self.training_config.lr_schedule,
            optimizer=optimizer,
            num_warmup_steps=self.training_config.lr_warmup_steps,
            num_training_steps=total_steps,
            scheduler_specific_kwargs=self.training_config.lr_scheduler_kwargs or None,
        )

        return {
            "scheduler": scheduler,
            "interval": "step",  # Update every step
            "frequency": 1,
            "name": "learning_rate",
        }

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Called when loading a checkpoint.

        Ensures that observation_space and action_space are converted from
        OmegaConf DictConfig to proper dataclass instances.

        Args:
            checkpoint: The loaded checkpoint dictionary
        """
        super().on_load_checkpoint(checkpoint)

    def forward(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass for inference.

        Args:
            obs_dict: Observation dictionary

        Returns:
            Predicted actions
        """
        return self.policy.predict_action(obs_dict)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Return training dataloader for Lightning."""
        return self._train_dataloader

    def val_dataloader(self) -> torch.utils.data.DataLoader | None:
        """Return validation dataloader for Lightning, or None if validation is disabled."""
        return self._val_dataloader
