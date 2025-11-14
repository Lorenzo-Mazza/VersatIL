"""PyTorch Lightning wrapper for Policy."""

import re
from typing import Any

import pytorch_lightning as pl
import torch
from hydra.utils import get_class
from omegaconf import OmegaConf
from transformers import get_scheduler

from refactoring.configs.task.task import ActionSpace, ObservationSpace
from refactoring.configs.training import TrainingConfig
from refactoring.metrics.accumulators import MetricsAccumulator
from refactoring.metrics.base import LossOutput
from refactoring.models.policy import Policy


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
        self.batch_size = None

    def training_step(self, batch: dict[str, dict[str, torch.Tensor]], batch_idx: int) -> torch.Tensor:
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
        self.log("train_loss", loss_output.total_loss, on_step=False, on_epoch=True, prog_bar=True)

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

    def validation_step(self, batch: dict[str, dict[str, torch.Tensor]], batch_idx: int) -> torch.Tensor:
        """Validation step.

        Args:
            batch: Batch dictionary with observations and actions
            batch_idx: Batch index

        Returns:
            Total loss tensor
        """
        loss_output: LossOutput = self.policy.compute_loss(batch)
        self.val_metrics.add_loss_output(loss_output)
        self.log("val_loss", loss_output.total_loss, on_step=False, on_epoch=True, prog_bar=True)
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

    def configure_optimizers(self) -> dict[str, Any]:  # type: ignore[override]
        """Configure optimizers and learning rate schedulers.

        Uses Hydra's get_class() to resolve optimizer from config._target_.
        Supports parameter groups with different learning rates.

        Returns:
            Dictionary with optimizer and optional lr_scheduler
        """
        optimizer_config = self.training_config.optimizer
        param_groups = self._create_parameter_groups(optimizer_config)

        # Convert config to OmegaConf, then to dict to extract optimizer kwargs
        # We need to exclude 'param_groups' since it's not a torch.optim parameter
        optimizer_config_omega = OmegaConf.structured(optimizer_config)
        optimizer_config_dict = OmegaConf.to_container(optimizer_config_omega, resolve=True)
        assert isinstance(optimizer_config_dict, dict), "Expected dict from OmegaConf.to_container"

        # Get optimizer class from _target_
        target = optimizer_config_dict.pop("_target_")
        optimizer_cls = get_class(target)

        # Remove custom field that's not passed to torch.optim
        optimizer_config_dict.pop("param_groups", None)

        # Manually instantiate optimizer to avoid Hydra's recursive conversion
        optimizer = optimizer_cls(param_groups, **optimizer_config_dict)

        if self.training_config.lr_schedule is None:
            return {"optimizer": optimizer}
        scheduler_config = self._create_scheduler_config(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }

    def _create_parameter_groups(self, optimizer_config) -> list[dict[str, Any]]:
        """Create parameter groups with different learning rates.

        Args:
            optimizer_config: Optimizer configuration

        Returns:
            List of parameter group dictionaries
        """
        # If no custom parameter groups, return all parameters
        if not optimizer_config.param_groups:
            return [{"params": self.policy.parameters()}]
        param_groups_dict: dict[str, list[torch.nn.Parameter]] = {}
        default_params: list[torch.nn.Parameter] = []
        for group_config in optimizer_config.param_groups:
            param_groups_dict[group_config.name] = []
        for name, param in self.policy.named_parameters():
            if not param.requires_grad:
                continue

            assigned = False
            for group_config in optimizer_config.param_groups:
                if group_config.params_pattern and re.search(group_config.params_pattern, name):
                    param_groups_dict[group_config.name].append(param)
                    assigned = True
                    break

            if not assigned:
                default_params.append(param)

        param_groups = []
        if default_params:
            param_groups.append({"params": default_params})

        for group_config in optimizer_config.param_groups:
            if param_groups_dict[group_config.name]:
                group_dict = {
                    "params": param_groups_dict[group_config.name],
                    "lr": group_config.lr,
                }
                if group_config.weight_decay is not None:
                    group_dict["weight_decay"] = group_config.weight_decay
                param_groups.append(group_dict)

        return param_groups

    def _create_scheduler_config(self, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
        """Create learning rate scheduler configuration.

        Args:
            optimizer: The optimizer

        Returns:
            Scheduler configuration dictionary for Lightning
        """

        if self.total_training_steps is not None:
            total_steps = self.total_training_steps
        else:
            total_steps = self.trainer.estimated_stepping_batches  # type: ignore[assignment]

        scheduler = get_scheduler(
            self.training_config.lr_schedule,  # type: ignore[arg-type]
            optimizer=optimizer,
            num_warmup_steps=self.training_config.lr_warmup_steps,
            num_training_steps=total_steps,
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

        # Convert observation_space and action_space to dataclass instances if needed
        if hasattr(self, 'policy') and self.policy is not None:
            if OmegaConf.is_config(self.policy.observation_space):
                config_dict = OmegaConf.to_container(self.policy.observation_space, resolve=True)
                self.policy.observation_space = ObservationSpace(**config_dict)  # type: ignore[arg-type]
            if OmegaConf.is_config(self.policy.action_space):
                config_dict = OmegaConf.to_container(self.policy.action_space, resolve=True)
                self.policy.action_space = ActionSpace(**config_dict)  # type: ignore[arg-type]


    def forward(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for inference.

        Args:
            obs_dict: Observation dictionary

        Returns:
            Predicted actions
        """
        return self.policy.predict_action(obs_dict)

    def train_dataloader(self):
        """Return training dataloader for Lightning."""
        return self._train_dataloader

    def val_dataloader(self):
        """Return validation dataloader for Lightning."""
        return self._val_dataloader
