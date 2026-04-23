"""Gradient norm logging callback."""

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback


class GradientNormCallback(Callback):
    """Callback to log gradient norms during training.

    Logs:
    - grad_norm: root step metric
    - train/grad_norm_step: Step metric under the train namespace
    - train/grad_norm_epoch: Mean sampled gradient norm over the epoch
    - train/grad_norm_max_epoch: Max sampled gradient norm over the epoch
    - train/grad_clip_active_ratio: Fraction of sampled steps above the clip threshold
    - Individual parameter group gradient norms
    """

    def __init__(self, log_every_n_steps: int = 50):
        """Initialize gradient norm callback.

        Args:
            log_every_n_steps: Log gradient norms every N steps
        """
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self._epoch_grad_norms: list[float] = []
        self._epoch_grad_clip_active: list[float] = []

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Log gradient norms before optimizer step.

        Args:
            trainer: Lightning trainer
            pl_module: Lightning module
            optimizer: The optimizer
        """
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        grad_norm = self._compute_grad_norm(pl_module)
        self._epoch_grad_norms.append(grad_norm)
        clip_val = getattr(trainer, "gradient_clip_val", None)
        clip_active = (
            isinstance(clip_val, (int, float))
            and clip_val > 0.0
            and grad_norm > clip_val
        )
        self._epoch_grad_clip_active.append(float(clip_active))

        pl_module.log(
            "grad_norm",
            grad_norm,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        pl_module.log(
            "train/grad_clip_active_step",
            float(clip_active),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        pl_module.log(
            "train/grad_norm_step",
            grad_norm,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )

        if hasattr(optimizer, "param_groups") and len(optimizer.param_groups) > 1:
            for idx, param_group in enumerate(optimizer.param_groups):
                group_grad_norm = self._compute_grad_norm_for_params(
                    param_group["params"]
                )
                pl_module.log(
                    f"grad_norm_group_{idx}",
                    group_grad_norm,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    logger=True,
                )
                pl_module.log(
                    f"train/grad_norm_group_{idx}_step",
                    group_grad_norm,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    logger=True,
                )

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Log epoch-level summaries so gradient norms appear with train charts."""
        if not self._epoch_grad_norms:
            return

        grad_norms = np.asarray(self._epoch_grad_norms, dtype=np.float32)
        metrics = {
            "train/grad_norm_epoch": float(grad_norms.mean()),
            "train/grad_norm_max_epoch": float(grad_norms.max()),
            "epoch": trainer.current_epoch,
        }
        if self._epoch_grad_clip_active:
            metrics["train/grad_clip_active_ratio"] = float(
                np.asarray(self._epoch_grad_clip_active, dtype=np.float32).mean()
            )
        if trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=trainer.current_epoch)
        self._epoch_grad_norms.clear()
        self._epoch_grad_clip_active.clear()

    def _compute_grad_norm(self, pl_module: pl.LightningModule) -> float:
        """Compute the total gradient norm across all parameters.

        Args:
            pl_module: Lightning module

        Returns:
            Total gradient norm
        """
        total_norm = 0.0
        for param in pl_module.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        return total_norm

    def _compute_grad_norm_for_params(self, params) -> float:
        """Compute gradient norm for a specific set of parameters.

        Args:
            params: List of parameters

        Returns:
            Gradient norm
        """
        total_norm = 0.0
        for param in params:
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        return total_norm
