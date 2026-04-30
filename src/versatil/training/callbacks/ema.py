"""Exponential moving average callback."""

import copy
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from torch.nn.modules.batchnorm import _BatchNorm


class EMACallback(Callback):
    """Exponential Moving Average callback for model weights.

    Maintains a moving average of model weights during training. The EMA model
    is used for validation and can provide more stable predictions.

    Based on @crowsonkb's notes on EMA Warmup:
        If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values
        for models you plan to train for a million or more steps (reaches decay factor 0.999 at
        31.6K steps, 0.9999 at 1M steps), gamma=1, power=3/4 for models you plan to train for
        less (reaches decay factor 0.999 at 10K steps, 0.9999 at 215.4k steps).
    """

    def __init__(
        self,
        power: float = 0.75,
        update_after_step: int = 0,
        inv_gamma: float = 1.0,
        min_value: float = 0.0,
        max_value: float = 0.9999,
    ):
        """Initialize EMA callback.

        Args:
            power: Exponential factor of EMA warmup (default: 0.75 for shorter training)
            update_after_step: Start EMA updates after this many steps
            inv_gamma: Inverse multiplicative factor of EMA warmup
            min_value: Minimum EMA decay rate
            max_value: Maximum EMA decay rate
        """
        super().__init__()
        self.power = power
        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.min_value = min_value
        self.max_value = max_value
        self.decay = 0.0
        self.ema_model: torch.nn.Module | None = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Create EMA model copy at start of training.

        Args:
            trainer: Lightning trainer
            pl_module: Lightning module (LightningPolicy)
        """
        self.ema_model = copy.deepcopy(pl_module.policy)
        self.ema_model.eval()
        self.ema_model.requires_grad_(False)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: torch.Tensor | dict[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Update EMA model after each training batch.

        Args:
            trainer: Lightning trainer
            pl_module: Lightning module
            outputs: Training step outputs
            batch: Current batch
            batch_idx: Batch index
        """
        if self.ema_model is None:
            return

        self.decay = self._get_decay(trainer.global_step)

        with torch.no_grad():
            for module, ema_module in zip(
                pl_module.policy.modules(), self.ema_model.modules()
            ):
                for buffer, ema_buffer in zip(
                    module.buffers(recurse=False),
                    ema_module.buffers(recurse=False),
                    strict=True,
                ):
                    ema_buffer.copy_(
                        buffer.to(device=ema_buffer.device, dtype=ema_buffer.dtype)
                    )

                for param, ema_param in zip(
                    module.parameters(recurse=False),
                    ema_module.parameters(recurse=False),
                    strict=True,
                ):
                    if isinstance(param, dict):
                        raise RuntimeError("Dict parameter not supported")

                    if isinstance(module, _BatchNorm) or not param.requires_grad:
                        ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                    else:
                        ema_param.mul_(self.decay)
                        ema_param.add_(
                            param.data.to(dtype=ema_param.dtype), alpha=1 - self.decay
                        )

        if trainer.global_step % 100 == 0:
            pl_module.log("ema_decay", self.decay, on_step=True, on_epoch=False)

    def _get_decay(self, global_step: int) -> float:
        """Compute the decay factor for the exponential moving average.

        Args:
            global_step: Current optimizer step count (from trainer.global_step).

        Returns:
            Decay factor between min_value and max_value
        """
        optimization_step = global_step
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power

        if step <= 0:
            return 0.0

        return max(self.min_value, min(value, self.max_value))

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict,
    ) -> None:
        """Inject EMA weights into the checkpoint.

        Args:
            trainer: Lightning trainer
            pl_module: Lightning module
            checkpoint: Checkpoint dictionary being saved
        """
        if self.ema_model is None:
            return
        ema_state = self.ema_model.state_dict()
        for key, value in ema_state.items():
            ckpt_key = f"policy.{key}"
            if ckpt_key in checkpoint["state_dict"]:
                checkpoint["state_dict"][ckpt_key] = value.clone()

    def on_validation_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Temporarily replace policy with EMA model for validation.

        Args:
            trainer: Lightning trainer
            pl_module: Lightning module
        """
        if self.ema_model is not None:
            self._original_policy = pl_module.policy
            self._sync_policy_runtime_state(
                source_policy=pl_module.policy,
                target_policy=self.ema_model,
            )
            pl_module.policy = self.ema_model

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Restore original policy after validation.

        Args:
            trainer: Lightning trainer
            pl_module: Lightning module
        """
        if hasattr(self, "_original_policy"):
            pl_module.policy = self._original_policy
            delattr(self, "_original_policy")

    @staticmethod
    def _sync_policy_runtime_state(
        source_policy: torch.nn.Module,
        target_policy: torch.nn.Module,
    ) -> None:
        """Copy non-state-dict runtime settings needed for validation.

        Args:
            source_policy: Live training policy that owns current runtime settings.
            target_policy: EMA policy that will be used for validation.
        """
        source_loss_module = getattr(source_policy, "loss_module", None)
        target_loss_module = getattr(target_policy, "loss_module", None)
        if source_loss_module is None or target_loss_module is None:
            return
        set_weights = getattr(target_loss_module, "set_weights", None)
        if not callable(set_weights):
            return
        weights = getattr(source_loss_module, "weights", None)
        if weights is None:
            return
        set_weights(copy.deepcopy(weights))
