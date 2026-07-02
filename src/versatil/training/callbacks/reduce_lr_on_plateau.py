"""ReduceLROnPlateau callback."""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torch.optim.lr_scheduler import ReduceLROnPlateau as TorchReduceLROnPlateau


class ReduceLROnPlateauCallback(Callback):
    """Callback to reduce learning rate when validation loss plateaus.

    Wraps PyTorch's ReduceLROnPlateau scheduler to work with Lightning.
    Reduces learning rate by a factor when validation metric hasn't improved
    for a given number of epochs (patience).
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        factor: float = 0.5,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        cooldown: int = 0,
        min_lr: float = 0.0,
        eps: float = 1e-8,
    ):
        """Initialize ReduceLROnPlateau callback.

        Args:
            monitor: Metric to monitor (default: "val_loss")
            mode: "min" to reduce LR when metric stops decreasing, "max" for increasing
            factor: Factor by which to reduce LR (new_lr = lr * factor)
            patience: Number of epochs with no improvement before reducing LR
            threshold: Threshold for measuring improvement
            threshold_mode: "rel" for relative threshold, "abs" for absolute
            cooldown: Number of epochs to wait before resuming normal operation after LR reduction
            min_lr: Minimum learning rate
            eps: Minimal decay applied to lr
        """
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.eps = eps
        self.scheduler: TorchReduceLROnPlateau | None = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Create ReduceLROnPlateau scheduler at start of training.

        Args:
            trainer: Lightning trainer
            pl_module: Lightning module
        """
        optimizers = pl_module.optimizers()
        optimizer = optimizers if not isinstance(optimizers, list) else optimizers[0]

        self.scheduler = TorchReduceLROnPlateau(
            optimizer,
            mode=self.mode,
            factor=self.factor,
            patience=self.patience,
            threshold=self.threshold,
            threshold_mode=self.threshold_mode,
            cooldown=self.cooldown,
            min_lr=self.min_lr,
            eps=self.eps,
        )

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Update scheduler with validation metric at end of epoch.

        Args:
            trainer: Lightning trainer
            pl_module: Lightning module
        """
        self._step_scheduler(trainer=trainer, pl_module=pl_module)

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Update scheduler on train epochs when no validation loop runs.

        Runs without a validation dataloader never trigger
        ``on_validation_epoch_end``, so a train-metric monitor must be
        stepped from the training loop instead.

        Args:
            trainer: Lightning trainer
            pl_module: Lightning module
        """
        if trainer.val_dataloaders is not None:
            return
        self._step_scheduler(trainer=trainer, pl_module=pl_module)

    def _step_scheduler(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Step the plateau scheduler with the monitored metric and log the LR."""
        if self.scheduler is None or trainer.sanity_checking:
            return

        if self.monitor not in trainer.callback_metrics:
            return

        current_metric = trainer.callback_metrics[self.monitor].item()
        self.scheduler.step(current_metric)

        optimizer = pl_module.optimizers()
        if not isinstance(optimizer, list):
            current_lr = optimizer.param_groups[0]["lr"]
        else:
            current_lr = optimizer[0].param_groups[0]["lr"]

        pl_module.log(
            "lr",
            current_lr,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
