"""PyTorch Lightning callbacks for training."""

import copy
import io
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
import wandb
from PIL import Image
from pytorch_lightning.callbacks import Callback, EarlyStopping
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim.lr_scheduler import ReduceLROnPlateau as TorchReduceLROnPlateau

from versatil.metrics.constants import MetadataKey

plt.set_loglevel("warning")


class ResumableEarlyStopping(EarlyStopping):
    """EarlyStopping that ignores checkpoint state, always using config values.

    Note: this allows to resume training beyond an initial early stopping state, which is
     otherwise not possible to overwrite from Lightning.
    """

    def load_state_dict(self, state_dict):
        pass


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
        # Deep copy the policy (not the whole LightningPolicy wrapper)
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

        # Compute decay factor using trainer's optimizer step count (not batch count)
        self.decay = self._get_decay(trainer.global_step)

        # Update EMA model parameters (no_grad to avoid in-place operation errors)
        with torch.no_grad():
            for module, ema_module in zip(
                pl_module.policy.modules(), self.ema_model.modules()
            ):
                # Copy BatchNorm running stats (buffers, not parameters)
                if isinstance(module, _BatchNorm):
                    for buffer, ema_buffer in zip(
                        module.buffers(), ema_module.buffers()
                    ):
                        ema_buffer.copy_(buffer)

                for param, ema_param in zip(
                    module.parameters(recurse=False),
                    ema_module.parameters(recurse=False),
                ):
                    if isinstance(param, dict):
                        raise RuntimeError("Dict parameter not supported")

                    if isinstance(module, _BatchNorm) or not param.requires_grad:
                        # Copy batchnorm learnable params and frozen params directly
                        ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                    else:
                        # EMA update: ema = decay * ema + (1 - decay) * param
                        ema_param.mul_(self.decay)
                        ema_param.add_(
                            param.data.to(dtype=ema_param.dtype), alpha=1 - self.decay
                        )

        # Log EMA decay factor
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
            # Store original policy
            self._original_policy = pl_module.policy
            # Use EMA model for validation
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
            # Restore original policy
            pl_module.policy = self._original_policy
            delattr(self, "_original_policy")


class ExpertUsageCallback(Callback):
    """Callback to log expert usage statistics for mixture-of-experts models.

    Logs expert usage ratios as bar plots to WandB at the end of each epoch.
    """

    def __init__(self, log_every_n_epochs: int = 1):
        """Initialize expert usage callback.

        Args:
            log_every_n_epochs: Log expert usage every N epochs
        """
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Log training expert usage at end of epoch.

        Args:
            trainer: Lightning trainer
            pl_module: Lightning module
        """
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        expert_usages = pl_module.train_metrics.compute_expert_usage()
        if expert_usages is not None:
            for key, expert_usage in expert_usages.items():
                fig = self._create_expert_usage_figure(expert_usage, f"Train {key}")
                if trainer.logger is not None:
                    wandb_image = _figure_to_wandb_image(fig)
                    trainer.logger.log_metrics(
                        {f"train_{key}": wandb_image},
                        step=trainer.current_epoch,
                    )
                plt.close(fig)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Log validation expert usage picture at end of epoch.

        Args:
            trainer: Lightning trainer
            pl_module: Lightning module
        """
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return
        expert_usages = pl_module.val_metrics.compute_expert_usage()
        if expert_usages is not None:
            for key, expert_usage in expert_usages.items():
                fig = self._create_expert_usage_figure(expert_usage, f"Val {key}")
                if trainer.logger is not None:
                    wandb_image = _figure_to_wandb_image(fig)
                    trainer.logger.log_metrics(
                        {f"val_{key}": wandb_image},
                        step=trainer.current_epoch,
                    )
                plt.close(fig)

    def _create_expert_usage_figure(
        self, expert_usage: np.ndarray, title: str
    ) -> plt.Figure:
        """Create a bar plot figure for expert usage.

        Args:
            expert_usage: Expert usage ratios as numpy array
            title: Title for the plot
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        expert_usage_list = [float(val) for val in expert_usage]
        sns.barplot(x=np.arange(len(expert_usage_list)), y=expert_usage_list, ax=ax)
        ax.set_xlabel("Expert Index")
        ax.set_ylabel("Average Usage Ratio")
        ax.set_title(title)
        plt.tight_layout()
        return fig


class ConfusionMatrixCallback(Callback):
    """Callback to log confusion matrices for phase classification models.

    Automatically detects when phase predictions are available in the metrics
    and logs confusion matrices to WandB.
    """

    def __init__(self, log_every_n_epochs: int = 1):
        """Initialize confusion matrix callback.

        Args:
            log_every_n_epochs: Log confusion matrix every N epochs
        """
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Log training confusion matrix at end of epoch.

        Args:
            trainer: Lightning trainer
            pl_module: Lightning module
        """
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        # Get confusion matrix from train metrics accumulator
        cm = pl_module.train_metrics.compute_confusion_matrix()
        if cm is not None:
            fig = self._create_confusion_matrix_figure(
                cm, "Train Phase Confusion Matrix"
            )
            if trainer.logger is not None:
                wandb_image = _figure_to_wandb_image(fig)
                trainer.logger.log_metrics(
                    {"train_phase_confusion_matrix": wandb_image},
                    step=trainer.current_epoch,
                )
            plt.close(fig)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Log validation confusion matrix at end of epoch.

        Args:
            trainer: Lightning trainer
            pl_module: Lightning module
        """
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return
        # Get confusion matrix from val metrics accumulator
        cm = pl_module.val_metrics.compute_confusion_matrix()
        if cm is not None:
            fig = self._create_confusion_matrix_figure(cm, "Val Phase Confusion Matrix")
            if trainer.logger is not None:
                wandb_image = _figure_to_wandb_image(fig)
                trainer.logger.log_metrics(
                    {"val_phase_confusion_matrix": wandb_image},
                    step=trainer.current_epoch,
                )
            plt.close(fig)

    def _create_confusion_matrix_figure(self, cm: np.ndarray, title: str) -> plt.Figure:
        """Create a seaborn heatmap figure for the confusion matrix.

        Args:
            cm: Confusion matrix as numpy array (n_phases, n_phases)
            title: Title for the plot

        Returns:
            Matplotlib figure
        """
        n_phases = cm.shape[0]

        # Normalize by row (true labels)
        cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1e-10)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=[f"Phase {i}" for i in range(n_phases)],
            yticklabels=[f"Phase {i}" for i in range(n_phases)],
            ax=ax,
            cbar_kws={"label": "Proportion"},
        )
        ax.set_xlabel("Predicted Phase")
        ax.set_ylabel("True Phase")
        ax.set_title(title)
        plt.tight_layout()
        return fig


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

        # Keep the old root metric for existing W&B panels.
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

        # Log per-parameter group if using parameter groups
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
        # Get optimizer from pl_module
        optimizers = pl_module.optimizers()
        optimizer = optimizers if not isinstance(optimizers, list) else optimizers[0]

        # Create scheduler
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
        if self.scheduler is None:
            return

        # Get monitored metric value
        if self.monitor not in trainer.callback_metrics:
            return

        current_metric = trainer.callback_metrics[self.monitor].item()

        # Step the scheduler
        self.scheduler.step(current_metric)

        # Log current learning rate
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


class LatentVisualizationCallback(Callback):
    """Visualize VAE latent space with phase coloring.

    Creates t-SNE projections of the latent space colored by dominant phase
    to show whether different action modes are disentangled.
    """

    def __init__(self, log_every_n_epochs: int = 5, max_samples: int = 5000):
        """Initialize latent visualization callback.

        Args:
            log_every_n_epochs: Log visualization every N epochs.
            max_samples: Maximum samples for t-SNE (subsamples if exceeded).
        """
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.max_samples = max_samples

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Create and log latent space visualization at end of training epoch."""
        self._log_latent(
            trainer=trainer,
            metrics_accumulator=pl_module.train_metrics,
            split="train",
        )

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Create and log latent space visualization at end of validation epoch."""
        self._log_latent(
            trainer=trainer,
            metrics_accumulator=pl_module.val_metrics,
            split="val",
        )

    def _log_latent(
        self,
        trainer: pl.Trainer,
        metrics_accumulator,
        split: str,
    ) -> None:
        """Compute and log latent-space visualizations for the given metrics accumulator.

        Args:
            trainer: Lightning trainer.
            metrics_accumulator: Either train_metrics or val_metrics.
            split: "train" or "val" — used as a prefix on logged metric keys.
        """
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        latent_data = metrics_accumulator.compute_latent_visualization_data()
        if latent_data is None:
            return
        z, z_prior, phase_per_sample = latent_data
        if z is None and z_prior is None:
            return

        figures = {}
        if z is not None:
            figures.update(
                self._build_latent_figures(
                    z=z,
                    phases=phase_per_sample,
                    prefix=f"{split}_posterior",
                    title=f"{split.title()} posterior latent space",
                )
            )
        if z_prior is not None:
            figures.update(
                self._build_latent_figures(
                    z=z_prior,
                    phases=phase_per_sample,
                    prefix=f"{split}_prior",
                    title=f"{split.title()} prior latent space",
                )
            )

        latent_stats_table = self._create_latent_stats_table(
            metrics_accumulator.metadata
        )

        if trainer.logger is not None:
            metrics = {key: _figure_to_wandb_image(fig) for key, fig in figures.items()}
            if latent_stats_table is not None:
                metrics[f"{split}_latent_space_statistics"] = latent_stats_table
            trainer.logger.log_metrics(metrics, step=trainer.current_epoch)

        for fig in figures.values():
            plt.close(fig)

    def _build_latent_figures(
        self,
        z: np.ndarray,
        phases: np.ndarray | None,
        prefix: str,
        title: str,
    ) -> dict[str, plt.Figure]:
        """Dispatch to histogram for 1D latents or t-SNE/PCA for higher dim.

        Args:
            z: Latent samples (N, latent_dim) or (N,).
            phases: Dominant phase per sample (N,), or None.
            prefix: Metric-key prefix (e.g. "posterior" or "prior").
            title: Human-readable figure title.

        Returns:
            Mapping from metric key to matplotlib figure.
        """
        latent_dim = z.shape[1] if z.ndim > 1 else 1
        if latent_dim == 1:
            return {
                f"{prefix}_latent_space_histogram": self._create_histogram_figure(
                    z=z, phases=phases, title=title
                )
            }
        return {
            f"{prefix}_latent_space_tsne": self._create_latent_figure(
                z=z, phases=phases, title=title
            ),
            f"{prefix}_latent_space_pca": self._create_pca_figure(
                z=z, phases=phases, title=title
            ),
            f"{prefix}_pca_explained_variance": self._create_pca_variance_figure(
                z=z, title=title
            ),
        }

    def _create_histogram_figure(
        self, z: np.ndarray, phases: np.ndarray | None, title: str = ""
    ) -> plt.Figure:
        """Create a histogram of a 1D latent distribution.

        When per-sample phase labels are provided, plots one translucent
        histogram per phase sharing the same bin edges so their shapes are
        directly comparable. Otherwise, plots a single histogram.

        Args:
            z: Latent samples (N, 1) or (N,).
            phases: Dominant phase per sample (N,), or None.
            title: Title for the plot.

        Returns:
            Matplotlib figure with the 1D latent histogram.
        """
        rng = np.random.default_rng(42)
        if z.shape[0] > self.max_samples:
            idx = rng.choice(z.shape[0], self.max_samples, replace=False)
            z = z[idx]
            if phases is not None:
                phases = phases[idx]

        values = z.reshape(-1)
        num_bins = min(50, max(10, int(np.sqrt(values.shape[0]))))

        fig, axis = plt.subplots(figsize=(10, 5))
        sns.histplot(
            x=values,
            hue=phases.astype(int) if phases is not None else None,
            palette="tab10" if phases is not None else None,
            bins=num_bins,
            stat="density",
            common_bins=True,
            common_norm=False,
            element="step",
            alpha=0.5,
            ax=axis,
        )
        phase_suffix = " (per phase)" if phases is not None else ""
        axis.set_title(f"{title} histogram{phase_suffix}")
        axis.set_xlabel("Latent value")
        axis.set_ylabel("Density")
        plt.tight_layout()
        return fig

    def _create_latent_figure(
        self, z: np.ndarray, phases: np.ndarray | None, title: str = ""
    ) -> plt.Figure:
        """Create t-SNE visualization of latent space.

        Args:
            z: Latent samples (N, latent_dim).
            phases: Dominant phase per sample (N,), or None.
            title: Title for the plot.

        Returns:
            Matplotlib figure with latent space visualization.
        """
        rng = np.random.default_rng(42)
        if z.shape[0] > self.max_samples:
            idx = rng.choice(z.shape[0], self.max_samples, replace=False)
            z = z[idx]
            if phases is not None:
                phases = phases[idx]

        latent_dim = z.shape[1] if z.ndim > 1 else 1
        n_tsne_components = min(2, latent_dim)
        perplexity = min(30, z.shape[0] - 1)
        reducer = TSNE(
            n_components=n_tsne_components, random_state=42, perplexity=perplexity
        )
        z_2d = reducer.fit_transform(z)

        fig, ax = plt.subplots(figsize=(10, 8))

        y_vals = z_2d[:, 1] if z_2d.shape[1] > 1 else np.zeros(z_2d.shape[0])
        if phases is not None:
            n_phases = int(phases.max()) + 1
            cmap = plt.cm.get_cmap("tab10", n_phases)
            scatter = ax.scatter(
                z_2d[:, 0],
                y_vals,
                c=phases,
                cmap=cmap,
                alpha=0.6,
                s=10,
                vmin=-0.5,
                vmax=n_phases - 0.5,
            )
            plt.colorbar(scatter, ax=ax, label="Phase", ticks=range(n_phases))
            ax.set_title(f"{title} t-SNE (colored by phase mode)")
        else:
            ax.scatter(z_2d[:, 0], y_vals, alpha=0.6, s=10)
            ax.set_title(f"{title}  t-SNE")

        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2" if n_tsne_components > 1 else "")
        plt.tight_layout()
        return fig

    def _create_pca_figure(
        self, z: np.ndarray, phases: np.ndarray | None, title: str = ""
    ) -> plt.Figure:
        """Create PCA 2D projection of latent space.

        Args:
            z: Latent samples (N, latent_dim).
            phases: Dominant phase per sample (N,), or None.
            title: Title for the plot.

        Returns:
            Matplotlib figure with PCA projection.
        """
        rng = np.random.default_rng(42)
        if z.shape[0] > self.max_samples:
            idx = rng.choice(z.shape[0], self.max_samples, replace=False)
            z = z[idx]
            if phases is not None:
                phases = phases[idx]

        latent_dim = z.shape[1] if z.ndim > 1 else 1
        n_pca_components = min(2, latent_dim)
        pca = PCA(n_components=n_pca_components)
        projected = pca.fit_transform(z)
        explained_variance = pca.explained_variance_ratio_
        fig, axis = plt.subplots(figsize=(10, 8))
        y_vals = (
            projected[:, 1] if projected.shape[1] > 1 else np.zeros(projected.shape[0])
        )
        if phases is not None:
            sns.scatterplot(
                x=projected[:, 0],
                y=y_vals,
                hue=phases.astype(int),
                palette="tab10",
                alpha=0.6,
                s=10,
                legend="full",
                ax=axis,
            )
            axis.set_title(f"{title} PCA (colored by phase mode)")
        else:
            sns.scatterplot(
                x=projected[:, 0],
                y=y_vals,
                alpha=0.6,
                s=10,
                ax=axis,
            )
            axis.set_title(f"{title} PCA")
        axis.set_xlabel(f"PC1 ({explained_variance[0]:.1%})")
        axis.set_ylabel(
            f"PC2 ({explained_variance[1]:.1%})" if n_pca_components > 1 else ""
        )
        plt.tight_layout()
        return fig

    def _create_pca_variance_figure(self, z: np.ndarray, title: str = "") -> plt.Figure:
        """Create PCA explained variance histogram per latent dimension.

        Args:
            z: Latent samples (N, latent_dim).
            title: Title prefix for the plot.

        Returns:
            Matplotlib figure with per-component variance bar chart.
        """
        pca = PCA()
        pca.fit(z)
        n_components = len(pca.explained_variance_ratio_)
        fig, axis = plt.subplots(figsize=(10, 5))
        sns.barplot(
            x=list(range(n_components)),
            y=pca.explained_variance_ratio_.tolist(),
            ax=axis,
        )
        axis.set_xlabel("Principal Component")
        axis.set_ylabel("Explained Variance Ratio")
        axis.set_title(f"{title} - Explained Variance Per Dimension")
        plt.tight_layout()
        return fig

    def _create_latent_stats_table(
        self, metadata: dict[str, list[torch.Tensor]]
    ) -> wandb.Table | None:
        """Create a WandB table with latent space statistics.

        Args:
            metadata: Accumulated metadata dict from the metrics accumulator.

        Returns:
            WandB Table with per-latent-type statistics, or None if no latent data.
        """
        key_mapping = [
            ("mu_posterior", MetadataKey.POSTERIOR_MU.value),
            ("z_posterior", MetadataKey.POSTERIOR_Z.value),
            ("mu_prior", MetadataKey.PRIOR_MU.value),
            ("z_prior", MetadataKey.PRIOR_Z.value),
        ]
        rows = []
        for label, metadata_key in key_mapping:
            if metadata_key not in metadata:
                continue
            concatenated = torch.cat(metadata[metadata_key], dim=0).float()
            if concatenated.ndim == 3:
                concatenated = concatenated.view(concatenated.shape[0], -1)
            array = concatenated.numpy()
            per_dim_std = array.std(axis=0)
            collapsed_dims = int((per_dim_std < 0.01).sum())
            rows.append(
                [
                    label,
                    str(array.shape),
                    f"{array.mean():.4f}",
                    f"{array.mean(axis=0).std():.4f}",
                    f"{array.std():.4f}",
                    f"{per_dim_std.mean():.4f}",
                    f"{array.min():.3f}",
                    f"{array.max():.3f}",
                    collapsed_dims,
                ]
            )
        if not rows:
            return None
        columns = [
            "name",
            "shape",
            "mean",
            "per_dim_std_of_mean",
            "std",
            "per_dim_mean_of_std",
            "min",
            "max",
            "collapsed_dims",
        ]
        return wandb.Table(columns=columns, data=rows)


def _figure_to_wandb_image(fig: plt.Figure) -> wandb.Image:
    """Convert matplotlib figure to WandB image.

    Args:
        fig: Matplotlib figure

    Returns:
        WandB image object
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    pil_img = Image.open(buf)
    return wandb.Image(pil_img)
