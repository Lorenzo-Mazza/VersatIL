"""Expert usage logging callback for mixture-of-experts models."""

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
from pytorch_lightning.callbacks import Callback

from versatil.training.callbacks.wandb_figure import figure_to_wandb_image


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
                    wandb_image = figure_to_wandb_image(fig)
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
        if trainer.sanity_checking:
            return
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return
        expert_usages = pl_module.val_metrics.compute_expert_usage()
        if expert_usages is not None:
            for key, expert_usage in expert_usages.items():
                fig = self._create_expert_usage_figure(expert_usage, f"Val {key}")
                if trainer.logger is not None:
                    wandb_image = figure_to_wandb_image(fig)
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
