"""Confusion matrix logging callback for phase classification models."""

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
from pytorch_lightning.callbacks import Callback

from versatil.training.callbacks.wandb_figure import figure_to_wandb_image


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

        cm = pl_module.train_metrics.compute_confusion_matrix()
        if cm is not None:
            fig = self._create_confusion_matrix_figure(
                cm, "Train Phase Confusion Matrix"
            )
            if trainer.logger is not None:
                wandb_image = figure_to_wandb_image(fig)
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
        cm = pl_module.val_metrics.compute_confusion_matrix()
        if cm is not None:
            fig = self._create_confusion_matrix_figure(cm, "Val Phase Confusion Matrix")
            if trainer.logger is not None:
                wandb_image = figure_to_wandb_image(fig)
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
