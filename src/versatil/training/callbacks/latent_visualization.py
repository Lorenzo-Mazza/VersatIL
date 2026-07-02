"""Latent-space visualization callback for variational policies."""

import logging
import re

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
import wandb
from pytorch_lightning.callbacks import Callback
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from versatil.metrics.accumulators import MetricsAccumulator
from versatil.metrics.constants import MetadataKey
from versatil.training.callbacks.wandb_figure import figure_to_wandb_image

logger = logging.getLogger(__name__)


class LatentVisualizationCallback(Callback):
    """Visualize latent spaces with optional metadata coloring.

    Creates t-SNE/PCA projections of latent spaces. Label metadata is
    configured explicitly so synthetic modes, phase labels, task ids, or other
    categorical annotations can be used without task-specific callback logic.
    """

    def __init__(
        self,
        log_every_n_epochs: int = 5,
        max_samples: int = 5000,
        label_keys: list[str] | None = None,
    ) -> None:
        """Initialize latent visualization callback.

        Args:
            log_every_n_epochs: Log visualization every N epochs.
            max_samples: Maximum samples for t-SNE (subsamples if exceeded).
            label_keys: Metadata keys to use as categorical color labels.
        """
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.max_samples = max_samples
        self.label_keys = (
            label_keys
            if label_keys is not None
            else [
                MetadataKey.LATENT_COLOR_LABEL.value,
                MetadataKey.PHASE_LABEL.value,
            ]
        )

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
        metrics_accumulator: MetricsAccumulator,
        split: str,
    ) -> None:
        """Compute and log latent-space visualizations for the given metrics accumulator.

        Args:
            trainer: Lightning trainer.
            metrics_accumulator: Either train_metrics or val_metrics.
            split: "train" or "val" — used as a prefix on logged metric keys.
        """
        if trainer.sanity_checking:
            return
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        latent_data = metrics_accumulator.compute_latent_visualization_data(
            label_keys=self.label_keys
        )
        if latent_data.posterior is None and latent_data.prior is None:
            return

        figures = {}
        try:
            # Figure creation stays inside the guard: degenerate accumulations
            # (e.g. a single latent sample) make t-SNE raise, and that must
            # not abort the training epoch.
            if latent_data.posterior is not None:
                figures.update(
                    self._build_latent_figures(
                        z=latent_data.posterior,
                        labels_by_key=latent_data.labels,
                        prefix=f"{split}_posterior",
                        title=f"{split.title()} posterior latent space",
                    )
                )
            if latent_data.prior is not None:
                figures.update(
                    self._build_latent_figures(
                        z=latent_data.prior,
                        labels_by_key=latent_data.labels,
                        prefix=f"{split}_prior",
                        title=f"{split.title()} prior latent space",
                    )
                )
            latent_stats_table = self._create_latent_stats_table(
                metrics_accumulator.metadata
            )
            if trainer.logger is not None:
                metrics = {
                    key: figure_to_wandb_image(fig) for key, fig in figures.items()
                }
                if latent_stats_table is not None:
                    metrics[f"{split}_latent_space_statistics"] = latent_stats_table
                trainer.logger.log_metrics(metrics, step=trainer.current_epoch)
        except Exception:
            logger.warning(
                "Skipping %s latent visualization logging at epoch %s.",
                split,
                trainer.current_epoch,
                exc_info=True,
            )
        finally:
            for fig in figures.values():
                plt.close(fig)

    def _build_latent_figures(
        self,
        z: np.ndarray,
        labels_by_key: dict[str, np.ndarray],
        prefix: str,
        title: str,
    ) -> dict[str, plt.Figure]:
        """Dispatch to histogram for 1D latents or t-SNE/PCA for higher dim.

        Args:
            z: Latent samples (N, latent_dim) or (N,).
            labels_by_key: Mapping from metadata key to per-sample labels.
            prefix: Metric-key prefix (e.g. "posterior" or "prior").
            title: Human-readable figure title.

        Returns:
            Mapping from metric key to matplotlib figure.
        """
        labels_for_plots: dict[str, np.ndarray | None] = (
            labels_by_key if labels_by_key else {"": None}
        )
        latent_dim = z.shape[1] if z.ndim > 1 else 1
        figures = {}
        for label_key, labels in labels_for_plots.items():
            metric_suffix = self._metric_suffix(label_key=label_key)
            label_name = self._label_display_name(label_key=label_key)
            if latent_dim == 1:
                figures[f"{prefix}_latent_space_histogram{metric_suffix}"] = (
                    self._create_histogram_figure(
                        z=z, labels=labels, label_name=label_name, title=title
                    )
                )
                continue
            figures[f"{prefix}_latent_space_tsne{metric_suffix}"] = (
                self._create_latent_figure(
                    z=z, labels=labels, label_name=label_name, title=title
                )
            )
            figures[f"{prefix}_latent_space_pca{metric_suffix}"] = (
                self._create_pca_figure(
                    z=z, labels=labels, label_name=label_name, title=title
                )
            )
        if latent_dim > 1:
            figures[f"{prefix}_pca_explained_variance"] = (
                self._create_pca_variance_figure(z=z, title=title)
            )
        return figures

    def _create_histogram_figure(
        self,
        z: np.ndarray,
        labels: np.ndarray | None,
        label_name: str,
        title: str = "",
    ) -> plt.Figure:
        """Create a histogram of a 1D latent distribution.

        When per-sample labels are provided, plots one translucent
        histogram per label sharing the same bin edges so their shapes are
        directly comparable. Otherwise, plots a single histogram.

        Args:
            z: Latent samples (N, 1) or (N,).
            labels: Categorical label per sample (N,), or None.
            label_name: Human-readable label name.
            title: Title for the plot.

        Returns:
            Matplotlib figure with the 1D latent histogram.
        """
        rng = np.random.default_rng(42)
        if z.shape[0] > self.max_samples:
            idx = rng.choice(z.shape[0], self.max_samples, replace=False)
            z = z[idx]
            if labels is not None:
                labels = labels[idx]

        values = z.reshape(-1)
        num_bins = min(50, max(10, int(np.sqrt(values.shape[0]))))
        encoded_labels, label_values = self._encode_labels(labels=labels)

        fig, axis = plt.subplots(figsize=(10, 5))
        sns.histplot(
            x=values,
            hue=encoded_labels,
            palette="tab10" if encoded_labels is not None else None,
            bins=num_bins,
            stat="density",
            common_bins=True,
            common_norm=False,
            element="step",
            alpha=0.5,
            ax=axis,
        )
        if label_values is not None:
            axis.legend(title=label_name, labels=label_values)
        label_suffix = f" (colored by {label_name})" if labels is not None else ""
        axis.set_title(f"{title} histogram{label_suffix}")
        axis.set_xlabel("Latent value")
        axis.set_ylabel("Density")
        plt.tight_layout()
        return fig

    def _create_latent_figure(
        self,
        z: np.ndarray,
        labels: np.ndarray | None,
        label_name: str,
        title: str = "",
    ) -> plt.Figure:
        """Create t-SNE visualization of latent space.

        Args:
            z: Latent samples (N, latent_dim).
            labels: Categorical label per sample (N,), or None.
            label_name: Human-readable label name.
            title: Title for the plot.

        Returns:
            Matplotlib figure with latent space visualization.
        """
        rng = np.random.default_rng(42)
        if z.shape[0] > self.max_samples:
            idx = rng.choice(z.shape[0], self.max_samples, replace=False)
            z = z[idx]
            if labels is not None:
                labels = labels[idx]

        latent_dim = z.shape[1] if z.ndim > 1 else 1
        n_tsne_components = min(2, latent_dim)
        perplexity = min(30, z.shape[0] - 1)
        reducer = TSNE(
            n_components=n_tsne_components, random_state=42, perplexity=perplexity
        )
        z_2d = reducer.fit_transform(z)

        fig, ax = plt.subplots(figsize=(10, 8))

        y_vals = z_2d[:, 1] if z_2d.shape[1] > 1 else np.zeros(z_2d.shape[0])
        encoded_labels, label_values = self._encode_labels(labels=labels)
        if encoded_labels is not None and label_values is not None:
            n_labels = len(label_values)
            cmap = plt.get_cmap("tab10", n_labels)
            scatter = ax.scatter(
                z_2d[:, 0],
                y_vals,
                c=encoded_labels,
                cmap=cmap,
                alpha=0.6,
                s=10,
                vmin=-0.5,
                vmax=n_labels - 0.5,
            )
            colorbar = plt.colorbar(
                scatter, ax=ax, label=label_name, ticks=range(n_labels)
            )
            colorbar.ax.set_yticklabels(label_values)
            ax.set_title(f"{title} t-SNE (colored by {label_name})")
        else:
            ax.scatter(z_2d[:, 0], y_vals, alpha=0.6, s=10)
            ax.set_title(f"{title}  t-SNE")

        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2" if n_tsne_components > 1 else "")
        plt.tight_layout()
        return fig

    def _create_pca_figure(
        self,
        z: np.ndarray,
        labels: np.ndarray | None,
        label_name: str,
        title: str = "",
    ) -> plt.Figure:
        """Create PCA 2D projection of latent space.

        Args:
            z: Latent samples (N, latent_dim).
            labels: Categorical label per sample (N,), or None.
            label_name: Human-readable label name.
            title: Title for the plot.

        Returns:
            Matplotlib figure with PCA projection.
        """
        rng = np.random.default_rng(42)
        if z.shape[0] > self.max_samples:
            idx = rng.choice(z.shape[0], self.max_samples, replace=False)
            z = z[idx]
            if labels is not None:
                labels = labels[idx]

        latent_dim = z.shape[1] if z.ndim > 1 else 1
        n_pca_components = min(2, latent_dim)
        pca = PCA(n_components=n_pca_components)
        projected = pca.fit_transform(z)
        explained_variance = pca.explained_variance_ratio_
        fig, axis = plt.subplots(figsize=(10, 8))
        y_vals = (
            projected[:, 1] if projected.shape[1] > 1 else np.zeros(projected.shape[0])
        )
        encoded_labels, label_values = self._encode_labels(labels=labels)
        if encoded_labels is not None:
            sns.scatterplot(
                x=projected[:, 0],
                y=y_vals,
                hue=encoded_labels,
                palette="tab10",
                alpha=0.6,
                s=10,
                legend="full",
                ax=axis,
            )
            if axis.get_legend() is not None and label_values is not None:
                axis.get_legend().set_title(label_name)
                for text, value in zip(axis.get_legend().texts, label_values):
                    text.set_text(value)
            axis.set_title(f"{title} PCA (colored by {label_name})")
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

    @staticmethod
    def _encode_labels(
        labels: np.ndarray | None,
    ) -> tuple[np.ndarray | None, list[str] | None]:
        """Encode categorical labels as integers for plotting."""
        if labels is None:
            return None, None
        flattened_labels = labels.reshape(-1)
        unique_values, encoded_labels = np.unique(flattened_labels, return_inverse=True)
        label_values = [str(value) for value in unique_values.tolist()]
        return encoded_labels, label_values

    @staticmethod
    def _metric_suffix(label_key: str) -> str:
        """Create a metric-key suffix for a label metadata key."""
        if not label_key:
            return ""
        safe_key = re.sub(r"[^0-9a-zA-Z_]+", "_", label_key).strip("_")
        return f"_by_{safe_key}"

    @staticmethod
    def _label_display_name(label_key: str) -> str:
        """Create a readable label name for plot titles."""
        if not label_key:
            return "label"
        return label_key.replace("_", " ")

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
