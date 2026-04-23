"""Latent-space visualization callback for variational policies."""

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
import wandb
from pytorch_lightning.callbacks import Callback
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from versatil.metrics.constants import MetadataKey
from versatil.training.callbacks.wandb_figure import figure_to_wandb_image


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
            metrics = {key: figure_to_wandb_image(fig) for key, fig in figures.items()}
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
