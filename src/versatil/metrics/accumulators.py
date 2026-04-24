"""Metrics accumulator for tracking training and validation metrics."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from versatil.metrics.base import LossOutput
from versatil.metrics.constants import MetadataKey, MetricKey


def to_scalar(value: Any) -> float:
    """Convert a value to a scalar float.

    Args:
        value: Tensor, numpy array, or scalar value

    Returns:
        Scalar float value
    """
    if isinstance(value, torch.Tensor):
        result: float = value.detach().item()
        return result
    elif isinstance(value, np.ndarray):
        return float(value)
    else:
        return float(value)


@dataclass
class LatentVisualizationData:
    """Latent arrays and optional categorical labels for plotting."""

    posterior: np.ndarray | None
    prior: np.ndarray | None
    labels: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class MetricsAccumulator:
    """Generic metrics accumulator for tracking losses and computing metrics across batches.

    This accumulator automatically tracks all component losses and metadata
    from LossOutput objects. It computes confusion matrices and accuracies
    for phase classification when phase metadata is available.
    """

    total_loss: float = 0.0
    component_metrics: dict[str, float] = field(default_factory=dict)
    num_batches: int = 0
    metadata: dict[str, list | torch.Tensor] = field(default_factory=dict)

    def add_loss_output(self, loss_output: LossOutput):
        """Add a loss output to the accumulator.

        Args:
            loss_output: LossOutput from a single batch
        """
        # Accumulate total loss
        self.total_loss += to_scalar(loss_output.total_loss)
        self.num_batches += 1

        # Accumulate component losses
        for key, value in loss_output.component_losses.items():
            if key not in self.component_metrics:
                self.component_metrics[key] = 0.0
            self.component_metrics[key] += to_scalar(value)

        # Store metadata for special metrics (e.g., phase predictions for confusion matrix)
        for key, value in loss_output.metadata.items():
            if key not in self.metadata:
                self.metadata[key] = []
            # Detach and move to CPU to save memory
            if isinstance(value, torch.Tensor):
                self.metadata[key].append(value.detach().cpu())
            else:
                self.metadata[key].append(value)

    def average(self) -> dict[str, float]:
        """Compute average metrics over all batches.

        Returns:
            Dictionary of averaged metric values
        """
        if self.num_batches == 0:
            return {}

        averaged = {
            MetricKey.TOTAL_LOSS.value: self.total_loss / self.num_batches,
        }

        for key, value in self.component_metrics.items():
            averaged[key] = value / self.num_batches

        return averaged

    def compute_phase_metrics(self) -> dict[str, float]:
        """Compute phase classification metrics from metadata if available.

        Returns:
            Dictionary with phase accuracy and per-phase accuracies, or empty dict
        """
        if MetadataKey.PHASE_LOGITS.value not in self.metadata:
            return {}

        # Concatenate all phase predictions and labels
        all_logits = torch.cat(self.metadata[MetadataKey.PHASE_LOGITS.value], dim=0)
        all_labels = torch.cat(self.metadata[MetadataKey.PHASE_LABEL.value], dim=0)

        # Compute predictions
        preds = torch.argmax(all_logits, dim=-1).flatten().numpy()
        labels = all_labels.flatten().numpy()

        # Overall accuracy
        accuracy = (preds == labels).mean()
        metrics = {MetricKey.PHASE_ACCURACY.value: float(accuracy)}

        # Per-phase accuracy
        n_phases = all_logits.shape[-1]
        for phase in range(n_phases):
            phase_mask = labels == phase
            if phase_mask.sum() > 0:
                phase_acc = (preds[phase_mask] == labels[phase_mask]).mean()
                metrics[f"phase_{phase}_accuracy"] = float(phase_acc)

        return metrics

    def compute_confusion_matrix(self) -> np.ndarray | None:
        """Compute confusion matrix from phase predictions if available.

        Returns:
            Confusion matrix as numpy array, or None if no phase data
        """
        if MetadataKey.PHASE_LOGITS.value not in self.metadata:
            return None

        # Concatenate all phase predictions and labels
        all_logits = torch.cat(self.metadata[MetadataKey.PHASE_LOGITS.value], dim=0)
        all_labels = torch.cat(self.metadata[MetadataKey.PHASE_LABEL.value], dim=0)

        # Compute predictions
        preds = torch.argmax(all_logits, dim=-1).flatten().numpy()
        labels = all_labels.flatten().numpy()

        # Compute confusion matrix
        n_phases = all_logits.shape[-1]
        cm = confusion_matrix(
            labels,
            preds,
            labels=list(range(n_phases)),
        )

        result: np.ndarray = cm
        return result

    def compute_expert_usage(self) -> dict[str, np.ndarray] | None:
        """Compute average expert usage from metadata if available.

        Returns:
            Expert usage ratio per expert as numpy array, or None if no expert usage data

        Note: this function supports multiple expert usage keys in metadata, but in practice we always only have one.
        """
        # TODO: drop support for multiple expert usage keys, it's unnecessary complexity
        expert_usages = {}
        for key in self.metadata:
            if key == MetadataKey.EXPERT_USAGE.value:
                all_usage = torch.stack(self.metadata[key], dim=0)
                expert_usages[key] = all_usage.float().mean(dim=0).numpy()
        if len(expert_usages.keys()) == 0:
            return None
        else:
            return expert_usages

    def compute_latent_visualization_data(
        self,
        label_keys: list[str] | None = None,
    ) -> LatentVisualizationData:
        """Compute latent visualization data with aligned categorical labels.

        Handles the shape mismatch between:
        - Latent z: (B, latent_dim) - one per sample
        - Labels: (B, T) or (B, T, 1) - multiple labels per sample

        Reduces sequence labels to one per sample using the mode.

        Returns:
            LatentVisualizationData with posterior/prior arrays and optional labels.
        """
        labels = {}
        for label_key in label_keys or []:
            if label_key in self.metadata:
                labels[label_key] = self._compute_label_per_sample(
                    metadata_key=label_key
                )
        return LatentVisualizationData(
            posterior=self._compute_latent_array(
                metadata_key=MetadataKey.POSTERIOR_Z.value
            ),
            prior=self._compute_latent_array(metadata_key=MetadataKey.PRIOR_Z.value),
            labels=labels,
        )

    def _compute_latent_array(self, metadata_key: str) -> np.ndarray | None:
        """Concatenate and flatten latent metadata for visualization."""
        if metadata_key not in self.metadata:
            return None
        latent = torch.cat(self.metadata[metadata_key], dim=0)
        if latent.ndim == 3:
            latent = latent.view(latent.shape[0], -1)
        return latent.float().numpy()

    def _compute_label_per_sample(self, metadata_key: str) -> np.ndarray:
        """Reduce sequence labels to one categorical value per sample."""
        labels = torch.cat(self.metadata[metadata_key], dim=0)
        if labels.ndim == 3 and labels.shape[-1] == 1:
            labels = labels.squeeze(-1)
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels.squeeze(-1)
        elif labels.ndim >= 2:
            labels = labels.view(labels.shape[0], -1)
            labels = torch.mode(labels, dim=1).values
        return labels.numpy()

    def compute_latent_statistics(self) -> dict[str, float]:
        """Compute scalar statistics from latent distribution metadata.

        Computes mean and std of mu, logvar, and z for both posterior and prior
        distributions when available. Useful for monitoring training stability
        and distribution behavior.

        Returns:
            Dictionary with scalar statistics for latent distributions.
        """
        stats: dict[str, float] = {}
        if MetadataKey.POSTERIOR_MU.value in self.metadata:
            all_mu = torch.cat(self.metadata[MetadataKey.POSTERIOR_MU.value], dim=0)
            stats["posterior_mu_mean"] = float(all_mu.mean().item())
            stats["posterior_mu_std"] = float(all_mu.std().item())

        if MetadataKey.POSTERIOR_LOGVAR.value in self.metadata:
            all_logvar = torch.cat(
                self.metadata[MetadataKey.POSTERIOR_LOGVAR.value], dim=0
            )
            stats["posterior_logvar_mean"] = float(all_logvar.mean().item())
            stats["posterior_logvar_std"] = float(all_logvar.std().item())
            all_std = (0.5 * all_logvar).exp()
            stats["posterior_std_mean"] = float(all_std.mean().item())

        if MetadataKey.POSTERIOR_Z.value in self.metadata:
            all_z = torch.cat(self.metadata[MetadataKey.POSTERIOR_Z.value], dim=0)
            stats["posterior_z_mean"] = float(all_z.float().mean().item())
            stats["posterior_z_std"] = float(all_z.float().std().item())

        if MetadataKey.PRIOR_MU.value in self.metadata:
            all_mu = torch.cat(self.metadata[MetadataKey.PRIOR_MU.value], dim=0)
            stats["prior_mu_mean"] = float(all_mu.mean().item())
            stats["prior_mu_std"] = float(all_mu.std().item())

        if MetadataKey.PRIOR_LOGVAR.value in self.metadata:
            all_logvar = torch.cat(self.metadata[MetadataKey.PRIOR_LOGVAR.value], dim=0)
            stats["prior_logvar_mean"] = float(all_logvar.mean().item())
            stats["prior_logvar_std"] = float(all_logvar.std().item())
            all_std = (0.5 * all_logvar).exp()
            stats["prior_std_mean"] = float(all_std.mean().item())

        if MetadataKey.PRIOR_Z.value in self.metadata:
            all_z = torch.cat(self.metadata[MetadataKey.PRIOR_Z.value], dim=0)
            stats["prior_z_mean"] = float(all_z.float().mean().item())
            stats["prior_z_std"] = float(all_z.float().std().item())

        return stats

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary of averaged metrics.

        Returns:
            Dictionary of metric values including optional phase and latent metrics
        """
        metrics = self.average()
        phase_metrics = self.compute_phase_metrics()
        if phase_metrics:
            metrics.update(phase_metrics)
        latent_stats = self.compute_latent_statistics()
        if latent_stats:
            metrics.update(latent_stats)
        return metrics

    def reset(self):
        """Reset accumulator to initial state."""
        self.total_loss = 0.0
        self.component_metrics = {}
        self.num_batches = 0
        self.metadata = {}
