"""Metrics accumulator for tracking training and validation metrics."""

import math
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
    component_batch_counts: dict[str, int] = field(default_factory=dict)
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
                self.component_batch_counts[key] = 0
            self.component_metrics[key] += to_scalar(value)
            self.component_batch_counts[key] += 1

        # Store metadata for special metrics (e.g., phase predictions for confusion matrix)
        for key, value in loss_output.metadata.items():
            if value is None:
                continue
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

        # Average each component over the batches that actually emitted it:
        # stochastic components (e.g. prior/posterior mixing) appear only in a
        # subset of batches and would otherwise be deflated.
        for key, value in self.component_metrics.items():
            averaged[key] = value / self.component_batch_counts[key]

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

    def compute_vq_codebook_metrics(self) -> dict[str, float]:
        """Compute VQ posterior code usage metrics from hard assignments.

        Aggregates posterior code indices over an epoch. The global metrics are
        averaged over residual VQ layers; per-layer code frequencies are emitted
        separately to make single-code collapse visible in W&B scalar plots.
        """
        indices_key = MetadataKey.VQ_CODE_INDICES.value
        num_codes_key = MetadataKey.VQ_NUM_CODES.value
        if indices_key not in self.metadata or num_codes_key not in self.metadata:
            return {}

        stored_indices = self.metadata[indices_key]
        stored_num_codes = self.metadata[num_codes_key]
        if not isinstance(stored_indices, list) or not isinstance(
            stored_num_codes, list
        ):
            return {}

        code_indices = torch.cat(stored_indices, dim=1).long()  # (L, total_samples)
        num_codes_per_batch = torch.stack(stored_num_codes).flatten()  # (num_batches,)
        num_codes = int(num_codes_per_batch[0].item())
        num_layers = code_indices.shape[0]
        max_entropy = math.log(num_codes) if num_codes > 1 else 0.0

        metrics: dict[str, float] = {}
        usage_ratios = []
        entropies = []
        entropy_ratios = []
        perplexities = []
        max_frequencies = []
        dead_codes = []

        for layer_index in range(num_layers):
            layer_indices = code_indices[layer_index]  # (total_samples,)
            counts = torch.bincount(layer_indices, minlength=num_codes)[
                :num_codes
            ]  # (K,)
            probabilities = counts.float() / counts.sum().clamp(min=1)  # (K,)
            nonzero_probabilities = probabilities[probabilities > 0]
            entropy = -(nonzero_probabilities * torch.log(nonzero_probabilities)).sum()
            entropy_ratio = entropy / max_entropy if max_entropy > 0.0 else entropy
            perplexity = torch.exp(entropy)
            usage_ratio = (counts > 0).float().mean()
            max_frequency = probabilities.max()
            dead_code_count = (counts == 0).sum().float()

            usage_ratios.append(usage_ratio)
            entropies.append(entropy)
            entropy_ratios.append(entropy_ratio)
            perplexities.append(perplexity)
            max_frequencies.append(max_frequency)
            dead_codes.append(dead_code_count)

            layer_prefix = f"vq_codebook/layer_{layer_index}"
            metrics[f"{layer_prefix}/usage_ratio"] = float(usage_ratio.item())
            metrics[f"{layer_prefix}/entropy"] = float(entropy.item())
            metrics[f"{layer_prefix}/entropy_ratio"] = float(entropy_ratio.item())
            metrics[f"{layer_prefix}/perplexity"] = float(perplexity.item())
            metrics[f"{layer_prefix}/max_frequency"] = float(max_frequency.item())
            metrics[f"{layer_prefix}/dead_codes"] = float(dead_code_count.item())
            for code_index, probability in enumerate(probabilities):
                metrics[f"{layer_prefix}/code_{code_index}_frequency"] = float(
                    probability.item()
                )

        usage_tensor = torch.stack(usage_ratios)  # (L,)
        entropy_tensor = torch.stack(entropies)  # (L,)
        entropy_ratio_tensor = torch.stack(entropy_ratios)  # (L,)
        perplexity_tensor = torch.stack(perplexities)  # (L,)
        max_frequency_tensor = torch.stack(max_frequencies)  # (L,)
        dead_code_tensor = torch.stack(dead_codes)  # (L,)

        metrics[MetricKey.VQ_CODEBOOK_USAGE.value] = float(usage_tensor.mean().item())
        metrics[MetricKey.VQ_CODEBOOK_ENTROPY.value] = float(
            entropy_tensor.mean().item()
        )
        metrics[MetricKey.VQ_CODEBOOK_ENTROPY_RATIO.value] = float(
            entropy_ratio_tensor.mean().item()
        )
        metrics[MetricKey.VQ_CODEBOOK_PERPLEXITY.value] = float(
            perplexity_tensor.mean().item()
        )
        metrics[MetricKey.VQ_CODEBOOK_MAX_FREQUENCY.value] = float(
            max_frequency_tensor.mean().item()
        )
        metrics[MetricKey.VQ_CODEBOOK_DEAD_CODES.value] = float(
            dead_code_tensor.mean().item()
        )
        return metrics

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
        vq_codebook_metrics = self.compute_vq_codebook_metrics()
        if vq_codebook_metrics:
            metrics.update(vq_codebook_metrics)
        return metrics

    def reset(self):
        """Reset accumulator to initial state."""
        self.total_loss = 0.0
        self.component_metrics = {}
        self.component_batch_counts = {}
        self.num_batches = 0
        self.metadata = {}
