"""Metrics accumulator for tracking training and validation metrics."""
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from refactoring.metrics.base import LossOutput
from refactoring.metrics.constants import MetadataKey, MetricKey


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
        all_labels = torch.cat(self.metadata[MetadataKey.PHASE_LABELS.value], dim=0)

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
        all_labels = torch.cat(self.metadata[MetadataKey.PHASE_LABELS.value], dim=0)

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
        """
        expert_usages = {}
        for key in self.metadata.keys():
            if MetadataKey.EXPERT_USAGE.value in key:
                all_usage = torch.stack(self.metadata[key], dim=0)
                logging.info(f"Computing expert usage for key: {key} with shape {all_usage.shape}")
                expert_usages[key] =  all_usage.mean(dim=0).numpy()
        if len(expert_usages.keys()) == 0:
            return None
        else:
            return expert_usages


    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary of averaged metrics.

        Returns:
            Dictionary of metric values including optional phase metrics
        """
        metrics = self.average()
        phase_metrics = self.compute_phase_metrics()
        if phase_metrics:
            metrics.update(phase_metrics)
        return metrics

    def reset(self):
        """Reset accumulator to initial state."""
        self.total_loss = 0.0
        self.component_metrics = {}
        self.num_batches = 0
        self.metadata = {}
