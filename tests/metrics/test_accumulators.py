"""Tests for metrics accumulator class."""

import pytest
import torch

from refactoring.metrics.accumulators import MetricsAccumulator
from refactoring.metrics.base import LossOutput
from refactoring.metrics.constants import MetricKey, MetadataKey


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class TestMetricsAccumulator:
    """Tests for the generic MetricsAccumulator class."""

    def test_single_batch_accumulation(self, device):
        """Test accumulating a single batch of losses."""
        metrics = MetricsAccumulator()

        loss_output = LossOutput(
            total_loss=torch.tensor(1.5, device=device),
            component_losses={
                "mse_loss": torch.tensor(1.0, device=device),
                "bce_loss": torch.tensor(0.5, device=device),
            },
        )

        metrics.add_loss_output(loss_output)

        metrics_dict = metrics.to_dict()
        assert metrics_dict[MetricKey.TOTAL_LOSS.value] == 1.5
        assert metrics_dict["mse_loss"] == 1.0
        assert metrics_dict["bce_loss"] == 0.5

    def test_multiple_batch_averaging(self, device):
        """Test averaging across multiple batches."""
        metrics = MetricsAccumulator()

        for i in range(5):
            loss_output = LossOutput(
                total_loss=torch.tensor(1.0 * (i + 1), device=device),
                component_losses={"mse_loss": torch.tensor(0.5 * (i + 1), device=device)},
            )
            metrics.add_loss_output(loss_output)

        metrics_dict = metrics.to_dict()
        assert metrics_dict[MetricKey.TOTAL_LOSS.value] == 3.0  # (1+2+3+4+5)/5
        assert metrics_dict["mse_loss"] == 1.5  # (0.5+1+1.5+2+2.5)/5

    def test_reset_functionality(self, device):
        """Test that reset clears all accumulated metrics."""
        metrics = MetricsAccumulator()

        loss_output = LossOutput(
            total_loss=torch.tensor(1.5, device=device),
            component_losses={"mse_loss": torch.tensor(1.0, device=device)},
        )
        metrics.add_loss_output(loss_output)

        metrics.reset()

        assert metrics.num_batches == 0
        assert metrics.total_loss == 0.0
        assert len(metrics.component_metrics) == 0
        assert len(metrics.metadata) == 0

    def test_accumulates_all_loss_types(self, device):
        """Test that all loss types are accumulated correctly."""
        metrics = MetricsAccumulator()

        loss_output = LossOutput(
            total_loss=torch.tensor(2.0, device=device),
            component_losses={
                MetricKey.MSE_LOSS.value: torch.tensor(1.0, device=device),
                MetricKey.KL_DIVERGENCE.value: torch.tensor(0.5, device=device),
                MetricKey.GRIPPER_BCE.value: torch.tensor(0.5, device=device),
            },
        )

        metrics.add_loss_output(loss_output)
        metrics_dict = metrics.to_dict()

        assert MetricKey.TOTAL_LOSS.value in metrics_dict
        assert MetricKey.MSE_LOSS.value in metrics_dict
        assert MetricKey.KL_DIVERGENCE.value in metrics_dict
        assert MetricKey.GRIPPER_BCE.value in metrics_dict


class TestPhaseMetrics:
    """Tests for phase classification metrics computation."""

    def test_phase_metrics_from_metadata(self, device):
        """Test that phase metrics are computed from metadata."""
        metrics = MetricsAccumulator()

        batch_size = 4
        horizon = 10
        n_phases = 5

        phase_logits = torch.randn(batch_size, horizon, n_phases, device=device)
        phase_labels = torch.randint(0, n_phases, (batch_size, horizon), device=device)

        loss_output = LossOutput(
            total_loss=torch.tensor(1.5, device=device),
            component_losses={MetricKey.PHASE_CROSS_ENTROPY.value: torch.tensor(1.0, device=device)},
            metadata={
                MetadataKey.PHASE_LOGITS.value: phase_logits,
                MetadataKey.PHASE_LABEL.value: phase_labels,
            },
        )

        metrics.add_loss_output(loss_output)
        metrics_dict = metrics.to_dict()

        # Should include phase accuracy
        assert MetricKey.PHASE_ACCURACY.value in metrics_dict
        assert 0.0 <= metrics_dict[MetricKey.PHASE_ACCURACY.value] <= 1.0

        # Should include per-phase accuracies
        for phase in range(n_phases):
            key = f"phase_{phase}_accuracy"
            if key in metrics_dict:  # May not be present if no samples of that phase
                assert 0.0 <= metrics_dict[key] <= 1.0

    def test_confusion_matrix_computation(self, device):
        """Test confusion matrix computation."""
        metrics = MetricsAccumulator()

        batch_size = 4
        horizon = 10
        n_phases = 3

        phase_logits = torch.randn(batch_size, horizon, n_phases, device=device)
        phase_labels = torch.randint(0, n_phases, (batch_size, horizon), device=device)

        loss_output = LossOutput(
            total_loss=torch.tensor(1.5, device=device),
            component_losses={MetricKey.PHASE_CROSS_ENTROPY.value: torch.tensor(1.0, device=device)},
            metadata={
                MetadataKey.PHASE_LOGITS.value: phase_logits,
                MetadataKey.PHASE_LABEL.value: phase_labels,
            },
        )

        metrics.add_loss_output(loss_output)
        cm = metrics.compute_confusion_matrix()

        assert cm is not None
        assert cm.shape == (n_phases, n_phases)

    def test_perfect_phase_accuracy(self, device):
        """Test phase accuracy when predictions are perfect."""
        metrics = MetricsAccumulator()

        batch_size = 4
        horizon = 10
        n_phases = 3

        # Create perfect predictions
        phase_labels = torch.randint(0, n_phases, (batch_size, horizon), device=device)
        phase_logits = torch.zeros(batch_size, horizon, n_phases, device=device)
        for i in range(batch_size):
            for j in range(horizon):
                phase_logits[i, j, phase_labels[i, j]] = 10.0  # High logit for correct class

        loss_output = LossOutput(
            total_loss=torch.tensor(0.0, device=device),
            component_losses={MetricKey.PHASE_CROSS_ENTROPY.value: torch.tensor(0.0, device=device)},
            metadata={
                MetadataKey.PHASE_LOGITS.value: phase_logits,
                MetadataKey.PHASE_LABEL.value: phase_labels,
            },
        )

        metrics.add_loss_output(loss_output)
        metrics_dict = metrics.to_dict()

        # Accuracy should be 1.0
        assert metrics_dict[MetricKey.PHASE_ACCURACY.value] == 1.0

    def test_multiple_batches_phase_accumulation(self, device):
        """Test that phase metrics accumulate across batches."""
        metrics = MetricsAccumulator()

        batch_size = 2
        horizon = 5
        n_phases = 3

        for _ in range(3):  # 3 batches
            phase_logits = torch.randn(batch_size, horizon, n_phases, device=device)
            phase_labels = torch.randint(0, n_phases, (batch_size, horizon), device=device)

            loss_output = LossOutput(
                total_loss=torch.tensor(1.0, device=device),
                component_losses={MetricKey.PHASE_CROSS_ENTROPY.value: torch.tensor(1.0, device=device)},
                metadata={
                    MetadataKey.PHASE_LOGITS.value: phase_logits,
                    MetadataKey.PHASE_LABEL.value: phase_labels,
                },
            )

            metrics.add_loss_output(loss_output)

        # Should have stored metadata from all batches
        assert len(metrics.metadata[MetadataKey.PHASE_LOGITS.value]) == 3
        assert len(metrics.metadata[MetadataKey.PHASE_LABEL.value]) == 3

        # Should compute metrics across all batches
        metrics_dict = metrics.to_dict()
        assert MetricKey.PHASE_ACCURACY.value in metrics_dict

        # Confusion matrix should be computed on all data
        cm = metrics.compute_confusion_matrix()
        assert cm is not None
        assert cm.sum() == batch_size * horizon * 3  # 3 batches

    def test_no_phase_data_returns_none(self, device):
        """Test that confusion matrix returns None when no phase data."""
        metrics = MetricsAccumulator()

        loss_output = LossOutput(
            total_loss=torch.tensor(1.5, device=device),
            component_losses={"mse_loss": torch.tensor(1.5, device=device)},
        )

        metrics.add_loss_output(loss_output)

        cm = metrics.compute_confusion_matrix()
        assert cm is None

        # Phase metrics should be empty
        phase_metrics = metrics.compute_phase_metrics()
        assert len(phase_metrics) == 0
