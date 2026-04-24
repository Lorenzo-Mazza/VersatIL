"""Tests for versatil.metrics.accumulators module."""

from collections.abc import Callable

import numpy as np
import pytest
import torch

from versatil.metrics.accumulators import MetricsAccumulator, to_scalar
from versatil.metrics.base import LossOutput
from versatil.metrics.constants import MetadataKey, MetricKey


@pytest.fixture
def phase_loss_output_factory(
    rng: np.random.Generator,
) -> Callable[..., LossOutput]:
    def factory(
        total_loss: float = 1.0,
        batch_size: int = 4,
        horizon: int = 3,
        num_phases: int = 3,
    ) -> LossOutput:
        logits_data = rng.standard_normal((batch_size, horizon, num_phases)).astype(
            np.float32
        )
        logits = torch.from_numpy(logits_data)
        labels = torch.argmax(logits, dim=-1)
        return LossOutput(
            total_loss=torch.tensor(total_loss),
            component_losses={},
            metadata={
                MetadataKey.PHASE_LOGITS.value: logits,
                MetadataKey.PHASE_LABEL.value: labels,
            },
        )

    return factory


@pytest.fixture
def latent_loss_output_factory(
    rng: np.random.Generator,
) -> Callable[..., LossOutput]:
    def factory(
        total_loss: float = 1.0,
        batch_size: int = 4,
        latent_dimension: int = 8,
        include_prior: bool = False,
    ) -> LossOutput:
        metadata = {}
        mu_data = rng.standard_normal((batch_size, latent_dimension)).astype(np.float32)
        logvar_data = rng.standard_normal((batch_size, latent_dimension)).astype(
            np.float32
        )
        z_data = rng.standard_normal((batch_size, latent_dimension)).astype(np.float32)
        metadata[MetadataKey.POSTERIOR_MU.value] = torch.from_numpy(mu_data)
        metadata[MetadataKey.POSTERIOR_LOGVAR.value] = torch.from_numpy(logvar_data)
        metadata[MetadataKey.POSTERIOR_Z.value] = torch.from_numpy(z_data)
        if include_prior:
            prior_mu_data = rng.standard_normal((batch_size, latent_dimension)).astype(
                np.float32
            )
            prior_logvar_data = rng.standard_normal(
                (batch_size, latent_dimension)
            ).astype(np.float32)
            prior_z_data = rng.standard_normal((batch_size, latent_dimension)).astype(
                np.float32
            )
            metadata[MetadataKey.PRIOR_MU.value] = torch.from_numpy(prior_mu_data)
            metadata[MetadataKey.PRIOR_LOGVAR.value] = torch.from_numpy(
                prior_logvar_data
            )
            metadata[MetadataKey.PRIOR_Z.value] = torch.from_numpy(prior_z_data)
        return LossOutput(
            total_loss=torch.tensor(total_loss),
            component_losses={},
            metadata=metadata,
        )

    return factory


@pytest.mark.unit
class TestToScalar:
    def test_converts_torch_tensor_to_float(self):
        assert to_scalar(torch.tensor(3.14)) == pytest.approx(3.14)

    def test_converts_numpy_array_to_float(self):
        assert to_scalar(np.array(2.71)) == pytest.approx(2.71)

    def test_converts_python_float_to_float(self):
        assert to_scalar(42.0) == pytest.approx(42.0)

    def test_converts_python_int_to_float(self):
        result = to_scalar(7)
        assert result == pytest.approx(7.0)

    def test_detaches_gradient_tracked_tensor(self):
        tensor = torch.tensor(1.5, requires_grad=True) * 2
        result = to_scalar(tensor)
        assert result == pytest.approx(3.0)


@pytest.mark.unit
class TestMetricsAccumulatorAddLossOutput:
    def test_accumulates_total_loss(self, loss_output_factory):
        accumulator = MetricsAccumulator()
        accumulator.add_loss_output(loss_output_factory(total_loss_value=2.0))
        accumulator.add_loss_output(loss_output_factory(total_loss_value=3.0))
        assert accumulator.total_loss == pytest.approx(5.0)
        assert accumulator.num_batches == 2

    def test_accumulates_component_losses(self, loss_output_factory):
        accumulator = MetricsAccumulator()
        accumulator.add_loss_output(
            loss_output_factory(
                total_loss_value=1.0, component_losses={"mse": 0.5, "l1": 0.3}
            )
        )
        accumulator.add_loss_output(
            loss_output_factory(
                total_loss_value=1.0, component_losses={"mse": 0.7, "l1": 0.1}
            )
        )
        assert accumulator.component_metrics["mse"] == pytest.approx(1.2)
        assert accumulator.component_metrics["l1"] == pytest.approx(0.4)

    def test_stores_tensor_metadata_on_cpu(self):
        accumulator = MetricsAccumulator()
        gpu_like_tensor = torch.tensor([1.0, 2.0])
        output = LossOutput(
            total_loss=torch.tensor(1.0),
            metadata={"test_key": gpu_like_tensor},
        )
        accumulator.add_loss_output(output)
        stored = accumulator.metadata["test_key"][0]
        assert stored.device.type == "cpu"

    def test_stores_non_tensor_metadata(self):
        accumulator = MetricsAccumulator()
        output = LossOutput(
            total_loss=torch.tensor(1.0),
            metadata={"string_key": "some_value"},
        )
        accumulator.add_loss_output(output)
        assert accumulator.metadata["string_key"] == ["some_value"]


@pytest.mark.unit
class TestMetricsAccumulatorAverage:
    def test_returns_empty_dict_when_no_batches(self):
        accumulator = MetricsAccumulator()
        assert accumulator.average() == {}

    def test_computes_correct_average(self, loss_output_factory):
        accumulator = MetricsAccumulator()
        accumulator.add_loss_output(
            loss_output_factory(total_loss_value=4.0, component_losses={"mse": 2.0})
        )
        accumulator.add_loss_output(
            loss_output_factory(total_loss_value=6.0, component_losses={"mse": 4.0})
        )
        averaged = accumulator.average()
        assert averaged[MetricKey.TOTAL_LOSS.value] == pytest.approx(5.0)
        assert averaged["mse"] == pytest.approx(3.0)


@pytest.mark.unit
class TestMetricsAccumulatorPhaseMetrics:
    def test_returns_empty_dict_when_no_phase_data(self):
        accumulator = MetricsAccumulator()
        assert accumulator.compute_phase_metrics() == {}

    def test_computes_perfect_accuracy_with_matching_predictions(self):
        accumulator = MetricsAccumulator()
        num_phases = 3
        logits = torch.zeros(4, 2, num_phases)
        labels = torch.zeros(4, 2, dtype=torch.long)
        for batch_index in range(4):
            for time_index in range(2):
                correct_phase = (batch_index + time_index) % num_phases
                logits[batch_index, time_index, correct_phase] = 10.0
                labels[batch_index, time_index] = correct_phase
        output = LossOutput(
            total_loss=torch.tensor(1.0),
            metadata={
                MetadataKey.PHASE_LOGITS.value: logits,
                MetadataKey.PHASE_LABEL.value: labels,
            },
        )
        accumulator.add_loss_output(output)
        metrics = accumulator.compute_phase_metrics()
        assert metrics[MetricKey.PHASE_ACCURACY.value] == pytest.approx(1.0)

    def test_computes_per_phase_accuracy(self):
        accumulator = MetricsAccumulator()
        # Phase 0: 2 correct out of 2, Phase 1: 0 correct out of 2
        logits = torch.tensor(
            [
                [[10.0, -10.0], [10.0, -10.0]],
                [[10.0, -10.0], [10.0, -10.0]],
            ]
        )  # (2, 2, 2) - always predicts phase 0
        labels = torch.tensor(
            [
                [0, 0],
                [1, 1],
            ]
        )  # (2, 2)
        output = LossOutput(
            total_loss=torch.tensor(1.0),
            metadata={
                MetadataKey.PHASE_LOGITS.value: logits,
                MetadataKey.PHASE_LABEL.value: labels,
            },
        )
        accumulator.add_loss_output(output)
        metrics = accumulator.compute_phase_metrics()
        assert metrics["phase_0_accuracy"] == pytest.approx(1.0)
        assert metrics["phase_1_accuracy"] == pytest.approx(0.0)
        assert metrics[MetricKey.PHASE_ACCURACY.value] == pytest.approx(0.5)


@pytest.mark.unit
class TestMetricsAccumulatorConfusionMatrix:
    def test_returns_none_when_no_phase_data(self):
        accumulator = MetricsAccumulator()
        assert accumulator.compute_confusion_matrix() is None

    def test_returns_correct_confusion_matrix(self):
        accumulator = MetricsAccumulator()
        logits = torch.tensor(
            [
                [[10.0, -10.0], [-10.0, 10.0]],
            ]
        )  # Predicts [0, 1] for batch 0
        labels = torch.tensor([[0, 1]])
        output = LossOutput(
            total_loss=torch.tensor(1.0),
            metadata={
                MetadataKey.PHASE_LOGITS.value: logits,
                MetadataKey.PHASE_LABEL.value: labels,
            },
        )
        accumulator.add_loss_output(output)
        confusion = accumulator.compute_confusion_matrix()
        assert confusion[0, 0] == 1
        assert confusion[1, 1] == 1
        assert confusion[0, 1] == 0
        assert confusion[1, 0] == 0


@pytest.mark.unit
class TestMetricsAccumulatorExpertUsage:
    def test_returns_none_when_no_expert_data(self):
        accumulator = MetricsAccumulator()
        assert accumulator.compute_expert_usage() is None

    def test_computes_average_expert_usage(self):
        accumulator = MetricsAccumulator()
        usage_1 = torch.tensor([0.8, 0.2])
        usage_2 = torch.tensor([0.6, 0.4])
        for usage in [usage_1, usage_2]:
            output = LossOutput(
                total_loss=torch.tensor(1.0),
                metadata={MetadataKey.EXPERT_USAGE.value: usage},
            )
            accumulator.add_loss_output(output)
        result = accumulator.compute_expert_usage()
        expected = np.array([0.7, 0.3])
        np.testing.assert_allclose(
            result[MetadataKey.EXPERT_USAGE.value], expected, atol=1e-6
        )


@pytest.mark.unit
class TestMetricsAccumulatorLatentVisualization:
    def test_returns_nones_when_no_latent_data(self) -> None:
        accumulator = MetricsAccumulator()
        result = accumulator.compute_latent_visualization_data()
        assert result.posterior is None
        assert result.prior is None
        assert result.labels == {}

    def test_returns_posterior_z_as_numpy(
        self,
        latent_loss_output_factory: Callable[..., LossOutput],
    ) -> None:
        accumulator = MetricsAccumulator()
        batch_size, latent_dimension = 4, 8
        accumulator.add_loss_output(
            latent_loss_output_factory(
                batch_size=batch_size, latent_dimension=latent_dimension
            )
        )
        result = accumulator.compute_latent_visualization_data()
        assert result.posterior.shape == (batch_size, latent_dimension)
        assert result.prior is None
        assert result.labels == {}

    def test_returns_prior_z_when_available(
        self,
        latent_loss_output_factory: Callable[..., LossOutput],
    ) -> None:
        accumulator = MetricsAccumulator()
        batch_size, latent_dimension = 4, 8
        accumulator.add_loss_output(
            latent_loss_output_factory(
                batch_size=batch_size,
                latent_dimension=latent_dimension,
                include_prior=True,
            )
        )
        result = accumulator.compute_latent_visualization_data()
        assert result.posterior.shape == (batch_size, latent_dimension)
        assert result.prior.shape == (batch_size, latent_dimension)

    def test_reduces_configured_sequence_labels_to_mode_per_sample(
        self,
        rng: np.random.Generator,
    ) -> None:
        accumulator = MetricsAccumulator()
        batch_size, horizon = 4, 5
        labels = torch.zeros(batch_size, horizon, dtype=torch.long)
        labels[0, :] = 0
        labels[1, :3] = 1
        labels[1, 3:] = 0
        labels[2, :] = 2
        labels[3, :4] = 0
        labels[3, 4] = 1
        z_data = torch.from_numpy(
            rng.standard_normal((batch_size, 4)).astype(np.float32)
        )
        output = LossOutput(
            total_loss=torch.tensor(1.0),
            metadata={
                MetadataKey.LATENT_COLOR_LABEL.value: labels,
                MetadataKey.POSTERIOR_Z.value: z_data,
            },
        )
        accumulator.add_loss_output(output)
        result = accumulator.compute_latent_visualization_data(
            label_keys=[MetadataKey.LATENT_COLOR_LABEL.value]
        )
        label = result.labels[MetadataKey.LATENT_COLOR_LABEL.value]
        assert label[0] == 0
        assert label[1] == 1
        assert label[2] == 2
        assert label[3] == 0

    def test_returns_phase_labels_when_requested(
        self,
        rng: np.random.Generator,
    ) -> None:
        accumulator = MetricsAccumulator()
        labels = torch.tensor([[0, 0, 1], [2, 2, 1]], dtype=torch.long)
        z_data = torch.from_numpy(rng.standard_normal((2, 4)).astype(np.float32))
        output = LossOutput(
            total_loss=torch.tensor(1.0),
            metadata={
                MetadataKey.PHASE_LABEL.value: labels,
                MetadataKey.POSTERIOR_Z.value: z_data,
            },
        )
        accumulator.add_loss_output(output)

        result = accumulator.compute_latent_visualization_data(
            label_keys=[MetadataKey.PHASE_LABEL.value]
        )

        label = result.labels[MetadataKey.PHASE_LABEL.value]
        np.testing.assert_array_equal(label, np.array([0, 2]))

    def test_squeezes_trailing_dim_from_configured_labels(
        self,
        rng: np.random.Generator,
    ) -> None:
        accumulator = MetricsAccumulator()
        labels = torch.zeros(2, 3, 1, dtype=torch.long)
        z_data = torch.from_numpy(rng.standard_normal((2, 4)).astype(np.float32))
        output = LossOutput(
            total_loss=torch.tensor(1.0),
            metadata={
                MetadataKey.LATENT_COLOR_LABEL.value: labels,
                MetadataKey.POSTERIOR_Z.value: z_data,
            },
        )
        accumulator.add_loss_output(output)
        result = accumulator.compute_latent_visualization_data(
            label_keys=[MetadataKey.LATENT_COLOR_LABEL.value]
        )
        label = result.labels[MetadataKey.LATENT_COLOR_LABEL.value]
        assert label.shape == (2,)

    def test_omits_unrequested_labels_from_visualization_data(
        self,
        rng: np.random.Generator,
    ) -> None:
        accumulator = MetricsAccumulator()
        labels = torch.zeros(2, 1, dtype=torch.long)
        z_data = torch.from_numpy(rng.standard_normal((2, 4)).astype(np.float32))
        output = LossOutput(
            total_loss=torch.tensor(1.0),
            metadata={
                MetadataKey.LATENT_COLOR_LABEL.value: labels,
                MetadataKey.POSTERIOR_Z.value: z_data,
            },
        )
        accumulator.add_loss_output(output)
        result = accumulator.compute_latent_visualization_data(label_keys=[])
        assert result.labels == {}

    def test_flattens_3d_latents(
        self,
        rng: np.random.Generator,
    ) -> None:
        accumulator = MetricsAccumulator()
        batch_size, temporal, hidden = 4, 3, 8
        z_3d_data = rng.standard_normal((batch_size, temporal, hidden)).astype(
            np.float32
        )
        z_3d = torch.from_numpy(z_3d_data)
        output = LossOutput(
            total_loss=torch.tensor(1.0),
            metadata={MetadataKey.POSTERIOR_Z.value: z_3d},
        )
        accumulator.add_loss_output(output)
        result = accumulator.compute_latent_visualization_data()
        assert result.posterior.shape == (batch_size, temporal * hidden)


@pytest.mark.unit
class TestMetricsAccumulatorLatentStatistics:
    def test_returns_empty_when_no_latent_data(self):
        accumulator = MetricsAccumulator()
        assert accumulator.compute_latent_statistics() == {}

    def test_computes_posterior_statistics(self, latent_loss_output_factory):
        accumulator = MetricsAccumulator()
        accumulator.add_loss_output(latent_loss_output_factory())
        stats = accumulator.compute_latent_statistics()
        assert "posterior_mu_mean" in stats
        assert "posterior_mu_std" in stats
        assert "posterior_logvar_mean" in stats
        assert "posterior_logvar_std" in stats
        assert "posterior_std_mean" in stats
        assert "posterior_z_mean" in stats
        assert "posterior_z_std" in stats

    def test_computes_prior_statistics_when_available(self, latent_loss_output_factory):
        accumulator = MetricsAccumulator()
        accumulator.add_loss_output(latent_loss_output_factory(include_prior=True))
        stats = accumulator.compute_latent_statistics()
        assert "prior_mu_mean" in stats
        assert "prior_mu_std" in stats
        assert "prior_logvar_mean" in stats
        assert "prior_logvar_std" in stats
        assert "prior_std_mean" in stats
        assert "prior_z_mean" in stats
        assert "prior_z_std" in stats

    def test_posterior_std_mean_is_exp_half_logvar(self):
        accumulator = MetricsAccumulator()
        logvar = torch.tensor([[0.0, 2.0], [-1.0, 1.0]])
        output = LossOutput(
            total_loss=torch.tensor(1.0),
            metadata={MetadataKey.POSTERIOR_LOGVAR.value: logvar},
        )
        accumulator.add_loss_output(output)
        stats = accumulator.compute_latent_statistics()
        expected_std = (0.5 * logvar).exp().mean().item()
        assert stats["posterior_std_mean"] == pytest.approx(expected_std)


@pytest.mark.unit
class TestMetricsAccumulatorToDict:
    def test_includes_averaged_and_phase_and_latent_metrics(
        self, phase_loss_output_factory, latent_loss_output_factory
    ):
        accumulator = MetricsAccumulator()
        accumulator.add_loss_output(phase_loss_output_factory())
        accumulator.add_loss_output(latent_loss_output_factory())
        result = accumulator.to_dict()
        assert MetricKey.TOTAL_LOSS.value in result
        assert MetricKey.PHASE_ACCURACY.value in result
        assert "posterior_mu_mean" in result


@pytest.mark.unit
class TestMetricsAccumulatorReset:
    def test_clears_all_accumulated_data(self, loss_output_factory):
        accumulator = MetricsAccumulator()
        accumulator.add_loss_output(
            loss_output_factory(total_loss_value=5.0, component_losses={"mse": 2.0})
        )
        accumulator.reset()
        assert accumulator.total_loss == 0.0
        assert accumulator.component_metrics == {}
        assert accumulator.num_batches == 0
        assert accumulator.metadata == {}

    def test_average_returns_empty_after_reset(self, loss_output_factory):
        accumulator = MetricsAccumulator()
        accumulator.add_loss_output(loss_output_factory(total_loss_value=5.0))
        accumulator.reset()
        assert accumulator.average() == {}
