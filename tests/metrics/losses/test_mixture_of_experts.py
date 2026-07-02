"""Tests for versatil.metrics.losses.mixture_of_experts module."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from versatil.configs.experiment import ExperimentConfig
from versatil.metrics.constants import MetadataKey, MetricKey
from versatil.metrics.losses.mixture_of_experts import MoELoss
from versatil.metrics.losses.regression import RegressionLoss
from versatil.models.decoding.constants import DecoderOutputKey
from versatil.training.callbacks.expert_usage import ExpertUsageCallback


@pytest.mark.unit
class TestMoELossGetRequiredKeys:
    def test_union_of_base_and_routing_keys(self):
        base_loss = RegressionLoss(action_keys=["position"])
        moe_loss = MoELoss(base_loss=base_loss)
        keys = moe_loss.get_required_keys()
        assert "position" in keys
        assert DecoderOutputKey.ROUTING_WEIGHTS.value in keys


class TestMoELossGetCallbacks:
    def test_returns_expert_usage_callback(self):
        base_loss = RegressionLoss(action_keys=["position"])
        moe_loss = MoELoss(base_loss=base_loss)
        experiment_config = MagicMock(spec=ExperimentConfig)
        callbacks = moe_loss.get_callbacks(experiment_config=experiment_config)
        assert len(callbacks) == 1
        assert isinstance(callbacks[0], ExpertUsageCallback)
        assert callbacks[0].log_every_n_epochs == 1


@pytest.mark.unit
class TestMoELossForward:
    def test_passes_through_base_loss(self):
        predictions = {
            "position": torch.ones(2, 3, 2),
            DecoderOutputKey.ROUTING_WEIGHTS.value: torch.tensor([[0.5, 0.5]]),
        }
        targets = {"position": torch.zeros(2, 3, 2)}
        base_loss = RegressionLoss(action_keys=["position"], mse_weight=1.0)
        moe_loss = MoELoss(base_loss=base_loss, entropy_weight=0.0)
        output = moe_loss(predictions, targets)
        # Base loss MSE = 1.0
        expected_mse = F.mse_loss(predictions["position"], targets["position"])
        assert output.total_loss.item() == pytest.approx(expected_mse.item())
        assert MetadataKey.EXPERT_USAGE.value in output.metadata

    def test_entropy_regularization_reduces_loss(self, rng):
        batch_size, horizon, action_dim, num_experts = 2, 3, 2, 3
        predictions = {
            "position": torch.ones(batch_size, horizon, action_dim),
            DecoderOutputKey.ROUTING_WEIGHTS.value: torch.softmax(
                torch.from_numpy(
                    rng.standard_normal((batch_size, num_experts)).astype(np.float32)
                ),
                dim=-1,
            ),
        }
        targets = {"position": torch.zeros(batch_size, horizon, action_dim)}
        base_loss = RegressionLoss(action_keys=["position"], mse_weight=1.0)
        moe_no_entropy = MoELoss(base_loss=base_loss, entropy_weight=0.0)
        moe_with_entropy = MoELoss(base_loss=base_loss, entropy_weight=1.0)
        output_no = moe_no_entropy(predictions, targets)
        output_with = moe_with_entropy(predictions, targets)
        # Entropy is subtracted, so total should be lower
        assert output_with.total_loss.item() < output_no.total_loss.item()
        assert MetricKey.EXPERTS_ENTROPY.value in output_with.component_losses

    def test_adds_weighted_mean_predictions_for_gmm(self):
        batch_size, horizon, num_experts, action_dim = 2, 3, 2, 2
        means = torch.ones(batch_size, horizon, num_experts, action_dim)
        means[:, :, 0] = 0.0  # expert 0 = 0
        means[:, :, 1] = 2.0  # expert 1 = 2
        routing_weights = torch.tensor([[0.5, 0.5]])
        predictions = {
            f"position_{DecoderOutputKey.MEAN.value}": means,
            DecoderOutputKey.ROUTING_WEIGHTS.value: routing_weights,
        }
        targets = {"position": torch.ones(batch_size, horizon, action_dim)}
        base_loss = RegressionLoss(action_keys=["position"], mse_weight=1.0)
        moe_loss = MoELoss(base_loss=base_loss, entropy_weight=0.0)
        output = moe_loss(predictions, targets)
        # Weighted mean = 0.5 * 0 + 0.5 * 2 = 1.0 = target => MSE ≈ 0
        assert output.total_loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_load_balance_minimum_at_uniform_per_trajectory_routing(self):
        batch_size, num_experts = 8, 4
        # Each example argmaxes to a different expert in round-robin → uniform usage
        routing_weights = torch.full((batch_size, num_experts), 1.0 / num_experts)
        argmax_targets = torch.arange(batch_size) % num_experts
        for b, k in enumerate(argmax_targets):
            routing_weights[b, k] += 1e-3  # break ties toward round-robin assignment
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        predictions = {
            "position": torch.zeros(batch_size, 3, 2),
            DecoderOutputKey.ROUTING_WEIGHTS.value: routing_weights,
        }
        targets = {"position": torch.zeros(batch_size, 3, 2)}
        base_loss = RegressionLoss(action_keys=["position"], mse_weight=1.0)
        moe_loss = MoELoss(
            base_loss=base_loss, entropy_weight=0.0, load_balance_weight=1.0
        )
        output = moe_loss(predictions, targets)
        load_balance = output.component_losses[
            MetricKey.EXPERTS_LOAD_BALANCE.value
        ].item()
        # f_k = 1/K, P_k ≈ 1/K, so K * sum(f_k * P_k) ≈ K * K * (1/K)² = 1.0
        assert load_balance == pytest.approx(1.0, rel=1e-3)

    def test_load_balance_penalises_collapsed_routing(self):
        batch_size, num_experts = 8, 4
        # All examples route exclusively to expert 0
        routing_weights = torch.zeros(batch_size, num_experts)
        routing_weights[:, 0] = 1.0
        predictions = {
            "position": torch.zeros(batch_size, 3, 2),
            DecoderOutputKey.ROUTING_WEIGHTS.value: routing_weights,
        }
        targets = {"position": torch.zeros(batch_size, 3, 2)}
        base_loss = RegressionLoss(action_keys=["position"], mse_weight=1.0)
        moe_loss = MoELoss(
            base_loss=base_loss, entropy_weight=0.0, load_balance_weight=1.0
        )
        output = moe_loss(predictions, targets)
        load_balance = output.component_losses[
            MetricKey.EXPERTS_LOAD_BALANCE.value
        ].item()
        # f_0 = 1, P_0 = 1, all others = 0 → K * sum = K * 1 = num_experts
        assert load_balance == pytest.approx(float(num_experts), rel=1e-5)

    def test_load_balance_handles_per_step_routing_with_padding(self):
        batch_size, horizon, num_experts = 2, 4, 4
        # Per-step routing collapsed at every valid timestep to expert 0;
        # padded timestep would (incorrectly) look balanced but must be ignored.
        routing_weights = torch.zeros(batch_size, horizon, num_experts)
        routing_weights[:, :3, 0] = 1.0
        routing_weights[:, 3, :] = 1.0 / num_experts
        is_pad = torch.tensor(
            [[False, False, False, True], [False, False, False, True]]
        )
        predictions = {
            "position": torch.zeros(batch_size, horizon, 2),
            DecoderOutputKey.ROUTING_WEIGHTS.value: routing_weights,
        }
        targets = {"position": torch.zeros(batch_size, horizon, 2)}
        base_loss = RegressionLoss(action_keys=["position"], mse_weight=1.0)
        moe_loss = MoELoss(
            base_loss=base_loss, entropy_weight=0.0, load_balance_weight=1.0
        )
        output = moe_loss(predictions, targets, is_pad=is_pad)
        load_balance = output.component_losses[
            MetricKey.EXPERTS_LOAD_BALANCE.value
        ].item()
        # Only valid positions count → all route to expert 0 → load_balance == K
        assert load_balance == pytest.approx(float(num_experts), rel=1e-5)
