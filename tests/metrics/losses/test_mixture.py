"""Tests for versatil.metrics.losses.mixture module."""

import math
import re

import numpy as np
import pytest
import torch

from versatil.data.constants import GripperType
from versatil.metrics.constants import MetricKey
from versatil.metrics.losses.mixture import (
    GaussianMixtureNLLoss,
    GripperMixtureNLLoss,
)
from versatil.models.decoding.constants import DecoderOutputKey


@pytest.mark.unit
class TestGaussianMixtureNLLossGetRequiredKeys:
    def test_returns_action_keys(self):
        loss = GaussianMixtureNLLoss(action_keys=["position", "orientation"])
        assert loss.get_required_keys() == {"position", "orientation"}


@pytest.mark.unit
class TestGaussianMixtureNLLossForward:
    def test_perfect_match_produces_low_nll(self):
        batch_size, horizon, num_experts, action_dim = 2, 3, 2, 2
        # Target matches expert 0's mean exactly
        target = torch.ones(batch_size, horizon, action_dim)
        means = torch.zeros(batch_size, horizon, num_experts, action_dim)
        means[:, :, 0] = 1.0  # Expert 0 = perfect match
        logvars = -2.0 * torch.ones(
            batch_size, horizon, num_experts, action_dim
        )  # small variance
        routing_weights = torch.tensor([[0.99, 0.01]])  # heavily weight expert 0
        predictions = {
            "position_mean": means,
            "position_logvar": logvars,
            DecoderOutputKey.ROUTING_WEIGHTS.value: routing_weights,
        }
        targets = {"position": target}
        loss = GaussianMixtureNLLoss(action_keys=["position"], learned_variance=True)
        output = loss(predictions, targets)
        assert output.total_loss.item() < 5.0

    def test_fixed_variance_mode(self):
        batch_size, horizon, num_experts, action_dim = 2, 3, 2, 2
        target = torch.zeros(batch_size, horizon, action_dim)
        means = torch.zeros(batch_size, horizon, num_experts, action_dim)
        means[:, :, 0] = 0.0  # Expert 0 matches target
        routing_weights = torch.tensor([[0.5, 0.5]])
        predictions = {
            "position": means,
            DecoderOutputKey.ROUTING_WEIGHTS.value: routing_weights,
        }
        targets = {"position": target}
        loss = GaussianMixtureNLLoss(
            action_keys=["position"],
            learned_variance=False,
            sigmas={"position": 1.0},
        )
        output = loss(predictions, targets)
        assert output.total_loss.isfinite()

    def test_fixed_variance_reads_mean_key_from_gaussian_head(self):
        batch_size, horizon, num_experts, action_dim = 2, 3, 2, 2
        target = torch.zeros(batch_size, horizon, action_dim)
        means = torch.zeros(batch_size, horizon, num_experts, action_dim)
        means[:, :, 0] = 0.0
        routing_weights = torch.tensor([[0.5, 0.5]])
        predictions = {
            "position_mean": means,
            "position_logvar": torch.zeros_like(means),
            DecoderOutputKey.ROUTING_WEIGHTS.value: routing_weights,
        }
        targets = {"position": target}
        loss = GaussianMixtureNLLoss(
            action_keys=["position"],
            learned_variance=False,
            sigmas={"position": 1.0},
        )
        output = loss(predictions, targets)
        assert output.total_loss.isfinite()

    def test_weight_scales_output(self, rng):
        batch_size, horizon, num_experts, action_dim = 2, 3, 2, 2
        target_data = rng.standard_normal((batch_size, horizon, action_dim)).astype(
            np.float32
        )
        target = torch.from_numpy(target_data)
        means = torch.zeros(batch_size, horizon, num_experts, action_dim)
        routing_weights = torch.ones(1, num_experts) / num_experts
        predictions = {
            "position": means,
            DecoderOutputKey.ROUTING_WEIGHTS.value: routing_weights,
        }
        targets = {"position": target}
        loss_w1 = GaussianMixtureNLLoss(
            action_keys=["position"], weight=1.0, learned_variance=False
        )
        loss_w2 = GaussianMixtureNLLoss(
            action_keys=["position"], weight=2.0, learned_variance=False
        )
        output_w1 = loss_w1(predictions, targets)
        output_w2 = loss_w2(predictions, targets)
        assert output_w2.total_loss.item() == pytest.approx(
            2.0 * output_w1.total_loss.item(), rel=1e-4
        )

    def test_per_trajectory_routing_uses_joint_logsumexp(self):
        batch_size, horizon, num_experts, action_dim = 1, 3, 2, 2
        target = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])
        means = torch.zeros(batch_size, horizon, num_experts, action_dim)
        means[0, :, 0, :] = target[0]
        means[0, :, 1, :] = -target[0]
        logvars = torch.zeros(batch_size, horizon, num_experts, action_dim)
        routing_weights = torch.tensor([[0.5, 0.5]])
        predictions = {
            "position_mean": means,
            "position_logvar": logvars,
            DecoderOutputKey.ROUTING_WEIGHTS.value: routing_weights,
        }
        loss = GaussianMixtureNLLoss(action_keys=["position"], learned_variance=True)
        actual = loss(predictions, {"position": target}).total_loss
        log_norm = -0.5 * action_dim * math.log(2 * math.pi)
        log_pi = math.log(0.5)
        sum_sq_match = float((target**2).sum().item())
        log_traj_match = horizon * log_norm - 0.5 * 0.0
        log_traj_other = horizon * log_norm - 0.5 * 4.0 * sum_sq_match
        expected = (
            -math.log(
                math.exp(log_pi + log_traj_match) + math.exp(log_pi + log_traj_other)
            )
            / horizon
        )
        assert actual.item() == pytest.approx(expected, rel=1e-4)

    def test_per_step_routing_uses_per_step_logsumexp(self):
        batch_size, horizon, num_experts, action_dim = 1, 3, 2, 2
        target = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])
        means = torch.zeros(batch_size, horizon, num_experts, action_dim)
        means[0, :, 0, :] = target[0]
        means[0, :, 1, :] = -target[0]
        logvars = torch.zeros(batch_size, horizon, num_experts, action_dim)
        routing_weights = torch.full((batch_size, horizon, num_experts), 0.5)
        predictions = {
            "position_mean": means,
            "position_logvar": logvars,
            DecoderOutputKey.ROUTING_WEIGHTS.value: routing_weights,
        }
        loss = GaussianMixtureNLLoss(action_keys=["position"], learned_variance=True)
        actual = loss(predictions, {"position": target}).total_loss
        log_norm = -0.5 * action_dim * math.log(2 * math.pi)
        log_pi = math.log(0.5)
        per_step_nll = 0.0
        for step in range(horizon):
            sq_match = float((target[0, step] ** 2).sum().item())
            log_match = log_norm - 0.5 * 0.0
            log_other = log_norm - 0.5 * 4.0 * sq_match
            log_step_mix = math.log(
                math.exp(log_pi + log_match) + math.exp(log_pi + log_other)
            )
            per_step_nll -= log_step_mix
        expected = per_step_nll / horizon
        assert actual.item() == pytest.approx(expected, rel=1e-4)

    @pytest.mark.parametrize("per_step_routing", [False, True])
    def test_multi_key_routing_uses_joint_component_assignment(
        self, per_step_routing: bool
    ):
        batch_size, horizon, num_experts, action_dim = 1, 1, 2, 1
        target = torch.zeros(batch_size, horizon, action_dim)
        position_means = 10.0 * torch.ones(batch_size, horizon, num_experts, action_dim)
        orientation_means = 10.0 * torch.ones(
            batch_size, horizon, num_experts, action_dim
        )
        position_means[:, :, 0, :] = 0.0
        orientation_means[:, :, 1, :] = 0.0
        logvars = torch.zeros(batch_size, horizon, num_experts, action_dim)
        if per_step_routing:
            routing_weights = torch.tensor([[[0.5, 0.5]]])
            log_routing_weights = torch.log(routing_weights[:, 0, :])
        else:
            routing_weights = torch.tensor([[0.5, 0.5]])
            log_routing_weights = torch.log(routing_weights)
        predictions = {
            "position_mean": position_means,
            "position_logvar": logvars,
            "orientation_mean": orientation_means,
            "orientation_logvar": logvars,
            DecoderOutputKey.ROUTING_WEIGHTS.value: routing_weights,
        }
        loss = GaussianMixtureNLLoss(
            action_keys=["position", "orientation"], learned_variance=True
        )
        actual = loss(
            predictions=predictions,
            targets={"position": target, "orientation": target},
        )
        log_normalization = -0.5 * action_dim * math.log(2 * math.pi)
        joint_log_component = torch.tensor(
            [[2 * log_normalization - 50.0, 2 * log_normalization - 50.0]]
        )
        expected = -torch.logsumexp(
            log_routing_weights + joint_log_component, dim=-1
        ).mean()
        assert actual.total_loss.item() == pytest.approx(expected.item(), rel=1e-4)
        assert actual.component_losses[
            MetricKey.GAUSSIAN_MIXTURE_NLL.value
        ].item() == pytest.approx(expected.item(), rel=1e-4)

    def test_padding_excludes_padded_timesteps_from_trajectory_sum(self):
        batch_size, horizon, num_experts, action_dim = 1, 4, 2, 2
        target = torch.zeros(batch_size, horizon, action_dim)
        means = torch.zeros(batch_size, horizon, num_experts, action_dim)
        means[0, 3, :, :] = 100.0
        logvars = torch.zeros(batch_size, horizon, num_experts, action_dim)
        routing_weights = torch.tensor([[0.5, 0.5]])
        predictions = {
            "position_mean": means,
            "position_logvar": logvars,
            DecoderOutputKey.ROUTING_WEIGHTS.value: routing_weights,
        }
        loss = GaussianMixtureNLLoss(action_keys=["position"], learned_variance=True)
        is_pad = torch.tensor([[False, False, False, True]])
        finite = loss(predictions, {"position": target}, is_pad=is_pad).total_loss
        no_pad_target = target[:, :3].clone()
        no_pad_means = means[:, :3].clone()
        no_pad_logvars = logvars[:, :3].clone()
        reference_predictions = {
            "position_mean": no_pad_means,
            "position_logvar": no_pad_logvars,
            DecoderOutputKey.ROUTING_WEIGHTS.value: routing_weights,
        }
        reference = loss(reference_predictions, {"position": no_pad_target}).total_loss
        assert finite.item() == pytest.approx(reference.item(), rel=1e-5)

    def test_trajectory_loss_separates_opposite_modes(self):
        torch.manual_seed(0)
        batch_size, horizon, num_experts, action_dim = 256, 60, 2, 2
        timesteps = torch.linspace(0, 2 * math.pi, horizon)
        cos = torch.cos(timesteps)
        sin = torch.sin(timesteps)
        mode_a = torch.stack([cos, sin], dim=-1)
        mode_b = torch.stack([cos, -sin], dim=-1)
        targets = torch.stack(
            [mode_a if i < batch_size // 2 else mode_b for i in range(batch_size)]
        )
        targets = targets + 0.01 * torch.randn_like(targets)
        means = torch.nn.Parameter(torch.randn(num_experts, horizon, action_dim) * 0.3)
        logvars = torch.nn.Parameter(torch.zeros(num_experts, horizon, action_dim))
        pi_logits = torch.nn.Parameter(torch.zeros(num_experts))
        optimizer = torch.optim.Adam([means, logvars, pi_logits], lr=1e-2)
        loss = GaussianMixtureNLLoss(action_keys=["position"], learned_variance=True)
        for _ in range(800):
            stacked_means = (
                means.unsqueeze(0).expand(batch_size, -1, -1, -1).permute(0, 2, 1, 3)
            )
            stacked_logvars = (
                logvars.unsqueeze(0).expand(batch_size, -1, -1, -1).permute(0, 2, 1, 3)
            )
            routing_weights = (
                torch.softmax(pi_logits, dim=0).unsqueeze(0).expand(batch_size, -1)
            )
            output = loss(
                {
                    "position_mean": stacked_means,
                    "position_logvar": stacked_logvars,
                    DecoderOutputKey.ROUTING_WEIGHTS.value: routing_weights,
                },
                {"position": targets},
            )
            optimizer.zero_grad()
            output.total_loss.backward()
            optimizer.step()
        with torch.no_grad():
            mode_a_reference = mode_a
            mode_b_reference = mode_b
            distances_to_a = [
                (means[k] - mode_a_reference).pow(2).mean().sqrt().item()
                for k in range(num_experts)
            ]
            distances_to_b = [
                (means[k] - mode_b_reference).pow(2).mean().sqrt().item()
                for k in range(num_experts)
            ]
        assignment_one = distances_to_a[0] + distances_to_b[1]
        assignment_two = distances_to_a[1] + distances_to_b[0]
        best_assignment = min(assignment_one, assignment_two)
        assert best_assignment < 0.05


@pytest.mark.unit
class TestGripperMixtureNLLossInit:
    def test_binary_gripper_stores_type(self, binary_gripper_metadata_factory):
        metadata = binary_gripper_metadata_factory()
        loss = GripperMixtureNLLoss(key="gripper", actions_metadata=metadata)
        assert loss.gripper_type == GripperType.BINARY.value

    def test_raises_on_missing_key(self, binary_gripper_metadata_factory):
        metadata = binary_gripper_metadata_factory()
        with pytest.raises(
            ValueError,
            match=re.escape("wrong is not available to the action space"),
        ):
            GripperMixtureNLLoss(key="wrong", actions_metadata=metadata)


@pytest.mark.unit
class TestGripperMixtureNLLossForward:
    def test_binary_gripper_produces_finite_loss(self, binary_gripper_metadata_factory):
        metadata = binary_gripper_metadata_factory()
        loss = GripperMixtureNLLoss(
            key="gripper", actions_metadata=metadata, weight=1.0
        )
        batch_size, horizon, num_experts = 2, 3, 2
        expert_logits = torch.zeros(batch_size, horizon, num_experts)
        routing_weights = torch.tensor([[0.5, 0.5]])
        targets = {"gripper": torch.ones(batch_size, horizon)}
        predictions = {
            "gripper": expert_logits,
            DecoderOutputKey.ROUTING_WEIGHTS.value: routing_weights,
        }
        output = loss(predictions, targets)
        assert output.total_loss.isfinite()
        assert MetricKey.GRIPPER_NLL.value in output.component_losses

    def test_continuous_gripper_fixed_variance(
        self, continuous_gripper_metadata_factory
    ):
        metadata = continuous_gripper_metadata_factory()
        loss = GripperMixtureNLLoss(
            key="gripper",
            actions_metadata=metadata,
            weight=1.0,
            learned_variance=False,
            sigma=0.5,
        )
        batch_size, horizon, num_experts = 2, 3, 2
        means = torch.zeros(batch_size, horizon, num_experts, 1)
        routing_weights = torch.tensor([[0.5, 0.5]])
        targets = {"gripper": torch.zeros(batch_size, horizon, 1)}
        predictions = {
            "gripper": means,
            DecoderOutputKey.ROUTING_WEIGHTS.value: routing_weights,
        }
        output = loss(predictions, targets)
        assert output.total_loss.isfinite()

    def test_raises_on_missing_target_key(self, binary_gripper_metadata_factory):
        metadata = binary_gripper_metadata_factory()
        loss = GripperMixtureNLLoss(key="gripper", actions_metadata=metadata)
        with pytest.raises(
            ValueError,
            match=re.escape("Targets must contain 'gripper' for GripperMixtureNLLoss."),
        ):
            loss(
                {DecoderOutputKey.ROUTING_WEIGHTS.value: torch.zeros(1)},
                {"wrong": torch.zeros(1)},
            )


class TestMixtureLossEdgeBranches:
    def test_padded_timesteps_excluded_from_per_step_mixture(self):
        batch_size, horizon, num_experts, action_dim = 1, 2, 2, 2
        target = torch.zeros(batch_size, horizon, action_dim)
        means = torch.zeros(batch_size, horizon, num_experts, action_dim)
        means[:, 1] = 100.0  # padded step is wildly wrong
        routing_weights = torch.tensor([[[0.5, 0.5], [0.5, 0.5]]])  # (B, T, K)
        is_pad = torch.tensor([[False, True]])
        predictions = {
            "position_mean": means,
            "position_logvar": torch.zeros(
                batch_size, horizon, num_experts, action_dim
            ),
            DecoderOutputKey.ROUTING_WEIGHTS.value: routing_weights,
        }
        loss = GaussianMixtureNLLoss(action_keys=["position"], learned_variance=True)
        padded_output = loss(predictions, {"position": target}, is_pad=is_pad)
        means_clean = means.clone()
        means_clean[:, 1] = 0.0
        predictions_clean = dict(predictions, position_mean=means_clean)
        clean_output = loss(predictions_clean, {"position": target}, is_pad=is_pad)
        torch.testing.assert_close(padded_output.total_loss, clean_output.total_loss)

    def test_continuous_gripper_learned_variance(
        self, continuous_gripper_metadata_factory
    ):
        metadata = continuous_gripper_metadata_factory()
        loss = GripperMixtureNLLoss(
            key="gripper",
            actions_metadata=metadata,
            weight=1.0,
            learned_variance=True,
        )
        batch_size, horizon, num_experts = 2, 3, 2
        predictions = {
            f"gripper_{DecoderOutputKey.MEAN.value}": torch.zeros(
                batch_size, horizon, num_experts, 1
            ),
            f"gripper_{DecoderOutputKey.LOGVAR.value}": torch.zeros(
                batch_size, horizon, num_experts, 1
            ),
            DecoderOutputKey.ROUTING_WEIGHTS.value: torch.tensor([[0.5, 0.5]]),
        }
        targets = {"gripper": torch.zeros(batch_size, horizon, 1)}
        output = loss(predictions, targets)
        assert output.total_loss.isfinite()

    def test_missing_target_key_raises(self, continuous_gripper_metadata_factory):
        metadata = continuous_gripper_metadata_factory()
        loss = GripperMixtureNLLoss(key="gripper", actions_metadata=metadata)
        with pytest.raises(ValueError, match="Targets must contain"):
            loss({}, {})
