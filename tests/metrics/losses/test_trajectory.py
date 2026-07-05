"""Tests for versatil.metrics.losses.trajectory module."""

import re

import pytest
import torch

from versatil.metrics.constants import MetricKey
from versatil.metrics.losses.trajectory import (
    TrajectoryLengthLoss,
    TrajectorySmoothness,
)


@pytest.mark.unit
class TestTrajectoryLengthLossForward:
    def test_identical_trajectories_produce_zero_loss(self):
        trajectory = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
        predictions = {"position": trajectory}
        targets = {"position": trajectory.clone()}
        loss = TrajectoryLengthLoss(action_key="position", weight=1.0)
        output = loss(predictions, targets)
        assert output.total_loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_different_length_trajectories_produce_positive_loss(self):
        pred = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
        target = torch.tensor([[[0.0, 0.0], [3.0, 0.0], [6.0, 0.0]]])
        predictions = {"position": pred}
        targets = {"position": target}
        loss = TrajectoryLengthLoss(action_key="position", weight=1.0)
        output = loss(predictions, targets)
        # pred length = 1 + 1 = 2, target length = 3 + 3 = 6
        # Per step norms: pred = [1, 1], target = [3, 3]
        # Mean of norms: pred = 1.0, target = 3.0
        # (1.0 - 3.0)^2 = 4.0
        assert output.total_loss.item() == pytest.approx(4.0)
        assert MetricKey.LENGTH_LOSS.value in output.component_losses

    def test_opposite_errors_do_not_cancel_across_batch(self):
        pred = torch.tensor(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]],
            ]
        )
        target = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            ]
        )
        loss = TrajectoryLengthLoss(action_key="position", weight=1.0)
        output = loss({"position": pred}, {"position": target})
        # Per-sample mean step lengths: pred [0, 2], target [1, 1].
        # Per-sample squared errors: [1, 1] -> mean 1, not (1 - 1)^2 = 0.
        assert output.total_loss.item() == pytest.approx(1.0)

    def test_padded_steps_do_not_contribute_to_length(self):
        pred = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [9.0, 9.0]]])
        target = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [-9.0, 9.0]]])
        is_pad = torch.tensor([[False, False, True]])
        loss = TrajectoryLengthLoss(action_key="position", weight=1.0)
        output = loss({"position": pred}, {"position": target}, is_pad=is_pad)
        # Only the first step is valid and matches exactly.
        assert output.total_loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_fully_padded_sample_is_excluded_from_batch_mean(self):
        pred = torch.tensor(
            [
                [[0.0, 0.0], [2.0, 0.0]],
                [[0.0, 0.0], [5.0, 0.0]],
            ]
        )
        target = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [1.0, 0.0]],
            ]
        )
        is_pad = torch.tensor([[False, False], [True, True]])
        loss = TrajectoryLengthLoss(action_key="position", weight=1.0)
        output = loss({"position": pred}, {"position": target}, is_pad=is_pad)
        # Sample 0: lengths 2 vs 1 -> error 1. Sample 1 has no valid steps.
        assert output.total_loss.item() == pytest.approx(1.0)
        assert not torch.isnan(output.total_loss)

    def test_horizon_one_returns_zero_loss(self):
        pred = torch.tensor([[[1.0, 2.0]]])
        target = torch.tensor([[[3.0, 4.0]]])
        loss = TrajectoryLengthLoss(action_key="position", weight=1.0)
        output = loss({"position": pred}, {"position": target})
        assert output.total_loss.item() == pytest.approx(0.0)
        assert not torch.isnan(output.total_loss)

    def test_raises_on_missing_key(self):
        loss = TrajectoryLengthLoss(action_key="position")
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Predictions and targets must contain key 'position' for TrajectoryLengthLoss."
            ),
        ):
            loss({"wrong": torch.zeros(1)}, {"wrong": torch.zeros(1)})

    def test_get_required_keys(self):
        loss = TrajectoryLengthLoss(action_key="position")
        assert loss.get_required_keys() == {"position"}


@pytest.mark.unit
class TestTrajectorySmoothnessForward:
    def test_padded_timesteps_do_not_contribute_to_smoothness(self):
        # Linear motion in valid steps; a huge kink only at the padded tail.
        pred = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [50.0, 50.0]]])
        target = pred.clone()
        is_pad = torch.tensor([[False, False, False, True]])
        loss = TrajectorySmoothness(action_key="position", weight=1.0)
        output = loss({"position": pred}, {"position": target}, is_pad=is_pad)
        assert output.total_loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_linear_trajectory_has_zero_smoothness(self):
        # Constant velocity => zero acceleration
        trajectory = torch.tensor([[[0.0], [1.0], [2.0], [3.0]]])
        predictions = {"position": trajectory}
        loss = TrajectorySmoothness(action_key="position", weight=1.0)
        output = loss(predictions, {})
        assert output.total_loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_non_linear_trajectory_has_positive_smoothness(self):
        # Accelerating trajectory
        trajectory = torch.tensor([[[0.0], [1.0], [4.0], [9.0]]])
        predictions = {"position": trajectory}
        loss = TrajectorySmoothness(action_key="position", weight=1.0)
        output = loss(predictions, {})
        assert output.total_loss.item() > 0.0

    def test_too_short_trajectory_returns_zero(self):
        trajectory = torch.tensor([[[0.0], [1.0]]])  # Only 2 timesteps
        predictions = {"position": trajectory}
        loss = TrajectorySmoothness(action_key="position", weight=1.0)
        output = loss(predictions, {})
        assert output.total_loss.item() == pytest.approx(0.0)

    def test_get_required_keys_returns_empty_set(self):
        loss = TrajectorySmoothness(action_key="position")
        assert loss.get_required_keys() == set()

    def test_raises_on_missing_key(self):
        loss = TrajectorySmoothness(action_key="position")
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Predictions must contain key 'position' for TrajectorySmoothness loss."
            ),
        ):
            loss({"wrong": torch.zeros(1)}, {})
