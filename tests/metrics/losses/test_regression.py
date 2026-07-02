"""Tests for versatil.metrics.losses.regression module."""

import re

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from versatil.metrics.constants import MetricKey
from versatil.metrics.losses.regression import RegressionLoss


@pytest.mark.unit
class TestRegressionLossInit:
    def test_stores_action_keys(self):
        loss = RegressionLoss(action_keys=["position", "orientation"])
        assert loss.action_keys == ["position", "orientation"]

    @pytest.mark.parametrize(
        "mse_weight, l1_weight, huber_weight",
        [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.5, 0.3, 0.2),
        ],
    )
    def test_stores_loss_weights(self, mse_weight, l1_weight, huber_weight):
        loss = RegressionLoss(
            action_keys=["position"],
            mse_weight=mse_weight,
            l1_weight=l1_weight,
            huber_weight=huber_weight,
        )
        assert loss.mse_weight == mse_weight
        assert loss.l1_weight == l1_weight
        assert loss.huber_weight == huber_weight


@pytest.mark.unit
class TestRegressionLossGetRequiredKeys:
    def test_returns_action_keys_as_set(self):
        loss = RegressionLoss(action_keys=["position", "orientation"])
        assert loss.get_required_keys() == {"position", "orientation"}


@pytest.mark.unit
class TestRegressionLossForward:
    def test_mse_only_computes_correct_value(self, action_tensor_factory):
        predictions = {"position": torch.tensor([[[1.0, 2.0]]])}
        targets = {"position": torch.tensor([[[3.0, 4.0]]])}
        loss = RegressionLoss(action_keys=["position"], mse_weight=1.0)
        output = loss(predictions, targets)
        # MSE = ((1-3)^2 + (2-4)^2) / 2 = (4 + 4) / 2 = 4.0
        expected_mse = F.mse_loss(predictions["position"], targets["position"])
        assert output.total_loss.item() == pytest.approx(expected_mse.item())

    def test_l1_only_computes_correct_value(self):
        predictions = {"position": torch.tensor([[[1.0, 2.0]]])}
        targets = {"position": torch.tensor([[[3.0, 5.0]]])}
        loss = RegressionLoss(
            action_keys=["position"],
            mse_weight=0.0,
            l1_weight=1.0,
        )
        output = loss(predictions, targets)
        # L1 = (|1-3| + |2-5|) / 2 = (2 + 3) / 2 = 2.5
        expected = F.l1_loss(predictions["position"], targets["position"])
        assert output.total_loss.item() == pytest.approx(expected.item())

    def test_huber_only_computes_correct_value(self):
        predictions = {"position": torch.tensor([[[0.0]]])}
        targets = {"position": torch.tensor([[[0.3]]])}
        delta = 1.0
        loss = RegressionLoss(
            action_keys=["position"],
            mse_weight=0.0,
            huber_weight=1.0,
            huber_delta=delta,
        )
        output = loss(predictions, targets)
        expected = F.huber_loss(
            predictions["position"], targets["position"], delta=delta
        )
        assert output.total_loss.item() == pytest.approx(expected.item())

    def test_per_key_weights_scale_loss(self):
        predictions = {
            "position": torch.ones(1, 1, 3),
            "orientation": torch.ones(1, 1, 1),
        }
        targets = {
            "position": torch.zeros(1, 1, 3),
            "orientation": torch.zeros(1, 1, 1),
        }
        loss = RegressionLoss(
            action_keys=["position", "orientation"],
            mse_weight=1.0,
            per_key_weights={"position": 2.0, "orientation": 0.5},
        )
        output = loss(predictions, targets)
        # position MSE = 1.0, orientation MSE = 1.0
        # total = 1.0 * 2.0 * 1.0 + 1.0 * 0.5 * 1.0 = 2.5
        assert output.total_loss.item() == pytest.approx(2.5)

    def test_component_losses_are_keyed_correctly(self):
        predictions = {"position": torch.ones(1, 1, 3)}
        targets = {"position": torch.zeros(1, 1, 3)}
        loss = RegressionLoss(
            action_keys=["position"],
            mse_weight=1.0,
            l1_weight=1.0,
        )
        output = loss(predictions, targets)
        assert f"position_{MetricKey.MSE_LOSS.value}" in output.component_losses
        assert f"position_{MetricKey.L1_LOSS.value}" in output.component_losses

    def test_raises_on_missing_key(self):
        predictions = {"wrong_key": torch.ones(1, 1, 3)}
        targets = {"position": torch.zeros(1, 1, 3)}
        loss = RegressionLoss(action_keys=["position"])
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Predictions and targets must contain key 'position' for RegressionLoss."
            ),
        ):
            loss(predictions, targets)

    def test_padding_mask_excludes_padded_positions(self):
        batch_size, horizon, action_dim = 1, 4, 2
        predictions = {"position": torch.ones(batch_size, horizon, action_dim)}
        targets = {"position": torch.zeros(batch_size, horizon, action_dim)}
        # Padded positions carry a huge error that must not leak into the loss.
        targets["position"][:, 2:] = 100.0
        is_pad = torch.tensor([[False, False, True, True]])
        loss = RegressionLoss(action_keys=["position"], mse_weight=1.0)
        output = loss(predictions, targets, is_pad=is_pad)
        # Valid positions: MSE (1 - 0)^2 = 1.0 per element, averaged over the
        # valid elements only.
        assert output.total_loss.item() == pytest.approx(1.0, abs=1e-6)

    def test_padding_mean_matches_unmasked_element_scale(self, rng):
        batch_size, horizon, action_dim = 2, 4, 3
        data = rng.standard_normal((batch_size, horizon, action_dim)).astype(np.float32)
        predictions = {"position": torch.from_numpy(data)}
        targets = {"position": torch.zeros(batch_size, horizon, action_dim)}
        all_valid = torch.zeros(batch_size, horizon, dtype=torch.bool)
        loss = RegressionLoss(action_keys=["position"], mse_weight=1.0)
        output_masked = loss(predictions, targets, is_pad=all_valid)
        output_unmasked = loss(predictions, targets)
        assert output_masked.total_loss.item() == pytest.approx(
            output_unmasked.total_loss.item(), abs=1e-6
        )

    def test_handles_long_dtype_targets(self):
        predictions = {"gripper": torch.tensor([[[0.8]]])}
        targets = {"gripper": torch.tensor([[[1]]], dtype=torch.long)}
        loss = RegressionLoss(action_keys=["gripper"], mse_weight=1.0)
        output = loss(predictions, targets)
        expected = F.mse_loss(predictions["gripper"], targets["gripper"].float())
        assert output.total_loss.item() == pytest.approx(expected.item())
