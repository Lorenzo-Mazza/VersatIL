"""Tests for versatil.metrics.losses.prior_denoising module."""

import re

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from versatil.metrics.constants import MetadataKey, MetricKey
from versatil.metrics.losses.prior_denoising import PriorDenoisingLoss
from versatil.models.decoding.constants import LatentKey


@pytest.mark.unit
class TestPriorDenoisingLossGetRequiredKeys:
    def test_returns_prediction_and_target_keys(self):
        loss = PriorDenoisingLoss()
        keys = loss.get_required_keys()
        assert LatentKey.PRIOR_PREDICTION.value in keys
        assert LatentKey.PRIOR_TARGET.value in keys


@pytest.mark.unit
class TestPriorDenoisingLossForward:
    def test_identical_prediction_and_target_produce_zero_loss(self):
        z = torch.ones(4, 8)
        predictions = {
            LatentKey.PRIOR_PREDICTION.value: z,
            LatentKey.PRIOR_TARGET.value: z.clone(),
        }
        loss = PriorDenoisingLoss(weight=1.0)
        output = loss(predictions, {})
        assert output.total_loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_different_prediction_and_target_produce_positive_loss(self, rng):
        pred = torch.from_numpy(rng.standard_normal((4, 8)).astype(np.float32))
        target = torch.from_numpy(rng.standard_normal((4, 8)).astype(np.float32))
        predictions = {
            LatentKey.PRIOR_PREDICTION.value: pred,
            LatentKey.PRIOR_TARGET.value: target,
        }
        loss = PriorDenoisingLoss(weight=1.0)
        output = loss(predictions, {})
        expected = F.mse_loss(pred, target)
        assert output.total_loss.item() == pytest.approx(expected.item())

    def test_logs_target_scale_normalized_prior_metrics(self):
        pred = torch.zeros(2, 2)
        target = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
        predictions = {
            LatentKey.PRIOR_PREDICTION.value: pred,
            LatentKey.PRIOR_TARGET.value: target,
        }
        loss = PriorDenoisingLoss(weight=1.0)
        output = loss(predictions, {})

        prior_mse = F.mse_loss(pred, target)
        target_var = target.var(unbiased=False)
        target_std = torch.sqrt(target_var + 1e-8)
        assert output.component_losses[
            MetricKey.PRIOR_DENOISING_TARGET_STD.value
        ].item() == pytest.approx(target_std.item())
        assert output.component_losses[
            MetricKey.PRIOR_DENOISING_NORMALIZED_MSE.value
        ].item() == pytest.approx((prior_mse / (target_var + 1e-8)).item())
        assert output.component_losses[
            MetricKey.PRIOR_DENOISING_NORMALIZED_RMSE.value
        ].item() == pytest.approx((torch.sqrt(prior_mse) / target_std).item())

    def test_normalized_prior_metrics_are_finite_for_constant_target(self):
        pred = torch.zeros(2, 2)
        target = torch.ones(2, 2)
        predictions = {
            LatentKey.PRIOR_PREDICTION.value: pred,
            LatentKey.PRIOR_TARGET.value: target,
        }
        loss = PriorDenoisingLoss(weight=1.0)
        output = loss(predictions, {})

        assert torch.isfinite(
            output.component_losses[MetricKey.PRIOR_DENOISING_NORMALIZED_MSE.value]
        )
        assert torch.isfinite(
            output.component_losses[MetricKey.PRIOR_DENOISING_NORMALIZED_RMSE.value]
        )

    def test_weight_scales_loss(self, rng):
        pred = torch.from_numpy(rng.standard_normal((4, 8)).astype(np.float32))
        target = torch.from_numpy(rng.standard_normal((4, 8)).astype(np.float32))
        predictions = {
            LatentKey.PRIOR_PREDICTION.value: pred,
            LatentKey.PRIOR_TARGET.value: target,
        }
        loss_w1 = PriorDenoisingLoss(weight=1.0)
        loss_w3 = PriorDenoisingLoss(weight=3.0)
        output_w1 = loss_w1(predictions, {})
        output_w3 = loss_w3(predictions, {})
        assert output_w3.total_loss.item() == pytest.approx(
            3.0 * output_w1.total_loss.item(), rel=1e-5
        )

    def test_raises_on_missing_prediction(self):
        loss = PriorDenoisingLoss()
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Predictions must contain '{LatentKey.PRIOR_PREDICTION.value}' for PriorDenoisingLoss."
            ),
        ):
            loss({LatentKey.PRIOR_TARGET.value: torch.zeros(1)}, {})

    def test_raises_on_missing_target(self):
        loss = PriorDenoisingLoss()
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Predictions must contain '{LatentKey.PRIOR_TARGET.value}' for PriorDenoisingLoss."
            ),
        ):
            loss({LatentKey.PRIOR_PREDICTION.value: torch.zeros(1)}, {})

    def test_metadata_includes_latent_variables_when_available(self, rng):
        pred = torch.from_numpy(rng.standard_normal((4, 8)).astype(np.float32))
        target = torch.from_numpy(rng.standard_normal((4, 8)).astype(np.float32))
        z_posterior = torch.from_numpy(rng.standard_normal((4, 8)).astype(np.float32))
        mu_posterior = torch.from_numpy(rng.standard_normal((4, 8)).astype(np.float32))
        predictions = {
            LatentKey.PRIOR_PREDICTION.value: pred,
            LatentKey.PRIOR_TARGET.value: target,
            LatentKey.POSTERIOR_LATENT.value: z_posterior,
            LatentKey.POSTERIOR_MU.value: mu_posterior,
        }
        loss = PriorDenoisingLoss(weight=1.0)
        output = loss(predictions, {})
        assert MetadataKey.POSTERIOR_Z.value in output.metadata
        assert MetadataKey.POSTERIOR_MU.value in output.metadata
