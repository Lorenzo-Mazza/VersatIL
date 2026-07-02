"""Tests for versatil.metrics.losses.latent_geometry module."""

import re
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import torch

from versatil.metrics.constants import MetricKey
from versatil.metrics.losses.latent_geometry import (
    PosteriorGeometryLoss,
    VICLatentLoss,
)
from versatil.models.decoding.constants import LatentKey


@pytest.mark.unit
class TestVICLatentLossGetRequiredKeys:
    def test_returns_key(self):
        loss = VICLatentLoss(key=LatentKey.POSTERIOR_MU.value)
        assert loss.get_required_keys() == {LatentKey.POSTERIOR_MU.value}


@pytest.mark.unit
class TestVICLatentLossForward:
    def test_zero_variance_produces_positive_variance_loss(self):
        # All vectors identical => std = 0, which is below gamma
        latent = torch.ones(8, 4)
        predictions = {LatentKey.POSTERIOR_MU.value: latent}
        loss = VICLatentLoss(
            key=LatentKey.POSTERIOR_MU.value,
            covariance_weight=0.0,
            variance_weight=1.0,
            gamma=0.3,
        )
        output = loss(predictions, {})
        assert output.total_loss.item() > 0.0
        assert MetricKey.VARIANCE_LOSS.value in output.component_losses

    def test_correlated_dimensions_produce_positive_covariance_loss(self):
        batch_size = 32
        # Two dimensions perfectly correlated: x2 = x1
        x1 = torch.linspace(-1, 1, batch_size).unsqueeze(1)
        latent = torch.cat([x1, x1], dim=1)
        predictions = {LatentKey.POSTERIOR_MU.value: latent}
        loss = VICLatentLoss(
            key=LatentKey.POSTERIOR_MU.value,
            covariance_weight=1.0,
            variance_weight=0.0,
        )
        output = loss(predictions, {})
        assert output.total_loss.item() > 0.0
        assert MetricKey.COVARIANCE_LOSS.value in output.component_losses

    def test_independent_high_variance_produces_low_loss(self, rng):
        batch_size, latent_dim = 64, 4
        # Independent, high-variance data
        data = rng.standard_normal((batch_size, latent_dim)).astype(np.float32) * 2.0
        latent = torch.from_numpy(data)
        predictions = {LatentKey.POSTERIOR_MU.value: latent}
        loss = VICLatentLoss(
            key=LatentKey.POSTERIOR_MU.value,
            covariance_weight=1.0,
            variance_weight=1.0,
            gamma=0.3,
        )
        output = loss(predictions, {})
        # With independent, high-variance data, both losses should be low
        assert output.component_losses[MetricKey.VARIANCE_LOSS.value].item() < 0.5

    def test_raises_on_missing_key(self):
        loss = VICLatentLoss(key=LatentKey.POSTERIOR_MU.value)
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Predictions must contain '{LatentKey.POSTERIOR_MU.value}' for VICLatentLoss."
            ),
        ):
            loss({}, {})


@pytest.mark.unit
class TestPosteriorGeometryLossGetRequiredKeys:
    def test_returns_key(self):
        loss = PosteriorGeometryLoss(key=LatentKey.POSTERIOR_MU.value)
        assert loss.get_required_keys() == {LatentKey.POSTERIOR_MU.value}


@pytest.mark.unit
class TestPosteriorGeometryLossInitialization:
    @pytest.mark.parametrize(
        "target_std, max_std, eps, expectation",
        [
            (1.0, 2.0, 1e-6, does_not_raise()),
            (
                0.0,
                2.0,
                1e-6,
                pytest.raises(
                    ValueError,
                    match=re.escape("target_std must be positive, got 0.0."),
                ),
            ),
            (
                1.0,
                0.0,
                1e-6,
                pytest.raises(
                    ValueError,
                    match=re.escape("max_std must be positive, got 0.0."),
                ),
            ),
            (
                1.0,
                2.0,
                0.0,
                pytest.raises(
                    ValueError,
                    match=re.escape("eps must be positive, got 0.0."),
                ),
            ),
        ],
    )
    def test_validates_positive_scale_configuration(
        self,
        target_std: float,
        max_std: float,
        eps: float,
        expectation: AbstractContextManager,
    ):
        with expectation:
            PosteriorGeometryLoss(
                target_std=target_std,
                max_std=max_std,
                eps=eps,
            )


@pytest.mark.unit
class TestPosteriorGeometryLossForward:
    def test_centered_unit_std_independent_latents_have_low_loss(self):
        latent = torch.tensor(
            [
                [-1.0, -1.0],
                [-1.0, 1.0],
                [1.0, -1.0],
                [1.0, 1.0],
            ]
        )
        predictions = {LatentKey.POSTERIOR_MU.value: latent}
        loss = PosteriorGeometryLoss(
            mean_weight=1.0,
            std_weight=1.0,
            target_std=1.0,
            max_std_weight=1.0,
            max_std=2.0,
            covariance_weight=1.0,
            eps=1e-6,
        )
        output = loss(predictions, {})
        assert output.total_loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_nonzero_mean_produces_mean_loss(self):
        latent = torch.ones(8, 2)
        predictions = {LatentKey.POSTERIOR_MU.value: latent}
        loss = PosteriorGeometryLoss(
            mean_weight=1.0,
            std_weight=0.0,
            max_std_weight=0.0,
            covariance_weight=0.0,
        )
        output = loss(predictions, {})
        assert (
            output.component_losses[MetricKey.POSTERIOR_GEOMETRY_MEAN_LOSS.value].item()
            > 0.0
        )

    def test_large_std_produces_target_and_max_std_losses(self):
        latent = torch.tensor(
            [
                [-3.0, -3.0],
                [-3.0, 3.0],
                [3.0, -3.0],
                [3.0, 3.0],
            ]
        )
        predictions = {LatentKey.POSTERIOR_MU.value: latent}
        loss = PosteriorGeometryLoss(
            mean_weight=0.0,
            std_weight=1.0,
            target_std=1.0,
            max_std_weight=1.0,
            max_std=2.0,
            covariance_weight=0.0,
        )
        output = loss(predictions, {})
        assert (
            output.component_losses[MetricKey.POSTERIOR_GEOMETRY_STD_LOSS.value].item()
            > 0.0
        )
        assert (
            output.component_losses[
                MetricKey.POSTERIOR_GEOMETRY_MAX_STD_LOSS.value
            ].item()
            > 0.0
        )

    def test_correlated_latents_produce_covariance_loss(self):
        x = torch.linspace(-1.0, 1.0, 8).unsqueeze(1)
        latent = torch.cat([x, x], dim=1)
        predictions = {LatentKey.POSTERIOR_MU.value: latent}
        loss = PosteriorGeometryLoss(
            mean_weight=0.0,
            std_weight=0.0,
            max_std_weight=0.0,
            covariance_weight=1.0,
        )
        output = loss(predictions, {})
        assert (
            output.component_losses[
                MetricKey.POSTERIOR_GEOMETRY_COVARIANCE_LOSS.value
            ].item()
            > 0.0
        )

    def test_raises_on_missing_key(self):
        loss = PosteriorGeometryLoss(key=LatentKey.POSTERIOR_MU.value)
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Predictions must contain '{LatentKey.POSTERIOR_MU.value}' "
                "for PosteriorGeometryLoss."
            ),
        ):
            loss({}, {})

    def test_raises_on_non_matrix_latents(self):
        loss = PosteriorGeometryLoss(key=LatentKey.POSTERIOR_MU.value)
        predictions = {
            LatentKey.POSTERIOR_MU.value: torch.zeros(2, 3, 4),
        }
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"PosteriorGeometryLoss expects '{LatentKey.POSTERIOR_MU.value}' "
                "with shape (batch_size, latent_dimension), got (2, 3, 4)."
            ),
        ):
            loss(predictions, {})
