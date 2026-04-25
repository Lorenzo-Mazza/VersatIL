"""Tests for versatil.models.layers.denoising.conditional_flow_matching module."""

import re
from collections.abc import Callable
from unittest.mock import patch

import pytest
import torch

from versatil.models.layers.denoising.conditional_flow_matching import (
    ConditionalFlowMatcher,
)


@pytest.fixture
def flow_matcher_factory() -> Callable[..., ConditionalFlowMatcher]:
    def factory(sigma: float = 0.0) -> ConditionalFlowMatcher:
        return ConditionalFlowMatcher(sigma=sigma)

    return factory


@pytest.fixture
def sample_pair_factory() -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    def factory(
        *,
        device: torch.device,
        batch_size: int = 2,
        feature_dimension: int = 3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.arange(
            batch_size * feature_dimension,
            device=device,
            dtype=torch.float32,
        ).reshape(batch_size, feature_dimension)
        return values, values + 10.0

    return factory


class TestConditionalFlowMatcher:
    def test_compute_mu_t_interpolates_between_endpoints(
        self,
        flow_matcher_factory: Callable[..., ConditionalFlowMatcher],
        sample_pair_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
    ):
        flow_matcher = flow_matcher_factory(sigma=0.0)
        source, target = sample_pair_factory(device=device)
        time = torch.tensor([0.0, 1.0], device=device)

        interpolation = flow_matcher.compute_mu_t(x0=source, x1=target, t=time)

        expected = torch.stack([source[0], target[1]], dim=0)
        assert torch.allclose(interpolation, expected)

    def test_sample_xt_adds_scaled_epsilon(
        self,
        flow_matcher_factory: Callable[..., ConditionalFlowMatcher],
        device: torch.device,
    ):
        sigma = 0.5
        flow_matcher = flow_matcher_factory(sigma=sigma)
        source = torch.zeros(2, 3, device=device)
        target = torch.full((2, 3), 2.0, device=device)
        time = torch.full((2,), 0.25, device=device)
        epsilon = torch.ones_like(source)

        sample = flow_matcher.sample_xt(
            x0=source,
            x1=target,
            t=time,
            epsilon=epsilon,
        )

        expected = torch.ones_like(source)
        assert torch.allclose(sample, expected)

    def test_compute_conditional_flow_returns_target_minus_source(
        self,
        flow_matcher_factory: Callable[..., ConditionalFlowMatcher],
        sample_pair_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
    ):
        flow_matcher = flow_matcher_factory(sigma=0.0)
        source, target = sample_pair_factory(device=device)
        time = torch.full((source.shape[0],), 0.5, device=device)
        sample = torch.zeros_like(source)

        conditional_flow = flow_matcher.compute_conditional_flow(
            x0=source,
            x1=target,
            t=time,
            xt=sample,
        )

        assert torch.allclose(conditional_flow, target - source)

    def test_compute_lambda_uses_sigma_schedule(
        self,
        flow_matcher_factory: Callable[..., ConditionalFlowMatcher],
        device: torch.device,
    ):
        sigma = 0.5
        flow_matcher = flow_matcher_factory(sigma=sigma)
        time = torch.full((2,), 0.5, device=device)

        lambda_value = flow_matcher.compute_lambda(time)

        expected = 2 * sigma / (sigma**2 + 1e-8)
        assert lambda_value == pytest.approx(expected)

    def test_sample_location_and_conditional_flow_uses_configured_time(
        self,
        flow_matcher_factory: Callable[..., ConditionalFlowMatcher],
        sample_pair_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
    ):
        sigma = 0.25
        flow_matcher = flow_matcher_factory(sigma=sigma)
        source, target = sample_pair_factory(device=device)
        time = torch.full((source.shape[0],), 0.5, device=device)
        epsilon = torch.ones_like(source)

        with patch.object(
            flow_matcher,
            "sample_noise_like",
            return_value=epsilon,
        ) as sample_noise_like_mock:
            sampled_time, sample, conditional_flow, noise = (
                flow_matcher.sample_location_and_conditional_flow(
                    x0=source,
                    x1=target,
                    t=time,
                    return_noise=True,
                )
            )

        expected_sample = 0.5 * source + 0.5 * target + sigma * epsilon
        sample_noise_like_mock.assert_called_once_with(source)
        assert torch.allclose(sampled_time, time)
        assert torch.allclose(sample, expected_sample)
        assert torch.allclose(conditional_flow, target - source)
        assert torch.allclose(noise, epsilon)

    def test_raises_when_time_batch_size_does_not_match_samples(
        self,
        flow_matcher_factory: Callable[..., ConditionalFlowMatcher],
        sample_pair_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
    ):
        flow_matcher = flow_matcher_factory(sigma=0.0)
        source, target = sample_pair_factory(device=device, batch_size=2)
        time = torch.full((3,), 0.5, device=device)
        error_message = "Time batch size 3 must match sample batch size 2."

        with pytest.raises(ValueError, match=re.escape(error_message)):
            flow_matcher.sample_location_and_conditional_flow(
                x0=source,
                x1=target,
                t=time,
            )
