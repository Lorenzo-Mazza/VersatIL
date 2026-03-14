"""Tests for versatil.models.layers.denoising.timestep_sampling module."""
import re
from contextlib import nullcontext as does_not_raise

import pytest
import torch

from versatil.models.layers.denoising.timestep_sampling import (
    TimestepSampler,
    sample_timesteps,
)


class TestTimestepSampler:

    @pytest.mark.parametrize(
        "member, expected_value",
        [
            (TimestepSampler.UNIFORM, "uniform"),
            (TimestepSampler.LOGIT_NORMAL, "logit_normal"),
            (TimestepSampler.BETA, "beta"),
        ],
    )
    def test_enum_values(self, member: TimestepSampler, expected_value: str):
        assert member.value == expected_value


class TestSampleTimestepsUniform:

    @pytest.mark.parametrize("batch_size", [1, 8])
    def test_output_shape(self, batch_size: int, device: torch.device):
        timesteps = sample_timesteps(
            batch_size=batch_size,
            device=device,
            sampler=TimestepSampler.UNIFORM.value,
        )
        assert timesteps.shape == (batch_size,)

    def test_values_in_unit_interval(self, device: torch.device):
        timesteps = sample_timesteps(
            batch_size=10000,
            device=device,
            sampler=TimestepSampler.UNIFORM.value,
        )
        assert timesteps.min() >= 0.0
        assert timesteps.max() <= 1.0

    def test_output_device(self, device: torch.device):
        timesteps = sample_timesteps(
            batch_size=4,
            device=device,
            sampler=TimestepSampler.UNIFORM.value,
        )
        assert timesteps.device.type == device.type

    def test_output_dtype_is_float(self, device: torch.device):
        timesteps = sample_timesteps(
            batch_size=4,
            device=device,
            sampler=TimestepSampler.UNIFORM.value,
        )
        assert timesteps.dtype == torch.float32

    def test_uniform_mean_near_half(self, device: torch.device):
        # E[Uniform(0,1)] = 0.5
        timesteps = sample_timesteps(
            batch_size=50000,
            device=device,
            sampler=TimestepSampler.UNIFORM.value,
        )
        assert abs(timesteps.mean().item() - 0.5) < 0.02

    def test_uniform_distribution_covers_full_range(self, device: torch.device):
        timesteps = sample_timesteps(
            batch_size=10000,
            device=device,
            sampler=TimestepSampler.UNIFORM.value,
        )
        assert timesteps.min() < 0.01
        assert timesteps.max() > 0.99


class TestSampleTimestepsLogitNormal:

    @pytest.mark.parametrize("batch_size", [1, 8])
    def test_output_shape(self, batch_size: int, device: torch.device):
        timesteps = sample_timesteps(
            batch_size=batch_size,
            device=device,
            sampler=TimestepSampler.LOGIT_NORMAL.value,
        )
        assert timesteps.shape == (batch_size,)

    def test_values_strictly_in_open_unit_interval(self, device: torch.device):
        # sigmoid never reaches exactly 0 or 1
        timesteps = sample_timesteps(
            batch_size=10000,
            device=device,
            sampler=TimestepSampler.LOGIT_NORMAL.value,
            logit_mean=0.0,
            logit_std=1.0,
        )
        assert timesteps.min() > 0.0
        assert timesteps.max() < 1.0

    def test_output_dtype_is_float(self, device: torch.device):
        timesteps = sample_timesteps(
            batch_size=4,
            device=device,
            sampler=TimestepSampler.LOGIT_NORMAL.value,
        )
        assert timesteps.dtype == torch.float32

    def test_zero_mean_centers_distribution_around_half(self, device: torch.device):
        # logit_mean=0 => sigmoid(N(0, sigma)) is symmetric around 0.5
        timesteps = sample_timesteps(
            batch_size=50000,
            device=device,
            sampler=TimestepSampler.LOGIT_NORMAL.value,
            logit_mean=0.0,
            logit_std=1.0,
        )
        mean_value = timesteps.mean().item()
        assert abs(mean_value - 0.5) < 0.02

    def test_positive_mean_shifts_distribution_higher(self, device: torch.device):
        timesteps_centered = sample_timesteps(
            batch_size=50000,
            device=device,
            sampler=TimestepSampler.LOGIT_NORMAL.value,
            logit_mean=0.0,
            logit_std=1.0,
        )
        timesteps_shifted = sample_timesteps(
            batch_size=50000,
            device=device,
            sampler=TimestepSampler.LOGIT_NORMAL.value,
            logit_mean=2.0,
            logit_std=1.0,
        )
        assert timesteps_shifted.mean() > timesteps_centered.mean()

    def test_negative_mean_shifts_distribution_lower(self, device: torch.device):
        timesteps_centered = sample_timesteps(
            batch_size=50000,
            device=device,
            sampler=TimestepSampler.LOGIT_NORMAL.value,
            logit_mean=0.0,
            logit_std=1.0,
        )
        timesteps_shifted = sample_timesteps(
            batch_size=50000,
            device=device,
            sampler=TimestepSampler.LOGIT_NORMAL.value,
            logit_mean=-2.0,
            logit_std=1.0,
        )
        assert timesteps_shifted.mean() < timesteps_centered.mean()

    def test_small_std_concentrates_distribution(self, device: torch.device):
        timesteps_wide = sample_timesteps(
            batch_size=50000,
            device=device,
            sampler=TimestepSampler.LOGIT_NORMAL.value,
            logit_mean=0.0,
            logit_std=2.0,
        )
        timesteps_narrow = sample_timesteps(
            batch_size=50000,
            device=device,
            sampler=TimestepSampler.LOGIT_NORMAL.value,
            logit_mean=0.0,
            logit_std=0.3,
        )
        assert timesteps_narrow.std() < timesteps_wide.std()


class TestSampleTimestepsBeta:

    @pytest.mark.parametrize("batch_size", [1, 8])
    def test_output_shape(self, batch_size: int, device: torch.device):
        timesteps = sample_timesteps(
            batch_size=batch_size,
            device=device,
            sampler=TimestepSampler.BETA.value,
        )
        assert timesteps.shape == (batch_size,)

    def test_values_in_range_zero_to_max_timestep(self, device: torch.device):
        max_timestep = 0.999
        timesteps = sample_timesteps(
            batch_size=10000,
            device=device,
            sampler=TimestepSampler.BETA.value,
            beta_alpha=1.5,
            beta_beta=1.0,
            max_timestep=max_timestep,
        )
        assert timesteps.min() >= 0.0
        assert timesteps.max() <= max_timestep

    def test_custom_max_timestep_limits_range(self, device: torch.device):
        max_timestep = 0.5
        timesteps = sample_timesteps(
            batch_size=10000,
            device=device,
            sampler=TimestepSampler.BETA.value,
            beta_alpha=1.5,
            beta_beta=1.0,
            max_timestep=max_timestep,
        )
        assert timesteps.max() <= max_timestep
        # With s=0.5, samples should cover most of [0, 0.5]
        assert timesteps.min() < 0.05

    def test_output_dtype_is_float(self, device: torch.device):
        timesteps = sample_timesteps(
            batch_size=4,
            device=device,
            sampler=TimestepSampler.BETA.value,
        )
        assert timesteps.dtype == torch.float32

    def test_output_device(self, device: torch.device):
        timesteps = sample_timesteps(
            batch_size=4,
            device=device,
            sampler=TimestepSampler.BETA.value,
        )
        assert timesteps.device.type == device.type

    def test_beta_distribution_mean_matches_analytical(self, device: torch.device):
        # t = s * (1 - u) where u ~ Beta(alpha, beta)
        # E[u] = alpha / (alpha + beta), so E[t] = s * (1 - alpha / (alpha + beta))
        # E[t] = s * beta / (alpha + beta)
        beta_alpha = 1.5
        beta_beta = 1.0
        max_timestep = 0.999
        expected_mean = max_timestep * beta_beta / (beta_alpha + beta_beta)

        timesteps = sample_timesteps(
            batch_size=50000,
            device=device,
            sampler=TimestepSampler.BETA.value,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
            max_timestep=max_timestep,
        )
        assert abs(timesteps.mean().item() - expected_mean) < 0.02

    def test_beta_distribution_emphasizes_low_timesteps(self, device: torch.device):
        # With alpha=1.5, beta=1.0: E[u] = 0.6 so E[1-u] = 0.4
        # => median of timesteps is below 0.5 * max_timestep
        timesteps = sample_timesteps(
            batch_size=50000,
            device=device,
            sampler=TimestepSampler.BETA.value,
            beta_alpha=1.5,
            beta_beta=1.0,
            max_timestep=0.999,
        )
        fraction_below_half = (timesteps < 0.5).float().mean().item()
        # E[t] ~ 0.4 * 0.999 ~ 0.4, so more than 50% should be below 0.5
        assert fraction_below_half > 0.55

    def test_max_timestep_scales_output(self, device: torch.device):
        # Same beta params, different max_timestep -> proportionally scaled output
        timesteps_full = sample_timesteps(
            batch_size=50000,
            device=device,
            sampler=TimestepSampler.BETA.value,
            beta_alpha=1.5,
            beta_beta=1.0,
            max_timestep=1.0,
        )
        timesteps_half = sample_timesteps(
            batch_size=50000,
            device=device,
            sampler=TimestepSampler.BETA.value,
            beta_alpha=1.5,
            beta_beta=1.0,
            max_timestep=0.5,
        )
        # Mean should scale proportionally
        ratio = timesteps_half.mean().item() / timesteps_full.mean().item()
        assert abs(ratio - 0.5) < 0.05


class TestSampleTimestepsValidation:

    @pytest.mark.parametrize(
        "sampler, expectation",
        [
            (TimestepSampler.UNIFORM.value, does_not_raise()),
            (TimestepSampler.LOGIT_NORMAL.value, does_not_raise()),
            (TimestepSampler.BETA.value, does_not_raise()),
            (
                "invalid_sampler",
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "Unknown sampler: invalid_sampler. "
                        f"Expected one of {[e.value for e in TimestepSampler]}"
                    ),
                ),
            ),
        ],
    )
    def test_sampler_validation(
        self,
        device: torch.device,
        sampler: str,
        expectation,
    ):
        with expectation:
            result = sample_timesteps(
                batch_size=4,
                device=device,
                sampler=sampler,
            )
            assert result.shape == (4,)
