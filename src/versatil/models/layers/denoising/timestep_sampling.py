"""Timestep sampling strategies for flow matching training.

References:
    Esser et al. "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis"
    https://arxiv.org/abs/2403.03206

    Black et al. "pi0: A Vision-Language-Action Flow Model for General Robot Control"
    https://arxiv.org/abs/2410.24164
"""

from dataclasses import dataclass
from enum import Enum

import torch


class TimestepSampler(Enum):
    """Timestep sampling strategies for continuous-time generative models."""

    UNIFORM = "uniform"
    LOGIT_NORMAL = "logit_normal"
    BETA = "beta"


@dataclass
class TimestepSamplingConfig:
    """Configuration for continuous timestep sampling."""

    sampler: str = TimestepSampler.BETA.value
    logit_mean: float = 0.0
    logit_std: float = 1.0
    beta_alpha: float = 1.5
    beta_beta: float = 1.0
    max_timestep: float = 0.999

    def __post_init__(self) -> None:
        validate_timestep_sampler(sampler=self.sampler)


def validate_timestep_sampler(sampler: str) -> None:
    """Validate a continuous timestep sampler name."""
    valid_samplers = [member.value for member in TimestepSampler]
    if sampler not in valid_samplers:
        raise ValueError(
            f"Unknown timestep sampler: {sampler}. Expected one of {valid_samplers}"
        )


def sample_timesteps_from_config(
    config: TimestepSamplingConfig,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample continuous timesteps from a reusable sampling configuration."""
    return sample_timesteps(
        batch_size=batch_size,
        device=device,
        sampler=config.sampler,
        logit_mean=config.logit_mean,
        logit_std=config.logit_std,
        beta_alpha=config.beta_alpha,
        beta_beta=config.beta_beta,
        max_timestep=config.max_timestep,
    )


def sample_timesteps(
    batch_size: int,
    device: torch.device,
    sampler: str = TimestepSampler.BETA.value,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    beta_alpha: float = 1.5,
    beta_beta: float = 1.0,
    max_timestep: float = 0.999,
) -> torch.Tensor:
    """Sample timesteps t in [0, 1] using various strategies.

    Args:
        batch_size: Number of samples.
        device: Target device.
        sampler: Sampling strategy name.
        logit_mean: Mean for logit-normal (shifts mode; 0 centers at t=0.5).
        logit_std: Std for logit-normal (smaller = more concentrated).
        beta_alpha: First shape parameter for Beta distribution.
        beta_beta: Second shape parameter for Beta distribution.
        max_timestep: Upper bound s for Beta sampling; timesteps above s
            are never sampled. Samples follow p(t) = Beta((s-t)/s; alpha, beta).

    Returns:
        Tensor of shape (batch_size,) with values in [0, 1].

    Raises:
        ValueError: If sampler is not a recognized strategy.
    """
    match sampler:
        case TimestepSampler.UNIFORM.value:
            return torch.rand(batch_size, device=device)

        case TimestepSampler.LOGIT_NORMAL.value:
            normal_samples = (
                torch.randn(batch_size, device=device) * logit_std + logit_mean
            )
            return torch.sigmoid(normal_samples)

        case TimestepSampler.BETA.value:
            beta_distribution = torch.distributions.Beta(
                concentration1=beta_alpha,
                concentration0=beta_beta,
            )
            # u ~ Beta(alpha, beta), then t = s * (1 - u) emphasizes low timesteps
            u = beta_distribution.sample((batch_size,)).to(device)
            return max_timestep * (1.0 - u)

        case _:
            raise ValueError(
                f"Unknown sampler: {sampler}. "
                f"Expected one of {[e.value for e in TimestepSampler]}"
            )
