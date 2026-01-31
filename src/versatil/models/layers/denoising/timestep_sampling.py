"""Timestep sampling strategies for flow matching training.

References:
    Esser et al. "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis"
    https://arxiv.org/abs/2403.03206
"""

from enum import Enum

import torch


class TimestepSampler(Enum):
    """Timestep sampling strategies for continuous-time generative models."""

    UNIFORM = "uniform"
    LOGIT_NORMAL = "logit_normal"


def sample_timesteps(
    batch_size: int,
    device: torch.device,
    sampler: str = TimestepSampler.LOGIT_NORMAL.value,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
) -> torch.Tensor:
    """Sample timesteps t in [0, 1] using various strategies.

    Args:
        batch_size: Number of samples.
        device: Target device.
        sampler: Sampling strategy name.
        logit_mean: Mean for logit-normal (shifts mode; 0 centers at t=0.5).
        logit_std: Std for logit-normal (smaller = more concentrated).

    Returns:
        Tensor of shape (batch_size,) with values in [0, 1].

    Raises:
        ValueError: If sampler is not a recognized strategy.
    """
    if sampler == TimestepSampler.UNIFORM.value:
        return torch.rand(batch_size, device=device)

    if sampler == TimestepSampler.LOGIT_NORMAL.value:
        normal_samples = torch.randn(batch_size, device=device) * logit_std + logit_mean
        return torch.sigmoid(normal_samples)

    raise ValueError(
        f"Unknown sampler: {sampler}. Expected one of {[e.value for e in TimestepSampler]}"
    )
