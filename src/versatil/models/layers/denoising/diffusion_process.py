"""This module provides reusable building blocks for implementing diffusion processes."""

import enum
from dataclasses import dataclass

import torch
from diffusers import DDIMScheduler, DDPMScheduler


@dataclass
class DiffusionSchedulerConfig:
    """Configuration for diffusion noise schedulers.

    Attributes:
        scheduler_type: Type of scheduler ("ddpm" or "ddim")
        num_train_timesteps: Number of diffusion steps during training
        num_inference_steps: Number of denoising steps during inference
        beta_start: Starting value of noise schedule
        beta_end: Ending value of noise schedule
        beta_schedule: Noise schedule type (e.g., "squaredcos_cap_v2", "linear")
        prediction_type: What model predicts ("epsilon" for noise, "sample" for clean data)
        clip_sample: Whether to clip samples during inference
        variance_type: Variance type for DDPM scheduler (e.g., "fixed_small")
        set_alpha_to_one: Whether to set final alpha to 1 (DDIM only)
        steps_offset: Offset for timestep calculation (DDIM only)
    """

    scheduler_type: str
    num_train_timesteps: int
    num_inference_steps: int
    beta_start: float
    beta_end: float
    beta_schedule: str
    prediction_type: str
    clip_sample: bool
    # DDPM-specific
    variance_type: str | None = None
    # DDIM-specific
    set_alpha_to_one: bool | None = None
    steps_offset: int | None = None


def create_noise_scheduler(
    config: DiffusionSchedulerConfig,
) -> DDPMScheduler | DDIMScheduler:
    """Factory function for creating diffusion noise schedulers.

    Creates either a DDPM or DDIM scheduler based on configuration.
    Both schedulers support the same forward diffusion process but differ
    in their reverse process (DDIM allows faster inference with fewer steps).

    Args:
        config: Scheduler configuration

    Returns:
        Configured noise scheduler (DDPM or DDIM)

    Raises:
        ValueError: If scheduler_type is not recognized

    Example:
        ```python
        config = DiffusionSchedulerConfig(
            scheduler_type="ddim",
            num_train_timesteps=100,
            ...
        )
        scheduler = create_noise_scheduler(config)
        ```
    """
    # Common scheduler arguments
    scheduler_kwargs = {
        "num_train_timesteps": config.num_train_timesteps,
        "beta_start": config.beta_start,
        "beta_end": config.beta_end,
        "beta_schedule": config.beta_schedule,
        "prediction_type": config.prediction_type,
        "clip_sample": config.clip_sample,
    }

    if config.scheduler_type == SchedulerType.DDPM.value:
        # DDPM-specific parameters
        if config.variance_type is not None:
            scheduler_kwargs["variance_type"] = config.variance_type
        return DDPMScheduler(**scheduler_kwargs)

    elif config.scheduler_type == SchedulerType.DDIM.value:
        # DDIM-specific parameters
        if config.set_alpha_to_one is not None:
            scheduler_kwargs["set_alpha_to_one"] = config.set_alpha_to_one
        if config.steps_offset is not None:
            scheduler_kwargs["steps_offset"] = config.steps_offset
        return DDIMScheduler(**scheduler_kwargs)

    else:
        valid_types = [e.value for e in SchedulerType]
        raise ValueError(
            f"Unknown scheduler_type: {config.scheduler_type}. "
            f"Expected one of {valid_types}"
        )


def add_noise_to_tensor(
    clean: torch.Tensor,
    noise_scheduler: DDPMScheduler | DDIMScheduler,
    timesteps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add noise to a clean tensor at specified timesteps.

    Implements the forward diffusion process:
        x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * epsilon

    where x_0 is the clean data, epsilon is sampled noise, and alpha_t
    controls the signal-to-noise ratio at timestep t.

    Args:
        clean: Clean input tensor of shape (..., features)
        noise_scheduler: Configured noise scheduler
        timesteps: Timesteps at which to add noise, shape (batch_size,)

    Returns:
        Tuple of (noisy_tensor, noise) where:
            - noisy_tensor: Input with added noise, same shape as clean
            - noise: The sampled noise that was added, same shape as clean
    """
    # Ensure tensor is float for noise generation (handles integer gripper actions)
    if not clean.is_floating_point():
        clean = clean.float()
    noise = torch.randn_like(clean)
    noisy = noise_scheduler.add_noise(clean, noise, timesteps)
    return noisy, noise


def sample_random_timesteps(
    batch_size: int,
    num_train_timesteps: int,
    device: torch.device,
) -> torch.IntTensor:
    """Sample random timesteps for training.

    Samples uniformly from [0, num_train_timesteps) for each batch element.
    This ensures the model learns to denoise at all noise levels.

    Args:
        batch_size: Number of timesteps to sample
        num_train_timesteps: Maximum timestep value (exclusive)
        device: Device to place timesteps on

    Returns:
        Tensor of shape (batch_size,) with sampled timesteps
    """
    timesteps = torch.randint(
        0,
        num_train_timesteps,
        (batch_size,),
        device=device,
    ).long()
    return timesteps


def setup_inference_timesteps(
    noise_scheduler: DDPMScheduler | DDIMScheduler,
    num_inference_steps: int,
) -> None:
    """Configure scheduler for inference denoising.

    Sets up the timestep schedule for the reverse diffusion process.
    For DDPM, this uses all timesteps. For DDIM, this can use fewer steps
    by skipping timesteps uniformly.

    Args:
        noise_scheduler: Configured noise scheduler (modified in-place)
        num_inference_steps: Number of denoising steps to use

    Side Effects:
        Modifies noise_scheduler.timesteps in-place
    """
    noise_scheduler.set_timesteps(num_inference_steps)


class SchedulerType(enum.StrEnum):
    """Diffusion scheduler types (compatible with diffusers API)."""

    DDIM = "ddim"
    DDPM = "ddpm"
