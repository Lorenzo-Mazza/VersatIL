"""Shared fixtures for denoising layer tests."""

from collections.abc import Callable

import pytest
import torch
from diffusers import DDIMScheduler, DDPMScheduler

from versatil.models.layers.denoising.diffusion_process import (
    DiffusionSchedulerConfig,
    SchedulerType,
    create_noise_scheduler,
)


@pytest.fixture
def scheduler_config_factory() -> Callable[..., DiffusionSchedulerConfig]:
    def factory(
        scheduler_type: str = SchedulerType.DDPM.value,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 10,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "squaredcos_cap_v2",
        prediction_type: str = "epsilon",
        clip_sample: bool = True,
        variance_type: str | None = None,
        set_alpha_to_one: bool | None = None,
        steps_offset: int | None = None,
    ) -> DiffusionSchedulerConfig:
        return DiffusionSchedulerConfig(
            scheduler_type=scheduler_type,
            num_train_timesteps=num_train_timesteps,
            num_inference_steps=num_inference_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            clip_sample=clip_sample,
            variance_type=variance_type,
            set_alpha_to_one=set_alpha_to_one,
            steps_offset=steps_offset,
        )

    return factory


@pytest.fixture
def ddpm_scheduler_factory(
    scheduler_config_factory: Callable[..., DiffusionSchedulerConfig],
) -> Callable[..., DDPMScheduler]:
    def factory(
        num_train_timesteps: int = 100,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "squaredcos_cap_v2",
        prediction_type: str = "epsilon",
        clip_sample: bool = True,
        variance_type: str | None = None,
    ) -> DDPMScheduler:
        config = scheduler_config_factory(
            scheduler_type=SchedulerType.DDPM.value,
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            clip_sample=clip_sample,
            variance_type=variance_type,
        )
        return create_noise_scheduler(config=config)

    return factory


@pytest.fixture
def ddim_scheduler_factory(
    scheduler_config_factory: Callable[..., DiffusionSchedulerConfig],
) -> Callable[..., DDIMScheduler]:
    def factory(
        num_train_timesteps: int = 100,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "squaredcos_cap_v2",
        prediction_type: str = "epsilon",
        clip_sample: bool = True,
        set_alpha_to_one: bool | None = None,
        steps_offset: int | None = None,
    ) -> DDIMScheduler:
        config = scheduler_config_factory(
            scheduler_type=SchedulerType.DDIM.value,
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            clip_sample=clip_sample,
            set_alpha_to_one=set_alpha_to_one,
            steps_offset=steps_offset,
        )
        return create_noise_scheduler(config=config)

    return factory


@pytest.fixture
def velocity_field_factory() -> Callable[
    ..., Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
]:
    def factory(
        field_type: str = "constant",
        constant_velocity: float = 1.0,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        match field_type:
            case "constant":

                def velocity_fn(z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                    return torch.full_like(z, constant_velocity)

                return velocity_fn
            case "linear":

                def velocity_fn(z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                    return z

                return velocity_fn
            case "time_dependent":

                def velocity_fn(z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                    return t.unsqueeze(-1).expand_as(z)

                return velocity_fn
            case _:
                raise ValueError(f"Unknown field_type: {field_type}")

    return factory
