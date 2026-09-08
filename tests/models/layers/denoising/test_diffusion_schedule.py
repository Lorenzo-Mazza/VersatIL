"""Tests for versatil.models.layers.denoising.diffusion_schedule module."""

import re
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from diffusers import DDIMScheduler, DDPMScheduler

from versatil.models.decoding.constants import PredictionType, VarianceType
from versatil.models.layers.denoising.diffusion_process import (
    DiffusionSchedulerConfig,
    SchedulerType,
    create_noise_scheduler,
)
from versatil.models.layers.denoising.diffusion_schedule import DiffusionSchedule


@pytest.fixture
def diffusion_schedule_factory(
    scheduler_config_factory: Callable[..., DiffusionSchedulerConfig],
) -> Callable[..., tuple[DiffusionSchedule, DDIMScheduler | DDPMScheduler]]:
    def factory(
        scheduler_type: str = SchedulerType.DDIM.value,
        num_train_timesteps: int = 23,
        num_inference_steps: int = 5,
        prediction_type: str = PredictionType.EPSILON.value,
        variance_type: str = VarianceType.FIXED_SMALL.value,
        clip_sample: bool = True,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
    ) -> tuple[DiffusionSchedule, DDIMScheduler | DDPMScheduler]:
        config = scheduler_config_factory(
            scheduler_type=scheduler_type,
            num_train_timesteps=num_train_timesteps,
            num_inference_steps=num_inference_steps,
            prediction_type=prediction_type,
            variance_type=variance_type,
            clip_sample=clip_sample,
            set_alpha_to_one=set_alpha_to_one,
            steps_offset=steps_offset,
        )
        scheduler = create_noise_scheduler(config=config)
        schedule = DiffusionSchedule(
            scheduler=scheduler, num_inference_steps=num_inference_steps
        )
        return schedule, scheduler

    return factory


class TestDiffusionScheduleUpdates:
    @pytest.mark.parametrize(
        "scheduler_type, variance_type",
        [
            (SchedulerType.DDIM.value, VarianceType.FIXED_SMALL.value),
            (SchedulerType.DDPM.value, VarianceType.FIXED_SMALL.value),
            (SchedulerType.DDPM.value, VarianceType.FIXED_SMALL_LOG.value),
            (SchedulerType.DDPM.value, VarianceType.FIXED_LARGE.value),
        ],
    )
    @pytest.mark.parametrize(
        "prediction_type", [member.value for member in PredictionType]
    )
    @pytest.mark.parametrize("clip_sample", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    @pytest.mark.parametrize(
        "device_name",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=[
                    pytest.mark.requires_gpu,
                    pytest.mark.skipif(
                        not torch.cuda.is_available(), reason="CUDA is unavailable"
                    ),
                ],
            ),
        ],
    )
    def test_each_update_matches_diffusers(
        self,
        diffusion_schedule_factory: Callable[
            ..., tuple[DiffusionSchedule, DDIMScheduler | DDPMScheduler]
        ],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        scheduler_type: str,
        variance_type: str,
        prediction_type: str,
        clip_sample: bool,
        dtype: torch.dtype,
        device_name: str,
    ) -> None:
        schedule, scheduler = diffusion_schedule_factory(
            scheduler_type=scheduler_type,
            num_train_timesteps=23,
            num_inference_steps=5,
            prediction_type=prediction_type,
            variance_type=variance_type,
            clip_sample=clip_sample,
        )
        schedule.to(device=device_name, dtype=dtype)
        sample, prediction, noise = (
            sequence_tensor_factory(
                batch_size=2, sequence_length=4, embedding_dimension=3
            ).to(device=device_name, dtype=dtype)  # (batch, horizon, action_dim)
            for _ in range(3)
        )
        for step_index, timestep in enumerate(schedule.timesteps):
            with patch(
                "diffusers.schedulers.scheduling_ddpm.randn_tensor", return_value=noise
            ):
                expected = scheduler.step(
                    model_output=prediction, timestep=timestep, sample=sample
                ).prev_sample  # (batch, horizon, action_dim)
            actual = schedule(
                model_output=prediction,
                sample=sample,
                step_index=step_index,
                noise=noise if schedule.stochastic else None,
            )  # (batch, horizon, action_dim)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            sample = expected

    @pytest.mark.parametrize("set_alpha_to_one", [True, False])
    @pytest.mark.parametrize("steps_offset", [0, 1])
    def test_ddim_respects_final_alpha_and_timestep_offset(
        self,
        diffusion_schedule_factory: Callable[
            ..., tuple[DiffusionSchedule, DDIMScheduler | DDPMScheduler]
        ],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        set_alpha_to_one: bool,
        steps_offset: int,
    ) -> None:
        schedule, scheduler = diffusion_schedule_factory(
            scheduler_type=SchedulerType.DDIM.value,
            num_train_timesteps=23,
            num_inference_steps=5,
            set_alpha_to_one=set_alpha_to_one,
            steps_offset=steps_offset,
        )
        sample = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=3
        )  # (batch, horizon, action_dim)
        prediction = torch.zeros_like(sample)  # (batch, horizon, action_dim)
        for step_index, timestep in enumerate(schedule.timesteps):
            expected = scheduler.step(
                model_output=prediction, timestep=timestep, sample=sample
            ).prev_sample  # (batch, horizon, action_dim)
            actual = schedule(
                model_output=prediction, sample=sample, step_index=step_index
            )  # (batch, horizon, action_dim)
            torch.testing.assert_close(actual, expected)
            sample = expected

    def test_precision_conversion_preserves_coefficients_and_checkpoint_keys(
        self,
        diffusion_schedule_factory: Callable[
            ..., tuple[DiffusionSchedule, DDIMScheduler | DDPMScheduler]
        ],
    ) -> None:
        schedule, _ = diffusion_schedule_factory(num_inference_steps=5)
        coefficients = schedule.coefficients
        schedule.to(dtype=torch.bfloat16)
        assert schedule.coefficients == coefficients
        assert schedule.state_dict() == {}


class TestDiffusionScheduleValidation:
    @pytest.mark.parametrize("num_inference_steps", [0, -1, 24])
    def test_rejects_invalid_step_count(
        self,
        diffusion_schedule_factory: Callable[
            ..., tuple[DiffusionSchedule, DDIMScheduler | DDPMScheduler]
        ],
        num_inference_steps: int,
    ) -> None:
        with pytest.raises(
            ValueError,
            match=re.escape(
                "num_inference_steps must be between 1 and 23, "
                f"got {num_inference_steps}."
            ),
        ):
            diffusion_schedule_factory(
                num_train_timesteps=23, num_inference_steps=num_inference_steps
            )

    @pytest.mark.parametrize(
        "variance_type", [VarianceType.LEARNED.value, VarianceType.LEARNED_RANGE.value]
    )
    def test_requires_fixed_variance_for_action_decoder_outputs(
        self,
        diffusion_schedule_factory: Callable[
            ..., tuple[DiffusionSchedule, DDIMScheduler | DDPMScheduler]
        ],
        variance_type: str,
    ) -> None:
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Diffusion action sampling requires fixed_small, fixed_small_log "
                f"or fixed_large variance, got {variance_type!r}. "
                "Decoder outputs contain denoising predictions without a learned "
                "variance component."
            ),
        ):
            diffusion_schedule_factory(
                scheduler_type=SchedulerType.DDPM.value, variance_type=variance_type
            )


@pytest.mark.integration
@pytest.mark.parametrize("scheduler_type", [member.value for member in SchedulerType])
def test_step_noise_remains_an_input_after_export_and_reload(
    diffusion_schedule_factory: Callable[
        ..., tuple[DiffusionSchedule, DDIMScheduler | DDPMScheduler]
    ],
    sequence_tensor_factory: Callable[..., torch.Tensor],
    scheduler_type: str,
    tmp_path: Path,
) -> None:
    schedule, _ = diffusion_schedule_factory(
        scheduler_type=scheduler_type, num_train_timesteps=23, num_inference_steps=5
    )
    sample, prediction, noise = (
        sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=3
        )  # (batch, horizon, action_dim)
        for _ in range(3)
    )
    inputs = (prediction, sample, 0, noise)
    program = torch.export.export(schedule, inputs, strict=False)
    artifact_path = tmp_path / "schedule.pt2"
    torch.export.save(program, artifact_path)
    reloaded = torch.export.load(artifact_path).module()
    changed_noise = -noise  # (batch, horizon, action_dim)
    expected = schedule(
        model_output=prediction, sample=sample, step_index=0, noise=changed_noise
    )  # (batch, horizon, action_dim)
    actual = reloaded(
        prediction, sample, 0, changed_noise
    )  # (batch, horizon, action_dim)
    torch.testing.assert_close(actual, expected)
    original = reloaded(*inputs)  # (batch, horizon, action_dim)
    assert torch.equal(actual, original) is not schedule.stochastic
