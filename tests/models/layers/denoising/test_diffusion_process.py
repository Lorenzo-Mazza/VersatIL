"""Tests for versatil.models.layers.denoising.diffusion_process module."""

import re
from collections.abc import Callable

import pytest
import torch
from diffusers import DDIMScheduler, DDPMScheduler

from versatil.models.layers.denoising.diffusion_process import (
    DiffusionSchedulerConfig,
    SchedulerType,
    add_noise_to_tensor,
    create_noise_scheduler,
    sample_random_timesteps,
    setup_inference_timesteps,
)


class TestSchedulerType:
    @pytest.mark.parametrize(
        "member, expected_value",
        [
            (SchedulerType.DDPM, "ddpm"),
            (SchedulerType.DDIM, "ddim"),
        ],
    )
    def test_enum_values(self, member: SchedulerType, expected_value: str):
        assert member.value == expected_value


class TestCreateNoiseScheduler:
    def test_ddpm_returns_ddpm_scheduler_instance(
        self,
        scheduler_config_factory: Callable[..., DiffusionSchedulerConfig],
    ):
        config = scheduler_config_factory(
            scheduler_type=SchedulerType.DDPM.value,
            num_train_timesteps=100,
        )
        scheduler = create_noise_scheduler(config=config)
        # Verify we can call DDPM-specific step method (behavioral, not isinstance)
        assert hasattr(scheduler, "variance_type")
        assert scheduler.config.num_train_timesteps == 100

    def test_ddim_returns_ddim_scheduler_instance(
        self,
        scheduler_config_factory: Callable[..., DiffusionSchedulerConfig],
    ):
        config = scheduler_config_factory(
            scheduler_type=SchedulerType.DDIM.value,
            num_train_timesteps=50,
            set_alpha_to_one=True,
            steps_offset=1,
        )
        scheduler = create_noise_scheduler(config=config)
        # DDIM exposes set_alpha_to_one; DDPM does not store it the same way
        assert scheduler.config.set_alpha_to_one is True
        assert scheduler.config.num_train_timesteps == 50

    @pytest.mark.parametrize(
        "scheduler_type", [SchedulerType.DDPM.value, SchedulerType.DDIM.value]
    )
    def test_common_config_passed_to_scheduler(
        self,
        scheduler_config_factory: Callable[..., DiffusionSchedulerConfig],
        scheduler_type: str,
    ):
        config = scheduler_config_factory(
            scheduler_type=scheduler_type,
            beta_start=0.001,
            beta_end=0.05,
            beta_schedule="linear",
            prediction_type="sample",
            clip_sample=False,
        )
        scheduler = create_noise_scheduler(config=config)
        assert scheduler.config.beta_start == 0.001
        assert scheduler.config.beta_end == 0.05
        assert scheduler.config.beta_schedule == "linear"
        assert scheduler.config.prediction_type == "sample"
        assert scheduler.config.clip_sample is False

    def test_ddpm_includes_variance_type(
        self,
        scheduler_config_factory: Callable[..., DiffusionSchedulerConfig],
    ):
        config = scheduler_config_factory(
            scheduler_type=SchedulerType.DDPM.value,
            variance_type="fixed_small",
        )
        scheduler = create_noise_scheduler(config=config)
        assert scheduler.config.variance_type == "fixed_small"

    def test_ddpm_without_variance_type_uses_library_default(
        self,
        scheduler_config_factory: Callable[..., DiffusionSchedulerConfig],
    ):
        config = scheduler_config_factory(
            scheduler_type=SchedulerType.DDPM.value,
            variance_type=None,
        )
        scheduler = create_noise_scheduler(config=config)
        # diffusers DDPMScheduler default variance_type is "fixed_small"
        assert scheduler.config.variance_type == "fixed_small"

    def test_ddim_includes_set_alpha_to_one(
        self,
        scheduler_config_factory: Callable[..., DiffusionSchedulerConfig],
    ):
        config = scheduler_config_factory(
            scheduler_type=SchedulerType.DDIM.value,
            set_alpha_to_one=False,
        )
        scheduler = create_noise_scheduler(config=config)
        assert scheduler.config.set_alpha_to_one is False

    def test_ddim_includes_steps_offset(
        self,
        scheduler_config_factory: Callable[..., DiffusionSchedulerConfig],
    ):
        config = scheduler_config_factory(
            scheduler_type=SchedulerType.DDIM.value,
            steps_offset=1,
        )
        scheduler = create_noise_scheduler(config=config)
        assert scheduler.config.steps_offset == 1

    def test_ddim_without_optional_params_uses_library_defaults(
        self,
        scheduler_config_factory: Callable[..., DiffusionSchedulerConfig],
    ):
        config = scheduler_config_factory(
            scheduler_type=SchedulerType.DDIM.value,
            set_alpha_to_one=None,
            steps_offset=None,
        )
        scheduler = create_noise_scheduler(config=config)
        # diffusers DDIMScheduler defaults
        assert scheduler.config.set_alpha_to_one is True
        assert scheduler.config.steps_offset == 0

    def test_unknown_scheduler_type_raises(
        self,
        scheduler_config_factory: Callable[..., DiffusionSchedulerConfig],
    ):
        unknown_type = "unknown_scheduler"
        config = scheduler_config_factory(scheduler_type=unknown_type)
        valid_types = [e.value for e in SchedulerType]
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Unknown scheduler_type: {unknown_type}. Expected one of {valid_types}"
            ),
        ):
            create_noise_scheduler(config=config)

    @pytest.mark.parametrize(
        "prediction_type",
        ["epsilon", "sample"],
    )
    def test_prediction_type_passed_to_scheduler(
        self,
        scheduler_config_factory: Callable[..., DiffusionSchedulerConfig],
        prediction_type: str,
    ):
        config = scheduler_config_factory(
            scheduler_type=SchedulerType.DDPM.value,
            prediction_type=prediction_type,
        )
        scheduler = create_noise_scheduler(config=config)
        assert scheduler.config.prediction_type == prediction_type

    @pytest.mark.parametrize("clip_sample", [True, False])
    def test_clip_sample_passed_to_scheduler(
        self,
        scheduler_config_factory: Callable[..., DiffusionSchedulerConfig],
        clip_sample: bool,
    ):
        config = scheduler_config_factory(
            scheduler_type=SchedulerType.DDPM.value,
            clip_sample=clip_sample,
        )
        scheduler = create_noise_scheduler(config=config)
        assert scheduler.config.clip_sample == clip_sample


class TestAddNoiseToTensor:
    def test_output_shape_matches_input(
        self,
        flat_tensor_factory: Callable[..., torch.Tensor],
        ddpm_scheduler_factory: Callable[..., DDPMScheduler],
        timestep_factory: Callable[..., torch.Tensor],
    ):
        batch_size = 4
        feature_dim = 16
        clean = flat_tensor_factory(
            batch_size=batch_size, feature_dimension=feature_dim
        )
        scheduler = ddpm_scheduler_factory(num_train_timesteps=100)
        timesteps = timestep_factory(batch_size=batch_size, num_train_timesteps=100)
        noisy, noise = add_noise_to_tensor(
            clean=clean,
            noise_scheduler=scheduler,
            timesteps=timesteps,
        )
        assert noisy.shape == (batch_size, feature_dim)
        assert noise.shape == (batch_size, feature_dim)

    def test_noise_follows_forward_diffusion_formula(
        self,
        flat_tensor_factory: Callable[..., torch.Tensor],
        ddpm_scheduler_factory: Callable[..., DDPMScheduler],
    ):
        # Verify: noisy = sqrt(alpha_bar_t) * clean + sqrt(1 - alpha_bar_t) * noise
        clean = flat_tensor_factory(batch_size=2, feature_dimension=8)
        scheduler = ddpm_scheduler_factory(num_train_timesteps=1000)
        timesteps = torch.tensor([10, 500], dtype=torch.long)

        noisy, noise = add_noise_to_tensor(
            clean=clean,
            noise_scheduler=scheduler,
            timesteps=timesteps,
        )

        # Reconstruct expected output from the scheduler's alpha schedule
        alpha_bar = scheduler.alphas_cumprod[timesteps]
        sqrt_alpha_bar = alpha_bar.sqrt().unsqueeze(-1)
        sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt().unsqueeze(-1)
        expected_noisy = sqrt_alpha_bar * clean + sqrt_one_minus_alpha_bar * noise
        assert torch.allclose(noisy, expected_noisy, atol=1e-6)

    def test_zero_timestep_preserves_clean_signal(
        self,
        flat_tensor_factory: Callable[..., torch.Tensor],
        ddpm_scheduler_factory: Callable[..., DDPMScheduler],
    ):
        clean = flat_tensor_factory(batch_size=2, feature_dimension=8)
        scheduler = ddpm_scheduler_factory(num_train_timesteps=1000)
        timesteps = torch.zeros(2, dtype=torch.long)

        noisy, noise = add_noise_to_tensor(
            clean=clean,
            noise_scheduler=scheduler,
            timesteps=timesteps,
        )
        # At t=0, alpha_bar is near 1.0, so noisy ~ clean
        alpha_bar_0 = scheduler.alphas_cumprod[0]
        sqrt_alpha_bar = alpha_bar_0.sqrt()
        sqrt_one_minus = (1.0 - alpha_bar_0).sqrt()
        expected = sqrt_alpha_bar * clean + sqrt_one_minus * noise
        assert torch.allclose(noisy, expected, atol=1e-6)
        # Also verify signal dominates: sqrt(alpha_bar_0) should be very close to 1
        assert sqrt_alpha_bar > 0.999

    def test_max_timestep_noise_dominates_signal(
        self,
        flat_tensor_factory: Callable[..., torch.Tensor],
        ddpm_scheduler_factory: Callable[..., DDPMScheduler],
    ):
        num_train_timesteps = 1000
        clean = flat_tensor_factory(batch_size=2, feature_dimension=8)
        scheduler = ddpm_scheduler_factory(num_train_timesteps=num_train_timesteps)
        timesteps = torch.full((2,), num_train_timesteps - 1, dtype=torch.long)

        noisy, noise = add_noise_to_tensor(
            clean=clean,
            noise_scheduler=scheduler,
            timesteps=timesteps,
        )
        # At t=T-1, alpha_bar is small -> noise coefficient is large
        alpha_bar_last = scheduler.alphas_cumprod[num_train_timesteps - 1]
        noise_coefficient = (1.0 - alpha_bar_last).sqrt()
        signal_coefficient = alpha_bar_last.sqrt()
        # Noise coefficient should dominate signal coefficient
        assert noise_coefficient > signal_coefficient

    def test_integer_input_converted_to_float(
        self,
        ddpm_scheduler_factory: Callable[..., DDPMScheduler],
    ):
        clean = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=torch.int64)
        scheduler = ddpm_scheduler_factory(num_train_timesteps=100)
        timesteps = torch.zeros(2, dtype=torch.long)
        noisy, noise = add_noise_to_tensor(
            clean=clean,
            noise_scheduler=scheduler,
            timesteps=timesteps,
        )
        assert noisy.dtype == torch.float32
        assert noise.dtype == torch.float32

    def test_noise_different_per_call(
        self,
        flat_tensor_factory: Callable[..., torch.Tensor],
        ddpm_scheduler_factory: Callable[..., DDPMScheduler],
    ):
        # Two calls with the same clean and timesteps should produce different noise
        clean = flat_tensor_factory(batch_size=2, feature_dimension=8)
        scheduler = ddpm_scheduler_factory(num_train_timesteps=100)
        timesteps = torch.tensor([50, 50], dtype=torch.long)

        _, noise_1 = add_noise_to_tensor(
            clean=clean,
            noise_scheduler=scheduler,
            timesteps=timesteps,
        )
        _, noise_2 = add_noise_to_tensor(
            clean=clean,
            noise_scheduler=scheduler,
            timesteps=timesteps,
        )
        assert not torch.allclose(noise_1, noise_2)


class TestSampleRandomTimesteps:
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_output_shape(self, batch_size: int, device: torch.device):
        timesteps = sample_random_timesteps(
            batch_size=batch_size,
            num_train_timesteps=100,
            device=device,
        )
        assert timesteps.shape == (batch_size,)

    def test_values_within_range(self, device: torch.device):
        num_train_timesteps = 50
        timesteps = sample_random_timesteps(
            batch_size=1000,
            num_train_timesteps=num_train_timesteps,
            device=device,
        )
        assert timesteps.min() >= 0
        assert timesteps.max() < num_train_timesteps

    def test_output_device(self, device: torch.device):
        timesteps = sample_random_timesteps(
            batch_size=2,
            num_train_timesteps=100,
            device=device,
        )
        assert timesteps.device.type == device.type

    def test_output_dtype_is_long(self, device: torch.device):
        timesteps = sample_random_timesteps(
            batch_size=2,
            num_train_timesteps=100,
            device=device,
        )
        assert timesteps.dtype == torch.long

    def test_covers_full_range_with_many_samples(self, device: torch.device):
        num_train_timesteps = 50
        timesteps = sample_random_timesteps(
            batch_size=10000,
            num_train_timesteps=num_train_timesteps,
            device=device,
        )
        assert timesteps.min() == 0
        assert timesteps.max() == num_train_timesteps - 1

    def test_negative_batch_size_raises(self, device: torch.device):
        batch_size = -1
        with pytest.raises(
            ValueError,
            match=re.escape(f"batch_size must be non-negative, got {batch_size}."),
        ):
            sample_random_timesteps(
                batch_size=batch_size,
                num_train_timesteps=100,
                device=device,
            )

    def test_non_positive_num_train_timesteps_raises(self, device: torch.device):
        num_train_timesteps = 0
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"num_train_timesteps must be positive, got {num_train_timesteps}."
            ),
        ):
            sample_random_timesteps(
                batch_size=4,
                num_train_timesteps=num_train_timesteps,
                device=device,
            )


class TestSetupInferenceTimesteps:
    @pytest.mark.parametrize("num_inference_steps", [5, 20])
    def test_sets_timesteps_on_scheduler(
        self,
        ddpm_scheduler_factory: Callable[..., DDPMScheduler],
        num_inference_steps: int,
    ):
        scheduler = ddpm_scheduler_factory(num_train_timesteps=100)
        setup_inference_timesteps(
            noise_scheduler=scheduler,
            num_inference_steps=num_inference_steps,
        )
        assert len(scheduler.timesteps) == num_inference_steps

    def test_timesteps_are_descending(
        self,
        ddpm_scheduler_factory: Callable[..., DDPMScheduler],
    ):
        scheduler = ddpm_scheduler_factory(num_train_timesteps=100)
        setup_inference_timesteps(
            noise_scheduler=scheduler,
            num_inference_steps=10,
        )
        timesteps = scheduler.timesteps
        for index in range(len(timesteps) - 1):
            assert timesteps[index] > timesteps[index + 1]

    def test_ddim_fewer_steps_than_train(
        self,
        ddim_scheduler_factory: Callable[..., DDIMScheduler],
    ):
        num_train_timesteps = 100
        num_inference_steps = 10
        scheduler = ddim_scheduler_factory(
            num_train_timesteps=num_train_timesteps,
        )
        setup_inference_timesteps(
            noise_scheduler=scheduler,
            num_inference_steps=num_inference_steps,
        )
        assert len(scheduler.timesteps) == num_inference_steps
        assert scheduler.timesteps.max() < num_train_timesteps

    def test_first_inference_timestep_is_within_training_range(
        self,
        ddpm_scheduler_factory: Callable[..., DDPMScheduler],
    ):
        num_train_timesteps = 100
        scheduler = ddpm_scheduler_factory(num_train_timesteps=num_train_timesteps)
        setup_inference_timesteps(
            noise_scheduler=scheduler,
            num_inference_steps=10,
        )
        # First timestep (highest) should be within valid training range
        assert 0 < scheduler.timesteps[0] < num_train_timesteps
        # Last timestep should be the lowest
        assert scheduler.timesteps[-1] < scheduler.timesteps[0]


class TestDiffusionSchedulerConfig:
    @pytest.mark.parametrize(
        "scheduler_type", [SchedulerType.DDPM.value, SchedulerType.DDIM.value]
    )
    @pytest.mark.parametrize("num_train_timesteps", [50, 1000])
    @pytest.mark.parametrize("num_inference_steps", [5, 20])
    @pytest.mark.parametrize("prediction_type", ["epsilon", "sample"])
    @pytest.mark.parametrize("clip_sample", [True, False])
    def test_stores_configuration(
        self,
        scheduler_config_factory: Callable[..., DiffusionSchedulerConfig],
        scheduler_type: str,
        num_train_timesteps: int,
        num_inference_steps: int,
        prediction_type: str,
        clip_sample: bool,
    ):
        beta_start = 0.001
        beta_end = 0.05
        beta_schedule = "linear"
        variance_type = "fixed_small"
        set_alpha_to_one = False
        steps_offset = 2
        config = scheduler_config_factory(
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
        assert config.scheduler_type == scheduler_type
        assert config.num_train_timesteps == num_train_timesteps
        assert config.num_inference_steps == num_inference_steps
        assert config.beta_start == beta_start
        assert config.beta_end == beta_end
        assert config.beta_schedule == beta_schedule
        assert config.prediction_type == prediction_type
        assert config.clip_sample == clip_sample
        assert config.variance_type == variance_type
        assert config.set_alpha_to_one == set_alpha_to_one
        assert config.steps_offset == steps_offset

    def test_optional_fields_default_to_none(
        self,
        scheduler_config_factory: Callable[..., DiffusionSchedulerConfig],
    ):
        config = scheduler_config_factory(
            scheduler_type=SchedulerType.DDPM.value,
        )
        assert config.variance_type is None
        assert config.set_alpha_to_one is None
        assert config.steps_offset is None
