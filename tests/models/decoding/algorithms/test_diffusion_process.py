"""Tests for shared diffusion process components."""

import pytest
import torch

from refactoring.models.decoding.algorithm.diffusion_process import (
    DiffusionSchedulerConfig,
    add_noise_to_tensor,
    create_noise_scheduler,
    sample_random_timesteps,
    setup_inference_timesteps,
)
from refactoring.models.decoding.constants import (
    BetaSchedule,
    PredictionType,
    SchedulerType,
    VarianceType,
)


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def scheduler_config_ddpm():
    """Default DDPM scheduler configuration."""
    return DiffusionSchedulerConfig(
        scheduler_type=SchedulerType.DDPM.value,
        num_train_timesteps=100,
        num_inference_steps=10,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule=BetaSchedule.SQUAREDCOS_CAP_V2.value,
        prediction_type=PredictionType.EPSILON.value,
        clip_sample=True,
        variance_type=VarianceType.FIXED_SMALL.value,
    )


@pytest.fixture
def scheduler_config_ddim():
    """Default DDIM scheduler configuration."""
    return DiffusionSchedulerConfig(
        scheduler_type=SchedulerType.DDIM.value,
        num_train_timesteps=100,
        num_inference_steps=10,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule=BetaSchedule.SQUAREDCOS_CAP_V2.value,
        prediction_type=PredictionType.EPSILON.value,
        clip_sample=True,
        set_alpha_to_one=True,
        steps_offset=0,
    )


class TestCreateNoiseScheduler:
    """Test noise scheduler creation."""

    def test_create_ddpm_scheduler(self, scheduler_config_ddpm):
        """Test DDPM scheduler creation."""
        scheduler = create_noise_scheduler(scheduler_config_ddpm)
        assert scheduler is not None
        assert scheduler.config.num_train_timesteps == 100
        assert scheduler.config.beta_start == 0.0001
        assert scheduler.config.beta_end == 0.02

    def test_create_ddim_scheduler(self, scheduler_config_ddim):
        """Test DDIM scheduler creation."""
        scheduler = create_noise_scheduler(scheduler_config_ddim)
        assert scheduler is not None
        assert scheduler.config.num_train_timesteps == 100
        assert scheduler.config.beta_start == 0.0001

    def test_invalid_scheduler_type_raises_error(self, scheduler_config_ddpm):
        """Test that invalid scheduler type raises ValueError."""
        scheduler_config_ddpm.scheduler_type = "invalid_type"
        with pytest.raises(ValueError, match="Unknown scheduler_type"):
            create_noise_scheduler(scheduler_config_ddpm)

    def test_scheduler_respects_prediction_type(self, scheduler_config_ddpm):
        """Test that prediction type is correctly set."""
        for pred_type in [PredictionType.EPSILON.value, PredictionType.SAMPLE.value]:
            scheduler_config_ddpm.prediction_type = pred_type
            scheduler = create_noise_scheduler(scheduler_config_ddpm)
            assert scheduler.config.prediction_type == pred_type


class TestAddNoiseToTensor:
    """Test noise addition to tensors."""

    def test_noise_addition_shape(self, scheduler_config_ddpm, device):
        """Test that noisy tensor has same shape as input."""
        scheduler = create_noise_scheduler(scheduler_config_ddpm)
        clean = torch.randn(8, 16, 3, device=device)
        timesteps = torch.randint(0, 100, (8,), device=device)

        noisy, noise = add_noise_to_tensor(clean, scheduler, timesteps)

        assert noisy.shape == clean.shape
        assert noise.shape == clean.shape

    def test_noise_addition_deterministic_with_seed(self, scheduler_config_ddpm, device):
        """Test that noise addition is reproducible with fixed seed."""
        scheduler = create_noise_scheduler(scheduler_config_ddpm)
        clean = torch.randn(8, 16, 3, device=device)
        timesteps = torch.randint(0, 100, (8,), device=device)

        # Add noise twice with same seed
        torch.manual_seed(42)
        noisy1, noise1 = add_noise_to_tensor(clean, scheduler, timesteps)

        torch.manual_seed(42)
        noisy2, noise2 = add_noise_to_tensor(clean, scheduler, timesteps)

        assert torch.allclose(noisy1, noisy2)
        assert torch.allclose(noise1, noise2)

    def test_noise_addition_different_timesteps(self, scheduler_config_ddpm, device):
        """Test that noise level varies with timestep."""
        scheduler = create_noise_scheduler(scheduler_config_ddpm)
        clean = torch.zeros(8, 16, 3, device=device)

        # At timestep 0, should have minimal noise
        timesteps_early = torch.zeros(8, dtype=torch.long, device=device)
        noisy_early, _ = add_noise_to_tensor(clean, scheduler, timesteps_early)

        # At final timestep, should have maximum noise
        timesteps_late = torch.full((8,), 99, dtype=torch.long, device=device)
        noisy_late, _ = add_noise_to_tensor(clean, scheduler, timesteps_late)

        # Later timesteps should have more noise
        early_magnitude = torch.abs(noisy_early).mean()
        late_magnitude = torch.abs(noisy_late).mean()
        assert late_magnitude > early_magnitude


class TestSampleRandomTimesteps:
    """Test random timestep sampling."""

    def test_timestep_sampling_shape(self, device):
        """Test that timesteps have correct shape."""
        batch_size = 32
        timesteps = sample_random_timesteps(batch_size, 100, device)

        assert timesteps.shape == (batch_size,)
        assert timesteps.dtype == torch.long

    def test_timestep_sampling_range(self, device):
        """Test that timesteps are in valid range."""
        num_train_timesteps = 100
        timesteps = sample_random_timesteps(64, num_train_timesteps, device)

        assert torch.all(timesteps >= 0)
        assert torch.all(timesteps < num_train_timesteps)

    def test_timestep_sampling_uniform_distribution(self, device):
        """Test that timesteps are approximately uniformly distributed."""
        num_train_timesteps = 100
        num_samples = 10000

        # Sample many timesteps
        timesteps = sample_random_timesteps(num_samples, num_train_timesteps, device)

        # Check that we get samples across the full range
        unique_values = torch.unique(timesteps)
        assert len(unique_values) > num_train_timesteps * 0.8  # At least 80% coverage


class TestSetupInferenceTimesteps:
    """Test inference timestep setup."""

    def test_setup_inference_timesteps_ddpm(self, scheduler_config_ddpm):
        """Test DDPM inference timestep setup."""
        scheduler = create_noise_scheduler(scheduler_config_ddpm)
        num_inference_steps = 10

        setup_inference_timesteps(scheduler, num_inference_steps)

        assert len(scheduler.timesteps) == num_inference_steps

    def test_setup_inference_timesteps_ddim(self, scheduler_config_ddim):
        """Test DDIM inference timestep setup."""
        scheduler = create_noise_scheduler(scheduler_config_ddim)
        num_inference_steps = 10

        setup_inference_timesteps(scheduler, num_inference_steps)

        assert len(scheduler.timesteps) == num_inference_steps

    def test_setup_different_num_inference_steps(self, scheduler_config_ddpm):
        """Test setup with different numbers of inference steps."""
        scheduler = create_noise_scheduler(scheduler_config_ddpm)

        for num_steps in [5, 10, 20, 50]:
            setup_inference_timesteps(scheduler, num_steps)
            assert len(scheduler.timesteps) == num_steps

    def test_timesteps_descending_order(self, scheduler_config_ddpm):
        """Test that timesteps are in descending order (for denoising)."""
        scheduler = create_noise_scheduler(scheduler_config_ddpm)
        setup_inference_timesteps(scheduler, 10)

        timesteps_list = scheduler.timesteps.tolist()
        assert timesteps_list == sorted(timesteps_list, reverse=True)


class TestDiffusionSchedulerConfig:
    """Test DiffusionSchedulerConfig dataclass."""

    def test_config_creation_with_all_params(self):
        """Test creating config with all parameters."""
        config = DiffusionSchedulerConfig(
            scheduler_type="ddpm",
            num_train_timesteps=1000,
            num_inference_steps=50,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            prediction_type="epsilon",
            clip_sample=False,
            variance_type="fixed_large",
            set_alpha_to_one=True,
            steps_offset=1,
        )

        assert config.scheduler_type == "ddpm"
        assert config.num_train_timesteps == 1000
        assert config.variance_type == "fixed_large"

    def test_config_creation_minimal_params(self):
        """Test creating config with minimal required parameters."""
        config = DiffusionSchedulerConfig(
            scheduler_type="ddim",
            num_train_timesteps=100,
            num_inference_steps=10,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
            clip_sample=True,
        )

        assert config.scheduler_type == "ddim"
        assert config.variance_type is None  # Optional params default to None
        assert config.set_alpha_to_one is None
