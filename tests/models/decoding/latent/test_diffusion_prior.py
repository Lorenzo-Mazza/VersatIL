"""Tests for DiffusionPrior learned prior for variational models."""
import pytest
import torch
from refactoring.models.decoding.latent.diffusion_prior import DiffusionPrior
from refactoring.models.decoding.constants import PRIOR_PREDICTION_KEY, PRIOR_TARGET_KEY
from refactoring.models.layers.activation import ActivationFunction


@pytest.mark.unit
class TestDiffusionPriorInitialization:
    """Tests for DiffusionPrior initialization."""

    def test_initialization_default_params(self, device):
        """Test DiffusionPrior initializes correctly with default parameters."""
        latent_dim = 32
        conditioning_dim = 128

        prior = DiffusionPrior(
            latent_dimension=latent_dim,
            conditioning_dim=conditioning_dim,
            device=str(device),
        )

        assert prior.latent_dimension == latent_dim
        assert prior.conditioning_dim == conditioning_dim
        assert prior.num_inference_steps == 10  # Default value
        assert prior.timestep_embed_dim == latent_dim

        # Check default hidden_dims
        expected_input_dim = latent_dim + conditioning_dim + latent_dim
        assert prior.denoising_network is not None

    def test_initialization_custom_hidden_dims(self, device):
        """Test DiffusionPrior with custom hidden dimensions."""
        latent_dim = 32
        conditioning_dim = 128
        custom_hidden_dims = [128, 256, 128]

        prior = DiffusionPrior(
            latent_dimension=latent_dim,
            conditioning_dim=conditioning_dim,
            hidden_dims=custom_hidden_dims,
            device=str(device),
        )

        assert prior.latent_dimension == latent_dim
        assert prior.conditioning_dim == conditioning_dim

    def test_initialization_custom_diffusion_params(self, device):
        """Test DiffusionPrior with custom diffusion parameters."""
        prior = DiffusionPrior(
            latent_dimension=16,
            conditioning_dim=64,
            num_train_timesteps=50,
            num_inference_steps=5,
            beta_start=0.0002,
            beta_end=0.01,
            beta_schedule="linear",
            device=str(device),
        )

        assert prior.num_inference_steps == 5
        assert prior.noise_scheduler.config.num_train_timesteps == 50
        assert prior.noise_scheduler.config.beta_start == 0.0002
        assert prior.noise_scheduler.config.beta_end == 0.01
        assert prior.noise_scheduler.config.beta_schedule == "linear"

    def test_submodules_created(self, device):
        """Test that all required submodules are created."""
        prior = DiffusionPrior(
            latent_dimension=32,
            conditioning_dim=128,
            device=str(device),
        )

        # Check timestep MLP exists
        assert hasattr(prior, "timestep_mlp")
        assert isinstance(prior.timestep_mlp, torch.nn.Sequential)

        # Check denoising network exists
        assert hasattr(prior, "denoising_network")

        # Check noise scheduler
        assert hasattr(prior, "noise_scheduler")


@pytest.mark.unit
class TestDiffusionPriorSamplePrior:
    """Tests for sample_prior method."""

    def test_sample_prior_with_conditioning(self, device):
        """Test sampling from prior with conditioning."""
        batch_size = 4
        latent_dim = 32
        conditioning_dim = 128

        prior = DiffusionPrior(
            latent_dimension=latent_dim,
            conditioning_dim=conditioning_dim,
             
            num_inference_steps=2,  # Fast for testing
            device=str(device),
        )

        conditioning = torch.randn(batch_size, conditioning_dim, device=device)

        # Sample from prior
        latent_samples = prior.sample_prior(batch_size=batch_size, conditioning=conditioning)

        # Check output shape
        assert latent_samples.shape == (batch_size, latent_dim)
        assert latent_samples.device.type == device.type

    def test_sample_prior_without_conditioning(self, device):
        """Test sampling from prior without conditioning (fallback to zeros)."""
        batch_size = 4
        latent_dim = 32
        conditioning_dim = 128

        prior = DiffusionPrior(
            latent_dimension=latent_dim,
            conditioning_dim=conditioning_dim,
             
            num_inference_steps=2,
            device=str(device),
        )

        # Sample without conditioning
        latent_samples = prior.sample_prior(batch_size=batch_size, conditioning=None)

        # Should still produce valid samples (using zero conditioning)
        assert latent_samples.shape == (batch_size, latent_dim)
        assert latent_samples.device.type == device.type

    def test_sample_prior_different_batch_sizes(self, device):
        """Test sampling with different batch sizes."""
        latent_dim = 32
        conditioning_dim = 128

        prior = DiffusionPrior(
            latent_dimension=latent_dim,
            conditioning_dim=conditioning_dim,
             
            num_inference_steps=2,
            device=str(device),
        )

        for batch_size in [1, 4, 8]:
            conditioning = torch.randn(batch_size, conditioning_dim, device=device)
            latent_samples = prior.sample_prior(batch_size=batch_size, conditioning=conditioning)
            assert latent_samples.shape == (batch_size, latent_dim)

    def test_sample_prior_deterministic_with_seed(self, device):
        """Test that sampling is deterministic when seed is set."""
        batch_size = 4
        latent_dim = 32
        conditioning_dim = 128

        prior = DiffusionPrior(
            latent_dimension=latent_dim,
            conditioning_dim=conditioning_dim,
             
            num_inference_steps=2,
            device=str(device),
        )

        conditioning = torch.randn(batch_size, conditioning_dim, device=device)

        # Sample twice with same seed
        torch.manual_seed(42)
        samples1 = prior.sample_prior(batch_size=batch_size, conditioning=conditioning)

        torch.manual_seed(42)
        samples2 = prior.sample_prior(batch_size=batch_size, conditioning=conditioning)

        # Should be identical
        assert torch.allclose(samples1, samples2, atol=1e-6)


@pytest.mark.unit
class TestDiffusionPriorForward:
    """Tests for forward method (training mode)."""

    def test_forward_returns_predictions_and_targets(self, device):
        """Test that forward returns predictions and targets (not loss)."""
        batch_size = 4
        latent_dim = 32
        conditioning_dim = 128

        prior = DiffusionPrior(
            latent_dimension=latent_dim,
            conditioning_dim=conditioning_dim,
            device=str(device),
        )

        # Create target latents and conditioning
        target_latents = torch.randn(batch_size, latent_dim, device=device)
        conditioning = torch.randn(batch_size, conditioning_dim, device=device)

        # Forward pass
        outputs = prior.forward(target_latents=target_latents, conditioning=conditioning)

        # Check outputs are dictionary with predictions and targets
        assert isinstance(outputs, dict)
        assert PRIOR_PREDICTION_KEY in outputs
        assert PRIOR_TARGET_KEY in outputs

        # Check shapes
        assert outputs[PRIOR_PREDICTION_KEY].shape == (batch_size, latent_dim)
        assert outputs[PRIOR_TARGET_KEY].shape == (batch_size, latent_dim)

    def test_forward_gradients_flow_to_prior(self, device):
        """Test that gradients flow to denoising network."""
        batch_size = 4
        latent_dim = 32
        conditioning_dim = 128

        prior = DiffusionPrior(
            latent_dimension=latent_dim,
            conditioning_dim=conditioning_dim,
            device=str(device),
        )

        # Create target latents WITH gradient tracking
        target_latents = torch.randn(batch_size, latent_dim, device=device, requires_grad=True)
        conditioning = torch.randn(batch_size, conditioning_dim, device=device)

        # Forward pass
        outputs = prior.forward(target_latents=target_latents, conditioning=conditioning)

        # Compute loss manually (MSE between prediction and target)
        loss = torch.nn.functional.mse_loss(
            outputs[PRIOR_PREDICTION_KEY],
            outputs[PRIOR_TARGET_KEY]
        )

        # Backward pass
        loss.backward()

        # Check that denoising network has gradients
        for param in prior.denoising_network.parameters():
            assert param.grad is not None

    def test_forward_with_different_timesteps(self, device):
        """Test that forward uses random timesteps (outputs vary)."""
        batch_size = 4
        latent_dim = 32
        conditioning_dim = 128

        prior = DiffusionPrior(
            latent_dimension=latent_dim,
            conditioning_dim=conditioning_dim,
            num_train_timesteps=100,
            device=str(device),
        )

        target_latents = torch.randn(batch_size, latent_dim, device=device)
        conditioning = torch.randn(batch_size, conditioning_dim, device=device)

        # Forward multiple times - predictions should vary due to random timesteps
        predictions = []
        for _ in range(5):
            outputs = prior.forward(target_latents=target_latents, conditioning=conditioning)
            predictions.append(outputs[PRIOR_PREDICTION_KEY].detach().cpu())

        # Not all predictions should be identical (random timestep sampling)
        # NOTE: This is a stochastic test, could fail with very low probability
        unique_predictions = len(set([pred.sum().item() for pred in predictions]))
        assert unique_predictions > 1, "All predictions are identical, expected variation due to random timesteps"


@pytest.mark.unit
class TestDiffusionPriorDevicePlacement:
    """Tests for device placement."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_placement_cuda(self):
        """Test that prior is correctly placed on CUDA device."""
        prior = DiffusionPrior(
            latent_dimension=32,
            conditioning_dim=128,
            device="cuda",
        )

        # Check device
        assert next(prior.parameters()).device.type == "cuda"

    def test_device_placement_cpu(self):
        """Test that prior is correctly placed on CPU device."""
        prior = DiffusionPrior(
            latent_dimension=32,
            conditioning_dim=128,
            device="cpu",
        )

        # Check device
        assert next(prior.parameters()).device.type == "cpu"


@pytest.mark.integration
class TestDiffusionPriorIntegration:
    """Integration tests for DiffusionPrior."""

    def test_training_loop_decreases_loss(self, device):
        """Test that prior loss decreases during training."""
        batch_size = 8
        latent_dim = 32
        conditioning_dim = 128

        prior = DiffusionPrior(
            latent_dimension=latent_dim,
            conditioning_dim=conditioning_dim,
             
            num_train_timesteps=50,
            device=str(device),
        )

        optimizer = torch.optim.Adam(prior.parameters(), lr=1e-3)

        # Generate fixed target latents and conditioning
        target_latents = torch.randn(batch_size, latent_dim, device=device)
        conditioning = torch.randn(batch_size, conditioning_dim, device=device)

        # Train for more steps to ensure decrease
        losses = []

        for step in range(100):
            optimizer.zero_grad()

            # Get predictions and targets
            outputs = prior.forward(target_latents=target_latents, conditioning=conditioning)

            # Compute loss (MSE between prediction and target)
            loss = torch.nn.functional.mse_loss(
                outputs[PRIOR_PREDICTION_KEY],
                outputs[PRIOR_TARGET_KEY]
            )

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Check that average loss in second half is lower than first half
        # This is more robust to stochasticity than comparing single points
        first_half_avg = sum(losses[:50]) / 50
        second_half_avg = sum(losses[50:]) / 50

        assert second_half_avg < first_half_avg, (
            f"Loss did not decrease on average: "
            f"first half avg={first_half_avg:.4f}, second half avg={second_half_avg:.4f}"
        )

    def test_prior_samples_have_reasonable_statistics(self, device):
        """Test that prior samples have reasonable statistics after some training."""
        batch_size = 16
        latent_dim = 32
        conditioning_dim = 128

        prior = DiffusionPrior(
            latent_dimension=latent_dim,
            conditioning_dim=conditioning_dim,
             
            num_inference_steps=5,
            device=str(device),
        )

        conditioning = torch.randn(batch_size, conditioning_dim, device=device)

        # Sample from prior
        samples = prior.sample_prior(batch_size=batch_size, conditioning=conditioning)

        # Check that samples are not NaN or Inf
        assert not torch.isnan(samples).any()
        assert not torch.isinf(samples).any()

        # Check that samples have reasonable magnitude (not all zeros)
        assert samples.abs().mean() > 0.01
