"""Tests for GaussianPrior - simple N(0,I) prior for variational models."""
import pytest
import torch
from refactoring.models.decoding.latent import GaussianPrior


@pytest.mark.unit
class TestGaussianPriorInitialization:
    """Tests for GaussianPrior initialization."""

    def test_initialization_default_params(self, device):
        """Test GaussianPrior initializes correctly with default parameters."""
        latent_dim = 32
        output_dim = 256

        prior = GaussianPrior(
            latent_dim=latent_dim,
            output_dim=output_dim,
            device=str(device),
        )

        assert prior.latent_dim == latent_dim
        assert prior.output_dim == output_dim
        assert prior.infer_constant_prior == False  # Default
        assert isinstance(prior.latent_output_projection, torch.nn.Linear)
        assert prior.latent_output_projection.in_features == latent_dim
        assert prior.latent_output_projection.out_features == output_dim

    def test_initialization_with_constant_prior(self, device):
        """Test GaussianPrior with constant prior mode (like ACT)."""
        prior = GaussianPrior(
            latent_dim=16,
            output_dim=128,
            device=str(device),
            infer_constant_prior=True,
        )

        assert prior.infer_constant_prior == True

    def test_latent_output_projection_created(self, device):
        """Test that latent output projection layer is created."""
        latent_dim = 64
        output_dim = 512

        prior = GaussianPrior(
            latent_dim=latent_dim,
            output_dim=output_dim,
            device=str(device),
        )

        # Check projection layer exists and has correct dimensions
        assert hasattr(prior, 'latent_output_projection')
        proj = prior.latent_output_projection
        assert isinstance(proj, torch.nn.Linear)
        assert proj.in_features == latent_dim
        assert proj.out_features == output_dim

    def test_device_placement_on_init(self, device):
        """Test that prior is moved to correct device during init."""
        prior = GaussianPrior(
            latent_dim=32,
            output_dim=256,
            device=str(device),
        )

        # Check all parameters are on correct device
        for param in prior.parameters():
            assert param.device.type == device.type

    @pytest.mark.parametrize("latent_dim,output_dim", [
        (16, 64),
        (32, 128),
        (64, 256),
        (128, 512),
        (256, 1024),
    ])
    def test_different_latent_output_dims(self, device, latent_dim, output_dim):
        """Test various dimension combinations."""
        prior = GaussianPrior(
            latent_dim=latent_dim,
            output_dim=output_dim,
            device=str(device),
        )

        assert prior.latent_dim == latent_dim
        assert prior.output_dim == output_dim

    def test_parameter_validation(self, device):
        """Test that all parameters are stored correctly."""
        latent_dim = 32
        output_dim = 256
        infer_constant = True

        prior = GaussianPrior(
            latent_dim=latent_dim,
            output_dim=output_dim,
            device=str(device),
            infer_constant_prior=infer_constant,
        )

        assert prior.latent_dim == latent_dim
        assert prior.output_dim == output_dim
        assert prior.infer_constant_prior == infer_constant
        assert prior.device == str(device)


@pytest.mark.unit
class TestGaussianPriorSamplePrior:
    """Tests for GaussianPrior.sample_prior() method."""

    def test_sample_prior_random_sampling(self, device):
        """Test standard N(0,I) random sampling."""
        prior = GaussianPrior(
            latent_dim=32,
            output_dim=256,
            device=str(device),
            infer_constant_prior=False,
        )

        batch_size = 4
        samples = prior.sample_prior(batch_size=batch_size)

        # Check shape
        assert samples.shape == (batch_size, prior.output_dim)

        # Check device
        assert samples.device.type == device.type

        # Check no NaN or Inf
        assert not torch.isnan(samples).any()
        assert not torch.isinf(samples).any()

    def test_sample_prior_constant_zero_latent(self, device):
        """Test constant zero latent sampling (ACT-style)."""
        prior = GaussianPrior(
            latent_dim=32,
            output_dim=256,
            device=str(device),
            infer_constant_prior=True,
        )

        batch_size = 4
        samples = prior.sample_prior(batch_size=batch_size)

        # With constant prior, the latent is zeros before projection
        # After projection, it won't necessarily be all zeros due to bias
        # But it should be deterministic
        samples2 = prior.sample_prior(batch_size=batch_size)
        assert torch.allclose(samples, samples2)

    def test_sample_prior_output_shape(self, device):
        """Test output shape is (batch_size, output_dim)."""
        latent_dim = 16
        output_dim = 128
        prior = GaussianPrior(
            latent_dim=latent_dim,
            output_dim=output_dim,
            device=str(device),
        )

        for batch_size in [1, 2, 8, 16]:
            samples = prior.sample_prior(batch_size=batch_size)
            assert samples.shape == (batch_size, output_dim)

    def test_sample_prior_output_device(self, device):
        """Test output is on correct device."""
        prior = GaussianPrior(
            latent_dim=32,
            output_dim=256,
            device=str(device),
        )

        samples = prior.sample_prior(batch_size=4)
        assert samples.device.type == device.type

    def test_sample_prior_ignores_conditioning(self, device):
        """Test that conditioning parameter has no effect."""
        prior = GaussianPrior(
            latent_dim=32,
            output_dim=256,
            device=str(device),
            infer_constant_prior=True,  # Use constant to make test deterministic
        )

        samples1 = prior.sample_prior(batch_size=4, conditioning=None)
        conditioning = torch.randn(4, 64, device=device)
        samples2 = prior.sample_prior(batch_size=4, conditioning=conditioning)

        # Constant prior should produce same samples regardless of conditioning
        assert torch.allclose(samples1, samples2)

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32])
    def test_sample_prior_different_batch_sizes(self, device, batch_size):
        """Test sampling with different batch sizes."""
        prior = GaussianPrior(
            latent_dim=32,
            output_dim=256,
            device=str(device),
        )

        samples = prior.sample_prior(batch_size=batch_size)
        assert samples.shape == (batch_size, 256)

    def test_sample_prior_deterministic_with_seed(self, device):
        """Test that same seed produces same samples."""
        prior = GaussianPrior(
            latent_dim=32,
            output_dim=256,
            device=str(device),
            infer_constant_prior=False,
        )

        torch.manual_seed(123)
        samples1 = prior.sample_prior(batch_size=4)

        torch.manual_seed(123)
        samples2 = prior.sample_prior(batch_size=4)

        assert torch.allclose(samples1, samples2)

    def test_sample_prior_random_without_seed(self, device):
        """Test that different calls produce different samples (for non-constant)."""
        prior = GaussianPrior(
            latent_dim=32,
            output_dim=256,
            device=str(device),
            infer_constant_prior=False,
        )

        samples1 = prior.sample_prior(batch_size=4)
        samples2 = prior.sample_prior(batch_size=4)

        # Should be different (with extremely high probability)
        assert not torch.allclose(samples1, samples2)

    def test_sample_prior_constant_always_same(self, device):
        """Test that constant mode always produces same output."""
        prior = GaussianPrior(
            latent_dim=32,
            output_dim=256,
            device=str(device),
            infer_constant_prior=True,
        )

        samples1 = prior.sample_prior(batch_size=4)
        samples2 = prior.sample_prior(batch_size=4)
        samples3 = prior.sample_prior(batch_size=4)

        # All should be identical
        assert torch.allclose(samples1, samples2)
        assert torch.allclose(samples2, samples3)


@pytest.mark.unit
class TestGaussianPriorForward:
    """Tests for GaussianPrior.forward() method."""

    def test_forward_returns_empty_dict(self, device):
        """Test that forward() returns empty dict (no training loss)."""
        prior = GaussianPrior(
            latent_dim=32,
            output_dim=256,
            device=str(device),
        )

        target_latents = torch.randn(4, 32, device=device)
        conditioning = torch.randn(4, 64, device=device)

        outputs = prior.forward(target_latents, conditioning)

        assert isinstance(outputs, dict)
        assert len(outputs) == 0
        assert outputs == {}

    def test_forward_ignores_target_latents(self, device):
        """Test that forward() returns same empty dict regardless of inputs."""
        prior = GaussianPrior(
            latent_dim=32,
            output_dim=256,
            device=str(device),
        )

        conditioning = torch.randn(4, 64, device=device)

        outputs1 = prior.forward(torch.randn(4, 32, device=device), conditioning)
        outputs2 = prior.forward(torch.zeros(4, 32, device=device), conditioning)
        outputs3 = prior.forward(torch.ones(4, 32, device=device), conditioning)

        assert outputs1 == outputs2 == outputs3 == {}

    def test_forward_ignores_conditioning(self, device):
        """Test that conditioning doesn't affect output."""
        prior = GaussianPrior(
            latent_dim=32,
            output_dim=256,
            device=str(device),
        )

        target_latents = torch.randn(4, 32, device=device)

        outputs1 = prior.forward(target_latents, torch.randn(4, 64, device=device))
        outputs2 = prior.forward(target_latents, torch.zeros(4, 64, device=device))
        outputs3 = prior.forward(target_latents, torch.ones(4, 64, device=device))

        assert outputs1 == outputs2 == outputs3 == {}

    def test_forward_no_gradient_flow_to_prior(self, device):
        """Test that forward() doesn't create computation graph."""
        prior = GaussianPrior(
            latent_dim=32,
            output_dim=256,
            device=str(device),
        )

        target_latents = torch.randn(4, 32, device=device, requires_grad=True)
        conditioning = torch.randn(4, 64, device=device, requires_grad=True)

        outputs = prior.forward(target_latents, conditioning)

        # Empty dict, so no tensors to check gradients on
        assert len(outputs) == 0

    def test_forward_accepts_various_input_shapes(self, device):
        """Test forward() works with various batch sizes."""
        prior = GaussianPrior(
            latent_dim=32,
            output_dim=256,
            device=str(device),
        )

        for batch_size in [1, 2, 4, 8]:
            target_latents = torch.randn(batch_size, 32, device=device)
            conditioning = torch.randn(batch_size, 64, device=device)
            outputs = prior.forward(target_latents, conditioning)
            assert outputs == {}

    def test_forward_output_type(self, device):
        """Test that forward() returns dict[str, torch.Tensor]."""
        prior = GaussianPrior(
            latent_dim=32,
            output_dim=256,
            device=str(device),
        )

        target_latents = torch.randn(4, 32, device=device)
        conditioning = torch.randn(4, 64, device=device)

        outputs = prior.forward(target_latents, conditioning)

        assert isinstance(outputs, dict)
        # All values should be tensors (but dict is empty)
        for value in outputs.values():
            assert isinstance(value, torch.Tensor)

    def test_forward_consistency(self, device):
        """Test that same inputs produce same output (empty dict)."""
        prior = GaussianPrior(
            latent_dim=32,
            output_dim=256,
            device=str(device),
        )

        target_latents = torch.randn(4, 32, device=device)
        conditioning = torch.randn(4, 64, device=device)

        outputs1 = prior.forward(target_latents, conditioning)
        outputs2 = prior.forward(target_latents, conditioning)

        assert outputs1 == outputs2 == {}


@pytest.mark.unit
class TestGaussianPriorConstantVsRandom:
    """Tests comparing constant vs random prior modes."""

    def test_constant_vs_random_sampling_difference(self, device):
        """Test that constant and random modes produce different outputs."""
        prior_constant = GaussianPrior(
            latent_dim=32,
            output_dim=256,
            device=str(device),
            infer_constant_prior=True,
        )

        prior_random = GaussianPrior(
            latent_dim=32,
            output_dim=256,
            device=str(device),
            infer_constant_prior=False,
        )

        # Copy weights to make projection identical
        prior_random.latent_output_projection.load_state_dict(
            prior_constant.latent_output_projection.state_dict()
        )

        samples_constant = prior_constant.sample_prior(batch_size=4)
        samples_random = prior_random.sample_prior(batch_size=4)

        # Should be different (random samples vs constant zeros)
        assert not torch.allclose(samples_constant, samples_random)

    def test_constant_prior_deterministic(self, device):
        """Test that constant mode is always deterministic."""
        prior = GaussianPrior(
            latent_dim=32,
            output_dim=256,
            device=str(device),
            infer_constant_prior=True,
        )

        samples = [prior.sample_prior(batch_size=4) for _ in range(5)]

        # All samples should be identical
        for i in range(1, len(samples)):
            assert torch.allclose(samples[0], samples[i])

    def test_random_prior_stochastic(self, device):
        """Test that random mode produces different samples each call."""
        prior = GaussianPrior(
            latent_dim=32,
            output_dim=256,
            device=str(device),
            infer_constant_prior=False,
        )

        samples = [prior.sample_prior(batch_size=4) for _ in range(3)]

        # At least some samples should be different
        all_same = all(torch.allclose(samples[0], s) for s in samples[1:])
        assert not all_same

    def test_both_modes_same_output_dim(self, device):
        """Test that both modes respect output_dim."""
        output_dim = 512

        prior_constant = GaussianPrior(
            latent_dim=32,
            output_dim=output_dim,
            device=str(device),
            infer_constant_prior=True,
        )

        prior_random = GaussianPrior(
            latent_dim=32,
            output_dim=output_dim,
            device=str(device),
            infer_constant_prior=False,
        )

        samples_constant = prior_constant.sample_prior(batch_size=4)
        samples_random = prior_random.sample_prior(batch_size=4)

        assert samples_constant.shape[1] == output_dim
        assert samples_random.shape[1] == output_dim

    def test_projection_applied_in_both_modes(self, device):
        """Test that linear projection is applied in both modes."""
        latent_dim = 32
        output_dim = 256

        prior_constant = GaussianPrior(
            latent_dim=latent_dim,
            output_dim=output_dim,
            device=str(device),
            infer_constant_prior=True,
        )

        prior_random = GaussianPrior(
            latent_dim=latent_dim,
            output_dim=output_dim,
            device=str(device),
            infer_constant_prior=False,
        )

        # Both should have projection layer
        assert hasattr(prior_constant, 'latent_output_projection')
        assert hasattr(prior_random, 'latent_output_projection')

        # Output dimensions should match output_dim (not latent_dim)
        assert prior_constant.sample_prior(1).shape[1] == output_dim
        assert prior_random.sample_prior(1).shape[1] == output_dim


@pytest.mark.unit
class TestGaussianPriorDevicePlacement:
    """Tests for device placement and transfers."""

    def test_device_placement_cpu(self):
        """Test explicit CPU device placement."""
        prior = GaussianPrior(
            latent_dim=32,
            output_dim=256,
            device="cpu",
        )

        # Check all parameters are on CPU
        for param in prior.parameters():
            assert param.device.type == "cpu"

        # Check samples are on CPU
        samples = prior.sample_prior(batch_size=4)
        assert samples.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_placement_cuda(self):
        """Test explicit CUDA device placement."""
        prior = GaussianPrior(
            latent_dim=32,
            output_dim=256,
            device="cuda",
        )

        # Check all parameters are on CUDA
        for param in prior.parameters():
            assert param.device.type == "cuda"

        # Check samples are on CUDA
        samples = prior.sample_prior(batch_size=4)
        assert samples.device.type == "cuda"

    def test_parameter_on_correct_device(self, device):
        """Test that latent_output_projection parameters are on correct device."""
        prior = GaussianPrior(
            latent_dim=32,
            output_dim=256,
            device=str(device),
        )

        # Check projection layer parameters
        for param in prior.latent_output_projection.parameters():
            assert param.device.type == device.type

    def test_sample_prior_respects_device(self, device):
        """Test that samples are generated on correct device."""
        prior = GaussianPrior(
            latent_dim=32,
            output_dim=256,
            device=str(device),
        )

        samples = prior.sample_prior(batch_size=4)
        assert samples.device.type == device.type

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_device_transfers_module(self):
        """Test that .to(device) works correctly."""
        # Start on CPU
        prior = GaussianPrior(
            latent_dim=32,
            output_dim=256,
            device="cpu",
        )

        # Move to CUDA
        prior = prior.to("cuda")
        # Update device attribute
        prior.device = "cuda"

        # Check all parameters moved
        for param in prior.parameters():
            assert param.device.type == "cuda"

        # Check samples are on CUDA
        samples = prior.sample_prior(batch_size=4)
        assert samples.device.type == "cuda"


@pytest.mark.unit
class TestGaussianPriorGradients:
    """Tests for gradient flow and backpropagation."""

    def test_sampling_output_has_gradients_from_projection(self, device):
        """Test that sample_prior() output has gradients from trainable projection."""
        prior = GaussianPrior(
            latent_dim=32,
            output_dim=256,
            device=str(device),
        )

        samples = prior.sample_prior(batch_size=4)

        # Samples should have gradients because projection layer is trainable
        assert samples.requires_grad

    def test_projection_layer_trainable(self, device):
        """Test that linear projection layer parameters are trainable."""
        prior = GaussianPrior(
            latent_dim=32,
            output_dim=256,
            device=str(device),
        )

        # Check weight and bias are trainable
        assert prior.latent_output_projection.weight.requires_grad
        assert prior.latent_output_projection.bias.requires_grad

    def test_projection_receives_gradients(self, device):
        """Test that backprop through projection works."""
        prior = GaussianPrior(
            latent_dim=32,
            output_dim=256,
            device=str(device),
        )

        # Create input with gradients
        z = torch.randn(4, 32, device=device, requires_grad=True)

        # Forward through projection
        output = prior.latent_output_projection(z)

        # Backward
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert prior.latent_output_projection.weight.grad is not None
        assert prior.latent_output_projection.bias.grad is not None
        assert z.grad is not None

    def test_forward_no_backprop(self, device):
        """Test that forward() doesn't require gradients."""
        prior = GaussianPrior(
            latent_dim=32,
            output_dim=256,
            device=str(device),
        )

        target_latents = torch.randn(4, 32, device=device, requires_grad=True)
        conditioning = torch.randn(4, 64, device=device, requires_grad=True)

        outputs = prior.forward(target_latents, conditioning)

        # Empty dict, no computation graph
        assert len(outputs) == 0


@pytest.mark.integration
class TestGaussianPriorIntegration:
    """Integration tests for GaussianPrior."""

    def test_with_variational_algorithm(self, device):
        """Test that GaussianPrior works with VariationalAlgorithm."""
        from refactoring.models.decoding.algorithm.variational import VariationalAlgorithm
        from refactoring.models.decoding.algorithm.behavior_cloning import BehavioralCloning
        from refactoring.models.decoding.latent import VAETransformerEncoder

        # Create VAE encoder
        vae_encoder = VAETransformerEncoder(
            latent_dim=16,
            embedding_dimension=64,
            prediction_horizon=10,
            device=str(device),
        )

        # Create VariationalAlgorithm with explicit GaussianPrior
        prior = GaussianPrior(
            latent_dim=16,
            output_dim=64,
            device=str(device),
        )

        algorithm = VariationalAlgorithm(
            base_algorithm=BehavioralCloning(),
            posterior_encoder=vae_encoder,
            prior=prior,
        )

        assert algorithm.prior is prior

    def test_auto_creation_in_variational_wrapper(self, device):
        """Test auto-creation when prior=None in VariationalAlgorithm."""
        from refactoring.models.decoding.algorithm.variational import VariationalAlgorithm
        from refactoring.models.decoding.algorithm.behavior_cloning import BehavioralCloning
        from refactoring.models.decoding.latent import VAETransformerEncoder

        vae_encoder = VAETransformerEncoder(
            latent_dim=16,
            embedding_dimension=64,
            prediction_horizon=10,
            device=str(device),
        )

        algorithm = VariationalAlgorithm(
            base_algorithm=BehavioralCloning(),
            posterior_encoder=vae_encoder,
            prior=None,  # Should auto-create GaussianPrior
        )

        # Check prior was auto-created
        assert isinstance(algorithm.prior, GaussianPrior)
        assert algorithm.prior.latent_dim == 16
        assert algorithm.prior.output_dim == 64

    def test_constant_prior_in_act_mode(self, device):
        """Test constant prior mode (ACT-style)."""
        prior = GaussianPrior(
            latent_dim=32,
            output_dim=256,
            device=str(device),
            infer_constant_prior=True,
        )

        # Multiple samples should be identical
        samples = [prior.sample_prior(batch_size=4) for _ in range(3)]

        for i in range(1, len(samples)):
            assert torch.allclose(samples[0], samples[i])

    def test_sampling_in_inference_loop(self, device):
        """Test multiple inference steps work correctly."""
        prior = GaussianPrior(
            latent_dim=32,
            output_dim=256,
            device=str(device),
        )

        # Simulate multiple inference steps
        for _ in range(10):
            samples = prior.sample_prior(batch_size=4)

            # Check valid samples each time
            assert samples.shape == (4, 256)
            assert not torch.isnan(samples).any()
            assert not torch.isinf(samples).any()
