"""Tests for VariationalAlgorithm wrapper."""

import pytest
import torch

from versatil.data.constants import ProprioceptiveType
from versatil.models.decoding.algorithm import (
    VariationalAlgorithm,
    BehavioralCloning,
    FlowMatching,
)
from versatil.models.decoding.constants import LatentKey
from versatil.models.decoding.latent import (
    VAETransformerEncoder,
    GaussianPrior,
    DiffusionPrior,
)


@pytest.fixture
def device():
    """Device fixture."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def batch_size():
    """Batch size for tests."""
    return 4


@pytest.fixture
def prediction_horizon():
    """Prediction horizon for tests."""
    return 10


@pytest.fixture
def latent_dim():
    """Latent dimension for tests."""
    return 16


@pytest.fixture
def embedding_dim():
    """Embedding dimension for tests."""
    return 64


@pytest.fixture
def vae_encoder(latent_dim, embedding_dim, prediction_horizon, device):
    """Create VAE encoder fixture."""
    return VAETransformerEncoder(
        latent_dimension=latent_dim,
        embedding_dimension=embedding_dim,
        prediction_horizon=prediction_horizon,
        number_of_heads=2,
        feedforward_dimension=128,
        number_of_encoder_layers=2,
        dropout_rate=0.0,
        device=device,
    )


@pytest.fixture
def gaussian_prior(latent_dim, embedding_dim, device):
    """Create Gaussian prior fixture."""
    return GaussianPrior(
        latent_dimension=latent_dim,
        output_dim=embedding_dim,
        device=device,
    )


@pytest.fixture
def diffusion_prior(latent_dim, embedding_dim, device):
    """Create Diffusion prior fixture.

    Conditioning dim must match sample_features:
    - flat_feature: 32
    - temporal_feature: 5 * 16 = 80 (flattened)
    Total: 112
    """
    return DiffusionPrior(
        latent_dimension=latent_dim,
        conditioning_dim=112,  # Matches flattened sample_features (32 + 5*16)
        output_dim=embedding_dim,
        hidden_dims=[32, 32],
        num_train_timesteps=10,  # Small for testing
        num_inference_steps=3,   # Small for testing
        device=device,
    )


@pytest.fixture
def sample_features(batch_size, device):
    """Create sample features."""
    return {
        "flat_feature": torch.randn(batch_size, 32, device=device),
        "temporal_feature": torch.randn(batch_size, 5, 16, device=device),
    }


@pytest.fixture
def sample_actions(batch_size, prediction_horizon, device):
    """Create sample actions."""
    return {
        ProprioceptiveType.POSITION.value: torch.randn(batch_size, prediction_horizon, 3, device=device),
        ProprioceptiveType.GRIPPER.value: torch.randn(batch_size, prediction_horizon, 1, device=device),
    }


class TestVariationalAlgorithmAutoGaussianPrior:
    """Test VariationalAlgorithm with auto-created Gaussian prior."""

    def test_auto_creates_gaussian_prior(self, vae_encoder):
        """Test that GaussianPrior is auto-created when prior=None."""
        alg = VariationalAlgorithm(
            base_algorithm=BehavioralCloning(),
            posterior_encoder=vae_encoder,
            prior=None,
        )

        assert alg.prior is not None
        assert isinstance(alg.prior, GaussianPrior)
        assert alg.prior.latent_dimension == vae_encoder.latent_dimension


class TestVariationalAlgorithmWithGaussianPrior:
    """Test VariationalAlgorithm with explicit Gaussian prior."""

    @pytest.fixture
    def algorithm(self, vae_encoder, gaussian_prior):
        """Create algorithm with Gaussian prior."""
        return VariationalAlgorithm(
            base_algorithm=BehavioralCloning(),
            posterior_encoder=vae_encoder,
            prior=gaussian_prior,
        )

    def test_forward_training_adds_latent_to_features(
        self,
        algorithm,
        sample_features,
        sample_actions,
    ):
        """Test that forward pass adds latent to features during training."""
        # Create a simple mock network that checks for latent
        class MockNetwork:
            def __init__(self):
                self.prediction_horizon = 10
                self.use_position_actions = True
                self.use_orientation_actions = False
                self.use_gripper_actions = True
                self.position_dim = 3
                self.gripper_dim = 1

            def __call__(self, features, actions=None):
                # Check that latent was added to features
                assert LatentKey.POSTERIOR_LATENT.value in features
                assert features[LatentKey.POSTERIOR_LATENT.value].ndim == 2  # (B, embedding_dim)
                return {
                    ProprioceptiveType.POSITION.value: actions[ProprioceptiveType.POSITION.value],
                    ProprioceptiveType.GRIPPER.value: actions[ProprioceptiveType.GRIPPER.value],
                }

        mock_network = MockNetwork()
        output = algorithm.forward(mock_network, sample_features, sample_actions)

        # Check outputs
        assert ProprioceptiveType.POSITION.value in output
        assert ProprioceptiveType.GRIPPER.value in output
        assert LatentKey.POSTERIOR_MU.value in output  # VAE outputs
        assert LatentKey.POSTERIOR_LOGVAR.value in output

    def test_predict_samples_from_prior(
        self,
        algorithm,
        sample_features,
        batch_size,
    ):
        """Test that predict samples from prior."""
        class MockNetwork:
            def __init__(self):
                self.prediction_horizon = 10

            def __call__(self, features, actions=None):
                assert LatentKey.POSTERIOR_LATENT.value in features
                assert actions is None  # Inference mode
                return {
                    ProprioceptiveType.POSITION.value: torch.randn(batch_size, 10, 3),
                }

        mock_network = MockNetwork()
        output = algorithm.predict(mock_network, sample_features)

        assert ProprioceptiveType.POSITION.value in output
        # Should not have VAE outputs during inference
        assert LatentKey.POSTERIOR_MU.value not in output
        assert LatentKey.POSTERIOR_LOGVAR.value not in output


class TestVariationalAlgorithmWithDiffusionPrior:
    """Test VariationalAlgorithm with learned Diffusion prior."""

    @pytest.fixture
    def algorithm(self, vae_encoder, diffusion_prior):
        """Create algorithm with Diffusion prior."""
        return VariationalAlgorithm(
            base_algorithm=FlowMatching(sigma=0.0, num_inference_steps=2),
            posterior_encoder=vae_encoder,
            prior=diffusion_prior,
        )

    def test_forward_training_includes_prior_outputs(
        self,
        algorithm,
        sample_features,
        sample_actions,
    ):
        """Test that forward pass includes prior predictions/targets during training."""

        class MockNetwork:
            def __init__(self):
                self.prediction_horizon = 10
                self.use_position_actions = True
                self.use_orientation_actions = False
                self.use_gripper_actions = True
                self.position_dim = 3
                self.gripper_dim = 1

            def __call__(self, features, actions=None):
                # Return dummy predictions
                return {
                    ProprioceptiveType.POSITION.value: torch.randn(4, 10, 3),
                    ProprioceptiveType.GRIPPER.value: torch.randn(4, 10, 1),
                }

        mock_network = MockNetwork()
        output = algorithm.forward(mock_network, sample_features, sample_actions)

        # Check that prior outputs are included
        assert LatentKey.PRIOR_PREDICTION.value in output
        assert LatentKey.PRIOR_TARGET.value in output


