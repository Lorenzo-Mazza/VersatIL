"""Tests for VariationalAlgorithm wrapper."""

import pytest
import torch

from refactoring.data.constants import POSITION_ACTION_KEY, GRIPPER_ACTION_KEY
from refactoring.models.decoding.algorithm import (
    VariationalAlgorithm,
    BehavioralCloning,
    FlowMatching,
)
from refactoring.models.decoding.constants import LATENT_KEY, MU_KEY, LOGVAR_KEY
from refactoring.models.decoding.latent import (
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
        POSITION_ACTION_KEY: torch.randn(batch_size, prediction_horizon, 3, device=device),
        GRIPPER_ACTION_KEY: torch.randn(batch_size, prediction_horizon, 1, device=device),
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
                assert LATENT_KEY in features
                assert features[LATENT_KEY].ndim == 2  # (B, embedding_dim)
                return {
                    POSITION_ACTION_KEY: actions[POSITION_ACTION_KEY],
                    GRIPPER_ACTION_KEY: actions[GRIPPER_ACTION_KEY],
                }

        mock_network = MockNetwork()
        output = algorithm.forward(mock_network, sample_features, sample_actions)

        # Check outputs
        assert POSITION_ACTION_KEY in output
        assert GRIPPER_ACTION_KEY in output
        assert MU_KEY in output  # VAE outputs
        assert LOGVAR_KEY in output

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
                assert LATENT_KEY in features
                assert actions is None  # Inference mode
                return {
                    POSITION_ACTION_KEY: torch.randn(batch_size, 10, 3),
                }

        mock_network = MockNetwork()
        output = algorithm.predict(mock_network, sample_features)

        assert POSITION_ACTION_KEY in output
        # Should not have VAE outputs during inference
        assert MU_KEY not in output
        assert LOGVAR_KEY not in output


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
        from refactoring.models.decoding.constants import (
            PRIOR_PREDICTION_KEY,
            PRIOR_TARGET_KEY,
        )

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
                    POSITION_ACTION_KEY: torch.randn(4, 10, 3),
                    GRIPPER_ACTION_KEY: torch.randn(4, 10, 1),
                }

        mock_network = MockNetwork()
        output = algorithm.forward(mock_network, sample_features, sample_actions)

        # Check that prior outputs are included
        assert PRIOR_PREDICTION_KEY in output
        assert PRIOR_TARGET_KEY in output


@pytest.mark.unit
class TestExtractConditioning:
    """Test the _extract_conditioning static method."""

    def test_extracts_flat_features(self):
        """Test extraction of flat 2D features."""
        features = {
            "flat1": torch.randn(4, 32),
            "flat2": torch.randn(4, 16),
        }
        cond = VariationalAlgorithm._extract_conditioning(features)

        assert cond is not None
        assert cond.shape == (4, 48)  # 32 + 16

    def test_flattens_temporal_features(self):
        """Test flattening of 3D temporal features."""
        features = {
            "temporal": torch.randn(4, 5, 8),  # (B, T, D)
        }
        cond = VariationalAlgorithm._extract_conditioning(features)

        assert cond is not None
        assert cond.shape == (4, 40)  # 5 * 8

    def test_ignores_spatial_features(self):
        """Test that 4D spatial features are ignored."""
        features = {
            "spatial": torch.randn(4, 3, 64, 64),  # (B, C, H, W)
            "flat": torch.randn(4, 16),
        }
        cond = VariationalAlgorithm._extract_conditioning(features)

        assert cond is not None
        assert cond.shape == (4, 16)  # Only flat feature

    def test_returns_none_when_no_valid_features(self):
        """Test that None is returned when no valid features."""
        features = {
            "spatial": torch.randn(4, 3, 64, 64),
        }
        cond = VariationalAlgorithm._extract_conditioning(features)

        assert cond is None

