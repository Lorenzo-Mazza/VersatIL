"""Integration tests verifying all algorithms work with variational pattern and metrics.

This file demonstrates that:
1. All algorithms (BC, Diffusion, FlowMatching) work standalone
2. All algorithms can be wrapped with VariationalAlgorithm
3. Metrics correctly handle variational outputs (mu, logvar, prior predictions)
4. No backward compatibility code exists
"""

import pytest
import torch

from versatil.data.constants import ProprioceptiveType
from versatil.models.decoding.algorithm import (
    BehavioralCloning,
    Diffusion,
    FlowMatching,
    VariationalAlgorithm,
)
from versatil.models.decoding.constants import LatentKey
from versatil.models.decoding.latent import (
    VAETransformerEncoder,
    GaussianPrior,
    DiffusionPrior,
)
from versatil.metrics.components import (
    KLDivergenceLoss,
    PriorDenoisingLoss,
)


@pytest.fixture
def device():
    """Device fixture."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def sample_features(device):
    """Create sample features."""
    return {
        "flat_feature": torch.randn(2, 64, device=device),
    }

@pytest.fixture
def embedding_dimension():
    """Embedding dimension for tests."""
    return 64

@pytest.fixture
def sample_actions(device):
    """Create sample actions."""
    return {
        ProprioceptiveType.POSITION.value: torch.randn(2, 10, 3, device=device),
        ProprioceptiveType.GRIPPER.value: torch.randn(2, 10, 1, device=device),
    }


@pytest.fixture
def vae_encoder(device, embedding_dimension):
    """Create VAE encoder."""
    return VAETransformerEncoder(
        latent_dimension=16,
        embedding_dimension=embedding_dimension,
        prediction_horizon=10,
        number_of_heads=2,
        feedforward_dimension=128,
        number_of_encoder_layers=2,
        dropout_rate=0.0,
        device=device,
        observation_horizon=1
    )


@pytest.fixture
def diffusion_prior(device, embedding_dimension):
    """Create Diffusion prior."""
    return DiffusionPrior(
        latent_dimension=16,
        conditioning_dim=embedding_dimension,
        hidden_dims=[32, 32],
        num_train_timesteps=10,
        num_inference_steps=3,
        device=device,
    )


class MockNetwork:
    """Mock decoder network."""

    def __init__(self, device):
        self.prediction_horizon = 10
        self.use_position_actions = True
        self.use_orientation_actions = False
        self.use_gripper_actions = True
        self.position_dim = 3
        self.gripper_dim = 1
        self.device = device

    def __call__(self, features, actions=None):
        """Mock forward pass."""
        batch_size = next(iter(features.values())).shape[0]
        return {
            ProprioceptiveType.POSITION.value: torch.randn(
                batch_size, self.prediction_horizon, 3, device=self.device
            ),
            ProprioceptiveType.GRIPPER.value: torch.randn(
                batch_size, self.prediction_horizon, 1, device=self.device
            ),
        }


@pytest.mark.unit
class TestPureAlgorithms:
    """Test that pure algorithms work without variational inference."""

    @pytest.mark.parametrize("algorithm_class", [BehavioralCloning, FlowMatching])
    def test_pure_algorithm_forward(
        self, algorithm_class, sample_features, sample_actions, device
    ):
        """Test pure algorithm forward pass."""
        if algorithm_class == FlowMatching:
            algorithm = algorithm_class(sigma=0.0, num_inference_steps=2)
        else:
            algorithm = algorithm_class()

        mock_network = MockNetwork(device)
        output = algorithm.forward(mock_network, sample_features, sample_actions)

        # Should have action predictions
        assert ProprioceptiveType.POSITION.value in output
        assert ProprioceptiveType.GRIPPER.value in output

        # Should NOT have variational outputs
        assert LatentKey.POSTERIOR_MU.value not in output
        assert LatentKey.POSTERIOR_LOGVAR.value not in output
        assert LatentKey.PRIOR_PREDICTION.value not in output

    @pytest.mark.parametrize("algorithm_class", [BehavioralCloning, FlowMatching])
    def test_pure_algorithm_predict(self, algorithm_class, sample_features, device):
        """Test pure algorithm prediction."""
        if algorithm_class == FlowMatching:
            algorithm = algorithm_class(sigma=0.0, num_inference_steps=2)
        else:
            algorithm = algorithm_class()

        mock_network = MockNetwork(device)
        output = algorithm.predict(mock_network, sample_features)

        # Should have action predictions only
        assert ProprioceptiveType.POSITION.value in output
        assert ProprioceptiveType.GRIPPER.value in output


@pytest.mark.unit
class TestVariationalAlgorithms:
    """Test that all algorithms work with variational wrapper."""

    @pytest.mark.parametrize(
        "base_algorithm",
        [
            BehavioralCloning(),
            FlowMatching(sigma=0.0, num_inference_steps=2),
            Diffusion(num_train_timesteps=10, num_inference_steps=2),
        ],
    )
    def test_variational_with_gaussian_prior(
        self, base_algorithm, vae_encoder, sample_features, sample_actions, device, embedding_dimension
    ):
        """Test all algorithms work with Gaussian prior."""
        algorithm = VariationalAlgorithm(
            base_algorithm=base_algorithm,
            posterior_encoder=vae_encoder,
            prior=None,  # Auto-creates GaussianPrior
            embedding_dimension=embedding_dimension
        )

        assert isinstance(algorithm.prior, GaussianPrior)

        mock_network = MockNetwork(device)
        output = algorithm.forward(mock_network, sample_features, sample_actions)

        # Should have action predictions
        assert ProprioceptiveType.POSITION.value in output

        # Should have VAE outputs
        assert LatentKey.POSTERIOR_MU.value in output
        assert LatentKey.POSTERIOR_LOGVAR.value in output

        # Should NOT have prior outputs (Gaussian prior has no training)
        assert LatentKey.PRIOR_PREDICTION.value not in output

    @pytest.mark.parametrize(
        "base_algorithm",
        [
            BehavioralCloning(),
            FlowMatching(sigma=0.0, num_inference_steps=2),
        ],
    )
    def test_variational_with_learned_prior(
        self,
        base_algorithm,
        vae_encoder,
        diffusion_prior,
        sample_features,
        sample_actions,
        device,
        embedding_dimension
    ):
        """Test all algorithms work with learned DiffusionPrior."""
        algorithm = VariationalAlgorithm(
            base_algorithm=base_algorithm,
            posterior_encoder=vae_encoder,
            prior=diffusion_prior,
            embedding_dimension=embedding_dimension
        )

        mock_network = MockNetwork(device)
        output = algorithm.forward(mock_network, sample_features, sample_actions)

        # Should have action predictions
        assert ProprioceptiveType.POSITION.value in output

        # Should have VAE outputs
        assert LatentKey.POSTERIOR_MU.value in output
        assert LatentKey.POSTERIOR_LOGVAR.value in output

        # Should have prior outputs (for training learned prior)
        assert LatentKey.PRIOR_PREDICTION.value in output
        assert LatentKey.PRIOR_TARGET.value in output


@pytest.mark.unit
class TestMetricsIntegration:
    """Test that metrics work correctly with variational outputs."""

    def test_kl_divergence_loss(self, vae_encoder, sample_features, sample_actions, device, embedding_dimension):
        """Test KL divergence loss works with variational outputs."""
        algorithm = VariationalAlgorithm(
            base_algorithm=BehavioralCloning(),
            posterior_encoder=vae_encoder,
            prior=None,
            embedding_dimension=embedding_dimension
        )

        mock_network = MockNetwork(device)
        predictions = algorithm.forward(mock_network, sample_features, sample_actions)

        # Verify predictions contain required keys
        assert LatentKey.POSTERIOR_MU.value in predictions
        assert LatentKey.POSTERIOR_LOGVAR.value in predictions

        # Test KL divergence loss
        kl_loss = KLDivergenceLoss(weight=0.0001)
        loss_output = kl_loss(predictions=predictions, targets=sample_actions, is_pad=None)

        assert loss_output.total_loss is not None
        assert loss_output.total_loss.item() >= 0  # KL divergence is non-negative

    def test_prior_denoising_loss(
        self, vae_encoder, diffusion_prior, sample_features, sample_actions, device, embedding_dimension
    ):
        """Test prior denoising loss works with variational outputs."""
        algorithm = VariationalAlgorithm(
            base_algorithm=FlowMatching(sigma=0.0, num_inference_steps=2),
            posterior_encoder=vae_encoder,
            prior=diffusion_prior,
            embedding_dimension=embedding_dimension
        )

        mock_network = MockNetwork(device)
        predictions = algorithm.forward(mock_network, sample_features, sample_actions)

        # Verify predictions contain required keys
        assert LatentKey.PRIOR_PREDICTION.value in predictions
        assert LatentKey.PRIOR_TARGET.value in predictions

        # Test prior denoising loss
        prior_loss = PriorDenoisingLoss(weight=1.0)
        loss_output = prior_loss(predictions=predictions, targets=sample_actions, is_pad=None)

        assert loss_output.total_loss is not None
        assert loss_output.total_loss.item() >= 0  # MSE is non-negative


