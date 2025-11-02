"""Tests for VAETransformerEncoder (transformer-based VAE latent action encoder)."""

import pytest
import torch

from refactoring.models.decoding.constants import LATENT_KEY, LOGVAR_KEY, MU_KEY
from refactoring.models.decoding.latent import VAETransformerEncoder


@pytest.fixture
def device():
    """Device for testing."""
    return "cpu"


@pytest.fixture
def batch_size():
    """Batch size for testing."""
    return 4


@pytest.fixture
def embedding_dimension():
    """Embedding dimension for VAE."""
    return 256


@pytest.fixture
def vae_latent_dimension():
    """VAE latent space dimension."""
    return 32


@pytest.fixture
def prediction_horizon():
    """Number of action timesteps."""
    return 10


@pytest.fixture
def actions_dict(batch_size, prediction_horizon, device):
    """Sample action dictionary."""
    return {
        "position_action": torch.randn(batch_size, prediction_horizon, 3, device=device),
        "orientation_action": torch.randn(batch_size, prediction_horizon, 4, device=device),
        "gripper_action": torch.randint(0, 2, (batch_size, prediction_horizon, 1), device=device).float(),
    }


@pytest.fixture
def observations_dict(batch_size, embedding_dimension, device):
    """Sample observation dictionary with flat features."""
    return {
        "proprio_features": torch.randn(batch_size, 64, device=device),
        "language_features": torch.randn(batch_size, 128, device=device),
    }


@pytest.mark.unit
class TestVAETransformerEncoderInitialization:
    """Test VAETransformerEncoder initialization."""

    def test_init_basic(self, embedding_dimension, vae_latent_dimension, prediction_horizon, device):
        """Test basic initialization."""
        encoder = VAETransformerEncoder(
            embedding_dimension=embedding_dimension,
            latent_dim=vae_latent_dimension,
            prediction_horizon=prediction_horizon,
            device=device,
        )

        assert encoder.embedding_dimension == embedding_dimension
        assert encoder.latent_dim == vae_latent_dimension
        assert encoder.prediction_horizon == prediction_horizon
        assert encoder.use_proprioceptive is False
        assert hasattr(encoder, "vae")

    def test_init_with_proprioceptive(self, embedding_dimension, vae_latent_dimension, prediction_horizon, device):
        """Test initialization with proprioceptive conditioning."""
        encoder = VAETransformerEncoder(
            embedding_dimension=embedding_dimension,
            latent_dim=vae_latent_dimension,
            prediction_horizon=prediction_horizon,
            device=device,
            use_proprioceptive=True,
        )

        assert encoder.use_proprioceptive is True

    def test_init_custom_params(self, embedding_dimension, vae_latent_dimension, prediction_horizon, device):
        """Test initialization with custom transformer parameters."""
        encoder = VAETransformerEncoder(
            embedding_dimension=embedding_dimension,
            latent_dim=vae_latent_dimension,
            prediction_horizon=prediction_horizon,
            device=device,
            number_of_heads=16,
            feedforward_dimension=1024,
            number_of_encoder_layers=6,
            dropout_rate=0.2,
            normalize_before=True,
        )

        assert encoder.vae.number_of_heads == 16
        assert encoder.vae.feedforward_dimension == 1024
        assert encoder.vae.number_of_encoder_layers == 6
        assert encoder.vae.dropout_rate == 0.2
        assert encoder.vae.normalize_before is True


@pytest.mark.unit
class TestVAETransformerEncoderEncode:
    """Test VAETransformerEncoder encoding functionality."""

    def test_encode_actions_only(
        self, embedding_dimension, vae_latent_dimension, prediction_horizon, device, batch_size, actions_dict
    ):
        """Test encoding actions without observations."""
        encoder = VAETransformerEncoder(
            embedding_dimension=embedding_dimension,
            latent_dim=vae_latent_dimension,
            prediction_horizon=prediction_horizon,
            device=device,
            use_proprioceptive=False,
        )

        output = encoder.encode(actions=actions_dict, observations=None)

        # Check all required keys are present
        assert LATENT_KEY in output
        assert MU_KEY in output
        assert LOGVAR_KEY in output

        # Check shapes
        assert output[LATENT_KEY].shape == (batch_size, embedding_dimension)
        assert output[MU_KEY].shape == (batch_size, vae_latent_dimension)
        assert output[LOGVAR_KEY].shape == (batch_size, vae_latent_dimension)

        # Check values are valid
        assert not torch.isnan(output[LATENT_KEY]).any()
        assert not torch.isnan(output[MU_KEY]).any()
        assert not torch.isnan(output[LOGVAR_KEY]).any()

    def test_encode_with_observations(
        self,
        embedding_dimension,
        vae_latent_dimension,
        prediction_horizon,
        device,
        batch_size,
        actions_dict,
        observations_dict,
    ):
        """Test encoding actions with observation conditioning."""
        encoder = VAETransformerEncoder(
            embedding_dimension=embedding_dimension,
            latent_dim=vae_latent_dimension,
            prediction_horizon=prediction_horizon,
            device=device,
            use_proprioceptive=True,
        )

        output = encoder.encode(actions=actions_dict, observations=observations_dict)

        # Check all required keys are present
        assert LATENT_KEY in output
        assert MU_KEY in output
        assert LOGVAR_KEY in output

        # Check shapes
        assert output[LATENT_KEY].shape == (batch_size, embedding_dimension)
        assert output[MU_KEY].shape == (batch_size, vae_latent_dimension)
        assert output[LOGVAR_KEY].shape == (batch_size, vae_latent_dimension)

    def test_encode_different_batch_sizes(
        self, embedding_dimension, vae_latent_dimension, prediction_horizon, device
    ):
        """Test encoding with different batch sizes."""
        encoder = VAETransformerEncoder(
            embedding_dimension=embedding_dimension,
            latent_dim=vae_latent_dimension,
            prediction_horizon=prediction_horizon,
            device=device,
        )

        for batch_size in [1, 4, 8, 16]:
            actions = {
                "position_action": torch.randn(batch_size, prediction_horizon, 3, device=device),
            }
            output = encoder.encode(actions=actions, observations=None)

            assert output[LATENT_KEY].shape[0] == batch_size
            assert output[MU_KEY].shape[0] == batch_size
            assert output[LOGVAR_KEY].shape[0] == batch_size


@pytest.mark.unit
class TestVAETransformerEncoderForward:
    """Test VAETransformerEncoder forward method (convenience wrapper)."""

    def test_forward_with_actions(
        self, embedding_dimension, vae_latent_dimension, prediction_horizon, device, batch_size, actions_dict
    ):
        """Test forward with actions (training mode)."""
        encoder = VAETransformerEncoder(
            embedding_dimension=embedding_dimension,
            latent_dim=vae_latent_dimension,
            prediction_horizon=prediction_horizon,
            device=device,
        )

        output = encoder.forward(actions=actions_dict, observations=None)

        # Should call encode() internally
        assert LATENT_KEY in output
        assert MU_KEY in output
        assert LOGVAR_KEY in output
        assert output[LATENT_KEY].shape == (batch_size, embedding_dimension)


@pytest.mark.unit
class TestVAETransformerEncoderGradients:
    """Test gradient flow through VAETransformerEncoder."""

    def test_gradients_flow_through_encode(
        self, embedding_dimension, vae_latent_dimension, prediction_horizon, device, batch_size, actions_dict
    ):
        """Test that gradients flow through encoding."""
        encoder = VAETransformerEncoder(
            embedding_dimension=embedding_dimension,
            latent_dim=vae_latent_dimension,
            prediction_horizon=prediction_horizon,
            device=device,
        )

        # Make actions require grad
        actions_grad = {k: v.clone().requires_grad_(True) for k, v in actions_dict.items()}

        output = encoder.encode(actions=actions_grad, observations=None)

        # Compute loss and backprop
        loss = output[LATENT_KEY].sum() + output[MU_KEY].sum() + output[LOGVAR_KEY].sum()
        loss.backward()

        # Check gradients exist
        for action_tensor in actions_grad.values():
            assert action_tensor.grad is not None
            assert not torch.isnan(action_tensor.grad).any()
