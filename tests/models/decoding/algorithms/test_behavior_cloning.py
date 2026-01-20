"""Tests for BehavioralCloning algorithm (pure, without variational inference).

Note: For tests of BC with variational inference (latent variables), see test_variational.py
which tests VariationalAlgorithm(BehavioralCloning(), VAE).
"""

import pytest
import torch
from unittest.mock import MagicMock

from versatil.data.constants import (
    GRIPPER_ACTION_KEY,
    ORIENTATION_ACTION_KEY,
    POSITION_ACTION_KEY,
)
from versatil.models.decoding.constants import LATENT_KEY, LOGVAR_KEY, MU_KEY
from versatil.models.decoding.algorithm.behavior_cloning import BehavioralCloning


@pytest.fixture
def device():
    """Device for testing."""
    return "cpu"


@pytest.fixture
def batch_size():
    """Batch size for testing."""
    return 4


@pytest.fixture
def prediction_horizon():
    """Prediction horizon."""
    return 10


@pytest.fixture
def embedding_dimension():
    """Embedding dimension."""
    return 256


@pytest.fixture
def features_dict(batch_size, embedding_dimension, device):
    """Sample features dictionary from encoding pipeline."""
    return {
        "rgb_features": torch.randn(batch_size, embedding_dimension, 7, 7, device=device),
        "proprio_features": torch.randn(batch_size, 64, device=device),
    }


@pytest.fixture
def actions_dict(batch_size, prediction_horizon, device):
    """Sample action dictionary."""
    return {
        POSITION_ACTION_KEY: torch.randn(batch_size, prediction_horizon, 3, device=device),
        ORIENTATION_ACTION_KEY: torch.randn(batch_size, prediction_horizon, 4, device=device),
        GRIPPER_ACTION_KEY: torch.randint(0, 2, (batch_size, prediction_horizon, 1), device=device).float(),
    }


@pytest.fixture
def mock_decoder(batch_size, prediction_horizon):
    """Mock decoder network."""
    decoder = MagicMock()

    def decoder_forward(features, actions):
        batch_size = list(features.values())[0].shape[0]
        return {
            POSITION_ACTION_KEY: torch.randn(batch_size, prediction_horizon, 3),
            ORIENTATION_ACTION_KEY: torch.randn(batch_size, prediction_horizon, 4),
            GRIPPER_ACTION_KEY: torch.randint(0, 2, (batch_size, prediction_horizon, 1)).float(),
        }

    decoder.side_effect = decoder_forward
    return decoder


@pytest.mark.unit
class TestBehavioralCloningInitialization:
    """Test BehavioralCloning initialization."""

    def test_init_without_parameters(self):
        """Test initialization without parameters (pure BC)."""
        algorithm = BehavioralCloning()

        # Should not have latent_encoder anymore
        assert not hasattr(algorithm, "latent_encoder")


@pytest.mark.unit
class TestBehavioralCloningForward:
    """Test BehavioralCloning forward pass (training)."""

    def test_forward_pure_bc(self, mock_decoder, features_dict, actions_dict):
        """Test forward for pure BC (no latent variables)."""
        algorithm = BehavioralCloning()

        predictions = algorithm.forward(network=mock_decoder, features=features_dict, actions=actions_dict)

        # Check decoder was called
        mock_decoder.assert_called_once()

        # Check predictions
        assert POSITION_ACTION_KEY in predictions
        assert ORIENTATION_ACTION_KEY in predictions
        assert GRIPPER_ACTION_KEY in predictions

        # Should NOT have latent keys (pure BC)
        assert LATENT_KEY not in predictions
        assert MU_KEY not in predictions
        assert LOGVAR_KEY not in predictions

    def test_forward_passes_features_unchanged(self, mock_decoder, features_dict, actions_dict):
        """Test that forward passes features unchanged to decoder."""
        algorithm = BehavioralCloning()

        algorithm.forward(network=mock_decoder, features=features_dict, actions=actions_dict)

        # Check that decoder received original features unchanged
        call_kwargs = mock_decoder.call_args.kwargs
        passed_features = call_kwargs["features"]

        # Should be the same feature keys
        assert set(passed_features.keys()) == set(features_dict.keys())
        assert "rgb_features" in passed_features
        assert "proprio_features" in passed_features


@pytest.mark.unit
class TestBehavioralCloningPredict:
    """Test BehavioralCloning predict (inference)."""

    def test_predict_pure_bc(self, mock_decoder, features_dict):
        """Test predict for pure BC."""
        algorithm = BehavioralCloning()

        predictions = algorithm.predict(network=mock_decoder, features=features_dict)

        # Check decoder was called with actions=None
        mock_decoder.assert_called_once()
        call_args = mock_decoder.call_args
        assert call_args[1]["actions"] is None

        # Check predictions
        assert POSITION_ACTION_KEY in predictions
        assert ORIENTATION_ACTION_KEY in predictions
        assert GRIPPER_ACTION_KEY in predictions

    def test_predict_passes_features_unchanged(self, mock_decoder, features_dict):
        """Test that predict passes features unchanged to decoder."""
        algorithm = BehavioralCloning()

        algorithm.predict(network=mock_decoder, features=features_dict)

        # Check that decoder received original features
        call_args = mock_decoder.call_args
        passed_features = call_args[0][0]  # First positional argument

        # Should be the same feature keys
        assert set(passed_features.keys()) == set(features_dict.keys())
        assert "rgb_features" in passed_features
        assert "proprio_features" in passed_features


@pytest.mark.unit
class TestBehavioralCloningDeterminism:
    """Test BehavioralCloning determinism."""

    def test_determinism_with_seed(self, features_dict, actions_dict, prediction_horizon):
        """Test that BC is deterministic with same seed."""
        algorithm = BehavioralCloning()

        decoder = MagicMock()
        decoder.return_value = {
            POSITION_ACTION_KEY: torch.randn(4, prediction_horizon, 3),
        }

        # Run twice with same seed
        torch.manual_seed(42)
        pred1 = algorithm.forward(network=decoder, features=features_dict, actions=actions_dict)

        torch.manual_seed(42)
        pred2 = algorithm.forward(network=decoder, features=features_dict, actions=actions_dict)

        # Since BC is deterministic (no sampling), features should be identical
        call1_features = decoder.call_args_list[0].kwargs["features"]
        call2_features = decoder.call_args_list[1].kwargs["features"]

        for key in call1_features:
            assert torch.allclose(call1_features[key], call2_features[key])
