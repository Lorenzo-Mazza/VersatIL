"""Tests for individual loss components."""

import pytest
import torch
import math

from versatil.data.constants import (
    POSITION_ACTION_KEY,
    GRIPPER_ACTION_KEY,
    PHASE_LABEL_KEY,
    GripperType, TOKENIZED_ACTIONS_KEY, IS_PAD_ACTION_KEY,
)
from versatil.models.decoding.constants import (
    PRIOR_PREDICTION_KEY,
    PRIOR_TARGET_KEY,
    PREDICTED_ACTION_TOKENS_KEY, ACTION_LOGITS_KEY,
)
from versatil.metrics.components import (
    RegressionLoss,
    GripperLoss,
    KLDivergenceLoss,
    TrajectoryLengthLoss,
    PhaseClassificationLoss,
    PriorDenoisingLoss,
    ActionTokenLoss,
)
from versatil.metrics.constants import MetricKey


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def horizon():
    return 10


@pytest.fixture
def position_dim():
    return 3


class TestRegressionLoss:
    def test_mse_loss_computation(self, device, batch_size, horizon, position_dim):
        loss_fn = RegressionLoss(
            action_keys=[POSITION_ACTION_KEY], mse_weight=1.0, l1_weight=0.0
        )

        predictions = {
            POSITION_ACTION_KEY: torch.randn(batch_size, horizon, position_dim, device=device)
        }
        targets = {
            POSITION_ACTION_KEY: torch.randn(batch_size, horizon, position_dim, device=device)
        }

        loss_output = loss_fn(predictions, targets)

        assert loss_output.total_loss.item() >= 0
        assert f"{POSITION_ACTION_KEY}_{MetricKey.MSE_LOSS.value}" in loss_output.component_losses

    def test_l1_loss_computation(self, device, batch_size, horizon, position_dim):
        loss_fn = RegressionLoss(
            action_keys=[POSITION_ACTION_KEY], mse_weight=0.0, l1_weight=1.0
        )

        predictions = {
            POSITION_ACTION_KEY: torch.randn(batch_size, horizon, position_dim, device=device)
        }
        targets = {
            POSITION_ACTION_KEY: torch.randn(batch_size, horizon, position_dim, device=device)
        }

        loss_output = loss_fn(predictions, targets)

        assert loss_output.total_loss.item() >= 0
        assert f"{POSITION_ACTION_KEY}_{MetricKey.L1_LOSS.value}" in loss_output.component_losses

    def test_padding_mask_computation(self, device, batch_size, horizon, position_dim):
        loss_fn = RegressionLoss(action_keys=[POSITION_ACTION_KEY], mse_weight=1.0)

        predictions = {
            POSITION_ACTION_KEY: torch.ones(batch_size, horizon, position_dim, device=device)
        }
        targets = {
            POSITION_ACTION_KEY: torch.zeros(batch_size, horizon, position_dim, device=device)
        }

        loss_no_pad = loss_fn(predictions, targets, is_pad=None)

        is_pad = torch.zeros(batch_size, horizon, dtype=torch.bool, device=device)
        is_pad[:, horizon // 2 :] = True
        loss_with_pad = loss_fn(predictions, targets, is_pad=is_pad)

        assert loss_no_pad.total_loss.item() == pytest.approx(1.0, abs=1e-5)
        assert loss_with_pad.total_loss.item() > 0


class TestGripperLoss:
    def test_binary_gripper_loss(self, device, batch_size, horizon):
        loss_fn = GripperLoss(gripper_type=GripperType.BINARY.value, bce_weight=1.0)

        predictions = {GRIPPER_ACTION_KEY: torch.randn(batch_size, horizon, 1, device=device)}
        targets = {
            GRIPPER_ACTION_KEY: torch.randint(0, 2, (batch_size, horizon, 1), device=device).float()
        }

        loss_output = loss_fn(predictions, targets)

        assert loss_output.total_loss.item() >= 0
        assert MetricKey.GRIPPER_BCE.value in loss_output.component_losses

    def test_continuous_gripper_loss(self, device, batch_size, horizon):
        loss_fn = GripperLoss(gripper_type=GripperType.CONTINUOUS.value, mse_weight=1.0)

        predictions = {GRIPPER_ACTION_KEY: torch.randn(batch_size, horizon, 1, device=device)}
        targets = {GRIPPER_ACTION_KEY: torch.randn(batch_size, horizon, 1, device=device)}

        loss_output = loss_fn(predictions, targets)

        assert loss_output.total_loss.item() >= 0
        assert MetricKey.GRIPPER_MSE.value in loss_output.component_losses


class TestKLDivergenceLoss:
    def test_kl_divergence_computation(self, device, batch_size):
        loss_fn = KLDivergenceLoss(weight=0.001)

        latent_dim = 32
        predictions = {
            "mu": torch.randn(batch_size, latent_dim, device=device),
            "logvar": torch.randn(batch_size, latent_dim, device=device),
        }
        targets = {}

        loss_output = loss_fn(predictions, targets)

        assert loss_output.total_loss.item() >= 0
        assert MetricKey.KL_DIVERGENCE.value in loss_output.component_losses


class TestTrajectoryLengthLoss:
    def test_length_loss_computation(self, device, batch_size, horizon, position_dim):
        loss_fn = TrajectoryLengthLoss(weight=0.1, action_key=POSITION_ACTION_KEY)

        predictions = {
            POSITION_ACTION_KEY: torch.randn(batch_size, horizon, position_dim, device=device)
        }
        targets = {
            POSITION_ACTION_KEY: torch.randn(batch_size, horizon, position_dim, device=device)
        }

        loss_output = loss_fn(predictions, targets)

        assert loss_output.total_loss.item() >= 0
        assert MetricKey.LENGTH_LOSS.value in loss_output.component_losses


class TestPhaseClassificationLoss:
    def test_phase_cross_entropy(self, device, batch_size, horizon):
        n_phases = 5
        loss_fn = PhaseClassificationLoss(cross_entropy_weight=1.0, entropy_weight=0.0)

        predictions = {PHASE_LABEL_KEY: torch.randn(batch_size, horizon, n_phases, device=device)}
        targets = {
            PHASE_LABEL_KEY: torch.randint(0, n_phases, (batch_size, horizon), device=device)
        }

        loss_output = loss_fn(predictions, targets)

        assert loss_output.total_loss.item() >= 0
        assert MetricKey.PHASE_CROSS_ENTROPY.value in loss_output.component_losses
        assert "phase_logits" in loss_output.metadata
        assert "phase_labels" in loss_output.metadata

    def test_phase_entropy_regularization(self, device, batch_size, horizon):
        n_phases = 5
        loss_fn = PhaseClassificationLoss(cross_entropy_weight=1.0, entropy_weight=0.1)

        predictions = {PHASE_LABEL_KEY: torch.randn(batch_size, horizon, n_phases, device=device)}
        targets = {
            PHASE_LABEL_KEY: torch.randint(0, n_phases, (batch_size, horizon), device=device)
        }

        loss_output = loss_fn(predictions, targets)

        assert MetricKey.PHASE_ENTROPY.value in loss_output.component_losses


class TestPriorDenoisingLoss:
    def test_prior_denoising_loss_computation(self, device, batch_size):
        """Test that PriorDenoisingLoss computes MSE correctly."""
        latent_dim = 32
        loss_fn = PriorDenoisingLoss(weight=1.0)

        predictions = {
            PRIOR_PREDICTION_KEY: torch.randn(batch_size, latent_dim, device=device),
            PRIOR_TARGET_KEY: torch.randn(batch_size, latent_dim, device=device),
        }
        targets = {}

        loss_output = loss_fn(predictions, targets)

        assert loss_output.total_loss.item() >= 0
        assert MetricKey.PRIOR_DENOISING_LOSS.value in loss_output.component_losses

    def test_prior_denoising_loss_weight_scaling(self, device, batch_size):
        """Test that weight parameter scales the loss correctly."""
        latent_dim = 32
        weight = 0.5

        loss_fn = PriorDenoisingLoss(weight=weight)

        predictions = {
            PRIOR_PREDICTION_KEY: torch.randn(batch_size, latent_dim, device=device),
            PRIOR_TARGET_KEY: torch.randn(batch_size, latent_dim, device=device),
        }
        targets = {}

        loss_output = loss_fn(predictions, targets)

        # Component loss should be unscaled, total loss should be scaled
        component_loss = loss_output.component_losses[MetricKey.PRIOR_DENOISING_LOSS.value]
        assert loss_output.total_loss.item() == pytest.approx(
            weight * component_loss.item(), abs=1e-5
        )

    def test_prior_denoising_loss_missing_prediction_key(self, device, batch_size):
        """Test that error is raised when PRIOR_PREDICTION_KEY is missing."""
        latent_dim = 32
        loss_fn = PriorDenoisingLoss(weight=1.0)

        predictions = {
            PRIOR_TARGET_KEY: torch.randn(batch_size, latent_dim, device=device),
        }
        targets = {}

        with pytest.raises(ValueError, match="must contain 'prior_prediction'"):
            loss_fn(predictions, targets)

    def test_prior_denoising_loss_missing_target_key(self, device, batch_size):
        """Test that error is raised when PRIOR_TARGET_KEY is missing."""
        latent_dim = 32
        loss_fn = PriorDenoisingLoss(weight=1.0)

        predictions = {
            PRIOR_PREDICTION_KEY: torch.randn(batch_size, latent_dim, device=device),
        }
        targets = {}

        with pytest.raises(ValueError, match="must contain 'prior_target'"):
            loss_fn(predictions, targets)

    def test_prior_denoising_loss_zero_when_identical(self, device, batch_size):
        """Test that loss is zero when prediction equals target."""
        latent_dim = 32
        loss_fn = PriorDenoisingLoss(weight=1.0)

        latent_values = torch.randn(batch_size, latent_dim, device=device)
        predictions = {
            PRIOR_PREDICTION_KEY: latent_values.clone(),
            PRIOR_TARGET_KEY: latent_values.clone(),
        }
        targets = {}

        loss_output = loss_fn(predictions, targets)

        assert loss_output.total_loss.item() == pytest.approx(0.0, abs=1e-6)
        assert loss_output.component_losses[MetricKey.PRIOR_DENOISING_LOSS.value].item() == pytest.approx(
            0.0, abs=1e-6
        )


class TestActionTokenLoss:
    def test_reads_targets_from_predictions(self, device, batch_size, horizon):
        """Test that ActionTokenLoss reads ground truth tokens from predictions dict."""
        vocab_size = 1024
        loss_fn = ActionTokenLoss()

        predictions = {
            ACTION_LOGITS_KEY: torch.randn(batch_size, horizon, vocab_size, device=device),
            TOKENIZED_ACTIONS_KEY: torch.randint(0, vocab_size, (batch_size, horizon), device=device),
        }
        targets = {}

        loss_output = loss_fn(predictions, targets, is_pad=None)

        assert loss_output.total_loss.item() >= 0
        assert MetricKey.ACTION_TOKEN_CROSS_ENTROPY.value in loss_output.component_losses

    def test_reads_padding_from_predictions(self, device, batch_size, horizon):
        """Test that ActionTokenLoss uses is_pad from predictions dict."""
        vocab_size = 1024
        loss_fn = ActionTokenLoss()
        target_tokens = torch.randint(0, vocab_size, (batch_size, horizon), device=device)
        is_pad = torch.zeros(batch_size, horizon, dtype=torch.bool, device=device)
        is_pad[:, :] = True

        predictions = {
            ACTION_LOGITS_KEY: torch.randn(batch_size, horizon, vocab_size, device=device),
            TOKENIZED_ACTIONS_KEY: target_tokens,
            IS_PAD_ACTION_KEY: is_pad,
        }
        targets = {}

        loss_output = loss_fn(predictions, targets, is_pad=None)
        assert math.isnan(loss_output.total_loss.item())

    def test_ignores_targets_parameter(self, device, batch_size, horizon):
        """Test that ActionTokenLoss ignores the targets parameter."""
        vocab_size = 1024
        loss_fn = ActionTokenLoss()
        predictions = {
            ACTION_LOGITS_KEY: torch.randn(batch_size, horizon, vocab_size, device=device),
            TOKENIZED_ACTIONS_KEY: torch.randint(0, vocab_size, (batch_size, horizon), device=device),
        }
        dummy_targets = {
            TOKENIZED_ACTIONS_KEY: torch.randint(0, vocab_size, (batch_size, horizon), device=device)
        }

        loss_output = loss_fn(predictions, dummy_targets, is_pad=None)

        assert loss_output.total_loss.item() >= 0

    def test_ignores_is_pad_parameter(self, device, batch_size, horizon):
        """Test that ActionTokenLoss ignores the is_pad parameter."""
        vocab_size = 1024
        loss_fn = ActionTokenLoss(ignore_index=-100)

        is_pad_in_predictions = torch.zeros(batch_size, horizon, dtype=torch.bool, device=device)
        is_pad_in_predictions[:, horizon // 2:] = True

        predictions = {
            PREDICTED_ACTION_TOKENS_KEY: torch.randn(batch_size, horizon, vocab_size, device=device),
            f"{PREDICTED_ACTION_TOKENS_KEY}_target": torch.randint(0, vocab_size, (batch_size, horizon), device=device),
            "is_pad": is_pad_in_predictions,
        }

        dummy_is_pad = torch.ones(batch_size, horizon, dtype=torch.bool, device=device)

        loss_output = loss_fn(predictions, {}, is_pad=dummy_is_pad)

        assert loss_output.total_loss.item() >= 0

    def test_missing_logits_key_raises_error(self, device, batch_size, horizon):
        """Test that error is raised when logits key is missing."""
        vocab_size = 1024
        loss_fn = ActionTokenLoss(ignore_index=-100)

        predictions = {
            f"{PREDICTED_ACTION_TOKENS_KEY}_target": torch.randint(0, vocab_size, (batch_size, horizon), device=device),
        }
        targets = {}

        with pytest.raises(ValueError, match="Predictions must contain keys"):
            loss_fn(predictions, targets)

    def test_missing_target_key_raises_error(self, device, batch_size, horizon):
        """Test that error is raised when target key is missing."""
        vocab_size = 1024
        loss_fn = ActionTokenLoss(ignore_index=-100)

        predictions = {
            PREDICTED_ACTION_TOKENS_KEY: torch.randn(batch_size, horizon, vocab_size, device=device),
        }
        targets = {}

        with pytest.raises(ValueError, match="Predictions must contain keys"):
            loss_fn(predictions, targets)

    def test_padding_reduces_loss(self, device, batch_size, horizon):
        """Test that padding mask correctly reduces loss."""
        vocab_size = 1024
        loss_fn = ActionTokenLoss(ignore_index=-100)

        logits = torch.randn(batch_size, horizon, vocab_size, device=device)
        target_tokens = torch.randint(0, vocab_size, (batch_size, horizon), device=device)

        predictions_no_pad = {
            PREDICTED_ACTION_TOKENS_KEY: logits.clone(),
            f"{PREDICTED_ACTION_TOKENS_KEY}_target": target_tokens.clone(),
        }
        loss_no_pad = loss_fn(predictions_no_pad, {}, is_pad=None)

        is_pad = torch.zeros(batch_size, horizon, dtype=torch.bool, device=device)
        is_pad[:, horizon // 2:] = True

        predictions_with_pad = {
            PREDICTED_ACTION_TOKENS_KEY: logits.clone(),
            f"{PREDICTED_ACTION_TOKENS_KEY}_target": target_tokens.clone(),
            "is_pad": is_pad,
        }
        loss_with_pad = loss_fn(predictions_with_pad, {}, is_pad=None)

        assert loss_with_pad.total_loss.item() < loss_no_pad.total_loss.item()

    def test_get_required_keys_returns_action_tokens(self):
        """Test that get_required_keys returns PREDICTED_ACTION_TOKENS_KEY."""
        loss_fn = ActionTokenLoss()

        required_keys = loss_fn.get_required_keys()

        assert required_keys == {ACTION_LOGITS_KEY}
