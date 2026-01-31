"""Tests for add-on loss components with heavy dependencies.

This module tests OptimalTransportLoss which requires geomloss+pykeops.
The entire file is skipped if geomloss is not installed.

Note: These tests may be slow due to PyKeOps compilation and OT computation.
"""

import pytest

# Skip entire file if geomloss not installed
pytest.importorskip("geomloss")

import torch

from versatil.data.constants import ProprioceptiveType
from versatil.metrics.constants import MetricKey


# Lazy import to avoid compilation during test collection
@pytest.fixture(scope="module")
def OptimalTransportLoss():
    """Lazy import OptimalTransportLoss to avoid compilation during test collection."""
    # Import from the add-ons module
    import importlib
    add_ons = importlib.import_module("versatil.metrics.add-ons")
    return add_ons.OptimalTransportLoss


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


@pytest.fixture
def orientation_dim():
    return 4


@pytest.fixture
def state_dim():
    return 256


@pytest.mark.slow
class TestOptimalTransportLossInitialization:
    """Test OptimalTransportLoss initialization and configuration."""

    def test_initialization_single_action_key(self, OptimalTransportLoss):
        """Test initialization with single action key."""
        loss_fn = OptimalTransportLoss(
            action_keys=[ProprioceptiveType.POSITION.value],
            weight=0.1,
            epsilon=0.01,
            lambda_state=1.0,
        )

        assert loss_fn.weight == 0.1
        assert loss_fn.lambda_state == 1.0
        assert loss_fn.action_keys == [ProprioceptiveType.POSITION.value]
        assert hasattr(loss_fn, "ot")

    def test_initialization_multiple_action_keys(self, OptimalTransportLoss):
        """Test initialization with multiple action keys."""
        loss_fn = OptimalTransportLoss(
            action_keys=[ProprioceptiveType.POSITION.value],
            weight=0.2,
            epsilon=0.02,
            lambda_state=0.5,
        )

        assert loss_fn.action_keys == [ProprioceptiveType.POSITION.value]
        assert loss_fn.weight == 0.2
        assert loss_fn.lambda_state == 0.5

    def test_initialization_zero_lambda_state(self, OptimalTransportLoss):
        """Test initialization with lambda_state=0 (no state augmentation)."""
        loss_fn = OptimalTransportLoss(
            action_keys=[ProprioceptiveType.POSITION.value],
            lambda_state=0.0,
        )

        assert loss_fn.lambda_state == 0.0

    def test_get_required_keys(self, OptimalTransportLoss):
        """Test get_required_keys returns action keys."""
        loss_fn = OptimalTransportLoss(
            action_keys=[ProprioceptiveType.POSITION.value],
        )

        required_keys = loss_fn.get_required_keys()

        assert ProprioceptiveType.POSITION.value in required_keys
        assert ProprioceptiveType.ORIENTATION.value in required_keys
        assert len(required_keys) == 2


@pytest.mark.slow
class TestOptimalTransportLossForward:
    """Test OptimalTransportLoss forward pass computation."""

    def test_forward_single_action(self, OptimalTransportLoss, device, batch_size, horizon, position_dim):
        """Test forward pass with single action key."""
        loss_fn = OptimalTransportLoss(
            action_keys=[ProprioceptiveType.POSITION.value],
            weight=0.1,
            lambda_state=0.0,  # No state augmentation
        )

        predictions = {
            ProprioceptiveType.POSITION.value: torch.randn(batch_size, horizon, position_dim, device=device)
        }
        targets = {
            ProprioceptiveType.POSITION.value: torch.randn(batch_size, horizon, position_dim, device=device)
        }

        loss_output = loss_fn(predictions, targets)

        # Check output structure
        assert hasattr(loss_output, "total_loss")
        assert hasattr(loss_output, "component_losses")

        # Check values
        assert loss_output.total_loss.item() >= 0
        assert MetricKey.OPTIMAL_TRANSPORT_LOSS.value in loss_output.component_losses
        assert loss_output.component_losses[MetricKey.OPTIMAL_TRANSPORT_LOSS.value].item() >= 0

        # Check total_loss is weighted correctly
        ot_loss = loss_output.component_losses[MetricKey.OPTIMAL_TRANSPORT_LOSS.value]
        assert torch.isclose(loss_output.total_loss, 0.1 * ot_loss, rtol=1e-4)

    def test_forward_multiple_actions(self, OptimalTransportLoss, device, batch_size, horizon, position_dim, orientation_dim):
        """Test forward pass with multiple action keys."""
        loss_fn = OptimalTransportLoss(
            action_keys=[ProprioceptiveType.POSITION.value],
            weight=0.2,
            lambda_state=0.0,
        )

        predictions = {
            ProprioceptiveType.POSITION.value: torch.randn(batch_size, horizon, position_dim, device=device),
            ProprioceptiveType.ORIENTATION.value: torch.randn(batch_size, horizon, orientation_dim, device=device),
        }
        targets = {
            ProprioceptiveType.POSITION.value: torch.randn(batch_size, horizon, position_dim, device=device),
            ProprioceptiveType.ORIENTATION.value: torch.randn(batch_size, horizon, orientation_dim, device=device),
        }

        loss_output = loss_fn(predictions, targets)

        assert loss_output.total_loss.item() >= 0
        assert MetricKey.OPTIMAL_TRANSPORT_LOSS.value in loss_output.component_losses

    def test_forward_with_padding_mask(self, OptimalTransportLoss, device, batch_size, horizon, position_dim):
        """Test forward pass with padding mask."""
        loss_fn = OptimalTransportLoss(
            action_keys=[ProprioceptiveType.POSITION.value],
            weight=0.1,
            lambda_state=0.0,
        )

        predictions = {
            ProprioceptiveType.POSITION.value: torch.randn(batch_size, horizon, position_dim, device=device)
        }
        targets = {
            ProprioceptiveType.POSITION.value: torch.randn(batch_size, horizon, position_dim, device=device)
        }

        # Create padding mask: last 3 timesteps are padded
        is_pad = torch.zeros(batch_size, horizon, dtype=torch.bool, device=device)
        is_pad[:, -3:] = True

        loss_output = loss_fn(predictions, targets, is_pad=is_pad)

        assert loss_output.total_loss.item() >= 0
        assert MetricKey.OPTIMAL_TRANSPORT_LOSS.value in loss_output.component_losses

    def test_forward_identical_predictions_targets(self, OptimalTransportLoss, device, batch_size, horizon, position_dim):
        """Test that identical predictions and targets give near-zero loss."""
        loss_fn = OptimalTransportLoss(
            action_keys=[ProprioceptiveType.POSITION.value],
            weight=1.0,
            lambda_state=0.0,
        )

        actions = torch.randn(batch_size, horizon, position_dim, device=device)
        predictions = {ProprioceptiveType.POSITION.value: actions}
        targets = {ProprioceptiveType.POSITION.value: actions.clone()}

        loss_output = loss_fn(predictions, targets)

        # OT loss between identical distributions should be very small
        assert loss_output.total_loss.item() < 1e-4

    def test_forward_weight_scaling(self, OptimalTransportLoss, device, batch_size, horizon, position_dim):
        """Test that weight parameter correctly scales the total loss."""
        # Loss with weight=1.0
        loss_fn_1 = OptimalTransportLoss(
            action_keys=[ProprioceptiveType.POSITION.value],
            weight=1.0,
            lambda_state=0.0,
        )

        # Loss with weight=0.5
        loss_fn_05 = OptimalTransportLoss(
            action_keys=[ProprioceptiveType.POSITION.value],
            weight=0.5,
            lambda_state=0.0,
        )

        predictions = {
            ProprioceptiveType.POSITION.value: torch.randn(batch_size, horizon, position_dim, device=device)
        }
        targets = {
            ProprioceptiveType.POSITION.value: torch.randn(batch_size, horizon, position_dim, device=device)
        }

        loss_output_1 = loss_fn_1(predictions, targets)
        loss_output_05 = loss_fn_05(predictions, targets)

        # Component losses should be identical
        comp_1 = loss_output_1.component_losses[MetricKey.OPTIMAL_TRANSPORT_LOSS.value]
        comp_05 = loss_output_05.component_losses[MetricKey.OPTIMAL_TRANSPORT_LOSS.value]
        assert torch.isclose(comp_1, comp_05, rtol=1e-4)

        # Total loss should be scaled by weight
        assert torch.isclose(loss_output_05.total_loss, 0.5 * loss_output_1.total_loss, rtol=1e-4)


@pytest.mark.slow
class TestOptimalTransportLossErrorHandling:
    """Test error handling in OptimalTransportLoss."""

    def test_missing_action_key_in_predictions(self, OptimalTransportLoss, device, batch_size, horizon, position_dim):
        """Test ValueError when action key missing in predictions."""
        loss_fn = OptimalTransportLoss(
            action_keys=[ProprioceptiveType.POSITION.value],
        )

        # Missing ProprioceptiveType.ORIENTATION.value in predictions
        predictions = {
            ProprioceptiveType.POSITION.value: torch.randn(batch_size, horizon, position_dim, device=device)
        }
        targets = {
            ProprioceptiveType.POSITION.value: torch.randn(batch_size, horizon, position_dim, device=device),
            ProprioceptiveType.ORIENTATION.value: torch.randn(batch_size, horizon, 4, device=device),
        }

        with pytest.raises(ValueError, match="must contain key"):
            loss_fn(predictions, targets)

    def test_missing_action_key_in_targets(self, OptimalTransportLoss, device, batch_size, horizon, position_dim):
        """Test ValueError when action key missing in targets."""
        loss_fn = OptimalTransportLoss(
            action_keys=[ProprioceptiveType.POSITION.value],
        )

        # Missing ProprioceptiveType.ORIENTATION.value in targets
        predictions = {
            ProprioceptiveType.POSITION.value: torch.randn(batch_size, horizon, position_dim, device=device),
            ProprioceptiveType.ORIENTATION.value: torch.randn(batch_size, horizon, 4, device=device),
        }
        targets = {
            ProprioceptiveType.POSITION.value: torch.randn(batch_size, horizon, position_dim, device=device),
        }

        with pytest.raises(ValueError, match="must contain key"):
            loss_fn(predictions, targets)



@pytest.mark.slow
class TestOptimalTransportLossGradients:
    """Test gradient computation for OptimalTransportLoss."""

    def test_gradients_flow_through_loss(self, OptimalTransportLoss, device, batch_size, horizon, position_dim):
        """Test that gradients flow through the OT loss."""
        loss_fn = OptimalTransportLoss(
            action_keys=[ProprioceptiveType.POSITION.value],
            weight=1.0,
            lambda_state=0.0,
        )

        predictions = {
            ProprioceptiveType.POSITION.value: torch.randn(batch_size, horizon, position_dim, device=device, requires_grad=True)
        }
        targets = {
            ProprioceptiveType.POSITION.value: torch.randn(batch_size, horizon, position_dim, device=device)
        }

        loss_output = loss_fn(predictions, targets)
        loss_output.total_loss.backward()

        # Check gradients exist and are non-zero
        assert predictions[ProprioceptiveType.POSITION.value].grad is not None
        assert not torch.allclose(predictions[ProprioceptiveType.POSITION.value].grad, torch.zeros_like(predictions[ProprioceptiveType.POSITION.value].grad))

