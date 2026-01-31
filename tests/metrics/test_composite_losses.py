"""Tests for composite loss classes."""

import pytest
import torch

from versatil.data.constants import (
    GripperType,
    ObsKey,
    ProprioceptiveType,
)
from versatil.metrics.composite import (
    ActionReconstructionLoss,
    PhaseActionLoss,
    CompositeLoss,
)
from versatil.metrics.components import RegressionLoss, GripperLoss
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


class TestActionReconstructionLoss:
    def test_basic_reconstruction_loss(self, device, batch_size, horizon, position_dim):
        loss_fn = ActionReconstructionLoss(
            action_keys=[ProprioceptiveType.POSITION.value],
            mse_weight=1.0,
            gripper_bce_weight=1.0,
            use_vae=False,
        )

        predictions = {
            ProprioceptiveType.POSITION.value: torch.randn(batch_size, horizon, position_dim, device=device),
            ProprioceptiveType.GRIPPER.value: torch.randn(batch_size, horizon, 1, device=device),
        }
        targets = {
            ProprioceptiveType.POSITION.value: torch.randn(batch_size, horizon, position_dim, device=device),
            ProprioceptiveType.GRIPPER.value: torch.randint(
                0, 2, (batch_size, horizon, 1), device=device
            ).float(),
        }

        loss_output = loss_fn(predictions, targets)

        assert loss_output.total_loss.item() >= 0
        assert len(loss_output.component_losses) > 0

    def test_with_vae_kl_divergence(self, device, batch_size, horizon, position_dim):
        loss_fn = ActionReconstructionLoss(
            action_keys=[ProprioceptiveType.POSITION.value],
            mse_weight=1.0,
            kl_weight=0.001,
            use_vae=True,
        )

        latent_dim = 32
        predictions = {
            ProprioceptiveType.POSITION.value: torch.randn(batch_size, horizon, position_dim, device=device),
            ProprioceptiveType.GRIPPER.value: torch.randn(batch_size, horizon, 1, device=device),
            "mu": torch.randn(batch_size, latent_dim, device=device),
            "logvar": torch.randn(batch_size, latent_dim, device=device),
        }
        targets = {
            ProprioceptiveType.POSITION.value: torch.randn(batch_size, horizon, position_dim, device=device),
            ProprioceptiveType.GRIPPER.value: torch.randint(
                0, 2, (batch_size, horizon, 1), device=device
            ).float(),
        }

        loss_output = loss_fn(predictions, targets)

        assert any(
            MetricKey.KL_DIVERGENCE.value in key
            for key in loss_output.component_losses.keys()
        )

    def test_with_trajectory_regularization(
        self, device, batch_size, horizon, position_dim
    ):
        loss_fn = ActionReconstructionLoss(
            action_keys=[ProprioceptiveType.POSITION.value],
            mse_weight=1.0,
            length_weight=0.1,
            smoothness_weight=0.01,
        )

        predictions = {
            ProprioceptiveType.POSITION.value: torch.randn(batch_size, horizon, position_dim, device=device),
            ProprioceptiveType.GRIPPER.value: torch.randn(batch_size, horizon, 1, device=device),
        }
        targets = {
            ProprioceptiveType.POSITION.value: torch.randn(batch_size, horizon, position_dim, device=device),
            ProprioceptiveType.GRIPPER.value: torch.randint(
                0, 2, (batch_size, horizon, 1), device=device
            ).float(),
        }

        loss_output = loss_fn(predictions, targets)

        assert any(
            MetricKey.LENGTH_LOSS.value in key
            for key in loss_output.component_losses.keys()
        )
        assert any(
            MetricKey.SMOOTHNESS_LOSS.value in key
            for key in loss_output.component_losses.keys()
        )


class TestPhaseActionLoss:
    def test_phase_action_loss_computation(
        self, device, batch_size, horizon, position_dim
    ):
        n_phases = 5
        loss_fn = PhaseActionLoss(
            action_keys=[ProprioceptiveType.POSITION.value],
            mse_weight=1.0,
            phase_ce_weight=1.0,
            use_vae=False,
        )

        predictions = {
            ProprioceptiveType.POSITION.value: torch.randn(batch_size, horizon, position_dim, device=device),
            ProprioceptiveType.GRIPPER.value: torch.randn(batch_size, horizon, 1, device=device),
            ObsKey.PHASE_LABEL.value: torch.randn(batch_size, horizon, n_phases, device=device),
        }
        targets = {
            ProprioceptiveType.POSITION.value: torch.randn(batch_size, horizon, position_dim, device=device),
            ProprioceptiveType.GRIPPER.value: torch.randint(
                0, 2, (batch_size, horizon, 1), device=device
            ).float(),
            ObsKey.PHASE_LABEL.value: torch.randint(
                0, n_phases, (batch_size, horizon), device=device
            ),
        }

        loss_output = loss_fn(predictions, targets)

        assert loss_output.total_loss.item() >= 0
        assert any(
            MetricKey.PHASE_CROSS_ENTROPY.value in key
            for key in loss_output.component_losses.keys()
        )


class TestCompositeLoss:
    def test_composite_loss_combination(self, device, batch_size, horizon, position_dim):
        loss_modules = {
            "regression": RegressionLoss(
                action_keys=[ProprioceptiveType.POSITION.value], mse_weight=1.0
            ),
            "gripper": GripperLoss(gripper_type=GripperType.BINARY.value),
        }
        weights = {"regression": 1.0, "gripper": 0.5}

        loss_fn = CompositeLoss(loss_modules=loss_modules, weights=weights)

        predictions = {
            ProprioceptiveType.POSITION.value: torch.randn(batch_size, horizon, position_dim, device=device),
            ProprioceptiveType.GRIPPER.value: torch.randn(batch_size, horizon, 1, device=device),
        }
        targets = {
            ProprioceptiveType.POSITION.value: torch.randn(batch_size, horizon, position_dim, device=device),
            ProprioceptiveType.GRIPPER.value: torch.randint(
                0, 2, (batch_size, horizon, 1), device=device
            ).float(),
        }

        loss_output = loss_fn(predictions, targets)

        assert loss_output.total_loss.item() >= 0
        assert len(loss_output.component_losses) >= 2
