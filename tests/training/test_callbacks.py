"""Tests for versatil.training.callbacks module."""

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from versatil.metrics.constants import MetadataKey
from versatil.training.callbacks import (
    ConfusionMatrixCallback,
    EMACallback,
    ExpertUsageCallback,
    GradientNormCallback,
    LatentVisualizationCallback,
    ReduceLROnPlateauCallback,
    ResumableEarlyStopping,
    _figure_to_wandb_image,
)


@pytest.fixture
def ema_callback_factory() -> Callable[..., EMACallback]:
    def factory(
        power: float = 0.75,
        update_after_step: int = 0,
        inv_gamma: float = 1.0,
        min_value: float = 0.0,
        max_value: float = 0.9999,
    ) -> EMACallback:
        return EMACallback(
            power=power,
            update_after_step=update_after_step,
            inv_gamma=inv_gamma,
            min_value=min_value,
            max_value=max_value,
        )

    return factory


@pytest.fixture
def simple_module_factory(rng: np.random.Generator) -> Callable[..., torch.nn.Module]:
    def factory(
        input_dimension: int = 4,
        output_dimension: int = 4,
    ) -> torch.nn.Module:
        module = torch.nn.Linear(input_dimension, output_dimension)
        # Deterministic initialization for reproducibility
        weight_data = torch.from_numpy(
            rng.standard_normal((output_dimension, input_dimension)).astype(np.float32)
        )
        bias_data = torch.from_numpy(
            rng.standard_normal((output_dimension,)).astype(np.float32)
        )
        module.weight.data.copy_(weight_data)
        module.bias.data.copy_(bias_data)
        return module

    return factory


@pytest.fixture
def pl_module_with_policy_factory(
    simple_module_factory: Callable,
) -> Callable[..., MagicMock]:
    def factory(
        policy: torch.nn.Module | None = None,
    ) -> MagicMock:
        if policy is None:
            policy = simple_module_factory()
        pl_module = MagicMock()
        pl_module.policy = policy
        pl_module.parameters.return_value = policy.parameters()
        pl_module.log = MagicMock()
        pl_module.log_dict = MagicMock()
        return pl_module

    return factory


@pytest.fixture
def mock_pl_module_factory() -> Callable[..., MagicMock]:
    """Factory for a vanilla ``pl_module`` MagicMock with configurable metric
    return values for train/val hook tests. Use when the test does not need
    a real ``policy`` submodule — for policy-backed tests use
    ``pl_module_with_policy_factory`` instead."""

    def factory(
        train_metrics_return: dict | np.ndarray | None = None,
        val_metrics_return: dict | np.ndarray | None = None,
        train_method_name: str = "compute_expert_usage",
        val_method_name: str = "compute_expert_usage",
    ) -> MagicMock:
        pl_module = MagicMock()
        getattr(
            pl_module.train_metrics, train_method_name
        ).return_value = train_metrics_return
        getattr(
            pl_module.val_metrics, val_method_name
        ).return_value = val_metrics_return
        return pl_module

    return factory


@pytest.fixture
def latent_data_factory(rng: np.random.Generator) -> Callable[..., np.ndarray]:
    def factory(
        num_samples: int = 12,
        latent_dimension: int = 4,
    ) -> np.ndarray:
        return rng.standard_normal((num_samples, latent_dimension)).astype(np.float32)

    return factory


@pytest.fixture
def phase_array_factory(rng: np.random.Generator) -> Callable[..., np.ndarray]:
    def factory(
        num_samples: int = 12,
        num_phases: int = 3,
    ) -> np.ndarray:
        return rng.integers(0, num_phases, size=num_samples).astype(np.int64)

    return factory


@pytest.fixture
def mock_latent_pl_module_factory(
    latent_data_factory: Callable[..., np.ndarray],
    phase_array_factory: Callable[..., np.ndarray],
) -> Callable[..., MagicMock]:
    """Factory for a ``pl_module`` MagicMock pre-wired for
    ``LatentVisualizationCallback.on_validation_epoch_end`` tests. Override
    ``posterior_latent`` / ``prior_latent`` with ``None`` to exercise the
    missing-branch paths."""

    def factory(
        posterior_latent: np.ndarray | None = ...,
        prior_latent: np.ndarray | None = ...,
        phases: np.ndarray | None = ...,
        metadata: dict | None = None,
        latent_dimension: int = 4,
    ) -> MagicMock:
        if posterior_latent is ...:
            posterior_latent = latent_data_factory(latent_dimension=latent_dimension)
        if prior_latent is ...:
            prior_latent = latent_data_factory(latent_dimension=latent_dimension)
        if phases is ...:
            phases = phase_array_factory()
        pl_module = MagicMock()
        pl_module.val_metrics.compute_latent_visualization_data.return_value = (
            posterior_latent,
            prior_latent,
            phases,
        )
        pl_module.val_metrics.metadata = metadata if metadata is not None else {}
        return pl_module

    return factory


@pytest.mark.unit
class TestResumableEarlyStopping:
    def test_load_state_dict_is_noop(self):
        callback = ResumableEarlyStopping(monitor="val_loss")

        # Should not raise, and should ignore the state dict completely
        callback.load_state_dict({"wait_count": 5, "best_score": 0.1})

        # The internal state should remain at initial values, not loaded values
        assert callback.wait_count == 0


@pytest.mark.unit
class TestEMACallbackInitialization:
    @pytest.mark.parametrize("power", [0.5, 0.999])
    @pytest.mark.parametrize("update_after_step", [0, 100])
    @pytest.mark.parametrize("max_value", [0.999, 0.9999])
    def test_stores_configuration(
        self,
        ema_callback_factory: Callable,
        power: float,
        update_after_step: int,
        max_value: float,
    ):
        callback = ema_callback_factory(
            power=power,
            update_after_step=update_after_step,
            max_value=max_value,
        )

        assert callback.power == power
        assert callback.update_after_step == update_after_step
        assert callback.max_value == max_value

    def test_ema_model_starts_as_none(
        self,
        ema_callback_factory: Callable,
    ):
        callback = ema_callback_factory()

        assert callback.ema_model is None
        assert callback.decay == 0.0


@pytest.mark.unit
class TestEMACallbackDecayComputation:
    def test_decay_is_zero_before_update_after_step(
        self,
        ema_callback_factory: Callable,
    ):
        callback = ema_callback_factory(update_after_step=100)

        # Steps 0 through 100 should return 0.0
        assert callback._get_decay(global_step=0) == 0.0
        assert callback._get_decay(global_step=50) == 0.0
        assert callback._get_decay(global_step=100) == 0.0

    def test_decay_increases_monotonically_with_steps(
        self,
        ema_callback_factory: Callable,
    ):
        callback = ema_callback_factory(power=0.75, update_after_step=0, inv_gamma=1.0)

        decay_10 = callback._get_decay(global_step=10)
        decay_100 = callback._get_decay(global_step=100)
        decay_1000 = callback._get_decay(global_step=1000)

        assert 0.0 < decay_10 < decay_100 < decay_1000

    def test_decay_clamped_to_max_value(
        self,
        ema_callback_factory: Callable,
    ):
        max_value = 0.99
        callback = ema_callback_factory(max_value=max_value)

        # Very large step should be clamped
        decay = callback._get_decay(global_step=1_000_000)

        assert decay <= max_value

    def test_decay_clamped_to_min_value(
        self,
        ema_callback_factory: Callable,
    ):
        min_value = 0.5
        callback = ema_callback_factory(min_value=min_value, update_after_step=0)

        # Early step where raw value is below min_value
        decay = callback._get_decay(global_step=2)

        assert decay >= min_value


@pytest.mark.unit
class TestEMACallbackOnFitStart:
    def test_creates_ema_model_as_deep_copy_of_policy(
        self,
        ema_callback_factory: Callable,
        pl_module_with_policy_factory: Callable,
        mock_trainer_factory: Callable,
        simple_module_factory: Callable,
    ):
        policy = simple_module_factory()
        pl_module = pl_module_with_policy_factory(policy=policy)
        callback = ema_callback_factory()

        callback.on_fit_start(
            trainer=mock_trainer_factory(),
            pl_module=pl_module,
        )

        assert callback.ema_model is not None
        # Verify it's a separate copy by checking weight values match
        for ema_param, policy_param in zip(
            callback.ema_model.parameters(), policy.parameters()
        ):
            assert torch.equal(ema_param.data, policy_param.data)

        # Verify it's actually a different object
        policy_weight_original = policy.weight.data[0, 0].item()
        policy.weight.data[0, 0] = 999.0
        assert callback.ema_model.weight.data[0, 0].item() == pytest.approx(
            policy_weight_original
        )

    def test_ema_model_set_to_eval_and_no_grad(
        self,
        ema_callback_factory: Callable,
        pl_module_with_policy_factory: Callable,
        mock_trainer_factory: Callable,
    ):
        callback = ema_callback_factory()
        pl_module = pl_module_with_policy_factory()

        callback.on_fit_start(
            trainer=mock_trainer_factory(),
            pl_module=pl_module,
        )

        assert not callback.ema_model.training
        for param in callback.ema_model.parameters():
            assert not param.requires_grad


@pytest.mark.unit
class TestEMACallbackOnTrainBatchEnd:
    def test_updates_ema_weights_with_exponential_average(
        self,
        ema_callback_factory: Callable,
        pl_module_with_policy_factory: Callable,
        mock_trainer_factory: Callable,
        simple_module_factory: Callable,
        rng: np.random.Generator,
    ):
        policy = simple_module_factory()
        pl_module = pl_module_with_policy_factory(policy=policy)
        callback = ema_callback_factory(power=0.75, update_after_step=0, inv_gamma=1.0)
        trainer = mock_trainer_factory()

        callback.on_fit_start(trainer=trainer, pl_module=pl_module)

        # Change policy weights to simulate a training step
        new_weight = torch.from_numpy(
            rng.standard_normal(policy.weight.shape).astype(np.float32)
        )
        policy.weight.data.copy_(new_weight)

        callback.on_train_batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs=None,
            batch=None,
            batch_idx=0,
        )

        # EMA weight should be between old EMA and new policy weight
        ema_weight_after = callback.ema_model.weight.data
        # At step 0 (first call), decay = _get_decay(0) = 0.0 (since step <= 0)
        # So EMA should be entirely the new param: ema = 0 * ema + 1 * param
        assert torch.allclose(ema_weight_after, new_weight, atol=1e-6)

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "max_epochs, num_samples, batch_size",
        [
            (2, 16, 4),
            (3, 12, 4),
            (1, 8, 2),
        ],
    )
    def test_ema_decay_matches_expected_value_after_real_training(
        self,
        ema_callback_factory: Callable,
        real_lightning_module_factory: Callable[
            ..., tuple[pl.LightningModule, DataLoader]
        ],
        max_epochs: int,
        num_samples: int,
        batch_size: int,
    ):
        power = 0.75
        callback = ema_callback_factory(power=power, update_after_step=0)
        module, dataloader = real_lightning_module_factory(
            num_samples=num_samples, batch_size=batch_size
        )
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[callback],
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(module, dataloader)

        # decay formula: step = max(0, global_step - update_after_step - 1)
        #                decay = 1 - (1 + step)^(-power)
        expected_global_step = max_epochs * (num_samples // batch_size)
        assert trainer.global_step == expected_global_step
        warmup_step = max(0, expected_global_step - 0 - 1)
        expected_decay = 1 - (1 + warmup_step) ** -power
        assert callback.decay == pytest.approx(expected_decay, abs=1e-6)
        # EMA weights should differ from training weights
        ema_weights = callback.ema_model.weight.data.cpu()
        training_weights = module.policy.weight.data.cpu()
        assert not torch.equal(ema_weights, training_weights)

    def test_batchnorm_running_stats_copied_directly(
        self,
        ema_callback_factory: Callable,
        mock_trainer_factory: Callable,
        rng: np.random.Generator,
    ):
        # Build a module that contains a BatchNorm layer
        policy = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.BatchNorm1d(4),
        )
        weight_data = torch.from_numpy(rng.standard_normal((4, 4)).astype(np.float32))
        policy[0].weight.data.copy_(weight_data)

        pl_module = MagicMock()
        pl_module.policy = policy
        pl_module.parameters.return_value = policy.parameters()
        pl_module.log = MagicMock()

        callback = ema_callback_factory(power=0.75, update_after_step=0, inv_gamma=1.0)
        trainer = mock_trainer_factory()

        callback.on_fit_start(trainer=trainer, pl_module=pl_module)

        # Run a forward pass through policy to update BN running stats
        input_data = torch.from_numpy(rng.standard_normal((8, 4)).astype(np.float32))
        policy.train()
        policy(input_data)

        # Advance to step 2 so decay > 0
        for step in range(3):
            callback.on_train_batch_end(
                trainer=trainer,
                pl_module=pl_module,
                outputs=None,
                batch=None,
                batch_idx=step,
            )

        # BN running_mean/running_var in EMA should match policy exactly
        # (copied directly, not blended via EMA decay)
        policy_bn = policy[1]
        ema_bn = callback.ema_model[1]
        assert torch.allclose(ema_bn.running_mean, policy_bn.running_mean, atol=1e-6)
        assert torch.allclose(ema_bn.running_var, policy_bn.running_var, atol=1e-6)

    def test_decay_uses_trainer_global_step(
        self,
        ema_callback_factory: Callable,
        pl_module_with_policy_factory: Callable,
        mock_trainer_factory: Callable,
    ):
        callback = ema_callback_factory(update_after_step=0)
        pl_module = pl_module_with_policy_factory()

        trainer_step_50 = mock_trainer_factory(global_step=50)
        callback.on_fit_start(trainer=trainer_step_50, pl_module=pl_module)
        callback.on_train_batch_end(
            trainer=trainer_step_50,
            pl_module=pl_module,
            outputs=None,
            batch=None,
            batch_idx=0,
        )
        decay_at_50 = callback.decay

        trainer_step_500 = mock_trainer_factory(global_step=500)
        callback.on_train_batch_end(
            trainer=trainer_step_500,
            pl_module=pl_module,
            outputs=None,
            batch=None,
            batch_idx=1,
        )
        decay_at_500 = callback.decay

        # Decay should increase with global_step
        assert decay_at_500 > decay_at_50
        # And match _get_decay computed directly from the step values
        assert decay_at_50 == callback._get_decay(global_step=50)
        assert decay_at_500 == callback._get_decay(global_step=500)

    def test_does_nothing_when_ema_model_is_none(
        self,
        ema_callback_factory: Callable,
        pl_module_with_policy_factory: Callable,
        mock_trainer_factory: Callable,
    ):
        callback = ema_callback_factory()
        pl_module = pl_module_with_policy_factory()
        policy_params_before = [p.clone() for p in pl_module.policy.parameters()]
        # Do NOT call on_fit_start, so ema_model stays None
        callback.on_train_batch_end(
            trainer=mock_trainer_factory(),
            pl_module=pl_module,
            outputs=None,
            batch=None,
            batch_idx=0,
        )
        # Nothing should have changed
        assert callback.ema_model is None
        for before, after in zip(policy_params_before, pl_module.policy.parameters()):
            torch.testing.assert_close(before, after)

    def test_logs_decay_at_global_step_100(
        self,
        ema_callback_factory: Callable,
        pl_module_with_policy_factory: Callable,
        mock_trainer_factory: Callable,
    ):
        callback = ema_callback_factory()
        pl_module = pl_module_with_policy_factory()
        trainer = mock_trainer_factory(global_step=100)

        callback.on_fit_start(trainer=trainer, pl_module=pl_module)
        callback.on_train_batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs=None,
            batch=None,
            batch_idx=0,
        )

        log_calls = [
            call_args
            for call_args in pl_module.log.call_args_list
            if call_args[0][0] == "ema_decay"
        ]
        assert len(log_calls) == 1

    def test_does_not_log_decay_at_non_100_step(
        self,
        ema_callback_factory: Callable,
        pl_module_with_policy_factory: Callable,
        mock_trainer_factory: Callable,
    ):
        callback = ema_callback_factory()
        pl_module = pl_module_with_policy_factory()
        trainer = mock_trainer_factory(global_step=99)

        callback.on_fit_start(trainer=trainer, pl_module=pl_module)
        callback.on_train_batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs=None,
            batch=None,
            batch_idx=0,
        )

        log_calls = [
            call_args
            for call_args in pl_module.log.call_args_list
            if call_args[0][0] == "ema_decay"
        ]
        assert len(log_calls) == 0


@pytest.mark.unit
class TestEMACallbackValidationSwap:
    def test_swaps_policy_with_ema_model_during_validation(
        self,
        ema_callback_factory: Callable,
        pl_module_with_policy_factory: Callable,
        mock_trainer_factory: Callable,
        simple_module_factory: Callable,
    ):
        policy = simple_module_factory()
        pl_module = pl_module_with_policy_factory(policy=policy)
        callback = ema_callback_factory()
        trainer = mock_trainer_factory()

        callback.on_fit_start(trainer=trainer, pl_module=pl_module)

        # Before validation, policy is the original
        assert pl_module.policy is policy

        callback.on_validation_start(trainer=trainer, pl_module=pl_module)

        # During validation, policy should be the EMA model
        assert pl_module.policy is callback.ema_model

        callback.on_validation_end(trainer=trainer, pl_module=pl_module)

        # After validation, policy should be restored
        assert pl_module.policy is policy

    def test_no_swap_when_ema_model_is_none(
        self,
        ema_callback_factory: Callable,
        pl_module_with_policy_factory: Callable,
        mock_trainer_factory: Callable,
        simple_module_factory: Callable,
    ):
        policy = simple_module_factory()
        pl_module = pl_module_with_policy_factory(policy=policy)
        callback = ema_callback_factory()
        trainer = mock_trainer_factory()

        # Do NOT call on_fit_start

        callback.on_validation_start(trainer=trainer, pl_module=pl_module)

        # Policy should remain unchanged
        assert pl_module.policy is policy


@pytest.mark.unit
class TestEMACallbackCheckpoint:
    def test_injects_ema_weights_into_checkpoint(
        self,
        ema_callback_factory: Callable,
        pl_module_with_policy_factory: Callable,
        mock_trainer_factory: Callable,
        simple_module_factory: Callable,
        rng: np.random.Generator,
    ):
        policy = simple_module_factory()
        pl_module = pl_module_with_policy_factory(policy=policy)
        callback = ema_callback_factory()
        trainer = mock_trainer_factory()

        callback.on_fit_start(trainer=trainer, pl_module=pl_module)

        # Modify EMA model weights to be different from policy
        ema_weight = torch.from_numpy(
            rng.standard_normal(callback.ema_model.weight.shape).astype(np.float32)
        )
        callback.ema_model.weight.data.copy_(ema_weight)

        checkpoint = {
            "state_dict": {
                "policy.weight": policy.weight.data.clone(),
                "policy.bias": policy.bias.data.clone(),
            }
        }

        callback.on_save_checkpoint(
            trainer=trainer,
            pl_module=pl_module,
            checkpoint=checkpoint,
        )

        # Checkpoint should now have EMA weights
        assert torch.allclose(
            checkpoint["state_dict"]["policy.weight"],
            ema_weight,
            atol=1e-6,
        )

    def test_no_checkpoint_modification_when_ema_model_is_none(
        self,
        ema_callback_factory: Callable,
        pl_module_with_policy_factory: Callable,
        mock_trainer_factory: Callable,
        rng: np.random.Generator,
    ):
        callback = ema_callback_factory()
        trainer = mock_trainer_factory()
        pl_module = pl_module_with_policy_factory()

        original_weight = torch.from_numpy(
            rng.standard_normal((4, 4)).astype(np.float32)
        )
        checkpoint = {"state_dict": {"policy.weight": original_weight.clone()}}

        callback.on_save_checkpoint(
            trainer=trainer,
            pl_module=pl_module,
            checkpoint=checkpoint,
        )

        # Checkpoint should be unchanged
        assert torch.equal(checkpoint["state_dict"]["policy.weight"], original_weight)


@pytest.mark.unit
class TestGradientNormCallback:
    @pytest.mark.parametrize("log_every_n_steps", [10, 50, 100])
    def test_stores_configuration(self, log_every_n_steps: int):
        callback = GradientNormCallback(log_every_n_steps=log_every_n_steps)

        assert callback.log_every_n_steps == log_every_n_steps

    def test_logs_at_correct_frequency(
        self,
        mock_trainer_factory: Callable,
    ):
        callback = GradientNormCallback(log_every_n_steps=10)
        pl_module = MagicMock()
        pl_module.parameters.return_value = iter([])

        # Should log at step 0
        trainer = mock_trainer_factory(global_step=0)
        callback.on_before_optimizer_step(
            trainer=trainer, pl_module=pl_module, optimizer=MagicMock()
        )
        pl_module.log.assert_called()

        pl_module.log.reset_mock()

        # Should NOT log at step 5
        trainer = mock_trainer_factory(global_step=5)
        callback.on_before_optimizer_step(
            trainer=trainer, pl_module=pl_module, optimizer=MagicMock()
        )
        pl_module.log.assert_not_called()

        # Should log at step 10
        trainer = mock_trainer_factory(global_step=10)
        callback.on_before_optimizer_step(
            trainer=trainer, pl_module=pl_module, optimizer=MagicMock()
        )
        pl_module.log.assert_called()

    def test_computes_correct_gradient_norm(
        self,
        mock_trainer_factory: Callable,
        rng: np.random.Generator,
    ):
        callback = GradientNormCallback(log_every_n_steps=1)

        # Create module with known gradient
        param1 = torch.nn.Parameter(torch.zeros(3))
        param1.grad = torch.from_numpy(np.array([3.0, 4.0, 0.0], dtype=np.float32))
        param2 = torch.nn.Parameter(torch.zeros(2))
        param2.grad = torch.from_numpy(np.array([0.0, 0.0], dtype=np.float32))

        pl_module = MagicMock()
        pl_module.parameters.return_value = [param1, param2]
        pl_module.log = MagicMock()

        expected_norm = (3.0**2 + 4.0**2) ** 0.5  # = 5.0

        trainer = mock_trainer_factory(global_step=0)
        optimizer = MagicMock()
        optimizer.param_groups = [{"params": [param1, param2]}]

        callback.on_before_optimizer_step(
            trainer=trainer, pl_module=pl_module, optimizer=optimizer
        )

        # Find the grad_norm log call
        log_calls = {
            call_args[0][0]: call_args[0][1]
            for call_args in pl_module.log.call_args_list
        }
        assert abs(log_calls["grad_norm"] - expected_norm) < 1e-5

    def test_logs_per_group_norms_with_multiple_param_groups(
        self,
        mock_trainer_factory: Callable,
    ):
        callback = GradientNormCallback(log_every_n_steps=1)

        param1 = torch.nn.Parameter(torch.zeros(2))
        param1.grad = torch.tensor([1.0, 0.0])
        param2 = torch.nn.Parameter(torch.zeros(2))
        param2.grad = torch.tensor([0.0, 2.0])

        pl_module = MagicMock()
        pl_module.parameters.return_value = [param1, param2]
        pl_module.log = MagicMock()

        optimizer = MagicMock()
        optimizer.param_groups = [
            {"params": [param1]},
            {"params": [param2]},
        ]

        trainer = mock_trainer_factory(global_step=0)
        callback.on_before_optimizer_step(
            trainer=trainer, pl_module=pl_module, optimizer=optimizer
        )

        log_calls = {
            call_args[0][0]: call_args[0][1]
            for call_args in pl_module.log.call_args_list
        }

        assert "grad_norm_group_0" in log_calls
        assert "grad_norm_group_1" in log_calls
        assert abs(log_calls["grad_norm_group_0"] - 1.0) < 1e-5
        assert abs(log_calls["grad_norm_group_1"] - 2.0) < 1e-5


@pytest.mark.unit
class TestExpertUsageCallback:
    @pytest.mark.parametrize("log_every_n_epochs", [1, 5])
    def test_stores_configuration(self, log_every_n_epochs: int):
        callback = ExpertUsageCallback(log_every_n_epochs=log_every_n_epochs)

        assert callback.log_every_n_epochs == log_every_n_epochs

    @pytest.mark.parametrize(
        "hook_name, metrics_attr",
        [
            ("on_train_epoch_end", "train_metrics"),
            ("on_validation_epoch_end", "val_metrics"),
        ],
    )
    def test_skips_when_epoch_does_not_match_frequency(
        self,
        mock_trainer_factory: Callable,
        mock_pl_module_factory: Callable,
        hook_name: str,
        metrics_attr: str,
    ):
        callback = ExpertUsageCallback(log_every_n_epochs=3)
        pl_module = mock_pl_module_factory()
        trainer = mock_trainer_factory(current_epoch=1)

        getattr(callback, hook_name)(trainer=trainer, pl_module=pl_module)

        metrics_object = getattr(pl_module, metrics_attr)
        metrics_object.compute_expert_usage.assert_not_called()

    @pytest.mark.parametrize(
        "hook_name, metrics_attr",
        [
            ("on_train_epoch_end", "train_metrics"),
            ("on_validation_epoch_end", "val_metrics"),
        ],
    )
    def test_calls_compute_when_epoch_matches_frequency(
        self,
        mock_trainer_factory: Callable,
        mock_pl_module_factory: Callable,
        hook_name: str,
        metrics_attr: str,
    ):
        callback = ExpertUsageCallback(log_every_n_epochs=2)
        pl_module = mock_pl_module_factory()
        trainer = mock_trainer_factory(current_epoch=4)

        getattr(callback, hook_name)(trainer=trainer, pl_module=pl_module)

        metrics_object = getattr(pl_module, metrics_attr)
        metrics_object.compute_expert_usage.assert_called_once()

    @pytest.mark.parametrize(
        "hook_name, metrics_attr",
        [
            ("on_train_epoch_end", "train_metrics"),
            ("on_validation_epoch_end", "val_metrics"),
        ],
    )
    def test_logs_to_wandb_when_expert_usage_available(
        self,
        mock_trainer_factory: Callable,
        mock_pl_module_factory: Callable,
        hook_name: str,
        metrics_attr: str,
    ):
        callback = ExpertUsageCallback(log_every_n_epochs=1)
        expert_usage = np.array([0.3, 0.5, 0.2])
        pl_module = mock_pl_module_factory()
        metrics_object = getattr(pl_module, metrics_attr)
        metrics_object.compute_expert_usage.return_value = {
            "expert_usage": expert_usage
        }
        trainer = mock_trainer_factory(current_epoch=0)

        with (
            patch.object(callback, "_create_expert_usage_figure") as mock_create,
            patch("versatil.training.callbacks._figure_to_wandb_image"),
        ):
            mock_create.return_value = MagicMock()
            getattr(callback, hook_name)(trainer=trainer, pl_module=pl_module)

        trainer.logger.log_metrics.assert_called_once()

    @pytest.mark.parametrize(
        "hook_name, metrics_attr",
        [
            ("on_train_epoch_end", "train_metrics"),
            ("on_validation_epoch_end", "val_metrics"),
        ],
    )
    def test_does_not_log_when_expert_usage_is_none(
        self,
        mock_trainer_factory: Callable,
        mock_pl_module_factory: Callable,
        hook_name: str,
        metrics_attr: str,
    ):
        callback = ExpertUsageCallback(log_every_n_epochs=1)
        pl_module = mock_pl_module_factory()
        metrics_object = getattr(pl_module, metrics_attr)
        metrics_object.compute_expert_usage.return_value = None
        trainer = mock_trainer_factory(current_epoch=0)

        getattr(callback, hook_name)(trainer=trainer, pl_module=pl_module)

        trainer.logger.log_metrics.assert_not_called()


@pytest.mark.unit
class TestConfusionMatrixCallback:
    @pytest.mark.parametrize("log_every_n_epochs", [1, 5])
    def test_stores_configuration(self, log_every_n_epochs: int):
        callback = ConfusionMatrixCallback(log_every_n_epochs=log_every_n_epochs)

        assert callback.log_every_n_epochs == log_every_n_epochs

    @pytest.mark.parametrize(
        "hook_name, metrics_attr",
        [
            ("on_train_epoch_end", "train_metrics"),
            ("on_validation_epoch_end", "val_metrics"),
        ],
    )
    def test_skips_when_epoch_does_not_match_frequency(
        self,
        mock_trainer_factory: Callable,
        mock_pl_module_factory: Callable,
        hook_name: str,
        metrics_attr: str,
    ):
        callback = ConfusionMatrixCallback(log_every_n_epochs=3)
        pl_module = mock_pl_module_factory()
        trainer = mock_trainer_factory(current_epoch=1)

        getattr(callback, hook_name)(trainer=trainer, pl_module=pl_module)

        metrics_object = getattr(pl_module, metrics_attr)
        metrics_object.compute_confusion_matrix.assert_not_called()

    @pytest.mark.parametrize(
        "hook_name, metrics_attr",
        [
            ("on_train_epoch_end", "train_metrics"),
            ("on_validation_epoch_end", "val_metrics"),
        ],
    )
    def test_logs_to_wandb_when_confusion_matrix_available(
        self,
        mock_trainer_factory: Callable,
        mock_pl_module_factory: Callable,
        hook_name: str,
        metrics_attr: str,
    ):
        callback = ConfusionMatrixCallback(log_every_n_epochs=1)
        confusion_matrix = np.array([[10, 2], [3, 15]])
        pl_module = mock_pl_module_factory()
        metrics_object = getattr(pl_module, metrics_attr)
        metrics_object.compute_confusion_matrix.return_value = confusion_matrix
        trainer = mock_trainer_factory(current_epoch=0)

        with (
            patch.object(callback, "_create_confusion_matrix_figure") as mock_create,
            patch("versatil.training.callbacks._figure_to_wandb_image"),
        ):
            mock_create.return_value = MagicMock()
            getattr(callback, hook_name)(trainer=trainer, pl_module=pl_module)

        trainer.logger.log_metrics.assert_called_once()

    @pytest.mark.parametrize(
        "hook_name, metrics_attr",
        [
            ("on_train_epoch_end", "train_metrics"),
            ("on_validation_epoch_end", "val_metrics"),
        ],
    )
    def test_does_not_log_when_confusion_matrix_is_none(
        self,
        mock_trainer_factory: Callable,
        mock_pl_module_factory: Callable,
        hook_name: str,
        metrics_attr: str,
    ):
        callback = ConfusionMatrixCallback(log_every_n_epochs=1)
        pl_module = mock_pl_module_factory()
        metrics_object = getattr(pl_module, metrics_attr)
        metrics_object.compute_confusion_matrix.return_value = None
        trainer = mock_trainer_factory(current_epoch=0)

        getattr(callback, hook_name)(trainer=trainer, pl_module=pl_module)

        trainer.logger.log_metrics.assert_not_called()

    def test_does_not_convert_figure_when_logger_is_none(
        self,
        mock_trainer_factory: Callable,
        mock_pl_module_factory: Callable,
    ):
        callback = ConfusionMatrixCallback(log_every_n_epochs=1)
        confusion_matrix = np.array([[5, 1], [2, 8]])
        pl_module = mock_pl_module_factory()
        pl_module.train_metrics.compute_confusion_matrix.return_value = confusion_matrix
        trainer = mock_trainer_factory(current_epoch=0, logger=None)

        with (
            patch.object(callback, "_create_confusion_matrix_figure") as mock_create,
            patch(
                "versatil.training.callbacks._figure_to_wandb_image"
            ) as mock_to_wandb,
        ):
            mock_create.return_value = MagicMock()
            callback.on_train_epoch_end(trainer=trainer, pl_module=pl_module)

        mock_to_wandb.assert_not_called()


@pytest.mark.unit
class TestReduceLROnPlateauCallback:
    @pytest.mark.parametrize("patience", [5, 15])
    @pytest.mark.parametrize("factor", [0.1, 0.5])
    def test_stores_configuration(self, patience: int, factor: float):
        callback = ReduceLROnPlateauCallback(patience=patience, factor=factor)

        assert callback.patience == patience
        assert callback.factor == factor
        assert callback.monitor == "val_loss"
        assert callback.mode == "min"
        assert callback.scheduler is None

    def test_creates_scheduler_on_fit_start(
        self,
        mock_trainer_factory: Callable,
    ):
        callback = ReduceLROnPlateauCallback(patience=10)
        pl_module = MagicMock()
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
        pl_module.optimizers.return_value = optimizer

        callback.on_fit_start(
            trainer=mock_trainer_factory(),
            pl_module=pl_module,
        )

        assert callback.scheduler is not None

    def test_reduces_lr_after_patience_exceeded(
        self,
        mock_trainer_factory: Callable,
    ):
        callback = ReduceLROnPlateauCallback(patience=2, factor=0.5, threshold=0.0)
        param = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([param], lr=0.1)

        pl_module = MagicMock()
        pl_module.optimizers.return_value = optimizer
        pl_module.log = MagicMock()

        callback.on_fit_start(trainer=mock_trainer_factory(), pl_module=pl_module)

        initial_lr = optimizer.param_groups[0]["lr"]

        # Simulate plateau: same val_loss for patience+1 epochs
        for _ in range(4):
            trainer = mock_trainer_factory(
                callback_metrics={"val_loss": torch.tensor(1.0)}
            )
            callback.on_validation_epoch_end(trainer=trainer, pl_module=pl_module)

        new_lr = optimizer.param_groups[0]["lr"]
        assert new_lr < initial_lr
        assert abs(new_lr - initial_lr * 0.5) < 1e-8

    def test_no_update_when_scheduler_is_none(
        self,
        mock_trainer_factory: Callable,
    ):
        callback = ReduceLROnPlateauCallback(patience=10)
        pl_module = MagicMock()

        trainer = mock_trainer_factory(callback_metrics={"val_loss": torch.tensor(0.5)})

        # Should not raise when scheduler is None
        callback.on_validation_epoch_end(trainer=trainer, pl_module=pl_module)

    def test_no_update_when_metric_not_available(
        self,
        mock_trainer_factory: Callable,
    ):
        callback = ReduceLROnPlateauCallback(patience=2, monitor="val_loss")
        param = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([param], lr=0.1)

        pl_module = MagicMock()
        pl_module.optimizers.return_value = optimizer

        callback.on_fit_start(trainer=mock_trainer_factory(), pl_module=pl_module)

        initial_lr = optimizer.param_groups[0]["lr"]

        # No "val_loss" in callback_metrics
        trainer = mock_trainer_factory(callback_metrics={})
        callback.on_validation_epoch_end(trainer=trainer, pl_module=pl_module)

        assert optimizer.param_groups[0]["lr"] == initial_lr

    def test_handles_optimizer_list(
        self,
        mock_trainer_factory: Callable,
    ):
        callback = ReduceLROnPlateauCallback(patience=10)
        param = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([param], lr=0.01)

        pl_module = MagicMock()
        # Return as list (multi-optimizer case)
        pl_module.optimizers.return_value = [optimizer]

        callback.on_fit_start(trainer=mock_trainer_factory(), pl_module=pl_module)

        assert callback.scheduler is not None

    def test_logs_learning_rate_from_optimizer_list(
        self,
        mock_trainer_factory: Callable,
    ):
        expected_lr = 0.0123
        param = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([param], lr=expected_lr)

        pl_module = MagicMock()
        pl_module.optimizers.return_value = [optimizer]
        pl_module.log = MagicMock()

        callback = ReduceLROnPlateauCallback(
            patience=100, factor=0.5, threshold=0.0, monitor="val_loss"
        )
        callback.on_fit_start(trainer=mock_trainer_factory(), pl_module=pl_module)

        trainer = mock_trainer_factory(callback_metrics={"val_loss": torch.tensor(0.5)})
        callback.on_validation_epoch_end(trainer=trainer, pl_module=pl_module)

        lr_log_calls = [
            call_args
            for call_args in pl_module.log.call_args_list
            if call_args[0][0] == "lr"
        ]
        assert len(lr_log_calls) == 1
        logged_lr = lr_log_calls[0][0][1]
        assert logged_lr == pytest.approx(expected_lr)


@pytest.mark.unit
class TestLatentVisualizationCallback:
    @pytest.mark.parametrize("log_every_n_epochs", [1, 10])
    @pytest.mark.parametrize("max_samples", [100, 5000])
    def test_stores_configuration(
        self,
        log_every_n_epochs: int,
        max_samples: int,
    ):
        callback = LatentVisualizationCallback(
            log_every_n_epochs=log_every_n_epochs,
            max_samples=max_samples,
        )

        assert callback.log_every_n_epochs == log_every_n_epochs
        assert callback.max_samples == max_samples

    def test_skips_logging_on_non_matching_epochs(
        self,
        mock_trainer_factory: Callable,
    ):
        callback = LatentVisualizationCallback(log_every_n_epochs=5)
        pl_module = MagicMock()
        trainer = mock_trainer_factory(current_epoch=3)

        callback.on_validation_epoch_end(trainer=trainer, pl_module=pl_module)

        pl_module.val_metrics.compute_latent_visualization_data.assert_not_called()

    def test_skips_logging_when_no_latent_data(
        self,
        mock_trainer_factory: Callable,
    ):
        callback = LatentVisualizationCallback(log_every_n_epochs=1)
        pl_module = MagicMock()
        pl_module.val_metrics.compute_latent_visualization_data.return_value = None

        trainer = mock_trainer_factory(current_epoch=0)

        callback.on_validation_epoch_end(trainer=trainer, pl_module=pl_module)

        trainer.logger.log_metrics.assert_not_called()


@pytest.mark.unit
class TestCreateConfusionMatrixFigure:
    def test_returns_matplotlib_figure(self):
        callback = ConfusionMatrixCallback()
        confusion_matrix = np.array([[10, 2], [3, 15]])
        fig = callback._create_confusion_matrix_figure(confusion_matrix, "Test Matrix")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_normalizes_rows_to_proportions(self):
        callback = ConfusionMatrixCallback()
        confusion_matrix = np.array([[8, 2], [4, 6]])
        fig = callback._create_confusion_matrix_figure(confusion_matrix, "Normalized")
        # Figure was created without errors; row sums are normalized
        axes = fig.get_axes()
        assert len(axes) > 0
        plt.close(fig)

    def test_handles_zero_row_without_division_error(self):
        callback = ConfusionMatrixCallback()
        confusion_matrix = np.array([[0, 0], [3, 7]])
        # Should not raise due to clip(min=1e-10)
        fig = callback._create_confusion_matrix_figure(confusion_matrix, "Zero Row")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_labels_match_number_of_phases(self):
        callback = ConfusionMatrixCallback()
        confusion_matrix = np.eye(4, dtype=int) * 10
        fig = callback._create_confusion_matrix_figure(confusion_matrix, "4 Phases")
        axis = fig.get_axes()[0]
        x_labels = [label.get_text() for label in axis.get_xticklabels()]
        y_labels = [label.get_text() for label in axis.get_yticklabels()]
        assert len(x_labels) == 4
        assert len(y_labels) == 4
        assert "Phase 0" in x_labels
        assert "Phase 3" in x_labels
        plt.close(fig)


@pytest.mark.unit
class TestCreateExpertUsageFigure:
    def test_returns_matplotlib_figure(self):
        callback = ExpertUsageCallback()
        expert_usage = np.array([0.3, 0.5, 0.2])
        fig = callback._create_expert_usage_figure(expert_usage, "Test Usage")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_figure_has_correct_title(self):
        callback = ExpertUsageCallback()
        expert_usage = np.array([0.6, 0.4])
        title = "Train Expert Usage"
        fig = callback._create_expert_usage_figure(expert_usage, title)
        axis = fig.get_axes()[0]
        assert axis.get_title() == title
        plt.close(fig)


@pytest.mark.unit
class TestFigureToWandbImage:
    @patch("versatil.training.callbacks.wandb")
    def test_converts_figure_to_wandb_image(self, mock_wandb):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        mock_wandb.Image.return_value = MagicMock()

        result = _figure_to_wandb_image(fig)

        mock_wandb.Image.assert_called_once()
        assert result is mock_wandb.Image.return_value
        plt.close(fig)


@pytest.mark.unit
class TestLatentVisualizationCallbackOnValidationEpochEnd:
    def test_logs_posterior_and_prior_figures_with_phases(
        self,
        mock_trainer_factory: Callable,
        mock_latent_pl_module_factory: Callable,
    ):
        callback = LatentVisualizationCallback(log_every_n_epochs=1, max_samples=100)
        pl_module = mock_latent_pl_module_factory()
        trainer = mock_trainer_factory(current_epoch=0)

        with patch("versatil.training.callbacks._figure_to_wandb_image"):
            callback.on_validation_epoch_end(trainer=trainer, pl_module=pl_module)

        trainer.logger.log_metrics.assert_called_once()
        logged_metrics = trainer.logger.log_metrics.call_args.args[0]
        expected_keys = {
            "posterior_latent_space_tsne",
            "posterior_latent_space_pca",
            "posterior_pca_explained_variance",
            "prior_latent_space_tsne",
            "prior_latent_space_pca",
            "prior_pca_explained_variance",
        }
        assert expected_keys.issubset(set(logged_metrics.keys()))
        assert trainer.logger.log_metrics.call_args.kwargs["step"] == 0

    def test_handles_latent_dimension_one_without_crash(
        self,
        mock_trainer_factory: Callable,
        mock_latent_pl_module_factory: Callable,
    ):
        callback = LatentVisualizationCallback(log_every_n_epochs=1, max_samples=100)
        pl_module = mock_latent_pl_module_factory(latent_dimension=1)
        trainer = mock_trainer_factory(current_epoch=0)

        with patch("versatil.training.callbacks._figure_to_wandb_image"):
            callback.on_validation_epoch_end(trainer=trainer, pl_module=pl_module)

        trainer.logger.log_metrics.assert_called_once()
        logged_metrics = trainer.logger.log_metrics.call_args.args[0]
        expected_keys = {
            "posterior_latent_space_tsne",
            "posterior_latent_space_pca",
            "posterior_pca_explained_variance",
        }
        assert expected_keys.issubset(set(logged_metrics.keys()))

    def test_skips_logging_when_both_latents_are_none(
        self,
        mock_trainer_factory: Callable,
        mock_latent_pl_module_factory: Callable,
    ):
        callback = LatentVisualizationCallback(log_every_n_epochs=1, max_samples=100)
        pl_module = mock_latent_pl_module_factory(
            posterior_latent=None, prior_latent=None
        )
        trainer = mock_trainer_factory(current_epoch=0)

        callback.on_validation_epoch_end(trainer=trainer, pl_module=pl_module)

        trainer.logger.log_metrics.assert_not_called()

    def test_logs_only_prior_when_posterior_missing(
        self,
        mock_trainer_factory: Callable,
        mock_latent_pl_module_factory: Callable,
    ):
        callback = LatentVisualizationCallback(log_every_n_epochs=1, max_samples=100)
        pl_module = mock_latent_pl_module_factory(posterior_latent=None)
        trainer = mock_trainer_factory(current_epoch=0)

        with patch("versatil.training.callbacks._figure_to_wandb_image"):
            callback.on_validation_epoch_end(trainer=trainer, pl_module=pl_module)

        logged_metrics = trainer.logger.log_metrics.call_args.args[0]
        prior_keys = {
            "prior_latent_space_tsne",
            "prior_latent_space_pca",
            "prior_pca_explained_variance",
        }
        posterior_keys = {
            "posterior_latent_space_tsne",
            "posterior_latent_space_pca",
            "posterior_pca_explained_variance",
        }
        assert prior_keys.issubset(set(logged_metrics.keys()))
        assert posterior_keys.isdisjoint(set(logged_metrics.keys()))

    def test_does_not_log_when_logger_is_none(
        self,
        mock_trainer_factory: Callable,
        mock_latent_pl_module_factory: Callable,
    ):
        callback = LatentVisualizationCallback(log_every_n_epochs=1, max_samples=100)
        pl_module = mock_latent_pl_module_factory(prior_latent=None)
        trainer = mock_trainer_factory(current_epoch=0, logger=None)

        with patch(
            "versatil.training.callbacks._figure_to_wandb_image"
        ) as mock_to_wandb:
            callback.on_validation_epoch_end(trainer=trainer, pl_module=pl_module)

        mock_to_wandb.assert_not_called()

    def test_logs_latent_stats_table_when_metadata_present(
        self,
        mock_trainer_factory: Callable,
        mock_latent_pl_module_factory: Callable,
        rng: np.random.Generator,
    ):
        callback = LatentVisualizationCallback(log_every_n_epochs=1, max_samples=100)
        mu = torch.from_numpy(rng.standard_normal((8, 4)).astype(np.float32))
        pl_module = mock_latent_pl_module_factory(
            prior_latent=None,
            metadata={MetadataKey.POSTERIOR_MU.value: [mu]},
        )
        trainer = mock_trainer_factory(current_epoch=0)

        with patch("versatil.training.callbacks._figure_to_wandb_image"):
            callback.on_validation_epoch_end(trainer=trainer, pl_module=pl_module)

        logged_metrics = trainer.logger.log_metrics.call_args.args[0]
        assert "latent_space_statistics" in logged_metrics

    def test_closes_figures_after_logging(
        self,
        mock_trainer_factory: Callable,
        mock_latent_pl_module_factory: Callable,
    ):
        callback = LatentVisualizationCallback(log_every_n_epochs=1, max_samples=100)
        pl_module = mock_latent_pl_module_factory(prior_latent=None)
        trainer = mock_trainer_factory(current_epoch=0)

        with (
            patch("versatil.training.callbacks._figure_to_wandb_image"),
            patch("versatil.training.callbacks.plt.close") as mock_close,
        ):
            callback.on_validation_epoch_end(trainer=trainer, pl_module=pl_module)

        # Posterior path generates 3 figures: tsne, pca, explained_variance
        assert mock_close.call_count == 3


@pytest.mark.unit
class TestCreateLatentFigure:
    def test_returns_figure_with_single_axes_and_titled_with_phase(
        self,
        latent_data_factory: Callable,
        phase_array_factory: Callable,
    ):
        callback = LatentVisualizationCallback(max_samples=100)
        z = latent_data_factory(num_samples=12, latent_dimension=4)
        phases = phase_array_factory(num_samples=12, num_phases=3)

        with patch("versatil.training.callbacks.TSNE") as mock_tsne_class:
            mock_instance = MagicMock()
            mock_instance.fit_transform.return_value = np.zeros(
                (12, 2), dtype=np.float32
            )
            mock_tsne_class.return_value = mock_instance

            fig = callback._create_latent_figure(
                z, phases, title="Posterior latent space"
            )

        assert isinstance(fig, plt.Figure)
        # Called with correct perplexity (min(30, n-1) = 11 here)
        _, call_kwargs = mock_tsne_class.call_args
        assert call_kwargs["perplexity"] == 11
        assert call_kwargs["n_components"] == 2
        # Main scatter axis + colorbar axis
        axes = fig.get_axes()
        assert len(axes) >= 1
        main_title = axes[0].get_title()
        assert "Posterior latent space" in main_title
        assert "phase" in main_title.lower()
        plt.close(fig)

    def test_returns_figure_without_phase_annotation_when_phases_none(
        self,
        latent_data_factory: Callable,
    ):
        callback = LatentVisualizationCallback(max_samples=100)
        z = latent_data_factory(num_samples=10, latent_dimension=3)

        with patch("versatil.training.callbacks.TSNE") as mock_tsne_class:
            mock_instance = MagicMock()
            mock_instance.fit_transform.return_value = np.zeros(
                (10, 2), dtype=np.float32
            )
            mock_tsne_class.return_value = mock_instance

            fig = callback._create_latent_figure(z, None, title="Prior")

        axes = fig.get_axes()
        # Only main axis (no colorbar)
        assert len(axes) == 1
        assert "Prior" in axes[0].get_title()
        plt.close(fig)

    def test_subsamples_when_exceeding_max_samples(
        self,
        latent_data_factory: Callable,
        phase_array_factory: Callable,
    ):
        max_samples = 20
        callback = LatentVisualizationCallback(max_samples=max_samples)
        z = latent_data_factory(num_samples=50, latent_dimension=4)
        phases = phase_array_factory(num_samples=50, num_phases=3)

        with patch("versatil.training.callbacks.TSNE") as mock_tsne_class:
            mock_instance = MagicMock()
            mock_instance.fit_transform.return_value = np.zeros(
                (max_samples, 2), dtype=np.float32
            )
            mock_tsne_class.return_value = mock_instance

            fig = callback._create_latent_figure(z, phases, title="X")

        # fit_transform should have been called with a (max_samples, latent_dim) array
        fitted = mock_instance.fit_transform.call_args.args[0]
        assert fitted.shape == (max_samples, 4)
        plt.close(fig)


@pytest.mark.unit
class TestCreatePcaFigure:
    def test_returns_figure_with_phase_colored_scatter(
        self,
        latent_data_factory: Callable,
        phase_array_factory: Callable,
    ):
        callback = LatentVisualizationCallback(max_samples=100)
        z = latent_data_factory(num_samples=12, latent_dimension=4)
        phases = phase_array_factory(num_samples=12, num_phases=3)

        fig = callback._create_pca_figure(z, phases, title="Posterior latent space")

        assert isinstance(fig, plt.Figure)
        axes = fig.get_axes()
        assert len(axes) >= 1
        main_title = axes[0].get_title()
        assert "Posterior latent space" in main_title
        # Axis labels include explained variance percentages
        assert "PC1" in axes[0].get_xlabel()
        assert "%" in axes[0].get_xlabel()
        assert "PC2" in axes[0].get_ylabel()
        plt.close(fig)

    def test_returns_figure_without_hue_when_phases_none(
        self,
        latent_data_factory: Callable,
    ):
        callback = LatentVisualizationCallback(max_samples=100)
        z = latent_data_factory(num_samples=10, latent_dimension=3)

        fig = callback._create_pca_figure(z, None, title="Prior")

        assert isinstance(fig, plt.Figure)
        axes = fig.get_axes()
        assert "Prior PCA" in axes[0].get_title()
        plt.close(fig)

    def test_subsamples_when_exceeding_max_samples(
        self,
        latent_data_factory: Callable,
        phase_array_factory: Callable,
    ):
        max_samples = 15
        callback = LatentVisualizationCallback(max_samples=max_samples)
        z = latent_data_factory(num_samples=40, latent_dimension=5)
        phases = phase_array_factory(num_samples=40, num_phases=2)

        with patch("versatil.training.callbacks.PCA") as mock_pca_class:
            mock_instance = MagicMock()
            mock_instance.fit_transform.return_value = np.zeros(
                (max_samples, 2), dtype=np.float32
            )
            mock_instance.explained_variance_ratio_ = np.array([0.5, 0.3])
            mock_pca_class.return_value = mock_instance

            callback._create_pca_figure(z, phases, title="X")

        fitted = mock_instance.fit_transform.call_args.args[0]
        assert fitted.shape == (max_samples, 5)


@pytest.mark.unit
class TestCreatePcaVarianceFigure:
    def test_returns_bar_chart_with_one_bar_per_component(
        self,
        latent_data_factory: Callable,
    ):
        callback = LatentVisualizationCallback()
        latent_dimension = 5
        num_samples = 20
        z = latent_data_factory(
            num_samples=num_samples, latent_dimension=latent_dimension
        )

        fig = callback._create_pca_variance_figure(z, title="Posterior")

        assert isinstance(fig, plt.Figure)
        axes = fig.get_axes()
        assert len(axes) == 1
        assert "Posterior" in axes[0].get_title()
        assert "Explained Variance" in axes[0].get_title()
        assert axes[0].get_xlabel() == "Principal Component"
        # PCA with n=5 features and 20 samples produces 5 components => 5 bars
        assert len(axes[0].patches) == latent_dimension
        plt.close(fig)


@pytest.mark.unit
class TestCreateLatentStatsTable:
    def test_returns_none_when_metadata_empty(self):
        callback = LatentVisualizationCallback()

        table = callback._create_latent_stats_table(metadata={})

        assert table is None

    def test_returns_wandb_table_with_expected_columns_and_rows(
        self,
        rng: np.random.Generator,
    ):
        callback = LatentVisualizationCallback()
        posterior_mu = torch.from_numpy(rng.standard_normal((8, 4)).astype(np.float32))
        prior_mu = torch.from_numpy(rng.standard_normal((8, 4)).astype(np.float32))
        metadata = {
            MetadataKey.POSTERIOR_MU.value: [posterior_mu],
            MetadataKey.PRIOR_MU.value: [prior_mu],
        }

        table = callback._create_latent_stats_table(metadata=metadata)

        assert table is not None
        expected_columns = [
            "name",
            "shape",
            "mean",
            "per_dim_std_of_mean",
            "std",
            "per_dim_mean_of_std",
            "min",
            "max",
            "collapsed_dims",
        ]
        assert list(table.columns) == expected_columns
        # Two metadata entries => two rows
        assert len(table.data) == 2
        row_labels = {row[0] for row in table.data}
        assert row_labels == {"mu_posterior", "mu_prior"}

    def test_flattens_three_dimensional_metadata(
        self,
        rng: np.random.Generator,
    ):
        callback = LatentVisualizationCallback()
        # 3D tensor should get reshaped to (N, -1) before stats computation
        posterior_z = torch.from_numpy(
            rng.standard_normal((6, 2, 3)).astype(np.float32)
        )
        metadata = {MetadataKey.POSTERIOR_Z.value: [posterior_z]}

        table = callback._create_latent_stats_table(metadata=metadata)

        assert table is not None
        assert len(table.data) == 1
        shape_field = table.data[0][1]
        assert shape_field == str((6, 6))

    def test_counts_collapsed_dimensions_below_threshold(
        self,
        rng: np.random.Generator,
    ):
        callback = LatentVisualizationCallback()
        # Build a tensor where two dimensions have near-zero std (< 0.01)
        base = rng.standard_normal((20, 4)).astype(np.float32)
        base[:, 0] = 0.001
        base[:, 1] = 0.005
        tensor = torch.from_numpy(base)
        metadata = {MetadataKey.POSTERIOR_MU.value: [tensor]}

        table = callback._create_latent_stats_table(metadata=metadata)

        collapsed_dims = table.data[0][-1]
        assert collapsed_dims == 2
