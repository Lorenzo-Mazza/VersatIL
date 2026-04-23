"""Tests for versatil.training.callbacks.ema module."""

from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from versatil.training.callbacks.ema import EMACallback


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

        decay = callback._get_decay(global_step=1_000_000)

        assert decay <= max_value

    def test_decay_clamped_to_min_value(
        self,
        ema_callback_factory: Callable,
    ):
        min_value = 0.5
        callback = ema_callback_factory(min_value=min_value, update_after_step=0)

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
        for ema_param, policy_param in zip(
            callback.ema_model.parameters(), policy.parameters()
        ):
            assert torch.equal(ema_param.data, policy_param.data)

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

        ema_weight_after = callback.ema_model.weight.data
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

        expected_global_step = max_epochs * (num_samples // batch_size)
        assert trainer.global_step == expected_global_step
        warmup_step = max(0, expected_global_step - 0 - 1)
        expected_decay = 1 - (1 + warmup_step) ** -power
        assert callback.decay == pytest.approx(expected_decay, abs=1e-6)
        ema_weights = callback.ema_model.weight.data.cpu()
        training_weights = module.policy.weight.data.cpu()
        assert not torch.equal(ema_weights, training_weights)

    def test_batchnorm_running_stats_copied_directly(
        self,
        ema_callback_factory: Callable,
        mock_trainer_factory: Callable,
        rng: np.random.Generator,
    ):
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

        input_data = torch.from_numpy(rng.standard_normal((8, 4)).astype(np.float32))
        policy.train()
        policy(input_data)

        for step in range(3):
            callback.on_train_batch_end(
                trainer=trainer,
                pl_module=pl_module,
                outputs=None,
                batch=None,
                batch_idx=step,
            )

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

        assert decay_at_500 > decay_at_50
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
        callback.on_train_batch_end(
            trainer=mock_trainer_factory(),
            pl_module=pl_module,
            outputs=None,
            batch=None,
            batch_idx=0,
        )
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

    def test_frozen_non_bn_params_are_copied_directly_not_ema_blended(
        self,
        ema_callback_factory: Callable,
        pl_module_with_policy_factory: Callable,
        mock_trainer_factory: Callable,
        rng: np.random.Generator,
    ):
        policy = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.Linear(4, 4),
        )
        for parameter in policy[0].parameters():
            parameter.requires_grad = False

        pl_module = pl_module_with_policy_factory(policy=policy)
        callback = ema_callback_factory(power=0.75, update_after_step=0, inv_gamma=1.0)
        trainer = mock_trainer_factory(global_step=100)

        callback.on_fit_start(trainer=trainer, pl_module=pl_module)
        trainable_ema_before = callback.ema_model[1].weight.data.clone()

        new_frozen_weight = torch.from_numpy(
            rng.standard_normal((4, 4)).astype(np.float32)
        )
        new_trainable_weight = torch.from_numpy(
            rng.standard_normal((4, 4)).astype(np.float32)
        )
        policy[0].weight.data.copy_(new_frozen_weight)
        policy[1].weight.data.copy_(new_trainable_weight)

        callback.on_train_batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs=None,
            batch=None,
            batch_idx=0,
        )

        torch.testing.assert_close(callback.ema_model[0].weight.data, new_frozen_weight)
        assert not torch.equal(callback.ema_model[1].weight.data, new_trainable_weight)
        assert not torch.equal(callback.ema_model[1].weight.data, trainable_ema_before)

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

        assert pl_module.policy is policy

        callback.on_validation_start(trainer=trainer, pl_module=pl_module)

        assert pl_module.policy is callback.ema_model

        callback.on_validation_end(trainer=trainer, pl_module=pl_module)

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

        callback.on_validation_start(trainer=trainer, pl_module=pl_module)

        assert pl_module.policy is policy

    def test_validation_end_is_noop_when_start_was_not_called(
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

        callback.on_validation_end(trainer=trainer, pl_module=pl_module)

        assert pl_module.policy is policy
        assert not hasattr(callback, "_original_policy")


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

        assert torch.allclose(
            checkpoint["state_dict"]["policy.weight"],
            ema_weight,
            atol=1e-6,
        )

    def test_skips_keys_missing_from_checkpoint_state_dict(
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
        unrelated_value = torch.zeros(1)
        checkpoint = {"state_dict": {"unrelated.key": unrelated_value.clone()}}

        callback.on_save_checkpoint(
            trainer=trainer,
            pl_module=pl_module,
            checkpoint=checkpoint,
        )

        assert set(checkpoint["state_dict"].keys()) == {"unrelated.key"}
        torch.testing.assert_close(
            checkpoint["state_dict"]["unrelated.key"], unrelated_value
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

        assert torch.equal(checkpoint["state_dict"]["policy.weight"], original_weight)
