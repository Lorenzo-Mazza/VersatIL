"""Tests for versatil.training.callbacks.gradient_norm module."""

from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from versatil.training.callbacks.gradient_norm import GradientNormCallback


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

        trainer = mock_trainer_factory(global_step=0)
        callback.on_before_optimizer_step(
            trainer=trainer, pl_module=pl_module, optimizer=MagicMock()
        )
        pl_module.log.assert_called()

        pl_module.log.reset_mock()

        trainer = mock_trainer_factory(global_step=5)
        callback.on_before_optimizer_step(
            trainer=trainer, pl_module=pl_module, optimizer=MagicMock()
        )
        pl_module.log.assert_not_called()

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

        param1 = torch.nn.Parameter(torch.zeros(3))
        param1.grad = torch.from_numpy(np.array([3.0, 4.0, 0.0], dtype=np.float32))
        param2 = torch.nn.Parameter(torch.zeros(2))
        param2.grad = torch.from_numpy(np.array([0.0, 0.0], dtype=np.float32))

        pl_module = MagicMock()
        pl_module.parameters.return_value = [param1, param2]
        pl_module.log = MagicMock()

        expected_norm = (3.0**2 + 4.0**2) ** 0.5

        trainer = mock_trainer_factory(global_step=0)
        optimizer = MagicMock()
        optimizer.param_groups = [{"params": [param1, param2]}]

        callback.on_before_optimizer_step(
            trainer=trainer, pl_module=pl_module, optimizer=optimizer
        )

        log_calls = {
            call_args[0][0]: call_args[0][1]
            for call_args in pl_module.log.call_args_list
        }
        assert abs(log_calls["grad_norm"] - expected_norm) < 1e-5
        assert abs(log_calls["train/grad_norm_step"] - expected_norm) < 1e-5
        assert log_calls["train/grad_clip_active_step"] == 0.0

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
        assert "train/grad_norm_group_0_step" in log_calls
        assert "train/grad_norm_group_1_step" in log_calls
        assert abs(log_calls["grad_norm_group_0"] - 1.0) < 1e-5
        assert abs(log_calls["grad_norm_group_1"] - 2.0) < 1e-5
        assert abs(log_calls["train/grad_norm_group_0_step"] - 1.0) < 1e-5
        assert abs(log_calls["train/grad_norm_group_1_step"] - 2.0) < 1e-5

    def test_epoch_end_is_noop_when_no_steps_logged(
        self,
        mock_trainer_factory: Callable,
    ):
        callback = GradientNormCallback(log_every_n_steps=50)
        pl_module = MagicMock()
        trainer = mock_trainer_factory(current_epoch=1)

        callback.on_train_epoch_end(trainer=trainer, pl_module=pl_module)

        trainer.logger.log_metrics.assert_not_called()

    def test_epoch_end_skips_logging_when_logger_is_none(
        self,
        mock_trainer_factory: Callable,
    ):
        callback = GradientNormCallback(log_every_n_steps=1)
        param = torch.nn.Parameter(torch.zeros(2))
        param.grad = torch.tensor([3.0, 4.0])
        pl_module = MagicMock()
        pl_module.parameters.return_value = [param]
        optimizer = MagicMock()
        optimizer.param_groups = [{"params": [param]}]
        trainer = mock_trainer_factory(current_epoch=0, global_step=0, logger=None)

        callback.on_before_optimizer_step(
            trainer=trainer, pl_module=pl_module, optimizer=optimizer
        )
        callback.on_train_epoch_end(trainer=trainer, pl_module=pl_module)

        assert callback._epoch_grad_norms == []
        assert callback._epoch_grad_clip_active == []

    def test_compute_grad_norm_skips_parameters_without_grad(
        self,
        mock_trainer_factory: Callable,
    ):
        callback = GradientNormCallback(log_every_n_steps=1)
        param_with_grad = torch.nn.Parameter(torch.zeros(2))
        param_with_grad.grad = torch.tensor([3.0, 4.0])
        param_without_grad = torch.nn.Parameter(torch.zeros(3))

        pl_module = MagicMock()
        pl_module.parameters.return_value = [param_with_grad, param_without_grad]
        pl_module.log = MagicMock()
        optimizer = MagicMock()
        optimizer.param_groups = [{"params": [param_with_grad, param_without_grad]}]
        trainer = mock_trainer_factory(global_step=0)

        callback.on_before_optimizer_step(
            trainer=trainer, pl_module=pl_module, optimizer=optimizer
        )

        log_calls = {
            call_args[0][0]: call_args[0][1]
            for call_args in pl_module.log.call_args_list
        }
        assert abs(log_calls["grad_norm"] - 5.0) < 1e-5

    def test_logs_epoch_summary_and_clears_buffer(
        self,
        mock_trainer_factory: Callable,
    ):
        callback = GradientNormCallback(log_every_n_steps=1)

        param = torch.nn.Parameter(torch.zeros(2))
        param.grad = torch.tensor([3.0, 4.0])

        pl_module = MagicMock()
        pl_module.parameters.return_value = [param]
        pl_module.log = MagicMock()

        optimizer = MagicMock()
        optimizer.param_groups = [{"params": [param]}]
        trainer = mock_trainer_factory(current_epoch=3, global_step=0)
        trainer.gradient_clip_val = 4.0

        callback.on_before_optimizer_step(
            trainer=trainer, pl_module=pl_module, optimizer=optimizer
        )
        callback.on_train_epoch_end(trainer=trainer, pl_module=pl_module)

        trainer.logger.log_metrics.assert_called_once()
        metrics = trainer.logger.log_metrics.call_args.args[0]
        assert metrics["train/grad_norm_epoch"] == pytest.approx(5.0)
        assert metrics["train/grad_norm_max_epoch"] == pytest.approx(5.0)
        assert metrics["train/grad_clip_active_ratio"] == pytest.approx(1.0)
        assert metrics["epoch"] == 3
        assert trainer.logger.log_metrics.call_args.kwargs["step"] == 3
        assert callback._epoch_grad_norms == []
        assert callback._epoch_grad_clip_active == []
