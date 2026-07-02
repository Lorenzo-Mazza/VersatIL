"""Tests for versatil.training.callbacks.reduce_lr_on_plateau module."""

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
import torch

from versatil.training.callbacks.reduce_lr_on_plateau import ReduceLROnPlateauCallback


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

        callback.on_validation_epoch_end(trainer=trainer, pl_module=pl_module)

        pl_module.log.assert_not_called()
        pl_module.optimizers.assert_not_called()

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

    def test_steps_on_train_epoch_end_without_validation_loop(
        self,
        mock_trainer_factory: Callable,
    ):
        callback = ReduceLROnPlateauCallback(
            patience=2, factor=0.5, threshold=0.0, monitor="train_loss"
        )
        param = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([param], lr=0.1)
        pl_module = MagicMock()
        pl_module.optimizers.return_value = optimizer
        pl_module.log = MagicMock()
        callback.on_fit_start(trainer=mock_trainer_factory(), pl_module=pl_module)
        initial_lr = optimizer.param_groups[0]["lr"]

        for _ in range(4):
            trainer = mock_trainer_factory(
                callback_metrics={"train_loss": torch.tensor(1.0)},
                val_dataloaders=None,
            )
            callback.on_train_epoch_end(trainer=trainer, pl_module=pl_module)

        assert optimizer.param_groups[0]["lr"] == pytest.approx(initial_lr * 0.5)

    def test_train_epoch_end_defers_to_validation_loop_when_present(
        self,
        mock_trainer_factory: Callable,
    ):
        callback = ReduceLROnPlateauCallback(patience=1, monitor="val_loss")
        pl_module = MagicMock()
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
        pl_module.optimizers.return_value = optimizer
        callback.on_fit_start(trainer=mock_trainer_factory(), pl_module=pl_module)
        callback.scheduler = MagicMock()

        trainer = mock_trainer_factory(
            callback_metrics={"val_loss": torch.tensor(1.0)},
            val_dataloaders=MagicMock(),
        )
        callback.on_train_epoch_end(trainer=trainer, pl_module=pl_module)

        callback.scheduler.step.assert_not_called()

    def test_skips_scheduler_step_during_sanity_check(
        self,
        mock_trainer_factory: Callable,
    ):
        callback = ReduceLROnPlateauCallback(patience=1, monitor="val_loss")
        pl_module = MagicMock()
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
        pl_module.optimizers.return_value = optimizer
        callback.on_fit_start(trainer=mock_trainer_factory(), pl_module=pl_module)
        callback.scheduler = MagicMock()

        trainer = mock_trainer_factory(
            callback_metrics={"val_loss": torch.tensor(1.0)},
            sanity_checking=True,
        )
        callback.on_validation_epoch_end(trainer=trainer, pl_module=pl_module)

        callback.scheduler.step.assert_not_called()


@pytest.mark.unit
class TestReduceLROnPlateauCallbackCheckpoint:
    def test_state_dict_is_empty_before_scheduler_exists(self):
        callback = ReduceLROnPlateauCallback(patience=10)

        assert callback.state_dict() == {}

    def test_load_empty_state_dict_leaves_no_resume_state(
        self,
        mock_trainer_factory: Callable,
    ):
        callback = ReduceLROnPlateauCallback(patience=3)
        callback.load_state_dict({})
        pl_module = MagicMock()
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
        pl_module.optimizers.return_value = optimizer

        callback.on_fit_start(trainer=mock_trainer_factory(), pl_module=pl_module)

        assert callback._resume_scheduler_state is None
        assert callback.scheduler.num_bad_epochs == 0

    def test_round_trip_restores_scheduler_progress(
        self,
        mock_trainer_factory: Callable,
    ):
        callback = ReduceLROnPlateauCallback(patience=5, factor=0.5, threshold=0.0)
        pl_module = MagicMock()
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
        pl_module.optimizers.return_value = optimizer
        callback.on_fit_start(trainer=mock_trainer_factory(), pl_module=pl_module)

        improving_trainer = mock_trainer_factory(
            callback_metrics={"val_loss": torch.tensor(0.5)}
        )
        callback.on_validation_epoch_end(trainer=improving_trainer, pl_module=pl_module)
        for _ in range(3):
            plateau_trainer = mock_trainer_factory(
                callback_metrics={"val_loss": torch.tensor(1.0)}
            )
            callback.on_validation_epoch_end(
                trainer=plateau_trainer, pl_module=pl_module
            )
        assert callback.scheduler.best == pytest.approx(0.5)
        assert callback.scheduler.num_bad_epochs == 3

        saved_state = callback.state_dict()

        resumed_callback = ReduceLROnPlateauCallback(
            patience=5, factor=0.5, threshold=0.0
        )
        resumed_callback.load_state_dict(saved_state)
        resumed_pl_module = MagicMock()
        resumed_optimizer = torch.optim.SGD(
            [torch.nn.Parameter(torch.zeros(1))], lr=0.1
        )
        resumed_pl_module.optimizers.return_value = resumed_optimizer
        resumed_callback.on_fit_start(
            trainer=mock_trainer_factory(), pl_module=resumed_pl_module
        )

        assert resumed_callback._resume_scheduler_state is None
        assert resumed_callback.scheduler.best == pytest.approx(0.5)
        assert resumed_callback.scheduler.num_bad_epochs == 3
        assert (
            resumed_callback.scheduler.cooldown_counter
            == callback.scheduler.cooldown_counter
        )

    def test_resumed_scheduler_reduces_lr_with_accumulated_bad_epochs(
        self,
        mock_trainer_factory: Callable,
    ):
        callback = ReduceLROnPlateauCallback(patience=3, factor=0.5, threshold=0.0)
        pl_module = MagicMock()
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
        pl_module.optimizers.return_value = optimizer
        callback.on_fit_start(trainer=mock_trainer_factory(), pl_module=pl_module)
        # The first step records the best metric; the next three accumulate
        # bad epochs so num_bad_epochs == patience right before the resume.
        for _ in range(4):
            plateau_trainer = mock_trainer_factory(
                callback_metrics={"val_loss": torch.tensor(1.0)}
            )
            callback.on_validation_epoch_end(
                trainer=plateau_trainer, pl_module=pl_module
            )
        assert callback.scheduler.num_bad_epochs == 3

        resumed_callback = ReduceLROnPlateauCallback(
            patience=3, factor=0.5, threshold=0.0
        )
        resumed_callback.load_state_dict(callback.state_dict())
        resumed_pl_module = MagicMock()
        resumed_optimizer = torch.optim.SGD(
            [torch.nn.Parameter(torch.zeros(1))], lr=0.1
        )
        resumed_pl_module.optimizers.return_value = resumed_optimizer
        resumed_callback.on_fit_start(
            trainer=mock_trainer_factory(), pl_module=resumed_pl_module
        )

        plateau_trainer = mock_trainer_factory(
            callback_metrics={"val_loss": torch.tensor(1.0)}
        )
        resumed_callback.on_validation_epoch_end(
            trainer=plateau_trainer, pl_module=resumed_pl_module
        )

        assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(0.05)
