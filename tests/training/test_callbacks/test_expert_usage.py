"""Tests for versatil.training.callbacks.expert_usage module."""

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from versatil.training.callbacks.expert_usage import ExpertUsageCallback


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
            patch("versatil.training.callbacks.expert_usage.figure_to_wandb_image"),
        ):
            mock_create.return_value = MagicMock()
            getattr(callback, hook_name)(trainer=trainer, pl_module=pl_module)

        trainer.logger.log_metrics.assert_called_once()

    def test_does_not_convert_figure_when_logger_is_none(
        self,
        mock_trainer_factory: Callable,
        mock_pl_module_factory: Callable,
    ):
        callback = ExpertUsageCallback(log_every_n_epochs=1)
        expert_usage = np.array([0.3, 0.5, 0.2])
        pl_module = mock_pl_module_factory()
        pl_module.train_metrics.compute_expert_usage.return_value = {
            "expert_usage": expert_usage
        }
        trainer = mock_trainer_factory(current_epoch=0, logger=None)

        with (
            patch.object(callback, "_create_expert_usage_figure") as mock_create,
            patch(
                "versatil.training.callbacks.expert_usage.figure_to_wandb_image"
            ) as mock_to_wandb,
        ):
            mock_create.return_value = MagicMock()
            callback.on_train_epoch_end(trainer=trainer, pl_module=pl_module)

        mock_to_wandb.assert_not_called()

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
