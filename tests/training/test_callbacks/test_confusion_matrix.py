"""Tests for versatil.training.callbacks.confusion_matrix module."""

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from versatil.training.callbacks.confusion_matrix import ConfusionMatrixCallback


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
            patch("versatil.training.callbacks.confusion_matrix.figure_to_wandb_image"),
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

    def test_does_not_log_during_sanity_checking(
        self,
        mock_trainer_factory: Callable,
        mock_pl_module_factory: Callable,
    ):
        callback = ConfusionMatrixCallback(log_every_n_epochs=1)
        pl_module = mock_pl_module_factory()
        pl_module.val_metrics.compute_confusion_matrix.return_value = np.array(
            [[10, 2], [3, 15]]
        )
        trainer = mock_trainer_factory(current_epoch=0, sanity_checking=True)

        callback.on_validation_epoch_end(trainer=trainer, pl_module=pl_module)

        pl_module.val_metrics.compute_confusion_matrix.assert_not_called()
        trainer.logger.log_metrics.assert_not_called()

    @pytest.mark.parametrize(
        "hook_name, metrics_attr",
        [
            ("on_train_epoch_end", "train_metrics"),
            ("on_validation_epoch_end", "val_metrics"),
        ],
    )
    def test_does_not_convert_figure_when_logger_is_none(
        self,
        mock_trainer_factory: Callable,
        mock_pl_module_factory: Callable,
        hook_name: str,
        metrics_attr: str,
    ):
        callback = ConfusionMatrixCallback(log_every_n_epochs=1)
        confusion_matrix = np.array([[5, 1], [2, 8]])
        pl_module = mock_pl_module_factory()
        metrics_object = getattr(pl_module, metrics_attr)
        metrics_object.compute_confusion_matrix.return_value = confusion_matrix
        trainer = mock_trainer_factory(current_epoch=0, logger=None)

        with (
            patch.object(callback, "_create_confusion_matrix_figure") as mock_create,
            patch(
                "versatil.training.callbacks.confusion_matrix.figure_to_wandb_image"
            ) as mock_to_wandb,
        ):
            mock_create.return_value = MagicMock()
            getattr(callback, hook_name)(trainer=trainer, pl_module=pl_module)

        mock_to_wandb.assert_not_called()


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
        axes = fig.get_axes()
        assert len(axes) > 0
        plt.close(fig)

    def test_handles_zero_row_without_division_error(self):
        callback = ConfusionMatrixCallback()
        confusion_matrix = np.array([[0, 0], [3, 7]])
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
