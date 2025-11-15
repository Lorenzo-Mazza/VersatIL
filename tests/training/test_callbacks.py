"""Tests for PyTorch Lightning callbacks."""

import pytest
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging
from unittest.mock import Mock, MagicMock, patch

from refactoring.training.callbacks import EMACallback, GradientNormCallback, ConfusionMatrixCallback, ReduceLROnPlateauCallback
from refactoring.training.lightning_policy import LightningPolicy
from refactoring.configs.training import TrainingConfig, AdamWConfig


@pytest.mark.unit
class TestEMACallback:
    """Test EMA callback functionality."""

    def test_initialization(self):
        """Test EMA callback initialization."""
        callback = EMACallback(power=0.75)

        assert callback.power == 0.75
        assert callback.update_after_step == 0
        assert callback.inv_gamma == 1.0
        assert callback.min_value == 0.0
        assert callback.max_value == 0.9999
        assert callback.ema_model is None

    def test_initialization_custom_params(self):
        """Test EMA callback with custom parameters."""
        callback = EMACallback(
            power=0.999,
            update_after_step=100,
            inv_gamma=0.5,
            min_value=0.1,
            max_value=0.999,
        )

        assert callback.power == 0.999
        assert callback.update_after_step == 100
        assert callback.inv_gamma == 0.5
        assert callback.min_value == 0.1
        assert callback.max_value == 0.999

    def test_ema_model_created_on_fit_start(self, simple_policy, simple_training_config):
        """Test that EMA model is created when training starts."""
        callback = EMACallback(power=0.75)
        lightning_policy = LightningPolicy(
            policy=simple_policy,
            training_config=simple_training_config,
        )

        trainer = Mock()

        # Simulate fit start
        callback.on_fit_start(trainer, lightning_policy)

        assert callback.ema_model is not None
        assert isinstance(callback.ema_model, type(simple_policy))

    def test_decay_computation(self):
        """Test EMA decay factor computation."""
        callback = EMACallback(power=0.75, inv_gamma=1.0)

        # Test decay at different steps
        decay_0 = callback._get_decay(0)
        decay_100 = callback._get_decay(100)
        decay_1000 = callback._get_decay(1000)

        # Decay should increase with steps
        assert 0.0 <= decay_0 <= 1.0
        assert decay_0 < decay_100 < decay_1000
        assert decay_1000 <= callback.max_value


@pytest.mark.unit
class TestGradientNormCallback:
    """Test gradient norm logging callback."""

    def test_initialization(self):
        """Test gradient norm callback initialization."""
        callback = GradientNormCallback(log_every_n_steps=50)

        assert callback.log_every_n_steps == 50

    def test_initialization_default_params(self):
        """Test gradient norm callback with default parameters."""
        callback = GradientNormCallback()

        assert callback.log_every_n_steps == 50  # Default value

    def test_gradient_logging_frequency(self, simple_policy, simple_training_config):
        """Test that gradients are logged at correct frequency."""
        callback = GradientNormCallback(log_every_n_steps=10)
        lightning_policy = LightningPolicy(
            policy=simple_policy,
            training_config=simple_training_config,
        )

        trainer = Mock()

        # Should log at step 0 (0 % 10 == 0)
        trainer.global_step = 0
        with patch.object(lightning_policy, 'log') as mock_log:
            callback.on_before_optimizer_step(trainer, lightning_policy, optimizer=None)
            mock_log.assert_called_once()

        # Should not log at step 5 (5 % 10 != 0)
        trainer.global_step = 5
        with patch.object(lightning_policy, 'log') as mock_log:
            callback.on_before_optimizer_step(trainer, lightning_policy, optimizer=None)
            mock_log.assert_not_called()

        # Should log at step 10 (10 % 10 == 0)
        trainer.global_step = 10
        with patch.object(lightning_policy, 'log') as mock_log:
            callback.on_before_optimizer_step(trainer, lightning_policy, optimizer=None)
            mock_log.assert_called_once()


@pytest.mark.unit
class TestConfusionMatrixCallback:
    """Test confusion matrix logging callback."""

    def test_initialization(self):
        """Test confusion matrix callback initialization."""
        callback = ConfusionMatrixCallback(log_every_n_epochs=5)

        assert callback.log_every_n_epochs == 5

    def test_initialization_default_params(self):
        """Test confusion matrix callback with default parameters."""
        callback = ConfusionMatrixCallback()

        assert callback.log_every_n_epochs == 1  # Default value


@pytest.mark.unit
class TestLearningRateMonitor:
    """Test PyTorch Lightning's LearningRateMonitor callback."""

    def test_initialization(self):
        """Test learning rate monitor initialization."""
        callback = LearningRateMonitor(logging_interval='step')

        assert callback.logging_interval == 'step'

    def test_initialization_epoch_interval(self):
        """Test learning rate monitor with epoch interval."""
        callback = LearningRateMonitor(logging_interval='epoch')

        assert callback.logging_interval == 'epoch'

    def test_log_momentum_disabled(self):
        """Test learning rate monitor with momentum logging disabled."""
        callback = LearningRateMonitor(logging_interval='step', log_momentum=False)

        assert callback.logging_interval == 'step'
        assert not callback.log_momentum

    def test_integration_with_optimizer(self, simple_policy, simple_training_config):
        """Test that LR monitor works with optimizer."""
        simple_training_config.lr_schedule = "cosine"
        simple_training_config.lr_warmup_steps = 100

        lightning_policy = LightningPolicy(
            policy=simple_policy,
            training_config=simple_training_config,
        )

        # Mock trainer
        trainer = Mock()
        trainer.estimated_stepping_batches = 1000
        lightning_policy._trainer = trainer

        # Configure optimizer with scheduler
        optimizer_config = lightning_policy.configure_optimizers()

        assert "optimizer" in optimizer_config
        assert "lr_scheduler" in optimizer_config

        # Verify scheduler name is set for LR monitor
        assert optimizer_config["lr_scheduler"]["name"] == "learning_rate"


@pytest.mark.unit
class TestStochasticWeightAveraging:
    """Test PyTorch Lightning's StochasticWeightAveraging callback."""

    def test_initialization(self):
        """Test SWA callback initialization."""
        callback = StochasticWeightAveraging(
            swa_lrs=0.05,
            swa_epoch_start=10,
            annealing_epochs=5,
        )

        # SWA callback uses private attributes with underscores
        assert callback._swa_lrs == 0.05
        assert callback._swa_epoch_start == 10
        assert callback._annealing_epochs == 5

    def test_initialization_default_annealing(self):
        """Test SWA callback with default annealing epochs."""
        callback = StochasticWeightAveraging(
            swa_lrs=0.05,
            swa_epoch_start=10,
        )

        # SWA callback uses private attributes with underscores
        assert callback._swa_lrs == 0.05
        assert callback._swa_epoch_start == 10
        assert callback._annealing_epochs == 10  # Default value

    def test_swa_start_calculation(self):
        """Test SWA start epoch calculation based on fraction."""
        num_epochs = 100
        swa_epoch_start_fraction = 0.8

        # Calculate start epoch as done in workspace
        swa_epoch_start = int(swa_epoch_start_fraction * num_epochs)

        callback = StochasticWeightAveraging(
            swa_lrs=0.05,
            swa_epoch_start=swa_epoch_start,
            annealing_epochs=10,
        )

        # SWA callback uses private attributes with underscores
        assert callback._swa_epoch_start == 80
        assert callback._annealing_epochs == 10


@pytest.mark.integration
class TestCallbacksIntegration:
    """Integration tests for callbacks working together."""

    def test_multiple_callbacks_together(self, simple_policy, simple_training_config):
        """Test that multiple callbacks can work together."""
        simple_training_config.use_ema = True
        simple_training_config.lr_schedule = "cosine"
        simple_training_config.lr_warmup_steps = 100

        lightning_policy = LightningPolicy(
            policy=simple_policy,
            training_config=simple_training_config,
        )

        # Create callbacks
        ema_callback = EMACallback(power=0.75)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        gradient_norm_callback = GradientNormCallback(log_every_n_steps=50)

        # Mock trainer
        trainer = Mock()
        trainer.global_step = 0
        trainer.estimated_stepping_batches = 1000
        trainer.callbacks = [ema_callback, lr_monitor, gradient_norm_callback]

        # Initialize EMA
        ema_callback.on_fit_start(trainer, lightning_policy)

        # Verify all callbacks initialized
        assert ema_callback.ema_model is not None
        assert lr_monitor.logging_interval == 'step'
        assert gradient_norm_callback.log_every_n_steps == 50

    def test_ema_with_swa(self, simple_policy, simple_training_config):
        """Test that EMA and SWA can coexist (though typically not used together)."""
        simple_training_config.use_ema = True

        lightning_policy = LightningPolicy(
            policy=simple_policy,
            training_config=simple_training_config,
        )

        # Create both callbacks
        ema_callback = EMACallback(power=0.75)
        swa_callback = StochasticWeightAveraging(
            swa_lrs=0.05,
            swa_epoch_start=10,
            annealing_epochs=5,
        )

        # Mock trainer
        trainer = Mock()
        trainer.callbacks = [ema_callback, swa_callback]

        # Initialize EMA
        ema_callback.on_fit_start(trainer, lightning_policy)

        # Both should be properly configured
        assert ema_callback.ema_model is not None
        assert swa_callback._swa_lrs == 0.05


@pytest.mark.unit
class TestTrainingConfigCallbackParameters:
    """Test training config parameters for callbacks."""

    def test_swa_config_parameters(self):
        """Test SWA configuration parameters."""
        training_config = TrainingConfig(
            num_epochs=100,
            optimizer=AdamWConfig(lr=1e-4),
            swa_lrs=0.05,
            swa_epoch_start=0.8,
            swa_annealing_epochs=10,
        )

        assert training_config.swa_lrs == 0.05
        assert training_config.swa_epoch_start == 0.8
        assert training_config.swa_annealing_epochs == 10

    def test_swa_disabled_by_default(self):
        """Test that SWA is disabled by default."""
        training_config = TrainingConfig(
            num_epochs=100,
            optimizer=AdamWConfig(lr=1e-4),
        )

        assert training_config.swa_lrs is None

    def test_tuning_config_parameters(self):
        """Test hyperparameter tuning configuration parameters."""
        training_config = TrainingConfig(
            num_epochs=100,
            optimizer=AdamWConfig(lr=1e-4),
            tune_lr=True,
            tune_batch_size=True,
        )

        assert training_config.tune_lr is True
        assert training_config.tune_batch_size is True

    def test_tuning_disabled_by_default(self):
        """Test that hyperparameter tuning is disabled by default."""
        training_config = TrainingConfig(
            num_epochs=100,
            optimizer=AdamWConfig(lr=1e-4),
        )

        assert training_config.tune_lr is False
        assert training_config.tune_batch_size is False

    def test_all_callback_parameters_together(self):
        """Test all callback-related parameters in training config."""
        training_config = TrainingConfig(
            num_epochs=100,
            optimizer=AdamWConfig(lr=1e-4),
            use_ema=True,
            ema_power=0.999,
            swa_lrs=0.05,
            swa_epoch_start=0.8,
            swa_annealing_epochs=10,
            tune_lr=False,
            tune_batch_size=False,
        )

        # EMA parameters
        assert training_config.use_ema is True
        assert training_config.ema_power == 0.999

        # SWA parameters
        assert training_config.swa_lrs == 0.05
        assert training_config.swa_epoch_start == 0.8
        assert training_config.swa_annealing_epochs == 10

        # Tuning parameters
        assert training_config.tune_lr is False
        assert training_config.tune_batch_size is False

    def test_reduce_lr_on_plateau_config_parameters(self):
        """Test ReduceLROnPlateau configuration parameters."""
        training_config = TrainingConfig(
            num_epochs=100,
            optimizer=AdamWConfig(lr=1e-4),
            reduce_lr_on_plateau=True,
            reduce_lr_patience=15,
        )

        assert training_config.reduce_lr_on_plateau is True
        assert training_config.reduce_lr_patience == 15

    def test_reduce_lr_on_plateau_disabled_by_default(self):
        """Test that ReduceLROnPlateau is disabled by default."""
        training_config = TrainingConfig(
            num_epochs=100,
            optimizer=AdamWConfig(lr=1e-4),
        )

        assert training_config.reduce_lr_on_plateau is False
        assert training_config.reduce_lr_patience == 10  # Default value


@pytest.mark.unit
class TestReduceLROnPlateauCallback:
    """Test ReduceLROnPlateau callback."""

    def test_initialization(self):
        """Test ReduceLROnPlateau callback initialization."""
        callback = ReduceLROnPlateauCallback(patience=15)

        assert callback.monitor == "val_loss"
        assert callback.mode == "min"
        assert callback.patience == 15
        assert callback.factor == 0.5
        assert callback.scheduler is None

    def test_initialization_custom_params(self):
        """Test ReduceLROnPlateau callback with custom parameters."""
        callback = ReduceLROnPlateauCallback(
            monitor="val_acc",
            mode="max",
            factor=0.1,
            patience=5,
            min_lr=1e-7,
        )

        assert callback.monitor == "val_acc"
        assert callback.mode == "max"
        assert callback.factor == 0.1
        assert callback.patience == 5
        assert callback.min_lr == 1e-7

    def test_scheduler_created_on_fit_start(self, simple_policy, simple_training_config):
        """Test that scheduler is created when training starts."""
        callback = ReduceLROnPlateauCallback(patience=10)
        lightning_policy = LightningPolicy(
            policy=simple_policy,
            training_config=simple_training_config,
        )

        trainer = Mock()

        # Simulate fit start
        callback.on_fit_start(trainer, lightning_policy)

        assert callback.scheduler is not None

    def test_lr_reduction_on_plateau(self, simple_policy, simple_training_config):
        """Test that LR is reduced when metric plateaus."""
        callback = ReduceLROnPlateauCallback(patience=2, factor=0.5)
        lightning_policy = LightningPolicy(
            policy=simple_policy,
            training_config=simple_training_config,
        )

        trainer = Mock()
        trainer.callback_metrics = {"val_loss": torch.tensor(1.0)}

        # Initialize scheduler
        callback.on_fit_start(trainer, lightning_policy)

        # Get initial LR
        optimizer = lightning_policy.optimizers()
        initial_lr = optimizer.param_groups[0]["lr"]

        # Simulate plateau (same val_loss for patience+1 epochs)
        with patch.object(lightning_policy, 'log'):
            for _ in range(3):  # patience=2, so LR should reduce on 3rd call
                callback.on_validation_epoch_end(trainer, lightning_policy)

        # Check LR was reduced
        new_lr = optimizer.param_groups[0]["lr"]
        assert new_lr < initial_lr
        assert new_lr == initial_lr * callback.factor