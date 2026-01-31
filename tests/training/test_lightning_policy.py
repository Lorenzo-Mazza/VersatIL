"""Tests for LightningPolicy wrapper."""

import torch
from unittest.mock import Mock

from versatil.training.lightning_policy import LightningPolicy
from versatil.data.constants import SampleKey
from versatil.configs.training import TrainingConfig, OptimizerConfig, AdamWConfig, ParameterGroupConfig


class TestLightningPolicyBasics:
    """Test basic LightningPolicy functionality."""

    def test_initialization(self, simple_policy, simple_training_config):
        lightning_policy = LightningPolicy(
            policy=simple_policy,
            training_config=simple_training_config,
        )

        assert lightning_policy.policy == simple_policy
        assert lightning_policy.training_config == simple_training_config
        assert lightning_policy.train_metrics is not None
        assert lightning_policy.val_metrics is not None

    def test_forward(self, simple_policy, simple_training_config, synthetic_training_batch):
        lightning_policy = LightningPolicy(
            policy=simple_policy,
            training_config=simple_training_config,
        )

        obs_dict = synthetic_training_batch[SampleKey.OBSERVATION.value]
        output = lightning_policy.forward(obs_dict)

        assert output is not None
        assert isinstance(output, dict)


class TestLightningPolicyTraining:
    """Test training step functionality."""

    def test_training_step(self, simple_policy, simple_training_config, synthetic_training_batch):
        lightning_policy = LightningPolicy(
            policy=simple_policy,
            training_config=simple_training_config,
        )

        loss = lightning_policy.training_step(synthetic_training_batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() > 0

        assert lightning_policy.train_metrics.num_batches == 1
        assert lightning_policy.train_metrics.total_loss > 0

    def test_training_epoch_end(self, simple_policy, simple_training_config, synthetic_training_batch):
        lightning_policy = LightningPolicy(
            policy=simple_policy,
            training_config=simple_training_config,
        )

        for i in range(3):
            lightning_policy.training_step(synthetic_training_batch, batch_idx=i)

        assert lightning_policy.train_metrics.num_batches == 3

        lightning_policy.on_train_epoch_end()

        assert lightning_policy.train_metrics.num_batches == 0
        assert lightning_policy.train_metrics.total_loss == 0.0

    def test_validation_step(self, simple_policy, simple_training_config, synthetic_training_batch):
        lightning_policy = LightningPolicy(
            policy=simple_policy,
            training_config=simple_training_config,
        )

        loss = lightning_policy.validation_step(synthetic_training_batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

        assert lightning_policy.val_metrics.num_batches == 1

    def test_validation_epoch_end(self, simple_policy, simple_training_config, synthetic_training_batch):
        lightning_policy = LightningPolicy(
            policy=simple_policy,
            training_config=simple_training_config,
        )

        for i in range(2):
            lightning_policy.validation_step(synthetic_training_batch, batch_idx=i)

        assert lightning_policy.val_metrics.num_batches == 2

        lightning_policy.on_validation_epoch_end()

        assert lightning_policy.val_metrics.num_batches == 0


class TestLightningPolicyOptimizer:
    """Test optimizer configuration."""

    def test_configure_optimizers_basic(self, simple_policy, simple_training_config):
        lightning_policy = LightningPolicy(
            policy=simple_policy,
            training_config=simple_training_config,
        )

        optimizer_config = lightning_policy.configure_optimizers()

        assert "optimizer" in optimizer_config
        assert isinstance(optimizer_config["optimizer"], torch.optim.AdamW)
        assert "lr_scheduler" not in optimizer_config

    def test_configure_optimizers_with_lr_schedule(self, simple_policy):
        optimizer_config = AdamWConfig(
            lr=1e-4,
        )

        training_config = TrainingConfig(
            num_epochs=10,
            optimizer=optimizer_config,
            lr_schedule="cosine",
            lr_warmup_steps=100,
            use_ema=False,
        )

        lightning_policy = LightningPolicy(
            policy=simple_policy,
            training_config=training_config,
        )

        mock_trainer = Mock()
        mock_trainer.estimated_stepping_batches = 1000
        lightning_policy._trainer = mock_trainer

        optimizer_config = lightning_policy.configure_optimizers()

        assert "optimizer" in optimizer_config
        assert "lr_scheduler" in optimizer_config
        assert "scheduler" in optimizer_config["lr_scheduler"]
        assert optimizer_config["lr_scheduler"]["interval"] == "step"

    def test_configure_optimizers_with_param_groups(self, simple_policy):
        param_groups = [
            ParameterGroupConfig(
                name="backbone",
                lr=1e-5,
                params_pattern=r".*backbone.*",
            ),
            ParameterGroupConfig(
                name="decoder",
                lr=1e-4,
                params_pattern=r".*decoder.*",
            ),
        ]

        optimizer_config = AdamWConfig(
            lr=1e-4,
            param_groups=param_groups,
        )

        training_config = TrainingConfig(
            num_epochs=10,
            optimizer=optimizer_config,
            use_ema=False,
        )

        lightning_policy = LightningPolicy(
            policy=simple_policy,
            training_config=training_config,
        )

        optimizer_dict = lightning_policy.configure_optimizers()
        optimizer = optimizer_dict["optimizer"]

        assert len(optimizer.param_groups) >= 1


class TestLightningPolicyGradientClipping:
    """Test gradient clipping configuration."""

    def test_gradient_clipping_disabled(self, simple_policy, simple_training_config):
        simple_training_config.clip_gradient_norm = False

        lightning_policy = LightningPolicy(
            policy=simple_policy,
            training_config=simple_training_config,
        )

        assert not simple_training_config.clip_gradient_norm

    def test_gradient_clipping_enabled(self, simple_policy):
        optimizer_config = AdamWConfig(lr=1e-4)

        training_config = TrainingConfig(
            num_epochs=10,
            optimizer=optimizer_config,
            clip_gradient_norm=True,
            clip_max_norm=1.0,
            use_ema=False,
        )

        assert training_config.clip_gradient_norm
        assert training_config.clip_max_norm == 1.0
