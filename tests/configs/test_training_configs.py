"""Tests for training configuration dataclasses."""
import dataclasses

import pytest

from versatil.configs.training import (
    TrainingConfig,
    OptimizerConfig,
    AdamWConfig,
    AdamConfig,
    SGDConfig,
    ParameterGroupConfig,
)


@pytest.mark.unit
class TestTrainingConfig:

    def test_config_can_be_instantiated(self):
        config = TrainingConfig()
        assert isinstance(config, TrainingConfig)
        assert config.num_epochs == 100
        assert config.use_ema is True

    def test_config_has_expected_fields(self):
        fields = {f.name for f in dataclasses.fields(TrainingConfig)}
        expected = {
            'num_epochs', 'gradient_accumulate_every', 'optimizer',
            'clip_gradient_norm', 'clip_max_norm', 'lr_schedule',
            'lr_warmup_steps', 'use_ema', 'ema_power'
        }
        assert expected.issubset(fields)


@pytest.mark.unit
class TestAdamWConfig:

    def test_config_has_correct_target(self):
        config = AdamWConfig()
        assert config.target_class == "torch.optim.AdamW"

    def test_config_can_be_instantiated(self):
        config = AdamWConfig()
        assert isinstance(config, AdamWConfig)
        assert config.lr == 1e-4
        assert config.weight_decay == 1e-4
        assert config.betas == (0.9, 0.999)


@pytest.mark.unit
class TestAdamConfig:

    def test_config_has_correct_target(self):
        config = AdamConfig()
        assert config.target_class == "torch.optim.Adam"

    def test_config_can_be_instantiated(self):
        config = AdamConfig()
        assert isinstance(config, AdamConfig)
        assert config.lr == 1e-4
        assert config.weight_decay == 0.0


@pytest.mark.unit
class TestSGDConfig:

    def test_config_has_correct_target(self):
        config = SGDConfig()
        assert config.target_class == "torch.optim.SGD"

    def test_config_can_be_instantiated(self):
        config = SGDConfig()
        assert isinstance(config, SGDConfig)
        assert config.lr == 1e-2
        assert config.momentum == 0.0


@pytest.mark.unit
class TestParameterGroupConfig:

    def test_config_can_be_instantiated(self):
        config = ParameterGroupConfig(name="backbone", lr=1e-5)
        assert config.name == "backbone"
        assert config.lr == 1e-5
        assert config.weight_decay is None
