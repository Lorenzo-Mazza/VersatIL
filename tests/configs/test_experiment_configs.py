"""Tests for experiment configuration dataclasses."""
import dataclasses

import pytest

from refactoring.configs.experiment import ExperimentConfig


@pytest.mark.unit
class TestExperimentConfig:

    def test_config_can_be_instantiated(self):
        config = ExperimentConfig(name="test_experiment", checkpoint_folder="/tmp/checkpoints")
        assert isinstance(config, ExperimentConfig)
        assert config.name == "test_experiment"
        assert config.seed == 42
        assert config.device == "cuda"

    def test_config_has_expected_fields(self):
        fields = {f.name for f in dataclasses.fields(ExperimentConfig)}
        expected = {
            'name', 'seed', 'checkpoint_folder', 'resume_from',
            'use_wandb', 'wandb_project', 'wandb_entity', 'device',
            'distributed', 'checkpoint_every', 'val_every', 'plot_every'
        }
        assert expected.issubset(fields)

    def test_default_values(self):
        config = ExperimentConfig(name="test", checkpoint_folder="/tmp")
        assert config.use_wandb is True
        assert config.distributed is False
        assert config.checkpoint_every == 100
        assert config.val_every == 1
