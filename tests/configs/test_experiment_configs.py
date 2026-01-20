"""Tests for experiment configuration dataclasses."""
import dataclasses

import pytest

from versatil.configs.experiment import ExperimentConfig
from versatil.training.constants import Float32MatmulPrecision, PrecisionType


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
            'distributed', 'precision', 'float32_matmul_precision',
            'checkpoint_every', 'val_every', 'plot_every'
        }
        assert expected.issubset(fields)

    def test_default_values(self):
        config = ExperimentConfig(name="test", checkpoint_folder="/tmp")
        assert config.use_wandb is True
        assert config.distributed is False
        assert config.checkpoint_every == 100
        assert config.val_every == 1
        assert config.precision == PrecisionType.FP16_MIXED.value
        assert config.float32_matmul_precision == Float32MatmulPrecision.MEDIUM.value

    def test_precision_config_defaults(self):
        """Test precision-related configuration defaults."""
        config = ExperimentConfig(name="test", checkpoint_folder="/tmp")
        assert config.precision == "16-mixed"
        assert config.float32_matmul_precision == "medium"

    def test_precision_config_can_be_overridden(self):
        """Test precision settings can be overridden."""
        config = ExperimentConfig(
            name="test",
            checkpoint_folder="/tmp",
            precision=PrecisionType.FP32.value,
            float32_matmul_precision=Float32MatmulPrecision.HIGHEST.value
        )
        assert config.precision == "32"
        assert config.float32_matmul_precision == "highest"

    def test_float32_matmul_precision_can_be_null(self):
        """Test float32_matmul_precision can be set to None to disable."""
        config = ExperimentConfig(
            name="test",
            checkpoint_folder="/tmp",
            float32_matmul_precision=None
        )
        assert config.float32_matmul_precision is None
