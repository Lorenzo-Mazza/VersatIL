"""Tests for versatil.configs.experiment module."""

import dataclasses

import pytest
from omegaconf import MISSING

from versatil.configs.experiment import ExperimentConfig
from versatil.training.constants import Float32MatmulPrecision, PrecisionType


@pytest.mark.unit
class TestExperimentConfig:
    def test_name_and_checkpoint_folder_default_to_missing(self):
        config = ExperimentConfig()
        assert config.name == MISSING
        assert config.checkpoint_folder == MISSING

    @pytest.mark.parametrize(
        "precision", [PrecisionType.FP32.value, PrecisionType.BF16_MIXED.value]
    )
    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("distributed", [True, False])
    def test_stores_configuration(self, precision, device, distributed):
        config = ExperimentConfig(
            name="experiment",
            checkpoint_folder="/tmp",
            precision=precision,
            device=device,
            distributed=distributed,
        )
        assert config.precision == precision
        assert config.device == device
        assert config.distributed == distributed

    def test_precision_default_is_fp16_mixed_string(self):
        config = ExperimentConfig(name="test", checkpoint_folder="/tmp")
        assert config.precision == PrecisionType.FP16_MIXED.value
        assert config.precision == "16-mixed"

    def test_float32_matmul_precision_default_is_medium_string(self):
        config = ExperimentConfig(name="test", checkpoint_folder="/tmp")
        assert config.float32_matmul_precision == Float32MatmulPrecision.MEDIUM.value
        assert config.float32_matmul_precision == "medium"

    @pytest.mark.parametrize(
        "float32_matmul_precision",
        [
            Float32MatmulPrecision.HIGHEST.value,
            None,
        ],
    )
    def test_float32_matmul_precision_nullable(self, float32_matmul_precision):
        config = ExperimentConfig(
            name="test",
            checkpoint_folder="/tmp",
            float32_matmul_precision=float32_matmul_precision,
        )
        assert config.float32_matmul_precision == float32_matmul_precision

    def test_has_all_expected_fields(self):
        field_names = {f.name for f in dataclasses.fields(ExperimentConfig)}
        expected = {
            "name",
            "seed",
            "checkpoint_folder",
            "resume_from",
            "use_wandb",
            "wandb_project",
            "wandb_entity",
            "device",
            "distributed",
            "precision",
            "float32_matmul_precision",
            "checkpoint_every",
            "val_every",
            "plot_every",
            "validate_loss_keys",
        }
        assert expected == field_names

    @pytest.mark.parametrize(
        "checkpoint_every, val_every, plot_every",
        [
            (50, 2, 100),
            (200, 5, 500),
        ],
    )
    def test_stores_interval_settings(self, checkpoint_every, val_every, plot_every):
        config = ExperimentConfig(
            name="test",
            checkpoint_folder="/tmp",
            checkpoint_every=checkpoint_every,
            val_every=val_every,
            plot_every=plot_every,
        )
        assert config.checkpoint_every == checkpoint_every
        assert config.val_every == val_every
        assert config.plot_every == plot_every
