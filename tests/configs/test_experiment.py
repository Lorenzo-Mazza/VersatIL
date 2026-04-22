"""Tests for versatil.configs.experiment module."""

import pytest
from omegaconf import MISSING

from versatil.configs.experiment import ExperimentConfig
from versatil.training.constants import Float32MatmulPrecision, PrecisionType


@pytest.mark.unit
class TestExperimentConfig:
    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("distributed", [True, False])
    @pytest.mark.parametrize(
        "precision",
        [PrecisionType.FP32.value, PrecisionType.BF16_MIXED.value],
    )
    @pytest.mark.parametrize(
        "float32_matmul_precision",
        [Float32MatmulPrecision.HIGHEST.value, None],
    )
    @pytest.mark.parametrize("save_checkpoints", [True, False])
    @pytest.mark.parametrize("use_wandb", [True, False])
    @pytest.mark.parametrize("validate_loss_keys", [True, False])
    def test_stores_configuration(
        self,
        device: str,
        distributed: bool,
        precision: str,
        float32_matmul_precision: str | None,
        save_checkpoints: bool,
        use_wandb: bool,
        validate_loss_keys: bool,
    ):
        config = ExperimentConfig(
            name="test",
            checkpoint_folder="/tmp",
            device=device,
            distributed=distributed,
            precision=precision,
            float32_matmul_precision=float32_matmul_precision,
            save_checkpoints=save_checkpoints,
            use_wandb=use_wandb,
            validate_loss_keys=validate_loss_keys,
        )
        assert config.device == device
        assert config.distributed == distributed
        assert config.precision == precision
        assert config.float32_matmul_precision == float32_matmul_precision
        assert config.save_checkpoints == save_checkpoints
        assert config.use_wandb == use_wandb
        assert config.validate_loss_keys == validate_loss_keys

    def test_defaults(self):
        config = ExperimentConfig()
        assert config.name == MISSING
        assert config.seed == 42
        assert config.checkpoint_folder == MISSING
        assert config.resume_from is None
        assert config.use_wandb is True
        assert config.wandb_project is None
        assert config.wandb_entity is None
        assert config.device == "cuda"
        assert config.distributed is False
        assert config.precision == PrecisionType.FP16_MIXED.value
        assert config.float32_matmul_precision == Float32MatmulPrecision.MEDIUM.value
        assert config.checkpoint_every == 100
        assert config.save_checkpoints is True
        assert config.val_every == 1
        assert config.plot_every == 200
        assert config.validate_loss_keys is True
