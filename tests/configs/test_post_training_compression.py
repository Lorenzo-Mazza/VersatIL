"""Tests for versatil.configs.post_training_compression module."""

from pathlib import Path

import hydra
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import versatil.configs  # noqa: F401
from versatil.configs.post_training_compression import (
    ModuleCompressorConfig,
    PostTrainingCompressorConfig,
    PreparationConfig,
)
from versatil.post_training_compression.compressor import (
    PostTrainingCompressor,
)
from versatil.quantization.strategies import PT2EStrategy

HYDRA_CONFIG_DIR = str(Path(__file__).parents[2] / "hydra_configs")


@pytest.mark.unit
class TestPreparationConfig:
    @pytest.mark.parametrize("replace_bn", [True, False])
    @pytest.mark.parametrize("fuse_conv", [True, False])
    def test_stores_configuration(self, replace_bn, fuse_conv):
        config = PreparationConfig(
            replace_frozen_batchnorm=replace_bn,
            fuse_conv_batchnorm=fuse_conv,
        )

        assert config.replace_frozen_batchnorm == replace_bn
        assert config.fuse_conv_batchnorm == fuse_conv


@pytest.mark.unit
class TestModuleCompressorConfig:
    def test_stores_module_path(self):
        config = ModuleCompressorConfig(
            module_path="encoding_pipeline.encoders.left.backbone",
        )

        assert config.module_path == "encoding_pipeline.encoders.left.backbone"

    def test_defaults_to_interpolation_for_inheritance(self):
        config = ModuleCompressorConfig(
            module_path="decoder",
        )

        assert config.preparation == "${preparation}"
        assert config.pruning == "${pruning}"
        assert config.quantization == "${quantization}"


@pytest.mark.unit
class TestPostTrainingCompressorConfig:
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    @pytest.mark.parametrize("calibration_steps", [64, 256])
    def test_stores_configuration(self, device, calibration_steps):
        config = PostTrainingCompressorConfig(
            checkpoint_path="/tmp/ckpt",
            device=device,
            calibration_steps=calibration_steps,
        )

        assert config.checkpoint_path == "/tmp/ckpt"
        assert config.device == device
        assert config.calibration_steps == calibration_steps

    def test_omegaconf_roundtrip(self):
        config = PostTrainingCompressorConfig(
            checkpoint_path="/tmp/ckpt",
            device="cpu",
            calibration_steps=64,
        )

        omega = OmegaConf.structured(config)
        assert omega.checkpoint_path == "/tmp/ckpt"
        assert omega.calibration_steps == 64
        assert omega.modules == []


@pytest.mark.unit
class TestPerModuleYamlInheritance:
    def test_modules_inherit_global_quantization(self):
        with initialize_config_dir(config_dir=HYDRA_CONFIG_DIR, version_base=None):
            yaml_config = compose(
                config_name="end_to_end_ptq/per_module_example",
                overrides=["checkpoint_path=/tmp/ckpt"],
            )

        compressor = hydra.utils.instantiate(yaml_config)

        assert isinstance(compressor, PostTrainingCompressor)
        for module in compressor.modules:
            if module.quantization is not None:
                assert isinstance(module.quantization, PT2EStrategy)
