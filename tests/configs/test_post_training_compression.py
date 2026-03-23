"""Tests for versatil.configs.post_training_compression module."""

import pytest
from omegaconf import OmegaConf

from versatil.configs.post_training_compression import (
    ModuleCompressorConfig,
    PostTrainingCompressorConfig,
    PreparationConfig,
)


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
