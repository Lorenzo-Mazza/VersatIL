"""Tests for versatil.configs.post_training_compression module."""

from pathlib import Path

import hydra
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import versatil.configs  # noqa: F401
from versatil.configs.post_training_compression import (
    CompressionTargetConfig,
    ExecutorchXNNPACKBackendConfig,
    PostTrainingCompressorConfig,
    PreparationConfig,
)
from versatil.post_training_compression.compressor import (
    PostTrainingCompressor,
)
from versatil.quantization.workflows.pt2e import PT2EQuantizationWorkflow

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
class TestCompressionTargetConfig:
    def test_stores_module_path(self):
        config = CompressionTargetConfig(
            module_path="encoding_pipeline.encoders.left.backbone",
        )

        assert config.module_path == "encoding_pipeline.encoders.left.backbone"

    def test_defaults_to_interpolation_for_inheritance(self):
        config = CompressionTargetConfig(
            module_path="decoder",
        )
        omega = OmegaConf.structured(config)

        assert config.preparation == "${preparation}"
        assert config.pruning == "${pruning}"
        assert "quantization" not in omega


@pytest.mark.unit
class TestExecutorchXNNPACKBackendConfig:
    def test_default_max_batch_size(self) -> None:
        config = ExecutorchXNNPACKBackendConfig()

        assert config.max_batch_size == 32

    @pytest.mark.parametrize("max_batch_size", [1, 8, 16])
    def test_stores_configuration(self, max_batch_size: int) -> None:
        config = ExecutorchXNNPACKBackendConfig(max_batch_size=max_batch_size)

        assert config.max_batch_size == max_batch_size


@pytest.mark.unit
class TestPostTrainingCompressorConfig:
    @pytest.mark.parametrize("calibration_steps", [64, 256])
    def test_stores_configuration(self, calibration_steps):
        config = PostTrainingCompressorConfig(
            checkpoint_path="/tmp/ckpt",
            calibration_steps=calibration_steps,
        )

        assert config.checkpoint_path == "/tmp/ckpt"
        assert config.calibration_steps == calibration_steps

    def test_omegaconf_roundtrip(self):
        config = PostTrainingCompressorConfig(
            checkpoint_path="/tmp/ckpt",
            calibration_steps=64,
        )

        omega = OmegaConf.structured(config)
        assert omega.checkpoint_path == "/tmp/ckpt"
        assert omega.calibration_steps == 64
        assert omega.modules == []


@pytest.mark.unit
class TestPerModuleYamlQuantizationTarget:
    def test_top_level_workflow_owns_quantization_targets(self):
        with initialize_config_dir(config_dir=HYDRA_CONFIG_DIR, version_base=None):
            yaml_config = compose(
                config_name="end_to_end_ptq/unstructured_prune_x86_decoder_only",
                overrides=["checkpoint_path=/tmp/ckpt"],
            )

        compressor = hydra.utils.instantiate(yaml_config)

        assert isinstance(compressor, PostTrainingCompressor)
        assert [module.module_path for module in compressor.modules] == ["decoder"]
        assert isinstance(compressor.quantization, PT2EQuantizationWorkflow)
        assert [target.module_path for target in compressor.quantization.targets] == [
            "decoder"
        ]
