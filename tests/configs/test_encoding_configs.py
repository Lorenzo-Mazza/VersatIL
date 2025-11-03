"""Tests for encoding configuration dataclasses."""
import dataclasses
import inspect

import pytest
from hydra.utils import instantiate

from refactoring.configs.encoding.pipeline import EncodingPipelineConfig
from refactoring.configs.encoding.image import CNNEncoderConfig
from refactoring.models.encoding.pipeline import EncodingPipeline
from refactoring.models.encoding.encoders.rgb.cnn import CNNEncoder


@pytest.mark.unit
class TestEncodingPipelineConfig:

    def test_config_has_correct_target(self):
        config = EncodingPipelineConfig(
            encoders={"test": CNNEncoderConfig(input_keys=["left"], backbone="timm/resnet18.a1_in1k")},
        )
        assert config._target_ == "refactoring.models.encoding.pipeline.EncodingPipeline"

    def test_config_params_match_class_signature(self):
        sig = inspect.signature(EncodingPipeline.__init__)
        params = set(sig.parameters.keys()) - {'self'}

        config = EncodingPipelineConfig(
            encoders={"test": CNNEncoderConfig(input_keys=["left"], backbone="timm/resnet18.a1_in1k")},
        )
        config_keys = {f.name for f in dataclasses.fields(config)} - {'_target_'}

        assert config_keys.issubset(params), f"Extra keys: {config_keys - params}"

    def test_fusion_stages_is_optional(self):
        config = EncodingPipelineConfig(
            encoders={"test": CNNEncoderConfig(input_keys=["left"], backbone="timm/resnet18.a1_in1k")},
        )
        assert config.fusion_stages is None


@pytest.mark.unit
class TestCNNEncoderConfig:

    def test_config_has_correct_target(self):
        config = CNNEncoderConfig(input_keys=["left"], backbone="timm/resnet18.a1_in1k")
        assert config._target_ == "refactoring.models.encoding.encoders.rgb.cnn.CNNEncoder"

    def test_config_instantiates_correctly(self):
        config = CNNEncoderConfig(input_keys=["left"], backbone="timm/resnet18.a1_in1k", image_height=224, image_width=224)
        encoder = CNNEncoder(**{k: v for k, v in config.__dict__.items() if k != '_target_'})
        assert isinstance(encoder, CNNEncoder)

    def test_config_params_match_class_signature(self):
        sig = inspect.signature(CNNEncoder.__init__)
        params = set(sig.parameters.keys()) - {'self'}

        config = CNNEncoderConfig(input_keys=["left"], backbone="timm/resnet18.a1_in1k")
        config_keys = {f.name for f in dataclasses.fields(config)} - {'_target_'}

        assert config_keys.issubset(params), f"Extra keys: {config_keys - params}"