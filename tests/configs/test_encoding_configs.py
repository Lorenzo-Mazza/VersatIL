import dataclasses
import inspect

import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from refactoring.configs.encoding.encoder import (
    DepthCNNEncoderConfig,
    LanguageEncoderConfig,
    ProprioEncoderConfig,
)
from refactoring.configs.encoding.fusion import (
    AttentionFusionConfig,
    ConcatFusionConfig,
)
from refactoring.configs.encoding.image import CNNEncoderConfig
from refactoring.configs.encoding.pipeline import EncodingPipelineConfig
from refactoring.models.encoding.encoders.rgb.cnn import CNNEncoder
from refactoring.models.encoding.fusion.attention import AttentionFusion
from refactoring.models.encoding.fusion.concat import ConcatFusion
from refactoring.models.encoding.pipeline import EncodingPipeline


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


@pytest.mark.unit
class TestDepthEncoderConfig:

    def test_config_has_correct_target(self):
        config = DepthCNNEncoderConfig(input_keys=["depth"], backbone="resnet18")
        assert config._target_ == "refactoring.models.encoding.encoders.depth.cnn.DepthCNNEncoder"

    def test_default_use_group_norm(self):
        config = DepthCNNEncoderConfig(input_keys=["depth"], backbone="resnet18")
        assert config.use_group_norm is True

    def test_default_spatial_softmax(self):
        config = DepthCNNEncoderConfig(input_keys=["depth"], backbone="resnet18")
        assert config.spatial_softmax is True


@pytest.mark.unit
class TestStateEncoderConfig:

    def test_config_has_correct_target(self):
        config = ProprioEncoderConfig(input_keys=["proprio"])
        assert config._target_ == "refactoring.models.encoding.encoders.proprioceptive.base.ProprioceptiveEncoder"

    def test_default_hidden_dims(self):
        config = ProprioEncoderConfig(input_keys=["proprio"])
        assert config.hidden_dims == [128]

    def test_default_activation(self):
        config = ProprioEncoderConfig(input_keys=["proprio"])
        assert config.activation == "relu"


@pytest.mark.unit
class TestLanguageEncoderConfig:

    def test_config_has_correct_target(self):
        config = LanguageEncoderConfig(input_keys=["language_instruction"])
        assert config._target_ == "refactoring.models.encoding.encoders.language.language.LanguageEncoder"


@pytest.mark.unit
class TestConcatFusionModule:

    def test_config_has_correct_target(self):
        config = ConcatFusionConfig(
            input_features=["feature1", "feature2"],
            output_name="fused",
            hidden_dim=256
        )
        assert config._target_ == "refactoring.models.encoding.fusion.concat.ConcatFusion"

    def test_config_instantiates_correctly(self):
        config = ConcatFusionConfig(
            input_features=["feature1", "feature2"],
            output_name="fused",
            hidden_dim=256
        )
        fusion = instantiate(config)
        assert isinstance(fusion, ConcatFusion)

    def test_config_params_match_class_signature(self):
        sig = inspect.signature(ConcatFusion.__init__)
        params = set(sig.parameters.keys()) - {'self'}
        config = ConcatFusionConfig(
            input_features=["feature1", "feature2"],
            output_name="fused",
            hidden_dim=256
        )
        config_dict = OmegaConf.structured(config)
        config_keys = set(config_dict.keys()) - {'_target_'}
        assert config_keys.issubset(params), f"Extra keys: {config_keys - params}"


@pytest.mark.unit
class TestAttentionFusionModule:

    def test_config_has_correct_target(self):
        config = AttentionFusionConfig(
            input_features=["feature1", "feature2"],
            output_name="fused",
            hidden_dim=256
        )
        assert config._target_ == "refactoring.models.encoding.fusion.attention.AttentionFusion"

    def test_config_instantiates_correctly(self):
        config = AttentionFusionConfig(
            input_features=["feature1", "feature2"],
            output_name="fused",
            hidden_dim=256
        )
        fusion = instantiate(config)
        assert isinstance(fusion, AttentionFusion)

    def test_config_params_match_class_signature(self):
        sig = inspect.signature(AttentionFusion.__init__)
        params = set(sig.parameters.keys()) - {'self'}
        config = AttentionFusionConfig(
            input_features=["feature1", "feature2"],
            output_name="fused",
            hidden_dim=256
        )
        config_dict = OmegaConf.structured(config)
        config_keys = set(config_dict.keys()) - {'_target_'}
        assert config_keys.issubset(params), f"Extra keys: {config_keys - params}"

    def test_default_num_heads(self):
        config = AttentionFusionConfig(
            input_features=["feature1", "feature2"],
            output_name="fused",
            hidden_dim=256
        )
        assert config.num_heads == 8

    def test_default_dropout(self):
        config = AttentionFusionConfig(
            input_features=["feature1", "feature2"],
            output_name="fused",
            hidden_dim=256
        )
        assert config.dropout == 0.1