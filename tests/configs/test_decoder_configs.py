import inspect
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from versatil.configs.decoding.action_head import ActionHeadConfig
from versatil.configs.decoding.decoder import (
    ACTConfig,
    FASTDETRDecoderConfig,
    FASTGPTDecoderConfig,
    MixtureOfExpertsDecoderConfig,
)
from versatil.models.decoding.constants import MoERoutingType
from versatil.models.decoding.decoders.factory.act import ACT
from versatil.models.decoding.decoders.factory.fast_detr_decoder import FASTDETRDecoder
from versatil.models.decoding.decoders.factory.fast_gpt_decoder import FASTGPTDecoder
from versatil.models.layers.activation import ActivationFunction


@pytest.mark.unit
class TestACTConfig:
    def test_config_has_correct_target(self):
        config = ACTConfig(
            action_heads={
                "position": ActionHeadConfig(input_dim=512, output_dim=3, blocks=[])
            },
            input_keys=["rgb_features"],
        )
        assert config._target_ == "versatil.models.decoding.decoders.factory.act.ACT"

    def test_config_params_match_class_signature(self):
        sig = inspect.signature(ACT.__init__)
        params = set(sig.parameters.keys()) - {"self"}
        config = ACTConfig(
            action_heads={
                "position": ActionHeadConfig(input_dim=512, output_dim=3, blocks=[])
            },
            input_keys=["rgb_features"],
        )
        config_dict = OmegaConf.structured(config)
        config_keys = set(config_dict.keys()) - {"_target_"}
        assert config_keys.issubset(params), f"Extra keys: {config_keys - params}"

    def test_default_activation_is_relu(self):
        config = ACTConfig(
            action_heads={
                "position": ActionHeadConfig(input_dim=512, output_dim=3, blocks=[])
            },
            input_keys=["rgb_features"],
        )
        assert config.activation == ActivationFunction.RELU.value

    def test_default_embedding_dimension(self):
        config = ACTConfig(
            action_heads={
                "position": ActionHeadConfig(input_dim=512, output_dim=3, blocks=[])
            },
            input_keys=["rgb_features"],
        )
        assert config.embedding_dimension == 512


@pytest.mark.unit
class TestFASTDETRDecoderConfig:
    def test_config_has_correct_target(self):
        config = FASTDETRDecoderConfig(
            action_heads={
                "action_logits": ActionHeadConfig(input_dim=256, output_dim=2048, blocks=[])
            },
            input_keys=["rgb_features"],
        )
        assert config._target_ == "versatil.models.decoding.decoders.factory.fast_detr_decoder.FASTDETRDecoder"

    def test_config_params_match_class_signature(self):
        sig = inspect.signature(FASTDETRDecoder.__init__)
        params = set(sig.parameters.keys()) - {"self"}
        config = FASTDETRDecoderConfig(
            action_heads={
                "action_logits": ActionHeadConfig(input_dim=256, output_dim=2048, blocks=[])
            },
            input_keys=["rgb_features"],
        )
        config_dict = OmegaConf.structured(config)
        config_keys = set(config_dict.keys()) - {"_target_", "action_heads"}
        assert config_keys.issubset(params), f"Extra keys: {config_keys - params}"

    def test_default_activation_is_relu(self):
        config = FASTDETRDecoderConfig(
            action_heads={
                "action_logits": ActionHeadConfig(input_dim=256, output_dim=2048, blocks=[])
            },
            input_keys=["rgb_features"],
        )
        assert config.activation == ActivationFunction.RELU.value

    def test_default_embedding_dimension(self):
        config = FASTDETRDecoderConfig(
            action_heads={
                "action_logits": ActionHeadConfig(input_dim=256, output_dim=2048, blocks=[])
            },
            input_keys=["rgb_features"],
        )
        assert config.embedding_dimension == 256

    def test_default_vocab_size(self):
        config = FASTDETRDecoderConfig(
            action_heads={
                "action_logits": ActionHeadConfig(input_dim=256, output_dim=2048, blocks=[])
            },
            input_keys=["rgb_features"],
        )
        assert config.vocab_size == 2048

    def test_yaml_config_loads(self):
        """Test that fast_detr_decoder_default.yaml loads correctly via Hydra."""
        project_root = Path(__file__).parent.parent.parent
        decoder_config_dir = project_root / "hydra_configs" / "policy" / "decoder"

        with initialize_config_dir(config_dir=str(decoder_config_dir), version_base=None):
            cfg = compose(config_name="fast_detr_decoder_default")
            assert cfg is not None
            assert cfg._target_ == "versatil.models.decoding.decoders.factory.fast_detr_decoder.FASTDETRDecoder"
            assert cfg.embedding_dimension == 512


@pytest.mark.unit
class TestFASTGPTDecoderConfig:
    def test_config_has_correct_target(self):
        config = FASTGPTDecoderConfig(
            action_heads={
                "action_logits": ActionHeadConfig(input_dim=256, output_dim=1024, blocks=[])
            },
            input_keys=["visual_embedding"],
        )
        assert config._target_ == "versatil.models.decoding.decoders.factory.fast_gpt_decoder.FASTGPTDecoder"

    def test_config_params_match_class_signature(self):
        sig = inspect.signature(FASTGPTDecoder.__init__)
        params = set(sig.parameters.keys()) - {"self"}
        config = FASTGPTDecoderConfig(
            action_heads={
                "action_logits": ActionHeadConfig(input_dim=256, output_dim=1024, blocks=[])
            },
            input_keys=["visual_embedding"],
        )
        config_dict = OmegaConf.structured(config)
        config_keys = set(config_dict.keys()) - {"_target_", "action_heads"}
        assert config_keys.issubset(params), f"Extra keys: {config_keys - params}"

    def test_default_activation_is_swiglu(self):
        config = FASTGPTDecoderConfig(
            action_heads={
                "action_logits": ActionHeadConfig(input_dim=256, output_dim=1024, blocks=[])
            },
            input_keys=["visual_embedding"],
        )
        assert config.activation == ActivationFunction.SWIGLU.value

    def test_default_normalization_is_rmsnorm(self):
        config = FASTGPTDecoderConfig(
            action_heads={
                "action_logits": ActionHeadConfig(input_dim=256, output_dim=1024, blocks=[])
            },
            input_keys=["visual_embedding"],
        )
        assert config.normalization_type == "rmsnorm"

    def test_default_attention_type_is_gqa(self):
        config = FASTGPTDecoderConfig(
            action_heads={
                "action_logits": ActionHeadConfig(input_dim=256, output_dim=1024, blocks=[])
            },
            input_keys=["visual_embedding"],
        )
        assert config.attention_type == "gqa"

    def test_default_embedding_dimension(self):
        config = FASTGPTDecoderConfig(
            action_heads={
                "action_logits": ActionHeadConfig(input_dim=256, output_dim=1024, blocks=[])
            },
            input_keys=["visual_embedding"],
        )
        assert config.embedding_dimension == 256

    def test_default_action_vocabulary_size(self):
        config = FASTGPTDecoderConfig(
            action_heads={
                "action_logits": ActionHeadConfig(input_dim=256, output_dim=2048, blocks=[])
            },
            input_keys=["visual_embedding"],
        )
        assert config.action_vocabulary_size == 2048

    def test_yaml_config_loads(self):
        """Test that fast_gpt_decoder_default.yaml loads correctly via Hydra."""
        project_root = Path(__file__).parent.parent.parent
        decoder_config_dir = project_root / "hydra_configs" / "policy" / "decoder"

        with initialize_config_dir(config_dir=str(decoder_config_dir), version_base=None):
            cfg = compose(config_name="fast_gpt_decoder_default")
            assert cfg is not None
            assert cfg._target_ == "versatil.models.decoding.decoders.factory.fast_gpt_decoder.FASTGPTDecoder"
            assert cfg.embedding_dimension == 128
            assert cfg.action_vocabulary_size == 1024
            assert cfg.activation == "swiglu"
            assert cfg.normalization_type == "rmsnorm"
            assert cfg.attention_type == "gqa"


@pytest.mark.unit
class TestMixtureOfExpertsDecoderConfig:
    def test_config_has_correct_target(self):
        config = MixtureOfExpertsDecoderConfig(
            action_heads={
                "position": ActionHeadConfig(input_dim=512, output_dim=3, blocks=[])
            },
            input_keys=["rgb_features"],
        )
        assert (
            config._target_
            == "versatil.models.decoding.decoders.mixture_of_experts.MoEDecoder"
        )

    def test_default_routing_type_is_soft(self):
        config = MixtureOfExpertsDecoderConfig(
            action_heads={
                "position": ActionHeadConfig(input_dim=512, output_dim=3, blocks=[])
            },
            input_keys=["rgb_features"],
        )
        assert config.routing_type == MoERoutingType.SOFT.value

    def test_default_top_k_is_2(self):
        config = MixtureOfExpertsDecoderConfig(
            action_heads={
                "position": ActionHeadConfig(input_dim=512, output_dim=3, blocks=[])
            },
            input_keys=["rgb_features"],
        )
        assert config.top_k == 2

    def test_temperature_defaults_to_1(self):
        config = MixtureOfExpertsDecoderConfig(
            action_heads={
                "position": ActionHeadConfig(input_dim=512, output_dim=3, blocks=[])
            },
            input_keys=["rgb_features"],
        )
        assert config.temperature == 1.0

    def test_learnable_temperature_defaults_to_false(self):
        config = MixtureOfExpertsDecoderConfig(
            action_heads={
                "position": ActionHeadConfig(input_dim=512, output_dim=3, blocks=[])
            },
            input_keys=["rgb_features"],
        )
        assert config.learnable_temperature is False