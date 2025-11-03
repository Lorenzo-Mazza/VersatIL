import inspect

import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from refactoring.configs.decoding.action_head import ActionHeadConfig
from refactoring.configs.decoding.decoder import (
    ACTConfig,
    MixtureOfExpertsDecoderConfig,
)
from refactoring.configs.task.task import ActionSpace, ObservationSpace
from refactoring.models.decoding.constants import MoERoutingType
from refactoring.models.decoding.decoders.factory.act import ACT
from refactoring.models.layers.activation import ActivationFunction


@pytest.mark.unit
class TestACTConfig:
    def test_config_has_correct_target(self):
        config = ACTConfig(
            action_heads={
                "position": ActionHeadConfig(input_dim=512, output_dim=3, blocks=[])
            },
            input_keys=["rgb_features"],
        )
        assert config._target_ == "refactoring.models.decoding.decoders.factory.act.ACT"

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
            == "refactoring.models.decoding.decoders.mixture_of_experts.MoEDecoder"
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