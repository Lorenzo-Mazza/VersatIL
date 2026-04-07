"""Tests for versatil.configs.decoding.decoder module."""

import pytest
from hydra.utils import instantiate
from omegaconf import MISSING, OmegaConf, flag_override

from versatil.configs.data.task import ActionSpaceConfig, ObservationSpaceConfig
from versatil.configs.decoding.action_head import ActionHeadConfig
from versatil.configs.decoding.decoder import (
    ACTConfig,
    ActionTransformerConfig,
    ConditionalActionUNetConfig,
    DecodingNetworkConfig,
    DiffusionActionTransformerConfig,
    DiscreteDETRActionTransformerConfig,
    DiTBlockActionTransformerConfig,
    FreeActionTransformerConfig,
    GPTActionTransformerConfig,
    LACTConfig,
    MixtureOfDensitiesActionTransformerConfig,
    MixtureOfExpertsDecoderConfig,
    MoEFreeActionTransformerConfig,
    PhaseACTConfig,
    Pi0DecoderConfig,
    SmolVLADecoderConfig,
)
from versatil.models.decoding.constants import (
    DecoderOutputKey,
    DiTType,
    GMMInitStrategy,
    MoERoutingType,
    TimeConditioning,
)
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType, PositionalEncodingType
from versatil.models.layers.normalization.constants import NormalizationType


def _instantiate_decoder_config(config, action_heads=None):
    """Resolve interpolation fields and instantiate a decoder config."""
    structured = OmegaConf.structured(config)
    with flag_override(structured, "struct", False):
        structured.observation_space = ObservationSpaceConfig()
        structured.action_space = ActionSpaceConfig()
        structured.observation_horizon = 2
        structured.prediction_horizon = 10
        structured.device = "cpu"
        structured.input_keys = ["rgb_features"]
        if action_heads is not None:
            structured.action_heads = action_heads
        elif not OmegaConf.is_missing(structured, "action_heads"):
            structured.action_heads = {}
    return instantiate(structured)


@pytest.mark.unit
class TestDecodingNetworkConfig:
    def test_target_defaults_to_missing(self):
        config = DecodingNetworkConfig(input_keys=["features"])
        assert config._target_ == MISSING

    def test_input_keys_required(self):
        config = DecodingNetworkConfig()
        assert config.input_keys == MISSING

    def test_action_heads_default_to_none(self):
        config = DecodingNetworkConfig(input_keys=["features"])
        assert config.action_heads is None

    def test_interpolation_references(self):
        config = DecodingNetworkConfig(input_keys=["features"])
        assert config.observation_space == "${policy.observation_space}"
        assert config.action_space == "${policy.action_space}"
        assert config.observation_horizon == "${policy.observation_horizon}"
        assert config.prediction_horizon == "${policy.prediction_horizon}"
        assert config.device == "${policy.device}"


@pytest.mark.unit
class TestACTConfig:
    def test_target_points_to_act(self):
        config = ACTConfig(input_keys=["features"])
        assert config._target_ == "versatil.models.decoding.decoders.factory.act.ACT"

    def test_activation_default_is_relu_string(self):
        config = ACTConfig(input_keys=["features"])
        assert config.activation == ActivationFunction.RELU.value

    @pytest.mark.parametrize("embedding_dimension", [256, 512])
    @pytest.mark.parametrize("number_of_heads", [4, 8])
    @pytest.mark.parametrize("number_of_encoder_layers", [2, 6])
    def test_stores_configuration(
        self, embedding_dimension, number_of_heads, number_of_encoder_layers
    ):
        config = ACTConfig(
            input_keys=["features"],
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_encoder_layers=number_of_encoder_layers,
        )
        assert config.embedding_dimension == embedding_dimension
        assert config.number_of_heads == number_of_heads
        assert config.number_of_encoder_layers == number_of_encoder_layers

    def test_inherits_from_decoding_network_config(self):
        config = ACTConfig(input_keys=["features"])
        assert isinstance(config, DecodingNetworkConfig)


@pytest.mark.unit
class TestPhaseACTConfig:
    def test_target_points_to_phase_act(self):
        config = PhaseACTConfig(input_keys=["features"])
        assert (
            config._target_
            == "versatil.models.decoding.decoders.factory.phase_act.PhaseACT"
        )

    def test_phase_routing_key_required(self):
        config = PhaseACTConfig(input_keys=["features"])
        assert config.phase_routing_key == MISSING

    def test_inherits_from_act_config(self):
        config = PhaseACTConfig(input_keys=["features"])
        assert isinstance(config, ACTConfig)


@pytest.mark.unit
class TestGPTActionTransformerConfig:
    def test_target_points_to_gpt_action_transformer(self):
        config = GPTActionTransformerConfig(input_keys=["features"])
        assert (
            config._target_
            == "versatil.models.decoding.decoders.factory.gpt_action_transformer.GPTActionTransformer"
        )

    def test_activation_default_is_swiglu_string(self):
        config = GPTActionTransformerConfig(input_keys=["features"])
        assert config.activation == ActivationFunction.SWIGLU.value

    def test_normalization_type_default_is_rms_norm_string(self):
        config = GPTActionTransformerConfig(input_keys=["features"])
        assert config.normalization_type == NormalizationType.RMS_NORM.value

    def test_attention_type_default_is_grouped_query_string(self):
        config = GPTActionTransformerConfig(input_keys=["features"])
        assert config.attention_type == AttentionType.GROUPED_QUERY.value

    def test_positional_encoding_default_is_rope_string(self):
        config = GPTActionTransformerConfig(input_keys=["features"])
        assert config.positional_encoding_type == PositionalEncodingType.ROPE.value

    @pytest.mark.parametrize("deterministic", [True, False])
    @pytest.mark.parametrize("learnable_temperature", [True, False])
    def test_stores_inference_options(self, deterministic, learnable_temperature):
        config = GPTActionTransformerConfig(
            input_keys=["features"],
            deterministic=deterministic,
            learnable_temperature=learnable_temperature,
        )
        assert config.deterministic == deterministic
        assert config.learnable_temperature == learnable_temperature


@pytest.mark.unit
class TestDiscreteDETRActionTransformerConfig:
    def test_target_points_to_discrete_detr(self):
        config = DiscreteDETRActionTransformerConfig(input_keys=["features"])
        assert (
            config._target_
            == "versatil.models.decoding.decoders.factory.discrete_detr_action_transformer.DiscreteDETRActionTransformer"
        )

    def test_activation_default_is_relu_string(self):
        config = DiscreteDETRActionTransformerConfig(input_keys=["features"])
        assert config.activation == ActivationFunction.RELU.value


@pytest.mark.unit
class TestActionTransformerConfig:
    def test_target_points_to_action_transformer(self):
        config = ActionTransformerConfig(input_keys=["features"])
        assert (
            config._target_
            == "versatil.models.decoding.decoders.factory.action_transformer.ActionTransformer"
        )

    def test_activation_default_is_swiglu_string(self):
        config = ActionTransformerConfig(input_keys=["features"])
        assert config.activation == ActivationFunction.SWIGLU.value


@pytest.mark.unit
class TestMixtureOfDensitiesActionTransformerConfig:
    def test_target_points_to_mode_act(self):
        config = MixtureOfDensitiesActionTransformerConfig(input_keys=["features"])
        assert (
            config._target_
            == "versatil.models.decoding.decoders.factory.mode_act.MixtureOfDensitiesActionTransformer"
        )

    def test_gmm_init_strategy_default_is_kmeans_plus_plus_string(self):
        config = MixtureOfDensitiesActionTransformerConfig(input_keys=["features"])
        assert config.gmm_init_strategy == GMMInitStrategy.KMEANS_PLUS_PLUS.value

    @pytest.mark.parametrize("num_mixture_components", [4, 16])
    def test_stores_mixture_components(self, num_mixture_components):
        config = MixtureOfDensitiesActionTransformerConfig(
            input_keys=["features"],
            num_mixture_components=num_mixture_components,
        )
        assert config.num_mixture_components == num_mixture_components


@pytest.mark.unit
class TestLACTConfig:
    def test_target_points_to_lact(self):
        config = LACTConfig(input_keys=["features"], latent_dimension=32)
        assert config._target_ == "versatil.models.decoding.decoders.factory.lact.LACT"

    def test_latent_dimension_required(self):
        config = LACTConfig(input_keys=["features"])
        assert config.latent_dimension == MISSING


@pytest.mark.unit
class TestFreeActionTransformerConfig:
    def test_target_points_to_free_action_transformer(self):
        config = FreeActionTransformerConfig(input_keys=["features"])
        assert (
            config._target_
            == "versatil.models.decoding.decoders.factory.free_action_transformer.FreeActionTransformer"
        )

    @pytest.mark.parametrize("use_global_latent", [True, False])
    def test_stores_global_latent_option(self, use_global_latent):
        config = FreeActionTransformerConfig(
            input_keys=["features"], use_global_latent=use_global_latent
        )
        assert config.use_global_latent == use_global_latent


@pytest.mark.unit
class TestMoEFreeActionTransformerConfig:
    def test_target_points_to_moe_free_action_transformer(self):
        config = MoEFreeActionTransformerConfig(input_keys=["features"])
        assert (
            config._target_
            == "versatil.models.decoding.decoders.factory.moe_free_action_transformer.MoEFreeActionTransformer"
        )

    def test_inherits_from_free_action_transformer_config(self):
        config = MoEFreeActionTransformerConfig(input_keys=["features"])
        assert isinstance(config, FreeActionTransformerConfig)


@pytest.mark.unit
class TestMixtureOfExpertsDecoderConfig:
    def test_target_points_to_moe_decoder(self):
        config = MixtureOfExpertsDecoderConfig(input_keys=["features"])
        assert config._target_ == "versatil.models.decoding.decoders.moe.MoEDecoder"

    def test_routing_type_default_is_soft_string(self):
        config = MixtureOfExpertsDecoderConfig(input_keys=["features"])
        assert config.routing_type == MoERoutingType.SOFT.value

    def test_required_fields(self):
        config = MixtureOfExpertsDecoderConfig(input_keys=["features"])
        assert config.base_expert == MISSING
        assert config.num_experts == MISSING
        assert config.gating_feature_key == MISSING


@pytest.mark.unit
class TestDiTBlockActionTransformerConfig:
    def test_target_points_to_dit_block_action_transformer(self):
        config = DiTBlockActionTransformerConfig(input_keys=["features"])
        assert (
            config._target_
            == "versatil.models.decoding.decoders.factory.dit_block_action_transformer.DiTBlockActionTransformer"
        )


@pytest.mark.unit
class TestDiffusionActionTransformerConfig:
    def test_target_points_to_diffusion_action_transformer(self):
        config = DiffusionActionTransformerConfig(input_keys=["features"])
        assert (
            config._target_
            == "versatil.models.decoding.decoders.factory.diffusion_action_transformer.DiffusionActionTransformer"
        )

    def test_diffusion_transformer_type_default_is_cross_attention_string(self):
        config = DiffusionActionTransformerConfig(input_keys=["features"])
        assert config.diffusion_transformer_type == DiTType.CROSS_ATTENTION.value


@pytest.mark.unit
class TestConditionalActionUNetConfig:
    def test_target_points_to_conditional_action_unet(self):
        config = ConditionalActionUNetConfig(input_keys=["features"])
        assert (
            config._target_
            == "versatil.models.decoding.decoders.factory.conditional_action_unet.ConditionalActionUNet"
        )

    def test_down_dimensions_default(self):
        config = ConditionalActionUNetConfig(input_keys=["features"])
        assert config.down_dimensions == [256, 512, 1024]


@pytest.mark.unit
class TestSmolVLADecoderConfig:
    def test_defaults(self):
        config = SmolVLADecoderConfig(input_keys=["features"])
        assert isinstance(config, DecodingNetworkConfig)
        assert (
            config._target_
            == "versatil.models.decoding.decoders.factory.smolvla_decoder.SmolVLADecoder"
        )
        assert config.normalization_type == NormalizationType.RMS_NORM.value
        assert config.proprioceptive_feature_key is None

    @pytest.mark.parametrize("expert_width_multiplier", [0.5, 0.75])
    @pytest.mark.parametrize("num_vlm_layers", [8, 16])
    @pytest.mark.parametrize("freeze_vlm", [True, False])
    def test_stores_configuration(
        self, expert_width_multiplier, num_vlm_layers, freeze_vlm
    ):
        config = SmolVLADecoderConfig(
            input_keys=["features"],
            expert_width_multiplier=expert_width_multiplier,
            num_vlm_layers=num_vlm_layers,
            freeze_vlm=freeze_vlm,
        )
        assert config.expert_width_multiplier == expert_width_multiplier
        assert config.num_vlm_layers == num_vlm_layers
        assert config.freeze_vlm == freeze_vlm


@pytest.mark.unit
class TestPi0DecoderConfig:
    def test_defaults(self):
        config = Pi0DecoderConfig(input_keys=["features"])
        assert isinstance(config, DecodingNetworkConfig)
        assert (
            config._target_
            == "versatil.models.decoding.decoders.factory.pi0.Pi0Decoder"
        )
        assert config.time_conditioning == TimeConditioning.CONCAT_MLP.value
        assert config.normalization_type == NormalizationType.RMS_NORM.value
        assert config.dropout == 0.0
        assert config.proprioceptive_feature_key is None

    @pytest.mark.parametrize("expert_hidden_size", [512, 1024])
    @pytest.mark.parametrize("time_conditioning", ["concat_mlp", "adanorm"])
    def test_stores_configuration(self, expert_hidden_size, time_conditioning):
        config = Pi0DecoderConfig(
            input_keys=["features"],
            expert_hidden_size=expert_hidden_size,
            time_conditioning=time_conditioning,
        )
        assert config.expert_hidden_size == expert_hidden_size
        assert config.time_conditioning == time_conditioning


@pytest.mark.unit
class TestDecoderInstantiation:
    ACTION_LOGITS_HEAD = {
        DecoderOutputKey.ACTION_LOGITS.value: ActionHeadConfig(input_dim=256)
    }

    @pytest.mark.parametrize(
        "config, expected_class_name, action_heads",
        [
            (ACTConfig(input_keys=["f"]), "ACT", None),
            (
                DiscreteDETRActionTransformerConfig(input_keys=["f"]),
                "DiscreteDETRActionTransformer",
                ACTION_LOGITS_HEAD,
            ),
            (
                ActionTransformerConfig(
                    input_keys=["f"],
                    attention_type=AttentionType.MULTI_HEAD.value,
                ),
                "ActionTransformer",
                None,
            ),
            (
                MixtureOfDensitiesActionTransformerConfig(input_keys=["f"]),
                "MixtureOfDensitiesActionTransformer",
                None,
            ),
            (LACTConfig(input_keys=["f"], latent_dimension=32), "LACT", None),
            (
                GPTActionTransformerConfig(
                    input_keys=["f"],
                    attention_type=AttentionType.MULTI_HEAD.value,
                ),
                "GPTActionTransformer",
                ACTION_LOGITS_HEAD,
            ),
            (
                FreeActionTransformerConfig(
                    input_keys=["f"],
                    attention_type=AttentionType.MULTI_HEAD.value,
                ),
                "FreeActionTransformer",
                ACTION_LOGITS_HEAD,
            ),
            (
                DiTBlockActionTransformerConfig(input_keys=["f"]),
                "DiTBlockActionTransformer",
                None,
            ),
            (
                DiffusionActionTransformerConfig(input_keys=["f"]),
                "DiffusionActionTransformer",
                None,
            ),
            (
                ConditionalActionUNetConfig(input_keys=["f"]),
                "ConditionalActionUNet",
                None,
            ),
        ],
    )
    def test_instantiates_decoder(self, config, expected_class_name, action_heads):
        instance = _instantiate_decoder_config(config=config, action_heads=action_heads)
        assert type(instance).__name__ == expected_class_name
