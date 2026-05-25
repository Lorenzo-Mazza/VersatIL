"""Tests for versatil.configs.encoding.encoder module."""

import importlib

import pytest
from hydra.utils import instantiate
from omegaconf import MISSING

from versatil.configs.adaptation import LoRAAdaptationConfig
from versatil.configs.encoding.encoder import (
    ConditionalCNNEncoderConfig,
    DFormerEncoderConfig,
    DinoV2SigLIPRGBEncoderConfig,
    EncoderConfig,
    FlatRGBEncoderConfig,
    GeometricRGBDEncoderConfig,
    ImageEncoderConfig,
    LanguageEncoderConfig,
    ProprioEncoderConfig,
    SpatialDepthEncoderConfig,
    SpatialRGBEncoderConfig,
    VLMEncoderConfig,
)
from versatil.data.constants import Cameras
from versatil.models.encoding.encoders.constants import (
    BatchNormHandling,
    DinoV2SigLIPBackboneType,
    LanguageEncoderType,
    PoolingMethod,
)
from versatil.models.encoding.encoders.proprioceptive.base import ProprioceptiveEncoder
from versatil.models.layers.activation import ActivationFunction


@pytest.mark.unit
class TestEncoderConfig:
    def test_target_defaults_to_missing(self):
        config = EncoderConfig()
        assert config._target_ == MISSING

    def test_input_keys_defaults_to_missing(self):
        config = EncoderConfig()
        assert config.input_keys == MISSING

    @pytest.mark.parametrize("pretrained", [True, False])
    @pytest.mark.parametrize("frozen", [True, False])
    def test_stores_configuration(self, pretrained, frozen):
        config = EncoderConfig(
            input_keys=["left"], pretrained=pretrained, frozen=frozen
        )
        assert config.pretrained == pretrained
        assert config.frozen == frozen

    def test_model_dtype_defaults_to_experiment_precision_interpolation(self):
        config = EncoderConfig()
        assert config.model_dtype == "${experiment.precision}"


@pytest.mark.unit
class TestSpatialRGBEncoderConfig:
    def test_target_points_to_cnn_encoder(self):
        config = SpatialRGBEncoderConfig(
            input_keys=["left"], backbone="timm/resnet18.a1_in1k"
        )
        assert (
            config._target_
            == "versatil.models.encoding.encoders.rgb.spatial.SpatialRGBEncoder"
        )

    def test_pooling_method_default_is_none_string(self):
        config = SpatialRGBEncoderConfig(
            input_keys=["left"], backbone="timm/resnet18.a1_in1k"
        )
        assert config.pooling_method == PoolingMethod.NONE.value

    def test_batch_norm_handling_default_is_frozen_string(self):
        config = SpatialRGBEncoderConfig(
            input_keys=["left"], backbone="timm/resnet18.a1_in1k"
        )
        assert config.batch_norm_handling == BatchNormHandling.FROZEN.value

    def test_intermediate_layer_index_default_is_none(self):
        config = SpatialRGBEncoderConfig(
            input_keys=["left"], backbone="timm/resnet18.a1_in1k"
        )
        assert config.intermediate_layer_index is None

    def test_inherits_from_image_encoder_config(self):
        config = SpatialRGBEncoderConfig(
            input_keys=["left"], backbone="timm/resnet18.a1_in1k"
        )
        assert isinstance(config, ImageEncoderConfig)
        assert isinstance(config, EncoderConfig)

    def test_stores_lora_config(self):
        lora_config = LoRAAdaptationConfig(enabled=True)
        config = SpatialRGBEncoderConfig(
            input_keys=["left"],
            backbone="timm/resnet18.a1_in1k",
            lora_config=lora_config,
        )
        assert config.lora_config is lora_config


@pytest.mark.unit
class TestConditionalCNNEncoderConfig:
    def test_target_points_to_conditional_cnn_encoder(self):
        config = ConditionalCNNEncoderConfig(
            input_keys=["left"],
            backbone="timm/resnet18.a1_in1k",
            condition_key="language",
            condition_dim=512,
        )
        assert (
            config._target_
            == "versatil.models.encoding.encoders.rgb.conditional_cnn.ConditionalCNNEncoder"
        )

    def test_condition_key_and_dim_required(self):
        config = ConditionalCNNEncoderConfig(
            input_keys=["left"], backbone="timm/resnet18.a1_in1k"
        )
        assert config.condition_key == MISSING
        assert config.condition_dim == MISSING

    def test_inherits_from_image_encoder_config(self):
        config = ConditionalCNNEncoderConfig(
            input_keys=["left"],
            backbone="timm/resnet18.a1_in1k",
            condition_key="language",
            condition_dim=512,
        )
        assert isinstance(config, ImageEncoderConfig)

    def test_stores_lora_config(self):
        lora_config = LoRAAdaptationConfig(enabled=True)
        config = ConditionalCNNEncoderConfig(
            input_keys=["left"],
            backbone="timm/resnet18.a1_in1k",
            condition_key="language",
            condition_dim=512,
            lora_config=lora_config,
        )
        assert config.lora_config is lora_config


@pytest.mark.unit
class TestFlatRGBEncoderConfig:
    def test_target_points_to_flat_rgb_encoder(self):
        config = FlatRGBEncoderConfig(input_keys=["left"])
        assert (
            config._target_
            == "versatil.models.encoding.encoders.rgb.flat.FlatRGBEncoder"
        )

    def test_pooling_method_default_is_none_string(self):
        config = FlatRGBEncoderConfig(input_keys=["left"])
        assert config.pooling_method == PoolingMethod.NONE.value

    def test_stores_lora_config(self):
        lora_config = LoRAAdaptationConfig(enabled=True)
        config = FlatRGBEncoderConfig(
            input_keys=["left"],
            backbone="timm/vit_base_patch14_dinov2.lvd142m",
            lora_config=lora_config,
        )
        assert config.lora_config is lora_config


@pytest.mark.unit
class TestDinoV2SigLIPRGBEncoderConfig:
    def test_target_points_to_dinov2_siglip_rgb_encoder(self):
        config = DinoV2SigLIPRGBEncoderConfig(input_keys=["left"])
        assert (
            config._target_
            == "versatil.models.encoding.encoders.rgb.dinov2_siglip.DinoV2SigLIPRGBEncoder"
        )

    def test_backbone_default_is_dinov2_siglip_224_string(self):
        config = DinoV2SigLIPRGBEncoderConfig(input_keys=["left"])
        assert (
            config.backbone == DinoV2SigLIPBackboneType.DINOV2_SIGLIP_VIT_SO_224PX.value
        )

    def test_stores_lora_config(self):
        lora_config = LoRAAdaptationConfig(enabled=True)
        config = DinoV2SigLIPRGBEncoderConfig(
            input_keys=["left"],
            backbone=DinoV2SigLIPBackboneType.DINOV2_SIGLIP_VIT_SO_384PX.value,
            lora_config=lora_config,
        )
        assert config.lora_config is lora_config


@pytest.mark.unit
class TestSpatialDepthEncoderConfig:
    def test_target_points_to_depth_cnn_encoder(self):
        config = SpatialDepthEncoderConfig(input_keys=["depth"], backbone="resnet18")
        assert (
            config._target_
            == "versatil.models.encoding.encoders.depth.spatial.SpatialDepthEncoder"
        )

    def test_pooling_method_default_is_none_string(self):
        config = SpatialDepthEncoderConfig(input_keys=["depth"], backbone="resnet18")
        assert config.pooling_method == PoolingMethod.NONE.value

    def test_batch_norm_handling_default_is_frozen_string(self):
        config = SpatialDepthEncoderConfig(input_keys=["depth"], backbone="resnet18")
        assert config.batch_norm_handling == BatchNormHandling.FROZEN.value

    def test_intermediate_layer_index_default_is_none(self):
        config = SpatialDepthEncoderConfig(input_keys=["depth"], backbone="resnet18")
        assert config.intermediate_layer_index is None

    def test_stores_lora_config(self):
        lora_config = LoRAAdaptationConfig(enabled=True)
        config = SpatialDepthEncoderConfig(
            input_keys=["depth"],
            backbone="resnet18",
            lora_config=lora_config,
        )
        assert config.lora_config is lora_config


@pytest.mark.unit
class TestDFormerEncoderConfig:
    def test_target_points_to_dformer_encoder(self):
        config = DFormerEncoderConfig()
        assert (
            config._target_
            == "versatil.models.encoding.encoders.cross_modal.rgbd.dformerv2.DFormerEncoder"
        )

    def test_default_input_keys_include_left_and_depth(self):
        config = DFormerEncoderConfig()
        assert Cameras.LEFT.value in config.input_keys
        assert Cameras.DEPTH.value in config.input_keys


@pytest.mark.unit
class TestGeometricRGBDEncoderConfig:
    def test_target_points_to_light_geometric_encoder(self):
        config = GeometricRGBDEncoderConfig()
        assert (
            config._target_
            == "versatil.models.encoding.encoders.cross_modal.rgbd.geometric_rgbd.GeometricRGBDEncoder"
        )

    def test_default_input_keys_include_left_and_depth(self):
        config = GeometricRGBDEncoderConfig()
        assert Cameras.LEFT.value in config.input_keys
        assert Cameras.DEPTH.value in config.input_keys

    def test_pooling_method_default_is_average_string(self):
        config = GeometricRGBDEncoderConfig()
        assert config.pooling_method == PoolingMethod.AVERAGE.value


@pytest.mark.unit
class TestProprioEncoderConfig:
    def test_target_points_to_proprioceptive_encoder(self):
        config = ProprioEncoderConfig(input_keys=["proprio"])
        assert (
            config._target_
            == "versatil.models.encoding.encoders.proprioceptive.base.ProprioceptiveEncoder"
        )

    def test_activation_default_is_relu_string(self):
        config = ProprioEncoderConfig(input_keys=["proprio"])
        assert config.activation == ActivationFunction.RELU.value


@pytest.mark.unit
class TestVLMEncoderConfig:
    def test_target_points_to_vlm_encoder(self):
        config = VLMEncoderConfig(input_keys=["left"], model_name="clip")
        assert (
            config._target_
            == "versatil.models.encoding.encoders.cross_modal.vision_language.vlm_encoder.VLMEncoder"
        )

    def test_model_name_required(self):
        config = VLMEncoderConfig(input_keys=["left"])
        assert config.model_name == MISSING


@pytest.mark.unit
class TestImageEncoderConfig:
    def test_target_defaults_to_missing(self):
        config = ImageEncoderConfig()
        assert config._target_ == MISSING

    def test_backbone_defaults_to_missing(self):
        config = ImageEncoderConfig()
        assert config.backbone == MISSING

    def test_inherits_from_encoder_config(self):
        assert issubclass(ImageEncoderConfig, EncoderConfig)


@pytest.mark.unit
class TestLanguageEncoderConfig:
    def test_target_points_to_language_encoder(self):
        config = LanguageEncoderConfig()
        assert (
            config._target_
            == "versatil.models.encoding.encoders.language.language.LanguageEncoder"
        )

    def test_model_name_default_is_bert_base_string(self):
        config = LanguageEncoderConfig()
        assert config.model_name == LanguageEncoderType.BERT_BASE.value

    def test_does_not_inherit_from_encoder_config(self):
        config = LanguageEncoderConfig()
        assert not isinstance(config, EncoderConfig)

    def test_model_dtype_defaults_to_experiment_precision_interpolation(self):
        config = LanguageEncoderConfig()
        assert config.model_dtype == "${experiment.precision}"


@pytest.mark.unit
class TestEncoderInstantiation:
    def test_proprio_encoder_instantiates(self):
        config = ProprioEncoderConfig(
            input_keys=["proprio"],
            output_dim=64,
            pretrained=False,
            model_dtype=None,
        )
        instance = instantiate(config)
        assert isinstance(instance, ProprioceptiveEncoder)
        assert instance.output_dim == 64


@pytest.mark.integration
class TestEncoderTargetResolutionIntegration:
    @pytest.mark.parametrize(
        "config_class, expected_class_name",
        [
            (
                lambda: SpatialRGBEncoderConfig(
                    input_keys=["left"], backbone="timm/resnet18.a1_in1k"
                ),
                "SpatialRGBEncoder",
            ),
            (
                lambda: ConditionalCNNEncoderConfig(
                    input_keys=["left"],
                    backbone="timm/resnet18.a1_in1k",
                    condition_key="lang",
                    condition_dim=512,
                ),
                "ConditionalCNNEncoder",
            ),
            (
                lambda: SpatialDepthEncoderConfig(
                    input_keys=["depth"], backbone="resnet18"
                ),
                "SpatialDepthEncoder",
            ),
            (lambda: DFormerEncoderConfig(), "DFormerEncoder"),
            (lambda: GeometricRGBDEncoderConfig(), "GeometricRGBDEncoder"),
            (
                lambda: VLMEncoderConfig(input_keys=["left"], model_name="clip"),
                "VLMEncoder",
            ),
            (lambda: LanguageEncoderConfig(), "LanguageEncoder"),
        ],
    )
    def test_target_resolves_to_importable_class(
        self, config_class, expected_class_name
    ):
        config = config_class()
        target = config._target_
        module_path, class_name = target.rsplit(".", 1)
        module = importlib.import_module(module_path)
        assert hasattr(module, class_name)
        assert class_name == expected_class_name
