"""Tests for versatil.models.encoding.encoders.depth.spatial module."""

import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock, patch

import pytest
import torch

from versatil.data.constants import CameraModality, Cameras
from versatil.data.metadata import BaseMetadata, CameraMetadata, DepthCameraMetadata
from versatil.models.adaptation.constants import LoRATargetModulePreset
from versatil.models.adaptation.lora import LoRAAdaptation
from versatil.models.encoding.encoders.constants import (
    BatchNormHandling,
    EncoderOutputKeys,
    PoolingMethod,
    SpatialBackboneType,
)
from versatil.models.encoding.encoders.depth.spatial import SpatialDepthEncoder
from versatil.models.encoding.explainability import (
    ActivationLayout,
    ExplanationTargetKind,
    resolve_timm_feature_info_layer,
)

CNN_BACKBONES = list(SpatialBackboneType)
SPATIAL_VALID_BACKBONES = [e.value for e in SpatialBackboneType]


def _mock_build_backbone(self, img_size: tuple[int, int] | None = None):
    """Side-effect to set self.backbone with expected attributes."""
    self.backbone = MagicMock()
    self.backbone.feature_info.channels.return_value = [64, 128, 256, 512]
    self.backbone.patch_embed = None


def _mock_setup_pooling(self, spatial_height: int, spatial_width: int):
    """Side-effect to create a mock pooling head with correct output dim."""
    self.pooling_head = MagicMock()
    self.pooling_head.return_value = torch.zeros(1, self.feature_dim)
    self.output_dim = self.feature_dim


@pytest.fixture
def spatial_depth_encoder_factory() -> Callable[..., SpatialDepthEncoder]:
    """Factory for SpatialDepthEncoder with mocked backbone."""

    def factory(
        input_keys: str | list[str] = Cameras.DEPTH.value,
        backbone: str = SpatialBackboneType.RESNET18.value,
        pooling_method: str = PoolingMethod.AVERAGE.value,
        batch_norm_handling: str = BatchNormHandling.FROZEN.value,
        intermediate_layer_index: int | None = None,
        pretrained: bool = False,
        frozen: bool = False,
        lora_config: LoRAAdaptation | None = None,
    ) -> SpatialDepthEncoder:
        with patch.object(SpatialDepthEncoder, "_build_backbone", _mock_build_backbone):
            return SpatialDepthEncoder(
                input_keys=input_keys,
                backbone=backbone,
                pooling_method=pooling_method,
                batch_norm_handling=batch_norm_handling,
                intermediate_layer_index=intermediate_layer_index,
                pretrained=pretrained,
                frozen=frozen,
                lora_config=lora_config,
            )

    return factory


def test_spatial_depth_encoder_exposes_layer4_gradcam_target(
    spatial_depth_encoder_factory: Callable[..., SpatialDepthEncoder],
):
    encoder = spatial_depth_encoder_factory()
    target_layer = MagicMock()
    encoder.backbone.layer4 = target_layer
    encoder.backbone.feature_info.module_name.return_value = "layer4"
    encoder.backbone.named_modules.return_value = [
        ("", encoder.backbone),
        ("layer4", target_layer),
    ]
    target = encoder.get_explainability_targets()[0]
    assert target.layer is target_layer
    assert target.target_kind == ExplanationTargetKind.SPATIAL_FEATURE_MAP.value
    assert target.activation_layout == ActivationLayout.NCHW.value


class TestSpatialDepthEncoderInitialization:
    @pytest.mark.parametrize(
        "backbone, expectation",
        [
            (SpatialBackboneType.RESNET18.value, does_not_raise()),
            (SpatialBackboneType.RESNET50.value, does_not_raise()),
            (
                "invalid_backbone",
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        f"Invalid backbone 'invalid_backbone'. Must be one of: {SPATIAL_VALID_BACKBONES}"
                    ),
                ),
            ),
        ],
    )
    def test_backbone_validation(
        self,
        backbone: str,
        expectation,
    ):
        with (
            expectation,
            patch.object(SpatialDepthEncoder, "_build_backbone", _mock_build_backbone),
        ):
            SpatialDepthEncoder(
                input_keys=Cameras.DEPTH.value,
                backbone=backbone,
            )

    @pytest.mark.parametrize(
        "input_keys",
        [
            Cameras.DEPTH.value,
            "left",
        ],
    )
    def test_input_keys_are_stored_without_observation_space_validation(
        self,
        spatial_depth_encoder_factory: Callable[..., SpatialDepthEncoder],
        input_keys: str | list[str],
    ):
        encoder = spatial_depth_encoder_factory(input_keys=input_keys)
        expected_keys = [input_keys] if isinstance(input_keys, str) else input_keys
        assert encoder.input_specification.keys == expected_keys
        assert encoder.input_specification.required_camera_modalities == [
            CameraModality.DEPTH
        ]

    @pytest.mark.parametrize("input_keys", [Cameras.DEPTH.value])
    @pytest.mark.parametrize(
        "backbone",
        [
            SpatialBackboneType.RESNET18.value,
            SpatialBackboneType.RESNET34.value,
        ],
    )
    @pytest.mark.parametrize(
        "pooling_method",
        [
            PoolingMethod.AVERAGE.value,
            PoolingMethod.NONE.value,
        ],
    )
    @pytest.mark.parametrize(
        "batch_norm_handling",
        [
            BatchNormHandling.FROZEN.value,
            BatchNormHandling.DEFAULT.value,
        ],
    )
    def test_stores_configuration(
        self,
        spatial_depth_encoder_factory: Callable[..., SpatialDepthEncoder],
        input_keys: str,
        backbone: str,
        pooling_method: str,
        batch_norm_handling: str,
    ):
        encoder = spatial_depth_encoder_factory(
            input_keys=input_keys,
            backbone=backbone,
            pooling_method=pooling_method,
            batch_norm_handling=batch_norm_handling,
        )
        expected_keys = [input_keys] if isinstance(input_keys, str) else input_keys
        assert encoder.backbone_name == backbone
        assert encoder.pooling_method == pooling_method
        assert encoder.batch_norm_handling == batch_norm_handling
        assert encoder.feature_dim == 512
        assert encoder.input_specification.keys == expected_keys
        assert encoder.intermediate_layer_index is None

    @pytest.mark.parametrize(
        "intermediate_layer_index, expected_feature_dim",
        [
            (None, 512),
            (-1, 512),
            (-2, 256),
            (1, 128),
        ],
    )
    def test_intermediate_layer_index_selects_feature_dimension(
        self,
        spatial_depth_encoder_factory: Callable[..., SpatialDepthEncoder],
        intermediate_layer_index: int | None,
        expected_feature_dim: int,
    ):
        encoder = spatial_depth_encoder_factory(
            intermediate_layer_index=intermediate_layer_index,
        )
        assert encoder.feature_dim == expected_feature_dim

    def test_input_specification_requires_depth_camera(
        self,
        spatial_depth_encoder_factory: Callable[..., SpatialDepthEncoder],
    ):
        encoder = spatial_depth_encoder_factory()
        assert encoder.input_specification.required_camera_modalities == [
            CameraModality.DEPTH
        ]

    @pytest.mark.unit
    def test_build_backbone_applies_lora_config(
        self,
        lora_passthrough: Callable[
            [torch.nn.Module, LoRAAdaptation | None, bool], torch.nn.Module
        ],
    ):
        lora_config = LoRAAdaptation(
            enabled=True,
            rank=2,
            alpha=4,
            target_modules=LoRATargetModulePreset.ALL_LINEAR.value,
        )
        backbone = MagicMock()
        backbone.feature_info.channels.return_value = [64, 128, 256, 512]

        with (
            patch(
                "versatil.models.encoding.encoders.spatial_backbone.timm.create_model",
                return_value=backbone,
            ),
            patch(
                "versatil.models.encoding.encoders.spatial_backbone.apply_lora_config",
                side_effect=lora_passthrough,
            ) as mock_apply_lora,
        ):
            encoder = SpatialDepthEncoder(
                input_keys=Cameras.DEPTH.value,
                backbone=SpatialBackboneType.RESNET18.value,
                pooling_method=PoolingMethod.AVERAGE.value,
                pretrained=False,
                frozen=False,
                lora_config=lora_config,
            )

        assert mock_apply_lora.call_args.kwargs["model"] is encoder.backbone
        assert mock_apply_lora.call_args.kwargs["lora_config"] is lora_config
        assert mock_apply_lora.call_args.kwargs["frozen"] is False


class TestSpatialDepthEncoderForward:
    @pytest.mark.parametrize("time_steps", [1, 3])
    def test_output_shape_with_temporal_dimension(
        self,
        spatial_depth_encoder_factory: Callable[..., SpatialDepthEncoder],
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        time_steps: int,
    ):
        batch_size = 2
        feature_dimension = 512
        encoder = spatial_depth_encoder_factory()
        mock_pooling = MagicMock()
        mock_pooling.return_value = torch.zeros(
            batch_size * time_steps,
            feature_dimension,
        )
        encoder.pooling_head = mock_pooling
        encoder.backbone.return_value = [
            torch.zeros(batch_size * time_steps, 512, 7, 7),
        ]
        inputs = image_input_factory(
            key=Cameras.DEPTH.value,
            channels=1,
            batch_size=batch_size,
            time_steps=time_steps,
        )
        output = encoder(inputs)
        features = output[EncoderOutputKeys.DEPTH.value]
        assert features.shape == (batch_size, time_steps, feature_dimension)

    def test_raises_when_pooling_head_not_initialized(
        self,
        spatial_depth_encoder_factory: Callable[..., SpatialDepthEncoder],
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        encoder = spatial_depth_encoder_factory()
        inputs = image_input_factory(key=Cameras.DEPTH.value, channels=1)
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "pooling_head is not initialized. Call set_image_size() before forward."
            ),
        ):
            encoder(inputs)

    def test_output_key_is_depth(
        self,
        spatial_depth_encoder_factory: Callable[..., SpatialDepthEncoder],
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        encoder = spatial_depth_encoder_factory()
        mock_pooling = MagicMock()
        mock_pooling.return_value = torch.zeros(2, 512)
        encoder.pooling_head = mock_pooling
        encoder.backbone.return_value = [torch.zeros(2, 512, 7, 7)]
        inputs = image_input_factory(key=Cameras.DEPTH.value, channels=1)
        output = encoder(inputs)
        assert EncoderOutputKeys.DEPTH.value in output
        assert EncoderOutputKeys.RGB.value not in output

    def test_uses_configured_intermediate_layer_output(
        self,
        spatial_depth_encoder_factory: Callable[..., SpatialDepthEncoder],
    ):
        encoder = spatial_depth_encoder_factory(intermediate_layer_index=1)
        selected_features = torch.ones(1, 128, 4, 4)
        encoder.backbone.return_value = [
            torch.zeros(1, 64, 8, 8),
            selected_features,
            torch.zeros(1, 256, 2, 2),
            torch.zeros(1, 512, 1, 1),
        ]
        mock_pooling = MagicMock()
        mock_pooling.return_value = torch.zeros(1, 128)
        encoder.pooling_head = mock_pooling

        encoder._encode_single_image(torch.zeros(1, 1, 64, 64))

        pooling_input = mock_pooling.call_args.args[0]
        torch.testing.assert_close(pooling_input, selected_features)


class TestSpatialDepthEncoderSetImageSize:
    def test_set_image_size_updates_output_dim(
        self,
        spatial_depth_encoder_factory: Callable[..., SpatialDepthEncoder],
    ):
        encoder = spatial_depth_encoder_factory()
        initial_output_dim = encoder.output_dim
        encoder.backbone.return_value = [torch.zeros(1, 512, 7, 7)]
        with patch.object(SpatialDepthEncoder, "_setup_pooling", _mock_setup_pooling):
            encoder.set_image_size(image_height=224, image_width=224)
        assert encoder.output_dim == initial_output_dim
        assert next(
            m
            for m in encoder.get_output_specification()
            if m.key == EncoderOutputKeys.DEPTH.value
        ).dimension == (encoder.output_dim,)


class TestSpatialDepthEncoderValidateInputMetadata:
    @pytest.mark.parametrize(
        "metadata, expected_error",
        [
            (
                DepthCameraMetadata(
                    camera_key="depth",
                    dtype="float32",
                    image_height=224,
                    image_width=224,
                ),
                None,
            ),
            (
                CameraMetadata(
                    camera_key="left",
                    dtype="uint8",
                    channels=3,
                    image_height=224,
                    image_width=224,
                ),
                None,
            ),
            (
                MagicMock(spec=BaseMetadata),
                "Expected CameraMetadata for 'depth', got MagicMock",
            ),
        ],
    )
    def test_validates_single_channel_camera_metadata(
        self,
        spatial_depth_encoder_factory: Callable[..., SpatialDepthEncoder],
        metadata,
        expected_error: str | None,
    ):
        encoder = spatial_depth_encoder_factory()
        result = encoder.validate_input_metadata(key="depth", metadata=metadata)
        assert result == expected_error


class TestSpatialDepthEncoderGetOutputSpecification:
    def test_returns_depth_feature_with_correct_dimension(
        self,
        spatial_depth_encoder_factory: Callable[..., SpatialDepthEncoder],
    ):
        encoder = spatial_depth_encoder_factory()
        specification = encoder.get_output_specification()
        feature_keys = [m.key for m in specification]
        assert feature_keys == [EncoderOutputKeys.DEPTH.value]
        assert next(
            m for m in specification if m.key == EncoderOutputKeys.DEPTH.value
        ).dimension == (encoder.output_dim,)


class TestSpatialDepthEncoderNHWCHandling:
    def test_permutes_nhwc_to_nchw_before_pooling(
        self,
        spatial_depth_encoder_factory: Callable[..., SpatialDepthEncoder],
    ):
        encoder = spatial_depth_encoder_factory()
        encoder._channels_last = True
        # (B, H, W, C) input — channels in last dim
        nhwc_features = torch.arange(24).reshape(1, 2, 3, 4).float()  # (1, 2, 3, 4)
        encoder.backbone.return_value = [nhwc_features]
        mock_pooling = MagicMock()
        mock_pooling.return_value = torch.zeros(1, 4)
        encoder.pooling_head = mock_pooling
        encoder._encode_single_image(torch.zeros(1, 1, 64, 64))
        pooling_input = mock_pooling.call_args[0][0]
        # After permute (B,H,W,C) -> (B,C,H,W): shape should be (1, 4, 2, 3)
        assert pooling_input.shape == (1, 4, 2, 3)
        torch.testing.assert_close(pooling_input, nhwc_features.permute(0, 3, 1, 2))

    def test_nchw_skips_permute(
        self,
        spatial_depth_encoder_factory: Callable[..., SpatialDepthEncoder],
    ):
        encoder = spatial_depth_encoder_factory()
        encoder._channels_last = False
        nchw_features = torch.arange(24).reshape(1, 4, 2, 3).float()  # (1, 4, 2, 3)
        encoder.backbone.return_value = [nchw_features]
        mock_pooling = MagicMock()
        mock_pooling.return_value = torch.zeros(1, 4)
        encoder.pooling_head = mock_pooling
        encoder._encode_single_image(torch.zeros(1, 1, 64, 64))
        pooling_input = mock_pooling.call_args[0][0]
        assert pooling_input.shape == (1, 4, 2, 3)
        torch.testing.assert_close(pooling_input, nchw_features)


class TestSpatialDepthEncoderStrictImageSize:
    def test_has_strict_image_size_true_when_patch_embed_strict(
        self,
        spatial_depth_encoder_factory: Callable[..., SpatialDepthEncoder],
    ):
        encoder = spatial_depth_encoder_factory()
        encoder.backbone.patch_embed = MagicMock(strict_img_size=True)
        assert encoder._has_strict_image_size() is True

    def test_has_strict_image_size_false_when_no_patch_embed(
        self,
        spatial_depth_encoder_factory: Callable[..., SpatialDepthEncoder],
    ):
        encoder = spatial_depth_encoder_factory()
        encoder.backbone.patch_embed = None
        assert encoder._has_strict_image_size() is False

    def test_has_strict_image_size_false_when_not_strict(
        self,
        spatial_depth_encoder_factory: Callable[..., SpatialDepthEncoder],
    ):
        encoder = spatial_depth_encoder_factory()
        encoder.backbone.patch_embed = MagicMock(strict_img_size=False)
        assert encoder._has_strict_image_size() is False


class TestSpatialDepthEncoderSetImageSizeDetection:
    def test_detects_nchw_layout(
        self,
        spatial_depth_encoder_factory: Callable[..., SpatialDepthEncoder],
    ):
        encoder = spatial_depth_encoder_factory()
        # Mock forward returns NCHW: (B, C=512, H=7, W=7)
        encoder.backbone.return_value = [torch.zeros(1, 512, 7, 7)]
        with patch.object(SpatialDepthEncoder, "_setup_pooling", _mock_setup_pooling):
            encoder.set_image_size(image_height=224, image_width=224)
        assert encoder._channels_last is False

    def test_detects_nhwc_layout(
        self,
        spatial_depth_encoder_factory: Callable[..., SpatialDepthEncoder],
    ):
        encoder = spatial_depth_encoder_factory()
        # Mock forward returns NHWC: (B, H=7, W=7, C=512)
        encoder.backbone.return_value = [torch.zeros(1, 7, 7, 512)]
        with patch.object(SpatialDepthEncoder, "_setup_pooling", _mock_setup_pooling):
            encoder.set_image_size(image_height=224, image_width=224)
        assert encoder._channels_last is True

    def test_raises_on_unrecognized_layout(
        self,
        spatial_depth_encoder_factory: Callable[..., SpatialDepthEncoder],
    ):
        encoder = spatial_depth_encoder_factory()
        # Mock forward returns shape where no dim matches expected channels (512)
        encoder.backbone.return_value = [torch.zeros(1, 256, 7, 7)]
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                f"Backbone '{SpatialBackboneType.RESNET18.value}' output shape "
                f"torch.Size([1, 256, 7, 7]) does not match expected channels "
                f"512 in either NCHW or NHWC layout."
            ),
        ):
            encoder.set_image_size(image_height=224, image_width=224)

    @pytest.mark.parametrize("frozen", [True, False])
    def test_strict_backbone_rebuilds_and_refreezes_on_set_image_size(
        self,
        spatial_depth_encoder_factory: Callable[..., SpatialDepthEncoder],
        frozen: bool,
    ):
        encoder = spatial_depth_encoder_factory(frozen=frozen)
        encoder.backbone.patch_embed = MagicMock(strict_img_size=True)
        encoder.backbone.return_value = [torch.zeros(1, 512, 8, 8)]

        def _rebuild_side_effect(img_size=None):
            _mock_build_backbone(encoder, img_size)
            encoder.backbone.return_value = [torch.zeros(1, 512, 8, 8)]

        mock_build = MagicMock(side_effect=_rebuild_side_effect)
        mock_freeze = MagicMock()
        with (
            patch.object(encoder, "_build_backbone", mock_build),
            patch.object(encoder, "_freeze_weights", mock_freeze),
            patch.object(SpatialDepthEncoder, "_setup_pooling", _mock_setup_pooling),
        ):
            encoder.set_image_size(image_height=256, image_width=256)
        mock_build.assert_called_once_with(img_size=(256, 256))
        if frozen:
            assert mock_freeze.call_count == 2
        else:
            mock_freeze.assert_not_called()


class TestSpatialDepthEncoderPoolingValidation:
    def test_rejects_incompatible_pooling_method(self):
        with (
            patch.object(
                PoolingMethod,
                "supports_spatial",
                new_callable=lambda: property(lambda self: self != PoolingMethod.MAX),
            ),
            pytest.raises(
                ValueError,
                match=re.escape(
                    f"Pooling method '{PoolingMethod.MAX.value}' is not compatible "
                    f"with spatial feature maps. Use one of: "
                    f"{[p.value for p in PoolingMethod if p.supports_spatial]}"
                ),
            ),
        ):
            SpatialDepthEncoder(
                input_keys=Cameras.DEPTH.value,
                backbone=SpatialBackboneType.RESNET18.value,
                pooling_method=PoolingMethod.MAX.value,
                pretrained=False,
            )


class TestSpatialDepthEncoderIntegration:
    @pytest.mark.integration
    @pytest.mark.parametrize("backbone", [b.value for b in CNN_BACKBONES])
    def test_exposes_real_gradcam_target_per_backbone(
        self,
        backbone: str,
    ):
        encoder = SpatialDepthEncoder(
            input_keys=Cameras.DEPTH.value,
            backbone=backbone,
            pooling_method=PoolingMethod.AVERAGE.value,
            batch_norm_handling=BatchNormHandling.DEFAULT.value,
            pretrained=False,
            frozen=False,
        )
        encoder.set_image_size(image_height=224, image_width=224)
        layer_index = encoder._resolve_intermediate_layer_index(
            intermediate_layer_index=encoder.intermediate_layer_index,
            output_count=len(encoder.backbone.feature_info.channels()),
        )
        expected_layer = resolve_timm_feature_info_layer(
            backbone=encoder.backbone,
            layer_index=layer_index,
        )

        target = encoder.get_explainability_targets()[0]

        assert target.layer is expected_layer
        assert target.target_kind == ExplanationTargetKind.SPATIAL_FEATURE_MAP.value
        assert target.activation_layout in {
            ActivationLayout.NCHW.value,
            ActivationLayout.NHWC.value,
        }

    @pytest.mark.integration
    @pytest.mark.parametrize("backbone", [b.value for b in CNN_BACKBONES])
    def test_forward_pass_per_backbone(
        self,
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        backbone: str,
    ):
        batch_size = 2
        encoder = SpatialDepthEncoder(
            input_keys=Cameras.DEPTH.value,
            backbone=backbone,
            pooling_method=PoolingMethod.AVERAGE.value,
            pretrained=False,
        )
        encoder.set_image_size(image_height=224, image_width=224)
        inputs = image_input_factory(
            key=Cameras.DEPTH.value,
            channels=1,
            batch_size=batch_size,
        )
        output = encoder(inputs)
        features = output[EncoderOutputKeys.DEPTH.value]
        assert features.shape == (batch_size, 1, encoder.output_dim)

    @pytest.mark.integration
    def test_lora_forward_pass_for_spatial_backbone(
        self,
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        parameter_count: Callable[[torch.nn.Module], int],
        trainable_parameter_count: Callable[[torch.nn.Module], int],
    ):
        lora_config = LoRAAdaptation(
            enabled=True,
            rank=2,
            alpha=4,
            target_modules=LoRATargetModulePreset.ALL_LINEAR.value,
        )
        encoder = SpatialDepthEncoder(
            input_keys=Cameras.DEPTH.value,
            backbone=SpatialBackboneType.DINOV3_CONVNEXT_SMALL.value,
            pooling_method=PoolingMethod.AVERAGE.value,
            batch_norm_handling=BatchNormHandling.DEFAULT.value,
            pretrained=False,
            frozen=False,
            lora_config=lora_config,
        )
        encoder.set_image_size(image_height=64, image_width=64)
        inputs = image_input_factory(
            key=Cameras.DEPTH.value,
            channels=1,
            batch_size=1,
            height=64,
            width=64,
        )
        output = encoder(inputs)
        trainable_parameter_names = [
            name
            for name, parameter in encoder.backbone.named_parameters()
            if parameter.requires_grad
        ]
        features = output[EncoderOutputKeys.DEPTH.value]
        trainable_parameters = trainable_parameter_count(encoder.backbone)
        total_parameters = parameter_count(encoder.backbone)
        assert features.shape == (1, 1, encoder.output_dim)
        assert trainable_parameter_names
        assert all("lora_" in name for name in trainable_parameter_names)
        assert 0 < trainable_parameters < total_parameters

    @pytest.mark.integration
    @pytest.mark.parametrize("time_steps", [1, 2])
    def test_temporal_reshaping(
        self,
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        time_steps: int,
    ):
        batch_size = 2
        encoder = SpatialDepthEncoder(
            input_keys=Cameras.DEPTH.value,
            backbone=SpatialBackboneType.RESNET18.value,
            pooling_method=PoolingMethod.AVERAGE.value,
            pretrained=False,
        )
        encoder.set_image_size(image_height=224, image_width=224)
        inputs = image_input_factory(
            key=Cameras.DEPTH.value,
            channels=1,
            batch_size=batch_size,
            time_steps=time_steps,
        )
        output = encoder(inputs)
        features = output[EncoderOutputKeys.DEPTH.value]
        assert features.shape == (batch_size, time_steps, encoder.output_dim)

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "batch_norm_handling",
        [
            BatchNormHandling.FROZEN.value,
            BatchNormHandling.DEFAULT.value,
            BatchNormHandling.CONVERT_TO_GROUPNORM.value,
        ],
    )
    def test_batch_norm_handling_variants(
        self,
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        batch_norm_handling: str,
    ):
        encoder = SpatialDepthEncoder(
            input_keys=Cameras.DEPTH.value,
            backbone=SpatialBackboneType.RESNET18.value,
            batch_norm_handling=batch_norm_handling,
            pretrained=False,
        )
        encoder.set_image_size(image_height=224, image_width=224)
        inputs = image_input_factory(
            key=Cameras.DEPTH.value,
            channels=1,
            batch_size=2,
        )
        output = encoder(inputs)
        assert EncoderOutputKeys.DEPTH.value in output

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "frozen, expected_requires_grad",
        [
            (False, True),
            (True, False),
        ],
    )
    def test_frozen_flag_controls_gradients(
        self,
        frozen: bool,
        expected_requires_grad: bool,
    ):
        encoder = SpatialDepthEncoder(
            input_keys=Cameras.DEPTH.value,
            backbone=SpatialBackboneType.RESNET18.value,
            pretrained=False,
            frozen=frozen,
        )
        for parameter in encoder.parameters():
            assert parameter.requires_grad is expected_requires_grad

    @pytest.mark.integration
    @pytest.mark.parametrize("frozen", [True, False])
    @pytest.mark.parametrize(
        "pooling_method",
        [PoolingMethod.AVERAGE.value, PoolingMethod.LEARNED_AGGREGATION.value],
    )
    def test_frozen_preserved_after_set_image_size(
        self,
        frozen: bool,
        pooling_method: str,
    ):
        encoder = SpatialDepthEncoder(
            input_keys=Cameras.DEPTH.value,
            backbone=SpatialBackboneType.RESNET18.value,
            pooling_method=pooling_method,
            pretrained=False,
            frozen=frozen,
        )
        encoder.set_image_size(image_height=224, image_width=224)
        for parameter in encoder.parameters():
            assert parameter.requires_grad is not frozen


class TestSpatialDepthEncoderBuildBackbone:
    def test_invalid_batch_norm_handling_raises(self):
        invalid_handling = "invalid_batch_norm_handling"
        with pytest.raises(
            ValueError,
            match=re.escape(f"Unknown batch norm handling: {invalid_handling}"),
        ):
            SpatialDepthEncoder(
                input_keys=Cameras.DEPTH.value,
                backbone=SpatialBackboneType.RESNET18.value,
                batch_norm_handling=invalid_handling,
                pretrained=False,
            )


def _real_depth_build_backbone(self, img_size: tuple[int, int] | None = None):
    """Side-effect installing a real nn.Conv2d (1-channel) so .to(dtype) has effect."""
    backbone = torch.nn.Conv2d(1, 16, kernel_size=3)
    backbone.feature_info = MagicMock()
    backbone.feature_info.channels.return_value = [16]
    backbone.patch_embed = None
    self.backbone = backbone


class TestSpatialDepthEncoderModelDtype:
    @pytest.mark.unit
    def test_apply_model_dtype_called_once_in_init(self):
        with (
            patch.object(SpatialDepthEncoder, "_build_backbone", _mock_build_backbone),
            patch.object(SpatialDepthEncoder, "_apply_model_dtype") as mock_apply,
        ):
            SpatialDepthEncoder(
                input_keys=Cameras.DEPTH.value,
                backbone=SpatialBackboneType.RESNET18.value,
                pretrained=False,
            )
        mock_apply.assert_called_once()

    @pytest.mark.unit
    def test_apply_model_dtype_called_again_in_set_image_size(self):
        with (
            patch.object(SpatialDepthEncoder, "_build_backbone", _mock_build_backbone),
            patch.object(SpatialDepthEncoder, "_apply_model_dtype") as mock_apply,
        ):
            encoder = SpatialDepthEncoder(
                input_keys=Cameras.DEPTH.value,
                backbone=SpatialBackboneType.RESNET18.value,
                pretrained=False,
            )
            mock_apply.reset_mock()
            encoder.backbone.return_value = [torch.zeros(1, 512, 7, 7)]
            with patch.object(
                SpatialDepthEncoder, "_setup_pooling", _mock_setup_pooling
            ):
                encoder.set_image_size(image_height=224, image_width=224)
        mock_apply.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "model_dtype, frozen, expected_dtype",
        [
            (None, False, torch.float32),
            ("32", False, torch.float32),
            ("bf16-mixed", True, torch.bfloat16),
            ("bf16-mixed", False, torch.float32),
        ],
    )
    def test_parameter_dtype_follows_precision_and_frozen_state(
        self,
        model_dtype: str | None,
        frozen: bool,
        expected_dtype: torch.dtype,
    ):
        with patch.object(
            SpatialDepthEncoder, "_build_backbone", _real_depth_build_backbone
        ):
            encoder = SpatialDepthEncoder(
                input_keys=Cameras.DEPTH.value,
                backbone=SpatialBackboneType.RESNET18.value,
                pooling_method=PoolingMethod.AVERAGE.value,
                batch_norm_handling=BatchNormHandling.DEFAULT.value,
                pretrained=False,
                frozen=frozen,
                model_dtype=model_dtype,
            )
        for parameter in encoder.parameters():
            assert parameter.dtype == expected_dtype

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "model_dtype, frozen, expected_dtype",
        [
            ("32", False, torch.float32),
            ("bf16-mixed", True, torch.bfloat16),
            ("bf16-mixed", False, torch.float32),
        ],
    )
    def test_backbone_rebuild_preserves_parameter_dtype(
        self,
        model_dtype: str,
        frozen: bool,
        expected_dtype: torch.dtype,
    ):
        with patch.object(
            SpatialDepthEncoder, "_build_backbone", _real_depth_build_backbone
        ):
            encoder = SpatialDepthEncoder(
                input_keys=Cameras.DEPTH.value,
                backbone=SpatialBackboneType.RESNET18.value,
                pooling_method=PoolingMethod.AVERAGE.value,
                batch_norm_handling=BatchNormHandling.DEFAULT.value,
                pretrained=False,
                frozen=frozen,
                model_dtype=model_dtype,
            )
            encoder._build_backbone(img_size=(224, 224))
            if frozen:
                encoder._freeze_weights()
            encoder._apply_model_dtype()
        for parameter in encoder.parameters():
            assert parameter.dtype == expected_dtype
