"""Tests for versatil.models.encoding.encoders.rgb.spatial module."""

import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock, patch

import pytest
import torch

from versatil.data.constants import RGB_CAMERAS
from versatil.data.metadata import BaseMetadata, CameraMetadata
from versatil.models.encoding.encoders.constants import (
    BatchNormHandling,
    EncoderOutputKeys,
    PoolingMethod,
    SpatialBackboneType,
)
from versatil.models.encoding.encoders.rgb.spatial import SpatialRGBEncoder

SPATIAL_BACKBONES = list(SpatialBackboneType)
SPATIAL_VALID_BACKBONES = [e.value for e in SpatialBackboneType]


def _mock_build_backbone(self, img_size: tuple[int, int] | None = None):
    """Side-effect to set self.backbone with expected attributes."""
    self.backbone = MagicMock()
    self.backbone.feature_info.channels.return_value = [64, 128, 256, 512]
    self.backbone.patch_embed = None


def _mock_setup_pooling(self, spatial_height: int, spatial_width: int):
    self.pooling_head = MagicMock()
    self.pooling_head.return_value = torch.zeros(1, self.feature_dim)
    self.output_dim = self.feature_dim


@pytest.fixture
def spatial_rgb_encoder_factory() -> Callable[..., SpatialRGBEncoder]:
    """Factory for SpatialRGBEncoder with mocked backbone."""

    def factory(
        input_keys: str | list[str] = "left",
        backbone: str = SpatialBackboneType.RESNET18.value,
        pooling_method: str = PoolingMethod.AVERAGE.value,
        batch_norm_handling: str = BatchNormHandling.FROZEN.value,
        pretrained: bool = False,
        frozen: bool = False,
    ) -> SpatialRGBEncoder:
        with patch.object(SpatialRGBEncoder, "_build_backbone", _mock_build_backbone):
            return SpatialRGBEncoder(
                input_keys=input_keys,
                backbone=backbone,
                pooling_method=pooling_method,
                batch_norm_handling=batch_norm_handling,
                pretrained=pretrained,
                frozen=frozen,
            )

    return factory


class TestSpatialRGBEncoderInitialization:
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
            patch.object(SpatialRGBEncoder, "_build_backbone", _mock_build_backbone),
        ):
            SpatialRGBEncoder(
                input_keys="left",
                backbone=backbone,
            )

    @pytest.mark.parametrize(
        "input_keys, expectation",
        [
            ("left", does_not_raise()),
            ("right", does_not_raise()),
            (["left", "right"], does_not_raise()),
            (
                "invalid_camera",
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        f"At least one from {RGB_CAMERAS} required, got {set()}"
                    ),
                ),
            ),
        ],
    )
    def test_input_keys_validation(
        self,
        spatial_rgb_encoder_factory: Callable[..., SpatialRGBEncoder],
        input_keys: str | list[str],
        expectation,
    ):
        with expectation:
            spatial_rgb_encoder_factory(input_keys=input_keys)

    @pytest.mark.parametrize("input_keys", ["left", "right"])
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
        spatial_rgb_encoder_factory: Callable[..., SpatialRGBEncoder],
        input_keys: str,
        backbone: str,
        pooling_method: str,
        batch_norm_handling: str,
    ):
        encoder = spatial_rgb_encoder_factory(
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


class TestSpatialRGBEncoderForward:
    @pytest.mark.parametrize("time_steps", [1, 3])
    def test_output_shape_with_temporal_dimension(
        self,
        spatial_rgb_encoder_factory: Callable[..., SpatialRGBEncoder],
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        time_steps: int,
    ):
        batch_size = 2
        feature_dimension = 512
        encoder = spatial_rgb_encoder_factory()
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
            batch_size=batch_size,
            time_steps=time_steps,
        )
        output = encoder(inputs)
        features = output[EncoderOutputKeys.RGB.value]
        assert features.shape == (batch_size, time_steps, feature_dimension)

    def test_raises_when_pooling_head_not_initialized(
        self,
        spatial_rgb_encoder_factory: Callable[..., SpatialRGBEncoder],
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        encoder = spatial_rgb_encoder_factory()
        inputs = image_input_factory()
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "pooling_head is not initialized. Call set_image_size() before forward."
            ),
        ):
            encoder(inputs)


class TestSpatialRGBEncoderGetOutputSpecification:
    def test_returns_rgb_feature_with_correct_dimension(
        self,
        spatial_rgb_encoder_factory: Callable[..., SpatialRGBEncoder],
    ):
        encoder = spatial_rgb_encoder_factory()
        specification = encoder.get_output_specification()
        feature_keys = [m.key for m in specification]
        assert feature_keys == [EncoderOutputKeys.RGB.value]
        assert next(
            m for m in specification if m.key == EncoderOutputKeys.RGB.value
        ).dimension == (encoder.output_dim,)


class TestSpatialRGBEncoderSetImageSize:
    def test_set_image_size_updates_output_dim(
        self,
        spatial_rgb_encoder_factory: Callable[..., SpatialRGBEncoder],
    ):
        encoder = spatial_rgb_encoder_factory()
        initial_output_dim = encoder.output_dim
        encoder.backbone.return_value = [torch.zeros(1, 512, 7, 7)]
        with patch.object(SpatialRGBEncoder, "_setup_pooling", _mock_setup_pooling):
            encoder.set_image_size(image_height=224, image_width=224)
        assert encoder.output_dim == initial_output_dim  # mock keeps feature_dim
        assert next(
            m
            for m in encoder.get_output_specification()
            if m.key == EncoderOutputKeys.RGB.value
        ).dimension == (encoder.output_dim,)


class TestSpatialRGBEncoderValidateInputMetadata:
    @pytest.mark.parametrize(
        "metadata, expected_error",
        [
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
                CameraMetadata(
                    camera_key="depth",
                    dtype="uint8",
                    channels=1,
                    image_height=224,
                    image_width=224,
                ),
                "Expected 3-channel RGB for 'left', got 1 channels",
            ),
            (
                MagicMock(spec=BaseMetadata),
                "Expected CameraMetadata for 'left', got MagicMock",
            ),
        ],
    )
    def test_validates_rgb_camera_metadata(
        self,
        spatial_rgb_encoder_factory: Callable[..., SpatialRGBEncoder],
        metadata,
        expected_error: str | None,
    ):
        encoder = spatial_rgb_encoder_factory()
        result = encoder.validate_input_metadata(key="left", metadata=metadata)
        assert result == expected_error


class TestSpatialRGBEncoderMultiCamera:
    @pytest.mark.parametrize(
        "input_keys, expected_feature_count, expected_multi_camera",
        [
            ("left", 1, False),
            (["left", "right"], 2, True),
        ],
    )
    def test_output_specification_scales_with_cameras(
        self,
        spatial_rgb_encoder_factory: Callable[..., SpatialRGBEncoder],
        input_keys: str | list[str],
        expected_feature_count: int,
        expected_multi_camera: bool,
    ):
        encoder = spatial_rgb_encoder_factory(input_keys=input_keys)
        specification = encoder.get_output_specification()
        feature_keys = [m.key for m in specification]
        assert len(feature_keys) == expected_feature_count
        assert encoder.is_multi_camera is expected_multi_camera
        if expected_multi_camera:
            camera_list = input_keys if isinstance(input_keys, list) else [input_keys]
            for camera_key in camera_list:
                feature_name = f"{EncoderOutputKeys.RGB.value}.{camera_key}"
                assert feature_name in feature_keys
        else:
            assert feature_keys == [EncoderOutputKeys.RGB.value]

    def test_multi_camera_forward_produces_per_camera_features(
        self,
        spatial_rgb_encoder_factory: Callable[..., SpatialRGBEncoder],
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 2
        feature_dimension = 512
        encoder = spatial_rgb_encoder_factory(input_keys=["left", "right"])
        mock_pooling = MagicMock()
        mock_pooling.return_value = torch.zeros(batch_size, feature_dimension)
        encoder.pooling_head = mock_pooling
        encoder.backbone.return_value = [
            torch.zeros(batch_size, 512, 7, 7),
        ]
        inputs = {
            **image_input_factory(key="left", batch_size=batch_size),
            **image_input_factory(key="right", batch_size=batch_size),
        }
        output = encoder(inputs)
        rgb = EncoderOutputKeys.RGB.value
        assert f"{rgb}.left" in output
        assert f"{rgb}.right" in output
        assert output[f"{rgb}.left"].shape[0] == batch_size
        assert output[f"{rgb}.right"].shape[0] == batch_size

    def test_multi_camera_backbone_called_per_camera(
        self,
        spatial_rgb_encoder_factory: Callable[..., SpatialRGBEncoder],
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 2
        encoder = spatial_rgb_encoder_factory(input_keys=["left", "right"])
        mock_pooling = MagicMock()
        mock_pooling.return_value = torch.zeros(batch_size, 512)
        encoder.pooling_head = mock_pooling
        encoder.backbone.return_value = [
            torch.zeros(batch_size, 512, 7, 7),
        ]
        inputs = {
            **image_input_factory(key="left", batch_size=batch_size),
            **image_input_factory(key="right", batch_size=batch_size),
        }
        encoder(inputs)
        assert encoder.backbone.call_count == 2


class TestSpatialRGBEncoderIntegration:
    @pytest.mark.integration
    @pytest.mark.parametrize("backbone", [b.value for b in SPATIAL_BACKBONES])
    def test_forward_pass_per_backbone(
        self,
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        backbone: str,
    ):
        batch_size = 2
        encoder = SpatialRGBEncoder(
            input_keys="left",
            backbone=backbone,
            pooling_method=PoolingMethod.AVERAGE.value,
            pretrained=False,
        )
        encoder.set_image_size(image_height=224, image_width=224)
        inputs = image_input_factory(batch_size=batch_size)
        output = encoder(inputs)
        features = output[EncoderOutputKeys.RGB.value]
        assert features.shape == (batch_size, 1, encoder.output_dim)

    @pytest.mark.integration
    @pytest.mark.parametrize("time_steps", [1, 2])
    def test_temporal_reshaping(
        self,
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        time_steps: int,
    ):
        batch_size = 2
        encoder = SpatialRGBEncoder(
            input_keys="left",
            backbone=SpatialBackboneType.RESNET18.value,
            pooling_method=PoolingMethod.AVERAGE.value,
            pretrained=False,
        )
        encoder.set_image_size(image_height=224, image_width=224)
        inputs = image_input_factory(
            batch_size=batch_size,
            time_steps=time_steps,
        )
        output = encoder(inputs)
        features = output[EncoderOutputKeys.RGB.value]
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
        encoder = SpatialRGBEncoder(
            input_keys="left",
            backbone=SpatialBackboneType.RESNET18.value,
            batch_norm_handling=batch_norm_handling,
            pretrained=False,
        )
        encoder.set_image_size(image_height=224, image_width=224)
        inputs = image_input_factory(batch_size=2)
        output = encoder(inputs)
        assert EncoderOutputKeys.RGB.value in output

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
        encoder = SpatialRGBEncoder(
            input_keys="left",
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
        encoder = SpatialRGBEncoder(
            input_keys="left",
            backbone=SpatialBackboneType.RESNET18.value,
            pooling_method=pooling_method,
            pretrained=False,
            frozen=frozen,
        )
        encoder.set_image_size(image_height=224, image_width=224)
        for parameter in encoder.parameters():
            assert parameter.requires_grad is not frozen


class TestSpatialRGBEncoderNHWCHandling:
    def test_permutes_nhwc_to_nchw_before_pooling(
        self,
        spatial_rgb_encoder_factory: Callable[..., SpatialRGBEncoder],
    ):
        encoder = spatial_rgb_encoder_factory()
        encoder._channels_last = True
        # (B, H, W, C) input — channels in last dim
        nhwc_features = torch.arange(24).reshape(1, 2, 3, 4).float()  # (1, 2, 3, 4)
        encoder.backbone.return_value = [nhwc_features]
        mock_pooling = MagicMock()
        mock_pooling.return_value = torch.zeros(1, 4)
        encoder.pooling_head = mock_pooling
        encoder._encode_single_image(torch.zeros(1, 3, 64, 64))
        pooling_input = mock_pooling.call_args[0][0]
        # After permute (B,H,W,C) → (B,C,H,W): shape should be (1, 4, 2, 3)
        assert pooling_input.shape == (1, 4, 2, 3)
        torch.testing.assert_close(pooling_input, nhwc_features.permute(0, 3, 1, 2))

    def test_nchw_skips_permute(
        self,
        spatial_rgb_encoder_factory: Callable[..., SpatialRGBEncoder],
    ):
        encoder = spatial_rgb_encoder_factory()
        encoder._channels_last = False
        nchw_features = torch.arange(24).reshape(1, 4, 2, 3).float()  # (1, 4, 2, 3)
        encoder.backbone.return_value = [nchw_features]
        mock_pooling = MagicMock()
        mock_pooling.return_value = torch.zeros(1, 4)
        encoder.pooling_head = mock_pooling
        encoder._encode_single_image(torch.zeros(1, 3, 64, 64))
        pooling_input = mock_pooling.call_args[0][0]
        assert pooling_input.shape == (1, 4, 2, 3)
        torch.testing.assert_close(pooling_input, nchw_features)


class TestSpatialRGBEncoderStrictImageSize:
    def test_has_strict_image_size_true_when_patch_embed_strict(
        self,
        spatial_rgb_encoder_factory: Callable[..., SpatialRGBEncoder],
    ):
        encoder = spatial_rgb_encoder_factory()
        encoder.backbone.patch_embed = MagicMock(strict_img_size=True)
        assert encoder._has_strict_image_size() is True

    def test_has_strict_image_size_false_when_no_patch_embed(
        self,
        spatial_rgb_encoder_factory: Callable[..., SpatialRGBEncoder],
    ):
        encoder = spatial_rgb_encoder_factory()
        encoder.backbone.patch_embed = None
        assert encoder._has_strict_image_size() is False

    def test_has_strict_image_size_false_when_not_strict(
        self,
        spatial_rgb_encoder_factory: Callable[..., SpatialRGBEncoder],
    ):
        encoder = spatial_rgb_encoder_factory()
        encoder.backbone.patch_embed = MagicMock(strict_img_size=False)
        assert encoder._has_strict_image_size() is False


class TestSpatialRGBEncoderSetImageSizeDetection:
    def test_detects_nchw_layout(
        self,
        spatial_rgb_encoder_factory: Callable[..., SpatialRGBEncoder],
    ):
        encoder = spatial_rgb_encoder_factory()
        # Mock forward returns NCHW: (B, C=512, H=7, W=7)
        encoder.backbone.return_value = [torch.zeros(1, 512, 7, 7)]
        with patch.object(SpatialRGBEncoder, "_setup_pooling", _mock_setup_pooling):
            encoder.set_image_size(image_height=224, image_width=224)
        assert encoder._channels_last is False

    def test_detects_nhwc_layout(
        self,
        spatial_rgb_encoder_factory: Callable[..., SpatialRGBEncoder],
    ):
        encoder = spatial_rgb_encoder_factory()
        # Mock forward returns NHWC: (B, H=7, W=7, C=512)
        encoder.backbone.return_value = [torch.zeros(1, 7, 7, 512)]
        with patch.object(SpatialRGBEncoder, "_setup_pooling", _mock_setup_pooling):
            encoder.set_image_size(image_height=224, image_width=224)
        assert encoder._channels_last is True

    def test_raises_on_unrecognized_layout(
        self,
        spatial_rgb_encoder_factory: Callable[..., SpatialRGBEncoder],
    ):
        encoder = spatial_rgb_encoder_factory()
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
        spatial_rgb_encoder_factory: Callable[..., SpatialRGBEncoder],
        frozen: bool,
    ):
        encoder = spatial_rgb_encoder_factory(frozen=frozen)
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
            patch.object(SpatialRGBEncoder, "_setup_pooling", _mock_setup_pooling),
        ):
            encoder.set_image_size(image_height=256, image_width=256)
        mock_build.assert_called_once_with(img_size=(256, 256))
        if frozen:
            assert mock_freeze.call_count == 2
        else:
            mock_freeze.assert_not_called()


class TestSpatialRGBEncoderPoolingValidation:
    def test_rejects_incompatible_pooling_method(self):
        # Currently all methods support spatial, so we test the mechanism
        # by temporarily making one return False
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
            SpatialRGBEncoder(
                input_keys="left",
                backbone=SpatialBackboneType.RESNET18.value,
                pooling_method=PoolingMethod.MAX.value,
                pretrained=False,
            )


class TestSpatialRGBEncoderBuildBackbone:
    def test_invalid_batch_norm_handling_raises(self):
        invalid_handling = "invalid_batch_norm_handling"
        with pytest.raises(
            ValueError,
            match=re.escape(f"Unknown batch norm handling: {invalid_handling}"),
        ):
            SpatialRGBEncoder(
                input_keys="left",
                backbone=SpatialBackboneType.RESNET18.value,
                batch_norm_handling=invalid_handling,
                pretrained=False,
            )
