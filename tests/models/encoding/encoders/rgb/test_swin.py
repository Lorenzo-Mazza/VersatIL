"""Tests for versatil.models.encoding.encoders.rgb.swin module."""

import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from versatil.data.constants import RGB_CAMERAS
from versatil.data.metadata import BaseMetadata, CameraMetadata
from versatil.models.encoding.encoders.constants import (
    EncoderOutputKeys,
    PoolingMethod,
    SwinBackboneType,
    ViTBackboneType,
)
from versatil.models.encoding.encoders.rgb.swin import SwinEncoder

SWIN_BACKBONES = list(SwinBackboneType)
SWIN_VALID_BACKBONES = [e.value for e in SwinBackboneType]

FEATURE_DIM = 768
SPATIAL_HEIGHT = 7
SPATIAL_WIDTH = 7


def _mock_build_backbone(self):
    self.backbone = MagicMock()
    self.backbone.num_features = FEATURE_DIM
    self.backbone.patch_embed = MagicMock()
    self.backbone.patch_embed.img_size = (224, 224)
    self.backbone.patch_embed.patch_size = (4, 4)
    self.expected_image_size = (224, 224)
    self.patch_size = (4, 4)


@pytest.fixture
def swin_encoder_factory() -> Callable[..., SwinEncoder]:
    def factory(
        input_keys: str | list[str] = "left",
        backbone: str = SwinBackboneType.SWIN_TINY.value,
        pooling_method: str = PoolingMethod.AVERAGE.value,
        pretrained: bool = False,
        frozen: bool = False,
    ) -> SwinEncoder:
        with patch.object(SwinEncoder, "_build_backbone", _mock_build_backbone):
            return SwinEncoder(
                input_keys=input_keys,
                backbone=backbone,
                pooling_method=pooling_method,
                pretrained=pretrained,
                frozen=frozen,
            )

    return factory


@pytest.fixture
def mock_backbone_spatial_output_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    def factory(
        batch_size: int = 2,
        spatial_height: int = SPATIAL_HEIGHT,
        spatial_width: int = SPATIAL_WIDTH,
        feature_dim: int = FEATURE_DIM,
    ) -> torch.Tensor:
        # Swin outputs (B, H, W, C) channels-last
        return torch.from_numpy(
            rng.standard_normal(
                (batch_size, spatial_height, spatial_width, feature_dim)
            ).astype(np.float32)
        )

    return factory


class TestSwinEncoderInitialization:
    @pytest.mark.parametrize(
        "backbone, expectation",
        [
            (SwinBackboneType.SWIN_TINY.value, does_not_raise()),
            (SwinBackboneType.SWIN_BASE.value, does_not_raise()),
            (
                ViTBackboneType.DINOV2_VITB14.value,
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        f"Invalid backbone '{ViTBackboneType.DINOV2_VITB14.value}'. "
                        f"Must be one of: {SWIN_VALID_BACKBONES}"
                    ),
                ),
            ),
            (
                "invalid_backbone",
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        f"Invalid backbone 'invalid_backbone'. "
                        f"Must be one of: {SWIN_VALID_BACKBONES}"
                    ),
                ),
            ),
        ],
    )
    def test_backbone_validation(self, backbone: str, expectation):
        with (
            expectation,
            patch.object(SwinEncoder, "_build_backbone", _mock_build_backbone),
        ):
            SwinEncoder(
                input_keys="left",
                pretrained=False,
                frozen=False,
                pooling_method=PoolingMethod.AVERAGE.value,
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
        swin_encoder_factory: Callable[..., SwinEncoder],
        input_keys: str | list[str],
        expectation,
    ):
        with expectation:
            swin_encoder_factory(input_keys=input_keys)

    @pytest.mark.parametrize("input_keys", ["left", "right"])
    @pytest.mark.parametrize(
        "backbone",
        [SwinBackboneType.SWIN_TINY.value, SwinBackboneType.SWIN_BASE.value],
    )
    @pytest.mark.parametrize(
        "pooling_method",
        [PoolingMethod.AVERAGE.value, PoolingMethod.MAX.value],
    )
    def test_stores_configuration(
        self,
        swin_encoder_factory: Callable[..., SwinEncoder],
        input_keys: str | list[str],
        backbone: str,
        pooling_method: str,
    ):
        encoder = swin_encoder_factory(
            input_keys=input_keys,
            backbone=backbone,
            pooling_method=pooling_method,
        )
        expected_keys = [input_keys] if isinstance(input_keys, str) else input_keys
        assert encoder.backbone_name == backbone
        assert encoder.pooling_method == pooling_method
        assert encoder.feature_dim == FEATURE_DIM
        assert encoder.input_specification.keys == expected_keys

    def test_pooling_head_is_none_before_set_image_size(
        self,
        swin_encoder_factory: Callable[..., SwinEncoder],
    ):
        encoder = swin_encoder_factory()
        assert encoder.pooling_head is None


class TestSwinEncoderForward:
    def test_raises_without_set_image_size(
        self,
        swin_encoder_factory: Callable[..., SwinEncoder],
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        encoder = swin_encoder_factory()
        inputs = image_input_factory(batch_size=2)
        with pytest.raises(
            RuntimeError,
            match="pooling_head is not initialized. Call set_image_size\\(\\) before forward.",
        ):
            encoder(inputs)

    @pytest.mark.parametrize("time_steps", [1, 3])
    def test_output_shape_with_temporal_dimension(
        self,
        swin_encoder_factory: Callable[..., SwinEncoder],
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        mock_backbone_spatial_output_factory: Callable[..., torch.Tensor],
        time_steps: int,
    ):
        batch_size = 2
        encoder = swin_encoder_factory(pooling_method=PoolingMethod.AVERAGE.value)
        encoder._setup_pooling(
            spatial_height=SPATIAL_HEIGHT, spatial_width=SPATIAL_WIDTH
        )
        effective_batch = batch_size * time_steps
        backbone_output = mock_backbone_spatial_output_factory(
            batch_size=effective_batch
        )
        encoder.backbone.forward_features.return_value = backbone_output
        inputs = image_input_factory(
            batch_size=batch_size,
            time_steps=time_steps,
        )
        output = encoder(inputs)
        features = output[EncoderOutputKeys.RGB.value]
        assert features.shape == (batch_size, time_steps, FEATURE_DIM)

    def test_permutes_nhwc_to_nchw_before_pooling(
        self,
        swin_encoder_factory: Callable[..., SwinEncoder],
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        rng: np.random.Generator,
    ):
        batch_size = 2
        encoder = swin_encoder_factory(pooling_method=PoolingMethod.AVERAGE.value)
        encoder._setup_pooling(
            spatial_height=SPATIAL_HEIGHT, spatial_width=SPATIAL_WIDTH
        )
        # Create NHWC output with distinct values per channel
        nhwc_output = torch.from_numpy(
            rng.standard_normal(
                (batch_size, SPATIAL_HEIGHT, SPATIAL_WIDTH, FEATURE_DIM)
            ).astype(np.float32)
        )
        encoder.backbone.forward_features.return_value = nhwc_output
        inputs = image_input_factory(batch_size=batch_size)
        output = encoder(inputs)
        features = output[EncoderOutputKeys.RGB.value]
        # Average pooling over spatial dims: mean of NHWC → permute → mean(H,W)
        expected = nhwc_output.permute(0, 3, 1, 2).mean(dim=[2, 3])
        # Output has temporal dim (B, T=1, D)
        assert torch.allclose(features[:, 0], expected, atol=1e-6)


class TestSwinEncoderGetOutputSpecification:
    def test_returns_rgb_feature_with_correct_dimension(
        self,
        swin_encoder_factory: Callable[..., SwinEncoder],
    ):
        encoder = swin_encoder_factory()
        specification = encoder.get_output_specification()
        feature_keys = [m.key for m in specification]
        assert feature_keys == [EncoderOutputKeys.RGB.value]
        assert next(
            m for m in specification if m.key == EncoderOutputKeys.RGB.value
        ).dimension == (encoder.output_dim,)


class TestSwinEncoderValidateInputMetadata:
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
                MagicMock(spec=BaseMetadata),
                "Expected CameraMetadata for 'left', got MagicMock",
            ),
        ],
    )
    def test_validates_camera_metadata(
        self,
        swin_encoder_factory: Callable[..., SwinEncoder],
        metadata,
        expected_error: str | None,
    ):
        encoder = swin_encoder_factory()
        result = encoder.validate_input_metadata(key="left", metadata=metadata)
        assert result == expected_error


class TestSwinEncoderMultiCamera:
    @pytest.mark.parametrize(
        "input_keys, expected_feature_count, expected_multi_camera",
        [
            ("left", 1, False),
            (["left", "right"], 2, True),
        ],
    )
    def test_output_specification_scales_with_cameras(
        self,
        swin_encoder_factory: Callable[..., SwinEncoder],
        input_keys: str | list[str],
        expected_feature_count: int,
        expected_multi_camera: bool,
    ):
        encoder = swin_encoder_factory(input_keys=input_keys)
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


class TestSwinEncoderIntegration:
    @pytest.mark.integration
    @pytest.mark.parametrize("backbone", [b.value for b in SWIN_BACKBONES])
    def test_forward_pass_per_backbone(
        self,
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        backbone: str,
    ):
        batch_size = 2
        encoder = SwinEncoder(
            input_keys="left",
            backbone=backbone,
            pooling_method=PoolingMethod.AVERAGE.value,
            pretrained=False,
            frozen=False,
        )
        encoder.set_image_size(image_height=224, image_width=224)
        inputs = image_input_factory(batch_size=batch_size)
        output = encoder(inputs)
        features = output[EncoderOutputKeys.RGB.value]
        assert features.shape == (batch_size, 1, encoder.output_dim)

    @pytest.mark.integration
    def test_swin_buffers_loaded_correctly(self):
        encoder = SwinEncoder(
            input_keys="left",
            backbone=SwinBackboneType.SWIN_TINY.value,
            pooling_method=PoolingMethod.AVERAGE.value,
            pretrained=True,
            frozen=False,
        )
        attn_modules = [
            module
            for _, module in encoder.backbone.named_modules()
            if hasattr(module, "relative_position_index")
        ]
        assert len(attn_modules) > 0
        for module in attn_modules:
            index = module.relative_position_index
            table_size = module.relative_position_bias_table.shape[0]
            out_of_bounds = ((index >= table_size) | (index < 0)).sum().item()
            assert out_of_bounds == 0

    @pytest.mark.integration
    def test_set_image_size_creates_pooling_head(self):
        encoder = SwinEncoder(
            input_keys="left",
            backbone=SwinBackboneType.SWIN_TINY.value,
            pooling_method=PoolingMethod.AVERAGE.value,
            pretrained=False,
            frozen=False,
        )
        assert encoder.pooling_head is None
        encoder.set_image_size(image_height=224, image_width=224)
        assert encoder.pooling_head is not None
        assert encoder.output_dim == FEATURE_DIM

    @pytest.mark.integration
    def test_set_image_size_rebuilds_with_different_size(self):
        encoder = SwinEncoder(
            input_keys="left",
            backbone=SwinBackboneType.SWIN_TINY.value,
            pooling_method=PoolingMethod.AVERAGE.value,
            pretrained=False,
            frozen=False,
        )
        encoder.set_image_size(image_height=224, image_width=224)
        assert encoder.expected_image_size == (224, 224)
        encoder.set_image_size(image_height=256, image_width=256)
        assert encoder.expected_image_size == (256, 256)

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
        encoder = SwinEncoder(
            input_keys="left",
            backbone=SwinBackboneType.SWIN_TINY.value,
            pooling_method=PoolingMethod.AVERAGE.value,
            pretrained=False,
            frozen=frozen,
        )
        for parameter in encoder.parameters():
            assert parameter.requires_grad is expected_requires_grad
