"""Tests for versatil.models.encoding.encoders.rgb.vit module."""

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
    CNNBackboneType,
    EncoderOutputKeys,
    PoolingMethod,
    SwinBackboneType,
    ViTBackboneType,
)
from versatil.models.encoding.encoders.rgb.vit import ViTEncoder

VIT_BACKBONES = list(ViTBackboneType)
VIT_VALID_BACKBONES = [e.value for e in ViTBackboneType]

FEATURE_DIM = 768
SEQUENCE_LENGTH = 196


def _mock_build_backbone(self):
    """Side-effect to set self.backbone with expected attributes."""
    self.backbone = MagicMock()
    self.backbone.num_features = FEATURE_DIM
    self.expected_image_size = None
    self.requires_strict_image_size = False
    self.patch_size = None


@pytest.fixture
def vit_encoder_factory() -> Callable[..., ViTEncoder]:
    """Factory for ViTEncoder with mocked backbone."""

    def factory(
        input_keys: str | list[str] = "left",
        backbone: str = ViTBackboneType.DINOV2_VITB14.value,
        pooling_method: str = PoolingMethod.DEFAULT.value,
        pretrained: bool = False,
        frozen: bool = False,
    ) -> ViTEncoder:
        with patch.object(ViTEncoder, "_build_backbone", _mock_build_backbone):
            return ViTEncoder(
                input_keys=input_keys,
                backbone=backbone,
                pooling_method=pooling_method,
                pretrained=pretrained,
                frozen=frozen,
            )

    return factory


@pytest.fixture
def mock_backbone_output_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for mock backbone output tensor (last_hidden_state)."""

    def factory(
        batch_size: int = 2,
        sequence_length: int = SEQUENCE_LENGTH + 1,
        feature_dim: int = FEATURE_DIM,
    ) -> torch.Tensor:
        return torch.from_numpy(
            rng.standard_normal((batch_size, sequence_length, feature_dim)).astype(
                np.float32
            )
        )

    return factory


class TestViTEncoderInitialization:
    @pytest.mark.parametrize(
        "backbone, expectation",
        [
            (ViTBackboneType.DINOV2_VITB14.value, does_not_raise()),
            (ViTBackboneType.VIT_BASE.value, does_not_raise()),
            (ViTBackboneType.DINOV2_VITS14.value, does_not_raise()),
            (
                CNNBackboneType.RESNET18.value,
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        f"Invalid backbone '{CNNBackboneType.RESNET18.value}'. "
                        f"Must be one of: {VIT_VALID_BACKBONES}"
                    ),
                ),
            ),
            (
                SwinBackboneType.SWIN_TINY.value,
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        f"Invalid backbone '{SwinBackboneType.SWIN_TINY.value}'. "
                        f"Must be one of: {VIT_VALID_BACKBONES}"
                    ),
                ),
            ),
            (
                "invalid_backbone",
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        f"Invalid backbone 'invalid_backbone'. "
                        f"Must be one of: {VIT_VALID_BACKBONES}"
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
            patch.object(ViTEncoder, "_build_backbone", _mock_build_backbone),
        ):
            ViTEncoder(
                input_keys="left",
                pretrained=False,
                frozen=False,
                pooling_method=PoolingMethod.DEFAULT.value,
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
        vit_encoder_factory: Callable[..., ViTEncoder],
        input_keys: str | list[str],
        expectation,
    ):
        with expectation:
            vit_encoder_factory(input_keys=input_keys)

    @pytest.mark.parametrize("input_keys", ["left", "right"])
    @pytest.mark.parametrize(
        "backbone",
        [
            ViTBackboneType.DINOV2_VITS14.value,
            ViTBackboneType.DINOV2_VITB14.value,
        ],
    )
    @pytest.mark.parametrize(
        "pooling_method",
        [
            PoolingMethod.DEFAULT.value,
            PoolingMethod.NONE.value,
        ],
    )
    def test_stores_configuration(
        self,
        vit_encoder_factory: Callable[..., ViTEncoder],
        input_keys: str | list[str],
        backbone: str,
        pooling_method: str,
    ):
        encoder = vit_encoder_factory(
            input_keys=input_keys,
            backbone=backbone,
            pooling_method=pooling_method,
        )
        expected_keys = [input_keys] if isinstance(input_keys, str) else input_keys
        assert encoder.backbone_name == backbone
        assert encoder.pooling_method == pooling_method
        assert encoder.feature_dim == FEATURE_DIM
        assert encoder.input_specification.keys == expected_keys

    def test_none_pooling_sets_output_dim_to_tuple(self):
        with patch.object(ViTEncoder, "_build_backbone", _mock_build_backbone):
            encoder = ViTEncoder(
                input_keys="left",
                backbone=ViTBackboneType.DINOV2_VITB14.value,
                pooling_method=PoolingMethod.NONE.value,
                pretrained=False,
                frozen=False,
            )
        assert encoder.output_dim == (-1, FEATURE_DIM)

    def test_non_none_pooling_sets_output_dim_to_int(self):
        with patch.object(ViTEncoder, "_build_backbone", _mock_build_backbone):
            encoder = ViTEncoder(
                input_keys="left",
                backbone=ViTBackboneType.DINOV2_VITB14.value,
                pooling_method=PoolingMethod.DEFAULT.value,
                pretrained=False,
                frozen=False,
            )
        assert encoder.output_dim == FEATURE_DIM


class TestViTEncoderForward:
    @pytest.mark.parametrize("time_steps", [1, 3])
    def test_output_shape_with_temporal_dimension(
        self,
        vit_encoder_factory: Callable[..., ViTEncoder],
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        mock_backbone_output_factory: Callable[..., torch.Tensor],
        time_steps: int,
    ):
        batch_size = 2
        encoder = vit_encoder_factory(pooling_method=PoolingMethod.DEFAULT.value)
        effective_batch = batch_size * time_steps
        backbone_output = mock_backbone_output_factory(batch_size=effective_batch)
        encoder.backbone.forward_features.return_value = backbone_output
        inputs = image_input_factory(
            batch_size=batch_size,
            time_steps=time_steps,
        )
        output = encoder(inputs)
        features = output[EncoderOutputKeys.RGB.value]
        assert features.shape == (batch_size, time_steps, FEATURE_DIM)

    def test_none_pooling_output_shape_with_time(
        self,
        vit_encoder_factory: Callable[..., ViTEncoder],
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        mock_backbone_output_factory: Callable[..., torch.Tensor],
    ):
        batch_size = 2
        time_steps = 3
        encoder = vit_encoder_factory(pooling_method=PoolingMethod.NONE.value)
        effective_batch = batch_size * time_steps
        backbone_output = mock_backbone_output_factory(batch_size=effective_batch)
        encoder.backbone.forward_features.return_value = backbone_output
        inputs = image_input_factory(
            batch_size=batch_size,
            time_steps=time_steps,
        )
        output = encoder(inputs)
        features = output[EncoderOutputKeys.RGB.value]
        assert features.shape == (batch_size, time_steps, SEQUENCE_LENGTH, FEATURE_DIM)


class TestViTEncoderGetOutputSpecification:
    def test_returns_rgb_feature_with_correct_dimension(
        self,
        vit_encoder_factory: Callable[..., ViTEncoder],
    ):
        encoder = vit_encoder_factory()
        specification = encoder.get_output_specification()
        feature_keys = [m.key for m in specification]
        assert feature_keys == [EncoderOutputKeys.RGB.value]
        output_dim = encoder.output_dim
        expected_dim = output_dim if isinstance(output_dim, tuple) else (output_dim,)
        assert (
            next(
                m for m in specification if m.key == EncoderOutputKeys.RGB.value
            ).dimension
            == expected_dim
        )


class TestViTEncoderBuildBackbone:
    @pytest.mark.integration
    def test_fixed_input_size_models_have_strict_image_size(self):
        encoder = ViTEncoder(
            input_keys="left",
            backbone=ViTBackboneType.DINOV2_VITS14.value,
            pooling_method=PoolingMethod.NONE.value,
            pretrained=False,
            frozen=False,
        )
        assert encoder.requires_strict_image_size is True
        assert encoder.expected_image_size is not None

    @pytest.mark.integration
    def test_set_image_size_rebuilds_backbone(self):
        encoder = ViTEncoder(
            input_keys="left",
            backbone=ViTBackboneType.DINOV2_VITS14.value,
            pooling_method=PoolingMethod.DEFAULT.value,
            pretrained=False,
            frozen=False,
        )
        original_size = encoder.expected_image_size
        encoder.set_image_size(image_height=256, image_width=256)
        assert encoder.expected_image_size == (256, 256)
        assert encoder.expected_image_size != original_size


class TestViTEncoderValidateInputMetadata:
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
        vit_encoder_factory: Callable[..., ViTEncoder],
        metadata,
        expected_error: str | None,
    ):
        encoder = vit_encoder_factory()
        result = encoder.validate_input_metadata(key="left", metadata=metadata)
        assert result == expected_error


class TestViTEncoderMultiCamera:
    @pytest.mark.parametrize(
        "input_keys, expected_feature_count, expected_multi_camera",
        [
            ("left", 1, False),
            (["left", "right"], 2, True),
        ],
    )
    def test_output_specification_scales_with_cameras(
        self,
        vit_encoder_factory: Callable[..., ViTEncoder],
        input_keys: str | list[str],
        expected_feature_count: int,
        expected_multi_camera: bool,
    ):
        encoder = vit_encoder_factory(input_keys=input_keys)
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
        vit_encoder_factory: Callable[..., ViTEncoder],
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        mock_backbone_output_factory: Callable[..., torch.Tensor],
    ):
        batch_size = 2
        encoder = vit_encoder_factory(
            input_keys=["left", "right"],
            pooling_method=PoolingMethod.DEFAULT.value,
        )
        backbone_output = mock_backbone_output_factory(batch_size=batch_size)
        encoder.backbone.forward_features.return_value = backbone_output
        inputs = {
            **image_input_factory(key="left", batch_size=batch_size),
            **image_input_factory(key="right", batch_size=batch_size),
        }
        output = encoder(inputs)
        rgb = EncoderOutputKeys.RGB.value
        assert f"{rgb}.left" in output
        assert f"{rgb}.right" in output

    def test_multi_camera_backbone_called_per_camera(
        self,
        vit_encoder_factory: Callable[..., ViTEncoder],
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        mock_backbone_output_factory: Callable[..., torch.Tensor],
    ):
        batch_size = 2
        encoder = vit_encoder_factory(
            input_keys=["left", "right"],
            pooling_method=PoolingMethod.DEFAULT.value,
        )
        backbone_output = mock_backbone_output_factory(batch_size=batch_size)
        encoder.backbone.forward_features.return_value = backbone_output
        inputs = {
            **image_input_factory(key="left", batch_size=batch_size),
            **image_input_factory(key="right", batch_size=batch_size),
        }
        encoder(inputs)
        assert encoder.backbone.forward_features.call_count == 2


class TestViTEncoderIntegration:
    @pytest.mark.integration
    @pytest.mark.parametrize("backbone", [b.value for b in VIT_BACKBONES])
    def test_forward_pass_per_backbone(
        self,
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        backbone: str,
    ):
        batch_size = 2
        encoder = ViTEncoder(
            input_keys="left",
            backbone=backbone,
            pooling_method=PoolingMethod.DEFAULT.value,
            pretrained=False,
            frozen=False,
        )
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
        encoder = ViTEncoder(
            input_keys="left",
            backbone=ViTBackboneType.DINOV2_VITS14.value,
            pooling_method=PoolingMethod.DEFAULT.value,
            pretrained=False,
            frozen=False,
        )
        inputs = image_input_factory(
            batch_size=batch_size,
            time_steps=time_steps,
        )
        output = encoder(inputs)
        features = output[EncoderOutputKeys.RGB.value]
        assert features.shape == (batch_size, time_steps, encoder.output_dim)

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
        encoder = ViTEncoder(
            input_keys="left",
            backbone=ViTBackboneType.DINOV2_VITS14.value,
            pooling_method=PoolingMethod.DEFAULT.value,
            pretrained=False,
            frozen=frozen,
        )
        for parameter in encoder.parameters():
            assert parameter.requires_grad is expected_requires_grad
