"""Tests for versatil.models.encoding.encoders.rgb.cnn module."""

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
    CNNBackboneType,
    EncoderOutputKeys,
    PoolingMethod,
)
from versatil.models.encoding.encoders.rgb.cnn import CNNEncoder

CNN_BACKBONES = list(CNNBackboneType)
CNN_VALID_BACKBONES = [e.value for e in CNNBackboneType]


def _mock_build_backbone(self):
    """Side-effect to set self.backbone with expected attributes."""
    self.backbone = MagicMock()
    self.backbone.feature_info.channels.return_value = [64, 128, 256, 512]


def _mock_setup_pooling(self, spatial_height: int, spatial_width: int):
    self.pooling_head = MagicMock()
    self.pooling_head.return_value = torch.zeros(1, self.feature_dim)
    self.output_dim = self.feature_dim


@pytest.fixture
def cnn_encoder_factory() -> Callable[..., CNNEncoder]:
    """Factory for CNNEncoder with mocked backbone."""

    def factory(
        input_keys: str | list[str] = "left",
        backbone: str = CNNBackboneType.RESNET18.value,
        pooling_method: str = PoolingMethod.AVERAGE.value,
        batch_norm_handling: str = BatchNormHandling.FROZEN.value,
        pretrained: bool = False,
        frozen: bool = False,
    ) -> CNNEncoder:
        with patch.object(CNNEncoder, "_build_backbone", _mock_build_backbone):
            return CNNEncoder(
                input_keys=input_keys,
                backbone=backbone,
                pooling_method=pooling_method,
                batch_norm_handling=batch_norm_handling,
                pretrained=pretrained,
                frozen=frozen,
            )

    return factory


class TestCNNEncoderInitialization:
    @pytest.mark.parametrize(
        "backbone, expectation",
        [
            (CNNBackboneType.RESNET18.value, does_not_raise()),
            (CNNBackboneType.RESNET50.value, does_not_raise()),
            (
                "invalid_backbone",
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        f"Invalid backbone 'invalid_backbone'. Must be one of: {CNN_VALID_BACKBONES}"
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
            patch.object(CNNEncoder, "_build_backbone", _mock_build_backbone),
        ):
            CNNEncoder(
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
        cnn_encoder_factory: Callable[..., CNNEncoder],
        input_keys: str | list[str],
        expectation,
    ):
        with expectation:
            cnn_encoder_factory(input_keys=input_keys)

    @pytest.mark.parametrize("input_keys", ["left", "right"])
    @pytest.mark.parametrize(
        "backbone",
        [
            CNNBackboneType.RESNET18.value,
            CNNBackboneType.RESNET34.value,
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
        cnn_encoder_factory: Callable[..., CNNEncoder],
        input_keys: str,
        backbone: str,
        pooling_method: str,
        batch_norm_handling: str,
    ):
        encoder = cnn_encoder_factory(
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


class TestCNNEncoderForward:
    @pytest.mark.parametrize("time_steps", [1, 3])
    def test_output_shape_with_temporal_dimension(
        self,
        cnn_encoder_factory: Callable[..., CNNEncoder],
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        time_steps: int,
    ):
        batch_size = 2
        feature_dimension = 512
        encoder = cnn_encoder_factory()
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
        cnn_encoder_factory: Callable[..., CNNEncoder],
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        encoder = cnn_encoder_factory()
        inputs = image_input_factory()
        with pytest.raises(
            RuntimeError,
            match="pooling_head is not initialized. Call set_image_size",
        ):
            encoder(inputs)


class TestCNNEncoderGetOutputSpecification:
    def test_returns_rgb_feature_with_correct_dimension(
        self,
        cnn_encoder_factory: Callable[..., CNNEncoder],
    ):
        encoder = cnn_encoder_factory()
        specification = encoder.get_output_specification()
        feature_keys = [m.key for m in specification]
        assert feature_keys == [EncoderOutputKeys.RGB.value]
        assert next(
            m for m in specification if m.key == EncoderOutputKeys.RGB.value
        ).dimension == (encoder.output_dim,)


class TestCNNEncoderSetImageSize:
    def test_set_image_size_updates_output_dim(
        self,
        cnn_encoder_factory: Callable[..., CNNEncoder],
    ):
        encoder = cnn_encoder_factory()
        initial_output_dim = encoder.output_dim
        encoder.backbone.return_value = [torch.zeros(1, 512, 7, 7)]
        with patch.object(CNNEncoder, "_setup_pooling", _mock_setup_pooling):
            encoder.set_image_size(image_height=224, image_width=224)
        assert encoder.output_dim == initial_output_dim  # mock keeps feature_dim
        assert next(
            m
            for m in encoder.get_output_specification()
            if m.key == EncoderOutputKeys.RGB.value
        ).dimension == (encoder.output_dim,)


class TestCNNEncoderValidateInputMetadata:
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
        cnn_encoder_factory: Callable[..., CNNEncoder],
        metadata,
        expected_error: str | None,
    ):
        encoder = cnn_encoder_factory()
        result = encoder.validate_input_metadata(key="left", metadata=metadata)
        assert result == expected_error


class TestCNNEncoderMultiCamera:
    @pytest.mark.parametrize(
        "input_keys, expected_feature_count, expected_multi_camera",
        [
            ("left", 1, False),
            (["left", "right"], 2, True),
        ],
    )
    def test_output_specification_scales_with_cameras(
        self,
        cnn_encoder_factory: Callable[..., CNNEncoder],
        input_keys: str | list[str],
        expected_feature_count: int,
        expected_multi_camera: bool,
    ):
        encoder = cnn_encoder_factory(input_keys=input_keys)
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
        cnn_encoder_factory: Callable[..., CNNEncoder],
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 2
        feature_dimension = 512
        encoder = cnn_encoder_factory(input_keys=["left", "right"])
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
        cnn_encoder_factory: Callable[..., CNNEncoder],
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 2
        encoder = cnn_encoder_factory(input_keys=["left", "right"])
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


class TestCNNEncoderIntegration:
    @pytest.mark.integration
    @pytest.mark.parametrize("backbone", [b.value for b in CNN_BACKBONES])
    def test_forward_pass_per_backbone(
        self,
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        backbone: str,
    ):
        batch_size = 2
        encoder = CNNEncoder(
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
        encoder = CNNEncoder(
            input_keys="left",
            backbone=CNNBackboneType.RESNET18.value,
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
        encoder = CNNEncoder(
            input_keys="left",
            backbone=CNNBackboneType.RESNET18.value,
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
        encoder = CNNEncoder(
            input_keys="left",
            backbone=CNNBackboneType.RESNET18.value,
            pretrained=False,
            frozen=frozen,
        )
        for parameter in encoder.parameters():
            assert parameter.requires_grad is expected_requires_grad


class TestCNNEncoderBuildBackbone:
    def test_invalid_batch_norm_handling_raises(self):
        invalid_handling = "invalid_batch_norm_handling"
        with pytest.raises(
            ValueError,
            match=re.escape(f"Unknown batch norm handling: {invalid_handling}"),
        ):
            CNNEncoder(
                input_keys="left",
                backbone=CNNBackboneType.RESNET18.value,
                batch_norm_handling=invalid_handling,
                pretrained=False,
            )
