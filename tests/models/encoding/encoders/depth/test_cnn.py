"""Tests for versatil.models.encoding.encoders.depth.cnn module."""

import re
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
import torch

from versatil.data.constants import Cameras
from versatil.data.metadata import BaseMetadata, CameraMetadata
from versatil.models.encoding.encoders.constants import (
    BatchNormHandling,
    CNNBackboneType,
    EncoderOutputKeys,
    PoolingMethod,
)
from versatil.models.encoding.encoders.depth.cnn import DepthCNNEncoder

CNN_BACKBONES = list(CNNBackboneType)


def _mock_build_backbone(self):
    """Side-effect to set self.backbone with expected attributes."""
    self.backbone = MagicMock()
    self.backbone.feature_info.channels.return_value = [64, 128, 256, 512]


def _mock_setup_pooling(self, spatial_height: int, spatial_width: int):
    """Side-effect to create a mock pooling head with correct output dim."""
    self.pooling_head = MagicMock()
    self.pooling_head.return_value = torch.zeros(1, self.feature_dim)
    self.output_dim = self.feature_dim


@pytest.fixture
def depth_cnn_encoder_factory() -> Callable[..., DepthCNNEncoder]:
    """Factory for DepthCNNEncoder with mocked backbone."""

    def factory(
        input_keys: str | list[str] = Cameras.DEPTH.value,
        backbone: str = CNNBackboneType.RESNET18.value,
        pooling_method: str = PoolingMethod.AVERAGE.value,
        batch_norm_handling: str = BatchNormHandling.FROZEN.value,
        pretrained: bool = False,
        frozen: bool = False,
    ) -> DepthCNNEncoder:
        with patch.object(DepthCNNEncoder, "_build_backbone", _mock_build_backbone):
            return DepthCNNEncoder(
                input_keys=input_keys,
                backbone=backbone,
                pooling_method=pooling_method,
                batch_norm_handling=batch_norm_handling,
                pretrained=pretrained,
                frozen=frozen,
            )

    return factory


class TestDepthCNNEncoderInitialization:
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
        depth_cnn_encoder_factory: Callable[..., DepthCNNEncoder],
        backbone: str,
        pooling_method: str,
        batch_norm_handling: str,
    ):
        encoder = depth_cnn_encoder_factory(
            backbone=backbone,
            pooling_method=pooling_method,
            batch_norm_handling=batch_norm_handling,
        )
        assert encoder.backbone_name == backbone
        assert encoder.pooling_method == pooling_method
        assert encoder.batch_norm_handling == batch_norm_handling
        assert encoder.feature_dim == 512

    def test_requires_depth_in_input_keys(self):
        with (
            pytest.raises(
                ValueError,
                match=re.escape("Missing required inputs: {'depth'}"),
            ),
            patch.object(DepthCNNEncoder, "_build_backbone", _mock_build_backbone),
        ):
            DepthCNNEncoder(input_keys="left")

    def test_input_specification_requires_depth_camera(
        self,
        depth_cnn_encoder_factory: Callable[..., DepthCNNEncoder],
    ):
        encoder = depth_cnn_encoder_factory()
        assert Cameras.DEPTH.value in encoder.input_specification.required


class TestDepthCNNEncoderForward:
    @pytest.mark.parametrize("time_steps", [1, 3])
    def test_output_shape_with_temporal_dimension(
        self,
        depth_cnn_encoder_factory: Callable[..., DepthCNNEncoder],
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        time_steps: int,
    ):
        batch_size = 2
        feature_dimension = 512
        encoder = depth_cnn_encoder_factory()
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
        depth_cnn_encoder_factory: Callable[..., DepthCNNEncoder],
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        encoder = depth_cnn_encoder_factory()
        inputs = image_input_factory(key=Cameras.DEPTH.value, channels=1)
        with pytest.raises(
            RuntimeError,
            match="pooling_head is not initialized. Call set_image_size",
        ):
            encoder(inputs)

    def test_output_key_is_depth(
        self,
        depth_cnn_encoder_factory: Callable[..., DepthCNNEncoder],
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        encoder = depth_cnn_encoder_factory()
        mock_pooling = MagicMock()
        mock_pooling.return_value = torch.zeros(2, 512)
        encoder.pooling_head = mock_pooling
        encoder.backbone.return_value = [torch.zeros(2, 512, 7, 7)]
        inputs = image_input_factory(key=Cameras.DEPTH.value, channels=1)
        output = encoder(inputs)
        assert EncoderOutputKeys.DEPTH.value in output
        assert EncoderOutputKeys.RGB.value not in output


class TestDepthCNNEncoderSetImageSize:
    def test_set_image_size_creates_pooling_head(
        self,
        depth_cnn_encoder_factory: Callable[..., DepthCNNEncoder],
    ):
        encoder = depth_cnn_encoder_factory()
        assert encoder.pooling_head is None
        encoder.backbone.return_value = [torch.zeros(1, 512, 7, 7)]
        with patch.object(DepthCNNEncoder, "_setup_pooling", _mock_setup_pooling):
            encoder.set_image_size(image_height=224, image_width=224)
        assert encoder.pooling_head is not None


class TestDepthCNNEncoderValidateInputMetadata:
    @pytest.mark.parametrize(
        "metadata, expected_error",
        [
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
                CameraMetadata(
                    camera_key="left",
                    dtype="uint8",
                    channels=3,
                    image_height=224,
                    image_width=224,
                ),
                "Expected single-channel depth for 'depth', got 3 channels",
            ),
            (
                MagicMock(spec=BaseMetadata),
                "Expected CameraMetadata for 'depth', got MagicMock",
            ),
        ],
    )
    def test_validates_single_channel_camera_metadata(
        self,
        depth_cnn_encoder_factory: Callable[..., DepthCNNEncoder],
        metadata,
        expected_error: str | None,
    ):
        encoder = depth_cnn_encoder_factory()
        result = encoder.validate_input_metadata(key="depth", metadata=metadata)
        assert result == expected_error


class TestDepthCNNEncoderGetOutputSpecification:
    def test_returns_depth_feature_with_correct_dimension(
        self,
        depth_cnn_encoder_factory: Callable[..., DepthCNNEncoder],
    ):
        encoder = depth_cnn_encoder_factory()
        specification = encoder.get_output_specification()
        feature_keys = [m.key for m in specification]
        assert feature_keys == [EncoderOutputKeys.DEPTH.value]
        assert next(
            m for m in specification if m.key == EncoderOutputKeys.DEPTH.value
        ).dimension == (encoder.output_dim,)


class TestDepthCNNEncoderIntegration:
    @pytest.mark.integration
    @pytest.mark.parametrize("backbone", [b.value for b in CNN_BACKBONES])
    def test_forward_pass_per_backbone(
        self,
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        backbone: str,
    ):
        batch_size = 2
        encoder = DepthCNNEncoder(
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
    @pytest.mark.parametrize("time_steps", [1, 2])
    def test_temporal_reshaping(
        self,
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        time_steps: int,
    ):
        batch_size = 2
        encoder = DepthCNNEncoder(
            input_keys=Cameras.DEPTH.value,
            backbone=CNNBackboneType.RESNET18.value,
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
        encoder = DepthCNNEncoder(
            input_keys=Cameras.DEPTH.value,
            backbone=CNNBackboneType.RESNET18.value,
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
        encoder = DepthCNNEncoder(
            input_keys=Cameras.DEPTH.value,
            backbone=CNNBackboneType.RESNET18.value,
            pretrained=False,
            frozen=frozen,
        )
        for parameter in encoder.parameters():
            assert parameter.requires_grad is expected_requires_grad


class TestDepthCNNEncoderBuildBackbone:
    def test_invalid_batch_norm_handling_raises(self):
        invalid_handling = "invalid_batch_norm_handling"
        with pytest.raises(
            ValueError,
            match=re.escape(f"Unknown batch norm handling: {invalid_handling}"),
        ):
            DepthCNNEncoder(
                input_keys=Cameras.DEPTH.value,
                backbone=CNNBackboneType.RESNET18.value,
                batch_norm_handling=invalid_handling,
                pretrained=False,
            )
