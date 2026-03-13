"""Tests for versatil.models.encoding.encoders.depth.cnn module."""
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
import torch

from versatil.data.constants import Cameras
from versatil.models.encoding.encoders.constants import (
    BatchNormHandling,
    EncoderOutputKeys,
    PoolingMethod,
    RGBBackboneType,
)
from versatil.models.encoding.encoders.depth.cnn import DepthCNNEncoder
from versatil.models.encoding.encoders.unconditional import Encoder


CNN_BACKBONES = [
    e for e in RGBBackboneType
    if "vit" not in e.value and "dino" not in e.value
]


def _mock_build_backbone(self):
    """Side-effect to set self.backbone with expected attributes."""
    self.backbone = MagicMock()
    self.backbone.num_features = [64, 128, 256, 512]


def _mock_setup_pooling(self):
    """Side-effect to set pooling-related attributes."""
    self.pooling_head = None
    self.output_dim = self.feature_dim


@pytest.fixture
def depth_cnn_encoder_factory() -> Callable[..., DepthCNNEncoder]:
    """Factory for DepthCNNEncoder with mocked backbone and pooling."""
    def factory(
        input_keys: str | list[str] = Cameras.DEPTH.value,
        backbone: str = RGBBackboneType.RESNET18.value,
        pooling_method: str = PoolingMethod.AVERAGE.value,
        batch_norm_handling: str = BatchNormHandling.FROZEN.value,
        pretrained: bool = False,
        frozen: bool = False,
    ) -> DepthCNNEncoder:
        with (
            patch.object(DepthCNNEncoder, "_build_backbone", _mock_build_backbone),
            patch.object(DepthCNNEncoder, "_setup_pooling", _mock_setup_pooling),
        ):
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

    @pytest.mark.parametrize("backbone", [
        RGBBackboneType.RESNET18.value,
        RGBBackboneType.RESNET34.value,
    ])
    @pytest.mark.parametrize("pooling_method", [
        PoolingMethod.AVERAGE.value,
        PoolingMethod.NONE.value,
    ])
    @pytest.mark.parametrize("batch_norm_handling", [
        BatchNormHandling.FROZEN.value,
        BatchNormHandling.DEFAULT.value,
    ])
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

    def test_inherits_from_encoder(
        self,
        depth_cnn_encoder_factory: Callable[..., DepthCNNEncoder],
    ):
        encoder = depth_cnn_encoder_factory()
        assert isinstance(encoder, Encoder)

    def test_requires_depth_in_input_keys(self):
        """Depth camera key must be present in input_keys."""
        with pytest.raises(ValueError, match="Missing required inputs"):
            with (
                patch.object(DepthCNNEncoder, "_build_backbone", _mock_build_backbone),
                patch.object(DepthCNNEncoder, "_setup_pooling", _mock_setup_pooling),
            ):
                DepthCNNEncoder(input_keys="left")

    def test_input_specification_requires_depth_camera(
        self,
        depth_cnn_encoder_factory: Callable[..., DepthCNNEncoder],
    ):
        encoder = depth_cnn_encoder_factory()
        assert Cameras.DEPTH.value in encoder.input_specification.required


class TestDepthCNNEncoderForward:

    @pytest.mark.parametrize("time_steps, expected_ndim", [
        (None, 2),
        (3, 3),
    ])
    def test_output_shape_with_and_without_time(
        self,
        depth_cnn_encoder_factory: Callable[..., DepthCNNEncoder],
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        time_steps: int | None,
        expected_ndim: int,
    ):
        batch_size = 2
        feature_dimension = 512
        encoder = depth_cnn_encoder_factory()
        mock_pooling = MagicMock()
        mock_pooling.return_value = torch.zeros(
            batch_size * (time_steps or 1), feature_dimension,
        )
        encoder.pooling_head = mock_pooling
        mock_backbone_output = MagicMock()
        mock_backbone_output.feature_maps = [
            torch.zeros(batch_size * (time_steps or 1), 512, 7, 7),
        ]
        encoder.backbone.return_value = mock_backbone_output
        inputs = image_input_factory(
            key=Cameras.DEPTH.value,
            channels=1,
            batch_size=batch_size,
            time_steps=time_steps,
        )
        output = encoder(inputs)
        features = output[EncoderOutputKeys.DEPTH.value]
        assert features.ndim == expected_ndim
        assert features.shape[0] == batch_size
        if time_steps is not None:
            assert features.shape[1] == time_steps

    def test_creates_pooling_head_on_first_forward(
        self,
        depth_cnn_encoder_factory: Callable[..., DepthCNNEncoder],
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        encoder = depth_cnn_encoder_factory()
        assert encoder.pooling_head is None
        mock_backbone_output = MagicMock()
        mock_backbone_output.feature_maps = [torch.zeros(2, 512, 7, 7)]
        encoder.backbone.return_value = mock_backbone_output
        with patch(
            "versatil.models.encoding.encoders.depth.cnn.create_pooling_head",
        ) as mock_create:
            mock_head = MagicMock()
            mock_head.return_value = torch.zeros(2, 512)
            mock_head.to.return_value = mock_head
            mock_create.return_value = mock_head
            inputs = image_input_factory(key=Cameras.DEPTH.value, channels=1)
            encoder(inputs)
            mock_create.assert_called_once()

    def test_output_key_is_depth(
        self,
        depth_cnn_encoder_factory: Callable[..., DepthCNNEncoder],
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        encoder = depth_cnn_encoder_factory()
        mock_pooling = MagicMock()
        mock_pooling.return_value = torch.zeros(2, 512)
        encoder.pooling_head = mock_pooling
        mock_backbone_output = MagicMock()
        mock_backbone_output.feature_maps = [torch.zeros(2, 512, 7, 7)]
        encoder.backbone.return_value = mock_backbone_output
        inputs = image_input_factory(key=Cameras.DEPTH.value, channels=1)
        output = encoder(inputs)
        assert EncoderOutputKeys.DEPTH.value in output
        assert EncoderOutputKeys.RGB.value not in output


class TestDepthCNNEncoderGetOutputSpecification:

    def test_returns_depth_feature_with_correct_dimension(
        self,
        depth_cnn_encoder_factory: Callable[..., DepthCNNEncoder],
    ):
        encoder = depth_cnn_encoder_factory()
        specification = encoder.get_output_specification()
        assert specification.features == [EncoderOutputKeys.DEPTH.value]
        assert specification.dimensions[EncoderOutputKeys.DEPTH.value] == encoder.output_dim


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
        inputs = image_input_factory(
            key=Cameras.DEPTH.value,
            channels=1,
            batch_size=batch_size,
        )
        output = encoder(inputs)
        features = output[EncoderOutputKeys.DEPTH.value]
        assert features.ndim == 2
        assert features.shape[0] == batch_size

    @pytest.mark.integration
    @pytest.mark.parametrize("time_steps", [None, 2])
    def test_temporal_reshaping(
        self,
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        time_steps: int | None,
    ):
        batch_size = 2
        encoder = DepthCNNEncoder(
            input_keys=Cameras.DEPTH.value,
            backbone=RGBBackboneType.RESNET18.value,
            pooling_method=PoolingMethod.AVERAGE.value,
            pretrained=False,
        )
        inputs = image_input_factory(
            key=Cameras.DEPTH.value,
            channels=1,
            batch_size=batch_size,
            time_steps=time_steps,
        )
        output = encoder(inputs)
        features = output[EncoderOutputKeys.DEPTH.value]
        if time_steps is not None:
            assert features.shape == (batch_size, time_steps, encoder.output_dim)
        else:
            assert features.shape == (batch_size, encoder.output_dim)

    @pytest.mark.integration
    @pytest.mark.parametrize("batch_norm_handling", [
        BatchNormHandling.FROZEN.value,
        BatchNormHandling.DEFAULT.value,
        BatchNormHandling.CONVERT_TO_GROUPNORM.value,
    ])
    def test_batch_norm_handling_variants(
        self,
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        batch_norm_handling: str,
    ):
        encoder = DepthCNNEncoder(
            input_keys=Cameras.DEPTH.value,
            backbone=RGBBackboneType.RESNET18.value,
            batch_norm_handling=batch_norm_handling,
            pretrained=False,
        )
        inputs = image_input_factory(
            key=Cameras.DEPTH.value,
            channels=1,
            batch_size=2,
        )
        output = encoder(inputs)
        assert EncoderOutputKeys.DEPTH.value in output

    @pytest.mark.integration
    @pytest.mark.parametrize("frozen, expected_requires_grad", [
        (False, True),
        (True, False),
    ])
    def test_frozen_flag_controls_gradients(
        self,
        frozen: bool,
        expected_requires_grad: bool,
    ):
        encoder = DepthCNNEncoder(
            input_keys=Cameras.DEPTH.value,
            backbone=RGBBackboneType.RESNET18.value,
            pretrained=False,
            frozen=frozen,
        )
        for parameter in encoder.parameters():
            assert parameter.requires_grad is expected_requires_grad
