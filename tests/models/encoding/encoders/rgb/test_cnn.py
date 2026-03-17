"""Tests for versatil.models.encoding.encoders.rgb.cnn module."""

import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock, patch

import pytest
import torch

from versatil.data.constants import RGB_CAMERAS
from versatil.models.encoding.encoders.constants import (
    BatchNormHandling,
    EncoderOutputKeys,
    PoolingMethod,
    RGBBackboneType,
)
from versatil.models.encoding.encoders.rgb.cnn import CNNEncoder

CNN_BACKBONES = [
    e for e in RGBBackboneType if "vit" not in e.value and "dino" not in e.value
]

CNN_VALID_BACKBONES = [e.value for e in RGBBackboneType if "vit" not in e.value]


def _mock_build_backbone(self):
    """Side-effect to set self.backbone with expected attributes."""
    self.backbone = MagicMock()
    self.backbone.num_features = [64, 128, 256, 512]


def _mock_setup_pooling(self):
    """Side-effect to set pooling-related attributes."""
    self.pooling_head = None
    self.output_dim = self.feature_dim


@pytest.fixture
def cnn_encoder_factory() -> Callable[..., CNNEncoder]:
    """Factory for CNNEncoder with mocked backbone and pooling."""

    def factory(
        input_keys: str | list[str] = "left",
        backbone: str = RGBBackboneType.RESNET18.value,
        pooling_method: str = PoolingMethod.AVERAGE.value,
        batch_norm_handling: str = BatchNormHandling.FROZEN.value,
        pretrained: bool = False,
        frozen: bool = False,
    ) -> CNNEncoder:
        with (
            patch.object(CNNEncoder, "_build_backbone", _mock_build_backbone),
            patch.object(CNNEncoder, "_setup_pooling", _mock_setup_pooling),
        ):
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
            (RGBBackboneType.RESNET18.value, does_not_raise()),
            (RGBBackboneType.RESNET50.value, does_not_raise()),
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
            patch.object(CNNEncoder, "_setup_pooling", _mock_setup_pooling),
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
            (
                ["left", "right"],
                pytest.raises(
                    ValueError,
                    match=re.escape(f"Exactly one from {RGB_CAMERAS} required, got"),
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

    def test_has_encoder_interface(
        self,
        cnn_encoder_factory: Callable[..., CNNEncoder],
    ):
        encoder = cnn_encoder_factory()
        assert hasattr(encoder, "forward")
        assert hasattr(encoder, "get_output_specification")
        assert hasattr(encoder, "input_specification")

    @pytest.mark.parametrize("input_keys", ["left", "right"])
    @pytest.mark.parametrize(
        "backbone",
        [
            RGBBackboneType.RESNET18.value,
            RGBBackboneType.RESNET34.value,
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
    @pytest.mark.parametrize(
        "time_steps, expected_ndim",
        [
            (None, 2),
            (3, 3),
        ],
    )
    def test_output_shape_with_and_without_time(
        self,
        cnn_encoder_factory: Callable[..., CNNEncoder],
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        time_steps: int | None,
        expected_ndim: int,
    ):
        batch_size = 2
        feature_dimension = 512
        encoder = cnn_encoder_factory()
        mock_pooling = MagicMock()
        mock_pooling.return_value = torch.zeros(
            batch_size * (time_steps or 1),
            feature_dimension,
        )
        encoder.pooling_head = mock_pooling
        mock_backbone_output = MagicMock()
        mock_backbone_output.feature_maps = [
            torch.zeros(batch_size * (time_steps or 1), 512, 7, 7),
        ]
        encoder.backbone.return_value = mock_backbone_output
        inputs = image_input_factory(
            batch_size=batch_size,
            time_steps=time_steps,
        )
        output = encoder(inputs)
        features = output[EncoderOutputKeys.RGB.value]
        assert features.ndim == expected_ndim
        assert features.shape[0] == batch_size
        if time_steps is not None:
            assert features.shape[1] == time_steps

    def test_creates_pooling_head_on_first_forward(
        self,
        cnn_encoder_factory: Callable[..., CNNEncoder],
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        encoder = cnn_encoder_factory()
        assert encoder.pooling_head is None
        mock_backbone_output = MagicMock()
        mock_backbone_output.feature_maps = [torch.zeros(2, 512, 7, 7)]
        encoder.backbone.return_value = mock_backbone_output
        with patch(
            "versatil.models.encoding.encoders.rgb.cnn.create_pooling_head",
        ) as mock_create:
            mock_head = MagicMock()
            mock_head.return_value = torch.zeros(2, 512)
            mock_head.to.return_value = mock_head
            mock_create.return_value = mock_head
            inputs = image_input_factory()
            encoder(inputs)
            mock_create.assert_called_once()


class TestCNNEncoderGetOutputSpecification:
    def test_returns_rgb_feature_with_correct_dimension(
        self,
        cnn_encoder_factory: Callable[..., CNNEncoder],
    ):
        encoder = cnn_encoder_factory()
        specification = encoder.get_output_specification()
        assert specification.features == [EncoderOutputKeys.RGB.value]
        assert (
            specification.dimensions[EncoderOutputKeys.RGB.value] == encoder.output_dim
        )


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
        inputs = image_input_factory(batch_size=batch_size)
        output = encoder(inputs)
        features = output[EncoderOutputKeys.RGB.value]
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
        encoder = CNNEncoder(
            input_keys="left",
            backbone=RGBBackboneType.RESNET18.value,
            pooling_method=PoolingMethod.AVERAGE.value,
            pretrained=False,
        )
        inputs = image_input_factory(
            batch_size=batch_size,
            time_steps=time_steps,
        )
        output = encoder(inputs)
        features = output[EncoderOutputKeys.RGB.value]
        if time_steps is not None:
            assert features.shape == (batch_size, time_steps, encoder.output_dim)
        else:
            assert features.shape == (batch_size, encoder.output_dim)

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
            backbone=RGBBackboneType.RESNET18.value,
            batch_norm_handling=batch_norm_handling,
            pretrained=False,
        )
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
            backbone=RGBBackboneType.RESNET18.value,
            pretrained=False,
            frozen=frozen,
        )
        for parameter in encoder.parameters():
            assert parameter.requires_grad is expected_requires_grad


class TestCNNEncoderBuildBackbone:
    def test_invalid_batch_norm_handling_raises(self):
        invalid_handling = "invalid_batch_norm_handling"

        def _mock_setup_pooling_only(self_inner):
            self_inner.pooling_head = None
            self_inner.output_dim = self_inner.feature_dim

        with (
            patch.object(CNNEncoder, "_setup_pooling", _mock_setup_pooling_only),
            pytest.raises(
                ValueError,
                match=re.escape(f"Unknown batch norm handling: {invalid_handling}"),
            ),
        ):
            CNNEncoder(
                input_keys="left",
                backbone=RGBBackboneType.RESNET18.value,
                batch_norm_handling=invalid_handling,
                pretrained=False,
            )
