"""Tests for versatil.models.encoding.encoders.rgb.conditional_cnn module."""

import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from versatil.data.constants import RGB_CAMERAS
from versatil.models.encoding.encoders.constants import (
    BatchNormHandling,
    CNNBackboneType,
    EncoderOutputKeys,
    PoolingMethod,
)
from versatil.models.encoding.encoders.rgb.conditional_cnn import (
    ConditionalCNNEncoder,
)

CONDITIONAL_CNN_BACKBONES = list(ConditionalCNNEncoder.BACKBONE_CONFIGS.keys())


def _mock_build_filmed_backbone(self):
    """Side-effect to set backbone layer attributes without timm."""
    self.conv1 = nn.Identity()
    self.bn1 = nn.Identity()
    self.relu = nn.Identity()
    self.maxpool = nn.Identity()
    self.layer1 = nn.ModuleList()
    self.layer2 = nn.ModuleList()
    self.layer3 = nn.ModuleList()
    self.layer4 = nn.ModuleList()


@pytest.fixture
def conditional_cnn_factory() -> Callable[..., ConditionalCNNEncoder]:
    """Factory for ConditionalCNNEncoder with mocked backbone."""

    def factory(
        input_keys: str | list[str] = "left",
        condition_key: str = "language_instruction",
        condition_dim: int = 64,
        backbone: str = CNNBackboneType.RESNET18.value,
        pooling_method: str = PoolingMethod.SPATIAL_SOFTMAX.value,
        batch_norm_handling: str = BatchNormHandling.FROZEN.value,
        pretrained: bool = False,
        frozen: bool = False,
    ) -> ConditionalCNNEncoder:
        with patch.object(
            ConditionalCNNEncoder,
            "_build_filmed_backbone",
            _mock_build_filmed_backbone,
        ):
            return ConditionalCNNEncoder(
                input_keys=input_keys,
                condition_key=condition_key,
                condition_dim=condition_dim,
                backbone=backbone,
                pooling_method=pooling_method,
                batch_norm_handling=batch_norm_handling,
                pretrained=pretrained,
                frozen=frozen,
            )

    return factory


@pytest.fixture
def conditioning_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for conditioning tensors."""

    def factory(
        batch_size: int = 2,
        condition_dim: int = 64,
        time_steps: int = 1,
    ) -> torch.Tensor:
        shape = (batch_size, time_steps, condition_dim)
        return torch.from_numpy(rng.standard_normal(shape).astype(np.float32))

    return factory


class TestConditionalCNNEncoderInitialization:
    @pytest.mark.parametrize(
        "backbone, expectation",
        [
            (CNNBackboneType.RESNET18.value, does_not_raise()),
            (CNNBackboneType.RESNET34.value, does_not_raise()),
            (
                "invalid_backbone",
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        f"Backbone invalid_backbone not supported for FiLM Conditioning. "
                        f"Supported: {list(ConditionalCNNEncoder.BACKBONE_CONFIGS.keys())}"
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
            patch.object(
                ConditionalCNNEncoder,
                "_build_filmed_backbone",
                _mock_build_filmed_backbone,
            ),
        ):
            ConditionalCNNEncoder(
                input_keys="left",
                condition_key="language_instruction",
                condition_dim=64,
                backbone=backbone,
            )

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
            PoolingMethod.SPATIAL_SOFTMAX.value,
            PoolingMethod.AVERAGE.value,
        ],
    )
    @pytest.mark.parametrize(
        "batch_norm_handling",
        [
            BatchNormHandling.FROZEN.value,
            BatchNormHandling.DEFAULT.value,
        ],
    )
    @pytest.mark.parametrize("condition_dim", [64, 128])
    def test_stores_configuration(
        self,
        conditional_cnn_factory: Callable[..., ConditionalCNNEncoder],
        input_keys: str,
        backbone: str,
        pooling_method: str,
        batch_norm_handling: str,
        condition_dim: int,
    ):
        encoder = conditional_cnn_factory(
            input_keys=input_keys,
            backbone=backbone,
            pooling_method=pooling_method,
            batch_norm_handling=batch_norm_handling,
            condition_dim=condition_dim,
        )
        assert encoder.backbone_name == backbone
        assert encoder.pooling_method == pooling_method
        assert encoder.batch_norm_handling == batch_norm_handling
        assert encoder.condition_dim == condition_dim
        assert encoder.condition_key == "language_instruction"
        assert encoder.feature_dim == 512
        assert encoder.input_specification.keys == [input_keys]

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
        conditional_cnn_factory: Callable[..., ConditionalCNNEncoder],
        input_keys: str | list[str],
        expectation,
    ):
        with expectation:
            conditional_cnn_factory(input_keys=input_keys)


class TestConditionalCNNEncoderForward:
    @pytest.mark.parametrize("time_steps", [1, 3])
    def test_output_shape_with_temporal_dimension(
        self,
        conditional_cnn_factory: Callable[..., ConditionalCNNEncoder],
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        conditioning_factory: Callable[..., torch.Tensor],
        time_steps: int,
    ):
        batch_size = 2
        feature_dimension = 512
        condition_dim = 64
        encoder = conditional_cnn_factory(condition_dim=condition_dim)

        effective_batch = batch_size * time_steps
        mock_pooling = MagicMock()
        mock_pooling.return_value = torch.zeros(effective_batch, feature_dimension)
        encoder.pooling_head = mock_pooling

        inputs = image_input_factory(
            batch_size=batch_size,
            time_steps=time_steps,
        )
        conditioning = conditioning_factory(
            batch_size=batch_size,
            condition_dim=condition_dim,
            time_steps=time_steps,
        )
        output = encoder(inputs=inputs, conditioning=conditioning)
        features = output[EncoderOutputKeys.RGB.value]
        assert features.shape == (batch_size, time_steps, feature_dimension)

    def test_conditioning_2d_replicated_over_time(
        self,
        conditional_cnn_factory: Callable[..., ConditionalCNNEncoder],
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        conditioning_factory: Callable[..., torch.Tensor],
    ):
        batch_size = 2
        time_steps = 3
        feature_dimension = 512
        condition_dim = 64
        encoder = conditional_cnn_factory(condition_dim=condition_dim)

        effective_batch = batch_size * time_steps
        mock_pooling = MagicMock()
        mock_pooling.return_value = torch.zeros(effective_batch, feature_dimension)
        encoder.pooling_head = mock_pooling

        inputs = image_input_factory(
            batch_size=batch_size,
            time_steps=time_steps,
        )
        conditioning = conditioning_factory(
            batch_size=batch_size,
            condition_dim=condition_dim,
        )
        output = encoder(inputs=inputs, conditioning=conditioning)
        features = output[EncoderOutputKeys.RGB.value]
        assert features.shape == (batch_size, time_steps, feature_dimension)

    def test_raises_when_pooling_head_not_initialized(
        self,
        conditional_cnn_factory: Callable[..., ConditionalCNNEncoder],
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        conditioning_factory: Callable[..., torch.Tensor],
    ):
        batch_size = 2
        condition_dim = 64
        encoder = conditional_cnn_factory(condition_dim=condition_dim)
        inputs = image_input_factory(batch_size=batch_size)
        conditioning = conditioning_factory(
            batch_size=batch_size,
            condition_dim=condition_dim,
        )
        with pytest.raises(
            RuntimeError,
            match="pooling_head is not initialized. Call set_image_size",
        ):
            encoder(inputs=inputs, conditioning=conditioning)


class TestConditionalCNNEncoderEncodeSingleImage:
    def test_raises_runtime_error(
        self,
        conditional_cnn_factory: Callable[..., ConditionalCNNEncoder],
    ):
        encoder = conditional_cnn_factory()
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "ConditionalCNNEncoder requires conditioning. Use encode() directly."
            ),
        ):
            encoder._encode_single_image(images=torch.zeros(2, 3, 32, 32))


class TestConditionalCNNEncoderMultiCamera:
    def test_multi_camera_forward_produces_per_camera_features(
        self,
        conditional_cnn_factory: Callable[..., ConditionalCNNEncoder],
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        conditioning_factory: Callable[..., torch.Tensor],
    ):
        batch_size = 2
        feature_dimension = 512
        condition_dim = 64
        encoder = conditional_cnn_factory(
            input_keys=["left", "right"],
            condition_dim=condition_dim,
        )
        mock_pooling = MagicMock()
        mock_pooling.return_value = torch.zeros(batch_size, feature_dimension)
        encoder.pooling_head = mock_pooling
        inputs = {
            **image_input_factory(key="left", batch_size=batch_size),
            **image_input_factory(key="right", batch_size=batch_size),
        }
        conditioning = conditioning_factory(
            batch_size=batch_size,
            condition_dim=condition_dim,
        )
        output = encoder(inputs=inputs, conditioning=conditioning)
        rgb = EncoderOutputKeys.RGB.value
        assert f"{rgb}.left" in output
        assert f"{rgb}.right" in output
        assert output[f"{rgb}.left"].shape[0] == batch_size
        assert output[f"{rgb}.right"].shape[0] == batch_size


class TestConditionalCNNEncoderGetOutputSpecification:
    def test_returns_rgb_feature_with_correct_dimension(
        self,
        conditional_cnn_factory: Callable[..., ConditionalCNNEncoder],
    ):
        encoder = conditional_cnn_factory()
        specification = encoder.get_output_specification()
        feature_keys = [m.key for m in specification]
        assert feature_keys == [EncoderOutputKeys.RGB.value]
        assert next(
            m for m in specification if m.key == EncoderOutputKeys.RGB.value
        ).dimension == (encoder.output_dim,)


class TestConditionalCNNEncoderIntegration:
    @pytest.mark.integration
    @pytest.mark.parametrize("backbone", CONDITIONAL_CNN_BACKBONES)
    def test_forward_pass_per_backbone(
        self,
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        conditioning_factory: Callable[..., torch.Tensor],
        backbone: str,
    ):
        batch_size = 2
        condition_dim = 64
        encoder = ConditionalCNNEncoder(
            input_keys="left",
            condition_key="language_instruction",
            condition_dim=condition_dim,
            backbone=backbone,
            pooling_method=PoolingMethod.AVERAGE.value,
            pretrained=False,
        ).cpu()
        encoder.set_image_size(image_height=224, image_width=224)
        inputs = image_input_factory(batch_size=batch_size)
        conditioning = conditioning_factory(
            batch_size=batch_size,
            condition_dim=condition_dim,
        )
        output = encoder(inputs=inputs, conditioning=conditioning)
        features = output[EncoderOutputKeys.RGB.value]
        assert features.shape == (batch_size, 1, encoder.output_dim)

    @pytest.mark.integration
    @pytest.mark.parametrize("time_steps", [1, 2])
    def test_temporal_reshaping(
        self,
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        conditioning_factory: Callable[..., torch.Tensor],
        time_steps: int,
    ):
        batch_size = 2
        condition_dim = 64
        encoder = ConditionalCNNEncoder(
            input_keys="left",
            condition_key="language_instruction",
            condition_dim=condition_dim,
            backbone=CNNBackboneType.RESNET18.value,
            pooling_method=PoolingMethod.AVERAGE.value,
            pretrained=False,
        ).cpu()
        encoder.set_image_size(image_height=224, image_width=224)
        inputs = image_input_factory(
            batch_size=batch_size,
            time_steps=time_steps,
        )
        conditioning = conditioning_factory(
            batch_size=batch_size,
            condition_dim=condition_dim,
            time_steps=time_steps,
        )
        output = encoder(inputs=inputs, conditioning=conditioning)
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
        conditioning_factory: Callable[..., torch.Tensor],
        batch_norm_handling: str,
    ):
        batch_size = 2
        condition_dim = 64
        encoder = ConditionalCNNEncoder(
            input_keys="left",
            condition_key="language_instruction",
            condition_dim=condition_dim,
            backbone=CNNBackboneType.RESNET18.value,
            batch_norm_handling=batch_norm_handling,
            pooling_method=PoolingMethod.AVERAGE.value,
            pretrained=False,
        ).cpu()
        encoder.set_image_size(image_height=224, image_width=224)
        inputs = image_input_factory(batch_size=batch_size)
        conditioning = conditioning_factory(
            batch_size=batch_size,
            condition_dim=condition_dim,
        )
        output = encoder(inputs=inputs, conditioning=conditioning)
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
        encoder = ConditionalCNNEncoder(
            input_keys="left",
            condition_key="language_instruction",
            condition_dim=64,
            backbone=CNNBackboneType.RESNET18.value,
            pretrained=False,
            frozen=frozen,
        ).cpu()
        for parameter in encoder.parameters():
            assert parameter.requires_grad is expected_requires_grad


class TestConditionalCNNEncoderApplyBatchNormHandling:
    def test_invalid_batch_norm_handling_raises(self):
        invalid_handling = "invalid_batch_norm_handling"
        with pytest.raises(
            ValueError,
            match=re.escape(f"Unknown batch norm handling: {invalid_handling}"),
        ):
            ConditionalCNNEncoder(
                input_keys="left",
                condition_key="language_instruction",
                condition_dim=64,
                backbone=CNNBackboneType.RESNET18.value,
                batch_norm_handling=invalid_handling,
                pretrained=False,
            )


class TestConditionalCNNEncoderCopyPretrainedWeights:
    @pytest.mark.integration
    def test_pretrained_weights_are_copied_to_filmed_blocks(self):
        encoder = ConditionalCNNEncoder(
            input_keys="left",
            condition_key="language_instruction",
            condition_dim=64,
            backbone=CNNBackboneType.RESNET18.value,
            pretrained=True,
        ).cpu()
        # Verify conv weights are non-zero (pretrained weights loaded)
        first_block = encoder.layer1[0]
        conv1_weight_norm = first_block.conv1.weight.data.abs().sum().item()
        assert conv1_weight_norm > 0.0
