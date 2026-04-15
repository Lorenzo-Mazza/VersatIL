"""Tests for versatil.models.encoding.encoders.rgb.conditional_cnn module."""

import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import timm
import torch
import torch.nn as nn

from versatil.data.constants import RGB_CAMERAS
from versatil.data.metadata import BaseMetadata, CameraMetadata
from versatil.models.encoding.encoders.base import EncodingMixin
from versatil.models.encoding.encoders.constants import (
    BatchNormHandling,
    EncoderOutputKeys,
    PoolingMethod,
    SpatialBackboneType,
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


@pytest.fixture(scope="session")
def _real_timm_resnet_template():
    """Session-scoped real timm ResNet18 (pretrained=False) used as a base
    model for _build_filmed_backbone tests. Constructed once per test session
    to amortize the ~1s instantiation cost."""
    return timm.create_model("resnet18", pretrained=False, num_classes=0)


@pytest.fixture
def mock_timm_resnet_backend(_real_timm_resnet_template):
    """Patches ``timm.create_model`` in conditional_cnn for the whole test.

    Returns the same real ResNet18 template on every call. The patch is
    started with ``patcher.start()`` and torn down via yield so it stays
    active across any ``set_image_size`` rebuilds the test triggers after
    the encoder is constructed.
    """

    class _Backend:
        def __init__(self, template):
            self.template = template
            self.create_model_mock = None

    backend = _Backend(_real_timm_resnet_template)
    patcher = patch(
        "versatil.models.encoding.encoders.rgb.conditional_cnn.timm.create_model",
        return_value=_real_timm_resnet_template,
    )
    backend.create_model_mock = patcher.start()
    yield backend
    patcher.stop()


@pytest.fixture
def conditional_cnn_factory(
    mock_timm_resnet_backend,
) -> Callable[..., ConditionalCNNEncoder]:
    """Factory for ConditionalCNNEncoder with mocked backbone.

    By default bypasses ``_build_filmed_backbone`` via a side-effect mock
    for fast shape/forward tests. Pass ``real_build=True`` to exercise the
    real ``_build_filmed_backbone`` method — the ``mock_timm_resnet_backend``
    fixture keeps ``timm.create_model`` patched to return a real (but
    untrained) ResNet18 for the whole test, so the full build/copy/BN
    handling paths run against real modules.
    """

    def factory(
        input_keys: str | list[str] = "left",
        condition_key: str = "language_instruction",
        condition_dim: int = 64,
        backbone: str = SpatialBackboneType.RESNET18.value,
        pooling_method: str = PoolingMethod.SPATIAL_SOFTMAX.value,
        batch_norm_handling: str = BatchNormHandling.FROZEN.value,
        pretrained: bool = False,
        frozen: bool = False,
        real_build: bool = False,
    ) -> ConditionalCNNEncoder:
        if not real_build:
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
            (SpatialBackboneType.RESNET18.value, does_not_raise()),
            (SpatialBackboneType.RESNET34.value, does_not_raise()),
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
            SpatialBackboneType.RESNET18.value,
            SpatialBackboneType.RESNET34.value,
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
            match=re.escape(
                "pooling_head is not initialized. Call set_image_size() before forward."
            ),
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
        assert f"{rgb}:left" in output
        assert f"{rgb}:right" in output
        assert output[f"{rgb}:left"].shape[0] == batch_size
        assert output[f"{rgb}:right"].shape[0] == batch_size


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


class TestConditionalCNNEncoderValidateInputMetadata:
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
        ],
    )
    def test_validates_rgb_camera_metadata(
        self,
        conditional_cnn_factory: Callable[..., ConditionalCNNEncoder],
        metadata,
        expected_error: str | None,
    ):
        encoder = conditional_cnn_factory()
        result = encoder.validate_input_metadata(key="left", metadata=metadata)
        assert result == expected_error


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
            backbone=SpatialBackboneType.RESNET18.value,
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
            backbone=SpatialBackboneType.RESNET18.value,
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
            backbone=SpatialBackboneType.RESNET18.value,
            pretrained=False,
            frozen=frozen,
        ).cpu()
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
        encoder = ConditionalCNNEncoder(
            input_keys="left",
            condition_key="language_instruction",
            condition_dim=64,
            backbone=SpatialBackboneType.RESNET18.value,
            pooling_method=pooling_method,
            pretrained=False,
            frozen=frozen,
        ).cpu()
        encoder.set_image_size(image_height=224, image_width=224)
        for parameter in encoder.parameters():
            assert parameter.requires_grad is not frozen


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
                backbone=SpatialBackboneType.RESNET18.value,
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
            backbone=SpatialBackboneType.RESNET18.value,
            pretrained=True,
        ).cpu()
        # Verify conv weights are non-zero (pretrained weights loaded)
        first_block = encoder.layer1[0]
        conv1_weight_norm = first_block.conv1.weight.data.abs().sum().item()
        assert conv1_weight_norm > 0.0


class TestConditionalCNNEncoderRealBuild:
    @pytest.mark.unit
    @pytest.mark.parametrize("pretrained", [False, True])
    def test_real_build_wires_layers_from_timm_backbone(
        self,
        conditional_cnn_factory: Callable[..., ConditionalCNNEncoder],
        mock_timm_resnet_backend,
        pretrained: bool,
    ):
        encoder = conditional_cnn_factory(
            real_build=True,
            pretrained=pretrained,
        )
        assert mock_timm_resnet_backend.create_model_mock.called
        assert isinstance(encoder.conv1, nn.Conv2d)
        assert isinstance(encoder.bn1, (nn.BatchNorm2d, nn.GroupNorm))
        assert len(encoder.layer1) == 2
        assert len(encoder.layer2) == 2
        assert len(encoder.layer3) == 2
        assert len(encoder.layer4) == 2
        assert encoder.feature_dim == 512

    @pytest.mark.unit
    def test_pretrained_copies_weights_into_filmed_blocks(
        self,
        conditional_cnn_factory: Callable[..., ConditionalCNNEncoder],
        mock_timm_resnet_backend,
    ):
        base_model = mock_timm_resnet_backend.template
        encoder = conditional_cnn_factory(real_build=True, pretrained=True)
        # conv weights in the FiLMed block must match the base model's
        # conv weights for all non-BN parameters.
        first_filmed_block = encoder.layer1[0]
        first_base_block = base_model.layer1[0]
        assert torch.allclose(
            first_filmed_block.conv1.weight.data,
            first_base_block.conv1.weight.data,
        )
        assert torch.allclose(
            first_filmed_block.conv2.weight.data,
            first_base_block.conv2.weight.data,
        )
        # layer2 has a downsample branch — verify it was copied too
        second_filmed_block = encoder.layer2[0]
        second_base_block = base_model.layer2[0]
        assert torch.allclose(
            second_filmed_block.downsample[0].weight.data,
            second_base_block.downsample[0].weight.data,
        )

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_norm_handling, expected_bn1_type",
        [
            (BatchNormHandling.FROZEN.value, nn.BatchNorm2d),
            (BatchNormHandling.DEFAULT.value, nn.BatchNorm2d),
            (BatchNormHandling.CONVERT_TO_GROUPNORM.value, nn.GroupNorm),
        ],
    )
    def test_batch_norm_handling_variants(
        self,
        conditional_cnn_factory: Callable[..., ConditionalCNNEncoder],
        batch_norm_handling: str,
        expected_bn1_type: type,
    ):
        encoder = conditional_cnn_factory(
            real_build=True,
            batch_norm_handling=batch_norm_handling,
        )
        assert isinstance(encoder.bn1, expected_bn1_type)

    @pytest.mark.unit
    def test_frozen_calls_freeze_weights_on_init(
        self,
        conditional_cnn_factory: Callable[..., ConditionalCNNEncoder],
    ):
        with patch.object(EncodingMixin, "_freeze_weights") as mock_freeze:
            conditional_cnn_factory(real_build=True, frozen=True)
        mock_freeze.assert_called_once()

    @pytest.mark.unit
    def test_set_image_size_creates_pooling_head(
        self,
        conditional_cnn_factory: Callable[..., ConditionalCNNEncoder],
    ):
        encoder = conditional_cnn_factory(real_build=True)
        assert encoder.pooling_head is None
        encoder.set_image_size(image_height=64, image_width=64)
        assert encoder.pooling_head is not None
        assert encoder.output_dim == encoder.pooling_head.output_dim

    @pytest.mark.unit
    def test_set_image_size_refreezes_when_frozen(
        self,
        conditional_cnn_factory: Callable[..., ConditionalCNNEncoder],
    ):
        with patch.object(EncodingMixin, "_freeze_weights") as mock_freeze:
            encoder = conditional_cnn_factory(real_build=True, frozen=True)
            calls_after_init = mock_freeze.call_count
            encoder.set_image_size(image_height=64, image_width=64)
        assert mock_freeze.call_count == calls_after_init + 1

    @pytest.mark.unit
    def test_encode_runs_full_conv_stack_with_conditioning(
        self,
        conditional_cnn_factory: Callable[..., ConditionalCNNEncoder],
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        conditioning_factory: Callable[..., torch.Tensor],
    ):
        batch_size = 2
        condition_dim = 64
        encoder = conditional_cnn_factory(
            real_build=True,
            condition_dim=condition_dim,
            pooling_method=PoolingMethod.AVERAGE.value,
        )
        encoder.set_image_size(image_height=64, image_width=64)
        inputs = image_input_factory(
            key="left", batch_size=batch_size, height=64, width=64
        )
        conditioning = conditioning_factory(
            batch_size=batch_size,
            condition_dim=condition_dim,
            time_steps=1,
        ).squeeze(1)
        output = encoder(inputs=inputs, conditioning=conditioning)
        rgb = output[EncoderOutputKeys.RGB.value]
        assert rgb.shape[0] == batch_size
        assert rgb.shape[-1] == encoder.feature_dim
