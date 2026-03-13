"""Tests for versatil.models.encoding.encoders.rgb.vit module."""
import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from transformers.models.timm_wrapper.modeling_timm_wrapper import (
    TimmWrapperModelOutput,
)

from versatil.data.constants import RGB_CAMERAS
from versatil.models.encoding.encoders.constants import (
    EncoderOutputKeys,
    PoolingMethod,
    RGBBackboneType,
)
from versatil.models.encoding.encoders.rgb.vit import ViTEncoder
from versatil.models.encoding.encoders.unconditional import Encoder


VIT_BACKBONES = [
    e
    for e in RGBBackboneType
    if "vit" in e.value or "dino" in e.value
]

VIT_VALID_BACKBONES = [
    e.value
    for e in RGBBackboneType
    if not any(x in e.value for x in ["efficientnet", "resnet", "edgenext", "mobilenet"])
]

FEATURE_DIM = 768
SEQUENCE_LENGTH = 196


def _mock_build_backbone(self):
    """Side-effect to set self.backbone with expected attributes."""
    self.backbone = MagicMock()
    self.backbone.config.num_features = FEATURE_DIM


def _mock_setup_feature_extractor(self):
    """Side-effect to set pooling-related attributes."""
    self.pooling_head = None
    self.output_dim = self.feature_dim


@pytest.fixture
def vit_encoder_factory() -> Callable[..., ViTEncoder]:
    """Factory for ViTEncoder with mocked backbone and feature extractor."""
    def factory(
        input_keys: str | list[str] = "left",
        backbone: str = RGBBackboneType.DINOV2_VITB14.value,
        pooling_method: str = PoolingMethod.DEFAULT.value,
        pretrained: bool = False,
        frozen: bool = False,
    ) -> ViTEncoder:
        with (
            patch.object(ViTEncoder, "_build_backbone", _mock_build_backbone),
            patch.object(
                ViTEncoder,
                "_setup_feature_extractor",
                _mock_setup_feature_extractor,
            ),
        ):
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
) -> Callable[..., TimmWrapperModelOutput]:
    """Factory for mock TimmWrapperModelOutput with last_hidden_state."""
    def factory(
        batch_size: int = 2,
        sequence_length: int = SEQUENCE_LENGTH + 1,
        feature_dim: int = FEATURE_DIM,
    ) -> TimmWrapperModelOutput:
        hidden_state = torch.from_numpy(
            rng.standard_normal(
                (batch_size, sequence_length, feature_dim)
            ).astype(np.float32)
        )
        return TimmWrapperModelOutput(last_hidden_state=hidden_state)
    return factory


class TestViTEncoderInitialization:

    @pytest.mark.parametrize("backbone, expectation", [
        (RGBBackboneType.DINOV2_VITB14.value, does_not_raise()),
        (RGBBackboneType.VIT_BASE.value, does_not_raise()),
        (RGBBackboneType.DINOV2_VITS14.value, does_not_raise()),
        (RGBBackboneType.RESNET18.value, pytest.raises(
            ValueError,
            match=re.escape(
                f"Invalid backbone '{RGBBackboneType.RESNET18.value}'. "
                f"Must be one Vision Transformer of the following: {VIT_VALID_BACKBONES}"
            ),
        )),
        ("invalid_backbone", pytest.raises(
            ValueError,
            match=re.escape(
                f"Invalid backbone 'invalid_backbone'. "
                f"Must be one Vision Transformer of the following: {VIT_VALID_BACKBONES}"
            ),
        )),
    ])
    def test_backbone_validation(
        self,
        backbone: str,
        expectation,
    ):
        with expectation:
            with (
                patch.object(ViTEncoder, "_build_backbone", _mock_build_backbone),
                patch.object(
                    ViTEncoder,
                    "_setup_feature_extractor",
                    _mock_setup_feature_extractor,
                ),
            ):
                ViTEncoder(
                    input_keys="left",
                    pretrained=False,
                    frozen=False,
                    pooling_method=PoolingMethod.DEFAULT.value,
                    backbone=backbone,
                )

    @pytest.mark.parametrize("input_keys, expectation", [
        ("left", does_not_raise()),
        ("right", does_not_raise()),
        (["left", "right"], pytest.raises(
            ValueError,
            match=re.escape(f"Exactly one from {RGB_CAMERAS} required, got"),
        )),
    ])
    def test_input_keys_validation(
        self,
        vit_encoder_factory: Callable[..., ViTEncoder],
        input_keys: str | list[str],
        expectation,
    ):
        with expectation:
            vit_encoder_factory(input_keys=input_keys)

    def test_inherits_from_encoder(
        self,
        vit_encoder_factory: Callable[..., ViTEncoder],
    ):
        encoder = vit_encoder_factory()
        assert isinstance(encoder, Encoder)

    @pytest.mark.parametrize("input_keys", ["left", "right"])
    @pytest.mark.parametrize("backbone", [
        RGBBackboneType.DINOV2_VITS14.value,
        RGBBackboneType.DINOV2_VITB14.value,
    ])
    @pytest.mark.parametrize("pooling_method", [
        PoolingMethod.DEFAULT.value,
        PoolingMethod.NONE.value,
    ])
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
        """When pooling_method is NONE, _setup_feature_extractor sets output_dim to a tuple."""
        with patch.object(ViTEncoder, "_build_backbone", _mock_build_backbone):
            encoder = ViTEncoder(
                input_keys="left",
                backbone=RGBBackboneType.DINOV2_VITB14.value,
                pooling_method=PoolingMethod.NONE.value,
                pretrained=False,
                frozen=False,
            )
        assert encoder.output_dim == (-1, FEATURE_DIM - 1)

    def test_non_none_pooling_sets_output_dim_to_int(self):
        """When pooling_method is not NONE, _setup_feature_extractor sets output_dim to an int."""
        with patch.object(ViTEncoder, "_build_backbone", _mock_build_backbone):
            encoder = ViTEncoder(
                input_keys="left",
                backbone=RGBBackboneType.DINOV2_VITB14.value,
                pooling_method=PoolingMethod.DEFAULT.value,
                pretrained=False,
                frozen=False,
            )
        assert encoder.output_dim == FEATURE_DIM


class TestViTEncoderExtractFeatures:

    def test_default_pooling_returns_cls_token(
        self,
        vit_encoder_factory: Callable[..., ViTEncoder],
        mock_backbone_output_factory: Callable[..., TimmWrapperModelOutput],
    ):
        batch_size = 2
        encoder = vit_encoder_factory(pooling_method=PoolingMethod.DEFAULT.value)
        outputs = mock_backbone_output_factory(batch_size=batch_size)
        features = encoder._extract_features(outputs=outputs)
        assert features.shape == (batch_size, FEATURE_DIM)
        # CLS token is the first token
        expected = outputs.last_hidden_state[:, 0]
        assert torch.allclose(features, expected)

    def test_average_pooling_returns_mean_of_patches(
        self,
        vit_encoder_factory: Callable[..., ViTEncoder],
        mock_backbone_output_factory: Callable[..., TimmWrapperModelOutput],
    ):
        batch_size = 2
        encoder = vit_encoder_factory(pooling_method=PoolingMethod.AVERAGE.value)
        outputs = mock_backbone_output_factory(batch_size=batch_size)
        features = encoder._extract_features(outputs=outputs)
        assert features.shape == (batch_size, FEATURE_DIM)
        # Average over patches (exclude CLS token at index 0)
        expected = outputs.last_hidden_state[:, 1:].mean(dim=1)
        assert torch.allclose(features, expected)

    def test_learned_aggregation_calls_pooling_head(
        self,
        vit_encoder_factory: Callable[..., ViTEncoder],
        mock_backbone_output_factory: Callable[..., TimmWrapperModelOutput],
    ):
        batch_size = 2
        encoder = vit_encoder_factory(
            pooling_method=PoolingMethod.LEARNED_AGGREGATION.value,
        )
        mock_pooling_head = MagicMock()
        mock_pooling_head.return_value = torch.zeros(batch_size, FEATURE_DIM - 1)
        encoder.pooling_head = mock_pooling_head
        outputs = mock_backbone_output_factory(batch_size=batch_size)
        features = encoder._extract_features(outputs=outputs)
        mock_pooling_head.assert_called_once()
        assert features.shape == (batch_size, FEATURE_DIM - 1)

    def test_learned_aggregation_raises_without_pooling_head(
        self,
        vit_encoder_factory: Callable[..., ViTEncoder],
        mock_backbone_output_factory: Callable[..., TimmWrapperModelOutput],
    ):
        encoder = vit_encoder_factory(
            pooling_method=PoolingMethod.LEARNED_AGGREGATION.value,
        )
        encoder.pooling_head = None
        outputs = mock_backbone_output_factory()
        with pytest.raises(RuntimeError, match="pooling_head must be initialized"):
            encoder._extract_features(outputs=outputs)

    def test_none_pooling_returns_all_patch_tokens(
        self,
        vit_encoder_factory: Callable[..., ViTEncoder],
        mock_backbone_output_factory: Callable[..., TimmWrapperModelOutput],
    ):
        batch_size = 2
        encoder = vit_encoder_factory(pooling_method=PoolingMethod.NONE.value)
        outputs = mock_backbone_output_factory(batch_size=batch_size)
        features = encoder._extract_features(outputs=outputs)
        assert features.shape == (batch_size, SEQUENCE_LENGTH, FEATURE_DIM)
        # All patch tokens (exclude CLS at index 0)
        expected = outputs.last_hidden_state[:, 1:]
        assert torch.allclose(features, expected)

    def test_invalid_pooling_method_raises_value_error(
        self,
        vit_encoder_factory: Callable[..., ViTEncoder],
        mock_backbone_output_factory: Callable[..., TimmWrapperModelOutput],
    ):
        encoder = vit_encoder_factory()
        encoder.pooling_method = "invalid_method"
        outputs = mock_backbone_output_factory()
        with pytest.raises(ValueError, match="Unknown feature extraction method"):
            encoder._extract_features(outputs=outputs)


class TestViTEncoderForward:

    @pytest.mark.parametrize("time_steps, expected_ndim", [
        (None, 2),
        (3, 3),
    ])
    def test_output_shape_with_and_without_time(
        self,
        vit_encoder_factory: Callable[..., ViTEncoder],
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        mock_backbone_output_factory: Callable[..., TimmWrapperModelOutput],
        time_steps: int | None,
        expected_ndim: int,
    ):
        batch_size = 2
        encoder = vit_encoder_factory(pooling_method=PoolingMethod.DEFAULT.value)
        effective_batch = batch_size * (time_steps or 1)
        backbone_output = mock_backbone_output_factory(batch_size=effective_batch)
        encoder.backbone.return_value = backbone_output
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

    def test_forward_returns_rgb_key(
        self,
        vit_encoder_factory: Callable[..., ViTEncoder],
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        mock_backbone_output_factory: Callable[..., TimmWrapperModelOutput],
    ):
        batch_size = 2
        encoder = vit_encoder_factory()
        backbone_output = mock_backbone_output_factory(batch_size=batch_size)
        encoder.backbone.return_value = backbone_output
        inputs = image_input_factory(batch_size=batch_size)
        output = encoder(inputs)
        assert EncoderOutputKeys.RGB.value in output

    def test_none_pooling_output_shape_with_time(
        self,
        vit_encoder_factory: Callable[..., ViTEncoder],
        image_input_factory: Callable[..., dict[str, torch.Tensor]],
        mock_backbone_output_factory: Callable[..., TimmWrapperModelOutput],
    ):
        batch_size = 2
        time_steps = 3
        encoder = vit_encoder_factory(pooling_method=PoolingMethod.NONE.value)
        effective_batch = batch_size * time_steps
        backbone_output = mock_backbone_output_factory(batch_size=effective_batch)
        encoder.backbone.return_value = backbone_output
        inputs = image_input_factory(
            batch_size=batch_size,
            time_steps=time_steps,
        )
        output = encoder(inputs)
        features = output[EncoderOutputKeys.RGB.value]
        # (B, T, Seq, Emb)
        assert features.ndim == 4
        assert features.shape[0] == batch_size
        assert features.shape[1] == time_steps
        assert features.shape[2] == SEQUENCE_LENGTH
        assert features.shape[3] == FEATURE_DIM


class TestViTEncoderGetOutputSpecification:

    def test_returns_rgb_feature_with_correct_dimension(
        self,
        vit_encoder_factory: Callable[..., ViTEncoder],
    ):
        encoder = vit_encoder_factory()
        specification = encoder.get_output_specification()
        assert specification.features == [EncoderOutputKeys.RGB.value]
        assert specification.dimensions[EncoderOutputKeys.RGB.value] == encoder.output_dim

    def test_output_specification_features_list_length(
        self,
        vit_encoder_factory: Callable[..., ViTEncoder],
    ):
        encoder = vit_encoder_factory()
        specification = encoder.get_output_specification()
        assert len(specification.features) == 1
        assert not specification.is_multi_output


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
        encoder = ViTEncoder(
            input_keys="left",
            backbone=RGBBackboneType.DINOV2_VITS14.value,
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
        if time_steps is not None:
            assert features.shape == (batch_size, time_steps, encoder.output_dim)
        else:
            assert features.shape == (batch_size, encoder.output_dim)

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
        encoder = ViTEncoder(
            input_keys="left",
            backbone=RGBBackboneType.DINOV2_VITS14.value,
            pooling_method=PoolingMethod.DEFAULT.value,
            pretrained=False,
            frozen=frozen,
        )
        for parameter in encoder.parameters():
            assert parameter.requires_grad is expected_requires_grad
