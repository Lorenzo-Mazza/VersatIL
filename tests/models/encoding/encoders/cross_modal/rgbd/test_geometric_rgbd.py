"""Tests for versatil.models.encoding.encoders.cross_modal.rgbd.geometric_rgbd module."""

import logging
import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock, patch

import pytest
import torch

from versatil.data.constants import CameraModality, Cameras
from versatil.data.metadata import (
    BaseMetadata,
    CameraMetadata,
    DepthCameraMetadata,
    RGBCameraMetadata,
)
from versatil.models.encoding.encoders.constants import (
    EncoderOutputKeys,
    PoolingMethod,
)
from versatil.models.encoding.encoders.cross_modal.rgbd.geometric_rgbd import (
    GeometricRGBDEncoder,
)
from versatil.models.encoding.explainability import (
    ActivationLayout,
    ExplanationTargetKind,
)
from versatil.models.layers.constants import AttentionDecompositionMode


@pytest.fixture
def light_geometric_encoder_factory(
    rgbd_camera_metadata_factory: Callable[..., dict[str, CameraMetadata]],
) -> Callable[..., GeometricRGBDEncoder]:
    """Factory for GeometricRGBDEncoder with small dimensions."""

    def factory(
        input_keys: str | list[str] | None = None,
        embedding_dimension: int = 32,
        num_heads: int = 2,
        ffn_dimension: int = 64,
        decomposition_mode: str = AttentionDecompositionMode.SEPARABLE.value,
        initial_decay: float = 2.0,
        decay_range: float = 4.0,
        patch_size: int = 16,
        pooling_method: str = PoolingMethod.AVERAGE.value,
        pretrained: bool = False,
        frozen: bool = False,
        model_dtype: str | None = None,
    ) -> GeometricRGBDEncoder:
        if input_keys is None:
            input_keys = [Cameras.LEFT.value, Cameras.DEPTH.value]
        encoder = GeometricRGBDEncoder(
            input_keys=input_keys,
            embedding_dimension=embedding_dimension,
            num_heads=num_heads,
            ffn_dimension=ffn_dimension,
            decomposition_mode=decomposition_mode,
            initial_decay=initial_decay,
            decay_range=decay_range,
            patch_size=patch_size,
            pooling_method=pooling_method,
            pretrained=pretrained,
            frozen=frozen,
            model_dtype=model_dtype,
        )
        encoder.set_camera_metadata(
            camera_metadata=rgbd_camera_metadata_factory(
                rgb_key=Cameras.LEFT.value,
                depth_key=Cameras.DEPTH.value,
                image_height=224,
                image_width=224,
            )
        )
        return encoder

    return factory


def test_geometric_rgbd_encoder_exposes_attention_block_gradcam_target(
    light_geometric_encoder_factory: Callable[..., GeometricRGBDEncoder],
):
    encoder = light_geometric_encoder_factory()
    target = encoder.get_explainability_targets()[0]
    assert target.layer is encoder.attention_block
    assert target.target_kind == ExplanationTargetKind.SPATIAL_FEATURE_MAP.value
    assert target.activation_layout == ActivationLayout.NHWC.value


class TestGeometricRGBDEncoderInitialization:
    def test_has_encoder_interface(
        self,
        light_geometric_encoder_factory: Callable[..., GeometricRGBDEncoder],
    ):
        encoder = light_geometric_encoder_factory()
        spec = encoder.get_output_specification()
        feature_keys = [m.key for m in spec]
        assert feature_keys == [EncoderOutputKeys.RGBD.value]

    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    @pytest.mark.parametrize(
        "decomposition_mode",
        [
            AttentionDecompositionMode.SEPARABLE.value,
            AttentionDecompositionMode.FULL.value,
        ],
    )
    @pytest.mark.parametrize(
        "pooling_method",
        [
            PoolingMethod.AVERAGE.value,
            PoolingMethod.SPATIAL_SOFTMAX.value,
        ],
    )
    @pytest.mark.parametrize("patch_size", [8, 16])
    def test_stores_configuration(
        self,
        light_geometric_encoder_factory: Callable[..., GeometricRGBDEncoder],
        embedding_dimension: int,
        decomposition_mode: str,
        pooling_method: str,
        patch_size: int,
    ):
        encoder = light_geometric_encoder_factory(
            embedding_dimension=embedding_dimension,
            decomposition_mode=decomposition_mode,
            pooling_method=pooling_method,
            patch_size=patch_size,
        )
        assert encoder.embedding_dimension == embedding_dimension
        assert encoder.decomposition_mode == AttentionDecompositionMode(
            decomposition_mode
        )
        assert encoder.pooling_method == pooling_method
        assert encoder.patch_size == patch_size
        assert encoder.pooling_head is None

    @pytest.mark.parametrize(
        "frozen, expectation",
        [
            (False, does_not_raise()),
            (
                True,
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "Freezing GeometricRGBDEncoder does not make sense "
                        "as it has no pretrained weights. Set frozen=False."
                    ),
                ),
            ),
        ],
    )
    def test_frozen_raises_value_error(
        self,
        light_geometric_encoder_factory: Callable[..., GeometricRGBDEncoder],
        frozen: bool,
        expectation,
    ):
        with expectation:
            light_geometric_encoder_factory(frozen=frozen)

    def test_pretrained_logs_warning(
        self,
        light_geometric_encoder_factory: Callable[..., GeometricRGBDEncoder],
        caplog,
    ):
        with caplog.at_level(logging.WARNING):
            encoder = light_geometric_encoder_factory(pretrained=True)
        assert "does not support pretrained weights" in caplog.text
        # Encoder is still created successfully despite the warning
        specification = encoder.get_output_specification()
        feature_keys = [m.key for m in specification]
        assert feature_keys == [EncoderOutputKeys.RGBD.value]

    def test_input_specification_requires_rgb_and_depth_modalities(
        self,
        light_geometric_encoder_factory: Callable[..., GeometricRGBDEncoder],
    ):
        encoder = light_geometric_encoder_factory()
        assert encoder.input_specification.required_camera_modalities == [
            CameraModality.RGB,
            CameraModality.DEPTH,
        ]

    def test_input_specification_requires_one_rgb_and_one_depth_modality(
        self,
        light_geometric_encoder_factory: Callable[..., GeometricRGBDEncoder],
    ):
        encoder = light_geometric_encoder_factory()
        assert encoder.input_specification.exactly_one_camera_modality == [
            CameraModality.RGB,
            CameraModality.DEPTH,
        ]


class TestGeometricRGBDEncoderMixin:
    def test_camera_keys_include_configured_rgb_and_depth(
        self,
        light_geometric_encoder_factory: Callable[..., GeometricRGBDEncoder],
    ):
        encoder = light_geometric_encoder_factory()
        assert encoder.camera_keys == [Cameras.LEFT.value, Cameras.DEPTH.value]

    def test_output_modality_is_rgbd(
        self,
        light_geometric_encoder_factory: Callable[..., GeometricRGBDEncoder],
    ):
        encoder = light_geometric_encoder_factory()
        assert encoder._output_modality == EncoderOutputKeys.RGBD.value

    def test_encode_single_image_raises(
        self,
        light_geometric_encoder_factory: Callable[..., GeometricRGBDEncoder],
    ):
        encoder = light_geometric_encoder_factory()
        with pytest.raises(
            NotImplementedError,
            match=re.escape(
                "GeometricRGBDEncoder processes RGB+depth jointly. "
                "Use encode() instead."
            ),
        ):
            encoder._encode_single_image(torch.zeros(1, 3, 32, 32))


class TestGeometricRGBDEncoderGetOutputSpecification:
    def test_returns_rgbd_feature_with_correct_dimension(
        self,
        light_geometric_encoder_factory: Callable[..., GeometricRGBDEncoder],
    ):
        encoder = light_geometric_encoder_factory()
        specification = encoder.get_output_specification()
        feature_keys = [m.key for m in specification]
        assert feature_keys == [EncoderOutputKeys.RGBD.value]
        assert next(
            m for m in specification if m.key == EncoderOutputKeys.RGBD.value
        ).dimension == (encoder.output_dim,)


class TestGeometricRGBDEncoderValidateInputMetadata:
    @pytest.mark.parametrize(
        "key, metadata, expected_error",
        [
            (
                Cameras.LEFT.value,
                RGBCameraMetadata(
                    camera_key="left",
                    dtype="uint8",
                    image_height=224,
                    image_width=224,
                ),
                None,
            ),
            (
                Cameras.LEFT.value,
                CameraMetadata(
                    camera_key="left",
                    dtype="uint8",
                    channels=1,
                    image_height=224,
                    image_width=224,
                ),
                None,
            ),
            (
                Cameras.DEPTH.value,
                DepthCameraMetadata(
                    camera_key="depth",
                    dtype="float32",
                    image_height=224,
                    image_width=224,
                ),
                None,
            ),
            (
                Cameras.DEPTH.value,
                CameraMetadata(
                    camera_key="depth",
                    dtype="uint8",
                    channels=3,
                    image_height=224,
                    image_width=224,
                ),
                None,
            ),
            (
                Cameras.LEFT.value,
                MagicMock(spec=BaseMetadata),
                f"Expected CameraMetadata for '{Cameras.LEFT.value}', got MagicMock",
            ),
        ],
    )
    def test_validates_rgb_and_depth_metadata(
        self,
        light_geometric_encoder_factory: Callable[..., GeometricRGBDEncoder],
        key: str,
        metadata,
        expected_error: str | None,
    ):
        encoder = light_geometric_encoder_factory()
        result = encoder.validate_input_metadata(key=key, metadata=metadata)
        assert result == expected_error


class TestGeometricRGBDEncoderForward:
    @pytest.mark.parametrize("time_steps", [1, 2])
    def test_output_shape_with_temporal_dimension(
        self,
        light_geometric_encoder_factory: Callable[..., GeometricRGBDEncoder],
        rgbd_input_factory: Callable[..., dict[str, torch.Tensor]],
        time_steps: int,
    ):
        batch_size = 2
        encoder = light_geometric_encoder_factory()
        encoder.set_image_size(image_height=224, image_width=224)
        inputs = rgbd_input_factory(
            batch_size=batch_size,
            time_steps=time_steps,
        )
        output = encoder(inputs)
        features = output[EncoderOutputKeys.RGBD.value]
        assert features.shape == (batch_size, time_steps, encoder.output_dim)

    def test_output_feature_dimension_matches_specification(
        self,
        light_geometric_encoder_factory: Callable[..., GeometricRGBDEncoder],
        rgbd_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 2
        encoder = light_geometric_encoder_factory()
        encoder.set_image_size(image_height=224, image_width=224)
        inputs = rgbd_input_factory(batch_size=batch_size)
        output = encoder(inputs)
        features = output[EncoderOutputKeys.RGBD.value]
        assert features.shape[-1] == encoder.output_dim

    def test_raises_when_pooling_head_not_initialized(
        self,
        light_geometric_encoder_factory: Callable[..., GeometricRGBDEncoder],
        rgbd_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        encoder = light_geometric_encoder_factory()
        inputs = rgbd_input_factory()
        with pytest.raises(
            RuntimeError,
            match="pooling_head is not initialized. Call set_image_size",
        ):
            encoder(inputs)

    def test_output_key_is_rgbd(
        self,
        light_geometric_encoder_factory: Callable[..., GeometricRGBDEncoder],
        rgbd_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        encoder = light_geometric_encoder_factory()
        encoder.set_image_size(image_height=224, image_width=224)
        inputs = rgbd_input_factory()
        output = encoder(inputs)
        assert EncoderOutputKeys.RGBD.value in output

    def test_temporal_reshaping_produces_correct_shape(
        self,
        light_geometric_encoder_factory: Callable[..., GeometricRGBDEncoder],
        rgbd_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 2
        time_steps = 3
        encoder = light_geometric_encoder_factory()
        encoder.set_image_size(image_height=224, image_width=224)
        inputs = rgbd_input_factory(
            batch_size=batch_size,
            time_steps=time_steps,
        )
        output = encoder(inputs)
        features = output[EncoderOutputKeys.RGBD.value]
        assert features.shape == (batch_size, time_steps, encoder.output_dim)


class TestGeometricRGBDEncoderIntegration:
    @pytest.mark.integration
    def test_exposes_real_attention_block_gradcam_target(
        self,
        rgbd_camera_metadata_factory: Callable[..., dict[str, CameraMetadata]],
    ):
        encoder = GeometricRGBDEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            embedding_dimension=32,
            num_heads=2,
            ffn_dimension=64,
            pooling_method=PoolingMethod.AVERAGE.value,
        )
        encoder.set_camera_metadata(
            camera_metadata=rgbd_camera_metadata_factory(
                rgb_key=Cameras.LEFT.value,
                depth_key=Cameras.DEPTH.value,
                image_height=224,
                image_width=224,
            )
        )

        target = encoder.get_explainability_targets()[0]

        assert target.layer is encoder.attention_block
        assert target.target_kind == ExplanationTargetKind.SPATIAL_FEATURE_MAP.value
        assert target.activation_layout == ActivationLayout.NHWC.value

    @pytest.mark.integration
    def test_forward_pass(
        self,
        rgbd_input_factory: Callable[..., dict[str, torch.Tensor]],
        rgbd_camera_metadata_factory: Callable[..., dict[str, CameraMetadata]],
    ):
        batch_size = 2
        encoder = GeometricRGBDEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            embedding_dimension=32,
            num_heads=2,
            ffn_dimension=64,
            pooling_method=PoolingMethod.AVERAGE.value,
        )
        encoder.set_camera_metadata(
            camera_metadata=rgbd_camera_metadata_factory(
                rgb_key=Cameras.LEFT.value,
                depth_key=Cameras.DEPTH.value,
                image_height=224,
                image_width=224,
            )
        )
        encoder.set_image_size(image_height=224, image_width=224)
        inputs = rgbd_input_factory(batch_size=batch_size)
        output = encoder(inputs)
        features = output[EncoderOutputKeys.RGBD.value]
        assert features.shape == (batch_size, 1, encoder.output_dim)

    @pytest.mark.integration
    @pytest.mark.parametrize("time_steps", [1, 2])
    def test_temporal_reshaping(
        self,
        rgbd_input_factory: Callable[..., dict[str, torch.Tensor]],
        rgbd_camera_metadata_factory: Callable[..., dict[str, CameraMetadata]],
        time_steps: int,
    ):
        batch_size = 2
        encoder = GeometricRGBDEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            embedding_dimension=32,
            num_heads=2,
            ffn_dimension=64,
            pooling_method=PoolingMethod.AVERAGE.value,
        )
        encoder.set_camera_metadata(
            camera_metadata=rgbd_camera_metadata_factory(
                rgb_key=Cameras.LEFT.value,
                depth_key=Cameras.DEPTH.value,
                image_height=224,
                image_width=224,
            )
        )
        encoder.set_image_size(image_height=224, image_width=224)
        inputs = rgbd_input_factory(
            batch_size=batch_size,
            time_steps=time_steps,
        )
        output = encoder(inputs)
        features = output[EncoderOutputKeys.RGBD.value]
        assert features.shape == (batch_size, time_steps, encoder.output_dim)


class TestGeometricRGBDEncoderModelDtype:
    @pytest.mark.unit
    def test_apply_model_dtype_called_once_in_init(self):
        with patch.object(GeometricRGBDEncoder, "_apply_model_dtype") as mock_apply:
            GeometricRGBDEncoder(
                input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
                embedding_dimension=32,
                num_heads=2,
                ffn_dimension=64,
                patch_size=16,
                pretrained=False,
            )
        mock_apply.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "model_dtype, expected_dtype",
        [
            (None, torch.float32),
            ("32", torch.float32),
            ("bf16-mixed", torch.bfloat16),
        ],
    )
    def test_all_parameters_share_model_dtype_after_init(
        self,
        light_geometric_encoder_factory: Callable[..., GeometricRGBDEncoder],
        model_dtype: str | None,
        expected_dtype: torch.dtype,
    ):
        encoder = light_geometric_encoder_factory(model_dtype=model_dtype)
        for parameter in encoder.parameters():
            assert parameter.dtype == expected_dtype

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "model_dtype, expected_dtype",
        [("32", torch.float32), ("bf16-mixed", torch.bfloat16)],
    )
    def test_set_image_size_preserves_model_dtype(
        self,
        light_geometric_encoder_factory: Callable[..., GeometricRGBDEncoder],
        model_dtype: str,
        expected_dtype: torch.dtype,
    ):
        encoder = light_geometric_encoder_factory(model_dtype=model_dtype)
        encoder.set_image_size(image_height=224, image_width=224)
        for parameter in encoder.parameters():
            assert parameter.dtype == expected_dtype
