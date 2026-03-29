"""Tests for versatil.models.encoding.encoders.cross_modal.rgbd.geometric_rgbd module."""

import logging
import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock

import pytest
import torch

from versatil.data.constants import RGB_CAMERAS, Cameras
from versatil.data.metadata import BaseMetadata, CameraMetadata
from versatil.models.encoding.encoders.constants import (
    EncoderOutputKeys,
    PoolingMethod,
)
from versatil.models.encoding.encoders.cross_modal.rgbd.geometric_rgbd import (
    GeometricRGBDEncoder,
)
from versatil.models.layers.constants import AttentionDecompositionMode


@pytest.fixture
def light_geometric_encoder_factory() -> Callable[..., GeometricRGBDEncoder]:
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
    ) -> GeometricRGBDEncoder:
        if input_keys is None:
            input_keys = [Cameras.LEFT.value, Cameras.DEPTH.value]
        return GeometricRGBDEncoder(
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
        )

    return factory


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

    def test_requires_depth_in_input_keys(
        self,
    ):
        with pytest.raises(
            ValueError,
            match=re.escape("Missing required inputs: {'depth'}"),
        ):
            GeometricRGBDEncoder(
                input_keys=Cameras.LEFT.value,
                embedding_dimension=32,
                num_heads=2,
                ffn_dimension=64,
            )

    def test_requires_rgb_camera_in_input_keys(
        self,
    ):
        with pytest.raises(
            ValueError,
            match=re.escape(f"Exactly one from {RGB_CAMERAS} required, got set()"),
        ):
            GeometricRGBDEncoder(
                input_keys=Cameras.DEPTH.value,
                embedding_dimension=32,
                num_heads=2,
                ffn_dimension=64,
            )

    def test_input_specification_requires_depth_camera(
        self,
        light_geometric_encoder_factory: Callable[..., GeometricRGBDEncoder],
    ):
        encoder = light_geometric_encoder_factory()
        assert Cameras.DEPTH.value in encoder.input_specification.required

    def test_input_specification_requires_one_rgb_camera(
        self,
        light_geometric_encoder_factory: Callable[..., GeometricRGBDEncoder],
    ):
        encoder = light_geometric_encoder_factory()
        assert encoder.input_specification.one_of_groups == [RGB_CAMERAS]


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
                Cameras.LEFT.value,
                CameraMetadata(
                    camera_key="left",
                    dtype="uint8",
                    channels=1,
                    image_height=224,
                    image_width=224,
                ),
                f"Expected 3-channel RGB for '{Cameras.LEFT.value}', got 1 channels",
            ),
            (
                Cameras.DEPTH.value,
                CameraMetadata(
                    camera_key="depth",
                    dtype="float32",
                    channels=1,
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
                f"Expected single-channel depth for '{Cameras.DEPTH.value}', got 3 channels",
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
    def test_forward_pass(
        self,
        rgbd_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 2
        encoder = GeometricRGBDEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            embedding_dimension=32,
            num_heads=2,
            ffn_dimension=64,
            pooling_method=PoolingMethod.AVERAGE.value,
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
        encoder.set_image_size(image_height=224, image_width=224)
        inputs = rgbd_input_factory(
            batch_size=batch_size,
            time_steps=time_steps,
        )
        output = encoder(inputs)
        features = output[EncoderOutputKeys.RGBD.value]
        assert features.shape == (batch_size, time_steps, encoder.output_dim)
