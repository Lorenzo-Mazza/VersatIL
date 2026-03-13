"""Tests for versatil.models.encoding.encoders.depth.light_geometric module."""
import logging
import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise

import pytest
import torch

from versatil.data.constants import Cameras, RGB_CAMERAS
from versatil.models.encoding.encoders.constants import (
    EncoderOutputKeys,
    PoolingMethod,
)
from versatil.models.encoding.encoders.depth.light_geometric import (
    LightGeometricEncoder,
)
from versatil.models.encoding.encoders.unconditional import Encoder
from versatil.models.layers.constants import AttentionDecompositionMode


@pytest.fixture
def light_geometric_encoder_factory() -> Callable[..., LightGeometricEncoder]:
    """Factory for LightGeometricEncoder with small dimensions."""
    def factory(
        input_keys: str | list[str] = [Cameras.LEFT.value, Cameras.DEPTH.value],
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
    ) -> LightGeometricEncoder:
        return LightGeometricEncoder(
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


class TestLightGeometricEncoderInitialization:

    def test_inherits_from_encoder(
        self,
        light_geometric_encoder_factory: Callable[..., LightGeometricEncoder],
    ):
        encoder = light_geometric_encoder_factory()
        assert isinstance(encoder, Encoder)

    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    @pytest.mark.parametrize("decomposition_mode", [
        AttentionDecompositionMode.SEPARABLE.value,
        AttentionDecompositionMode.FULL.value,
    ])
    @pytest.mark.parametrize("pooling_method", [
        PoolingMethod.AVERAGE.value,
        PoolingMethod.SPATIAL_SOFTMAX.value,
    ])
    @pytest.mark.parametrize("patch_size", [8, 16])
    def test_stores_configuration(
        self,
        light_geometric_encoder_factory: Callable[..., LightGeometricEncoder],
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
        assert encoder.decomposition_mode == AttentionDecompositionMode(decomposition_mode)
        assert encoder.pooling_method == pooling_method
        assert encoder.patch_size == patch_size
        assert encoder.pooling_head is None

    @pytest.mark.parametrize("frozen, expectation", [
        (False, does_not_raise()),
        (True, pytest.raises(
            ValueError,
            match=re.escape(
                "Freezing LightGeometricEncoder does not make sense "
                "as it has no pretrained weights. Set frozen=False."
            ),
        )),
    ])
    def test_frozen_raises_value_error(
        self,
        light_geometric_encoder_factory: Callable[..., LightGeometricEncoder],
        frozen: bool,
        expectation,
    ):
        with expectation:
            light_geometric_encoder_factory(frozen=frozen)

    def test_pretrained_logs_warning(
        self,
        light_geometric_encoder_factory: Callable[..., LightGeometricEncoder],
        caplog,
    ):
        with caplog.at_level(logging.WARNING):
            encoder = light_geometric_encoder_factory(pretrained=True)
        assert "does not support pretrained weights" in caplog.text
        assert isinstance(encoder, LightGeometricEncoder)

    def test_requires_depth_in_input_keys(
        self,
    ):
        with pytest.raises(
            ValueError,
            match=re.escape(f"Missing required inputs: {{'depth'}}"),
        ):
            LightGeometricEncoder(
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
            LightGeometricEncoder(
                input_keys=Cameras.DEPTH.value,
                embedding_dimension=32,
                num_heads=2,
                ffn_dimension=64,
            )

    def test_input_specification_requires_depth_camera(
        self,
        light_geometric_encoder_factory: Callable[..., LightGeometricEncoder],
    ):
        encoder = light_geometric_encoder_factory()
        assert Cameras.DEPTH.value in encoder.input_specification.required

    def test_input_specification_requires_one_rgb_camera(
        self,
        light_geometric_encoder_factory: Callable[..., LightGeometricEncoder],
    ):
        encoder = light_geometric_encoder_factory()
        assert encoder.input_specification.one_of_groups == [RGB_CAMERAS]


class TestLightGeometricEncoderGetOutputSpecification:

    def test_returns_rgbd_feature_with_correct_dimension(
        self,
        light_geometric_encoder_factory: Callable[..., LightGeometricEncoder],
    ):
        encoder = light_geometric_encoder_factory()
        specification = encoder.get_output_specification()
        assert specification.features == [EncoderOutputKeys.RGBD.value]
        assert specification.dimensions[EncoderOutputKeys.RGBD.value] == encoder.output_dim


class TestLightGeometricEncoderForward:

    @pytest.mark.parametrize("time_steps, expected_ndim", [
        (None, 2),
        (2, 3),
    ])
    def test_output_shape_with_and_without_time(
        self,
        light_geometric_encoder_factory: Callable[..., LightGeometricEncoder],
        rgbd_input_factory: Callable[..., dict[str, torch.Tensor]],
        time_steps: int | None,
        expected_ndim: int,
    ):
        batch_size = 2
        encoder = light_geometric_encoder_factory()
        inputs = rgbd_input_factory(
            batch_size=batch_size,
            time_steps=time_steps,
        )
        output = encoder(inputs)
        features = output[EncoderOutputKeys.RGBD.value]
        assert features.ndim == expected_ndim
        assert features.shape[0] == batch_size
        if time_steps is not None:
            assert features.shape[1] == time_steps

    def test_output_feature_dimension_matches_specification(
        self,
        light_geometric_encoder_factory: Callable[..., LightGeometricEncoder],
        rgbd_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 2
        encoder = light_geometric_encoder_factory()
        inputs = rgbd_input_factory(batch_size=batch_size)
        output = encoder(inputs)
        features = output[EncoderOutputKeys.RGBD.value]
        assert features.shape[-1] == encoder.output_dim

    def test_creates_pooling_head_on_first_forward(
        self,
        light_geometric_encoder_factory: Callable[..., LightGeometricEncoder],
        rgbd_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        encoder = light_geometric_encoder_factory()
        assert encoder.pooling_head is None
        inputs = rgbd_input_factory()
        encoder(inputs)
        assert encoder.pooling_head is not None

    def test_output_key_is_rgbd(
        self,
        light_geometric_encoder_factory: Callable[..., LightGeometricEncoder],
        rgbd_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        encoder = light_geometric_encoder_factory()
        inputs = rgbd_input_factory()
        output = encoder(inputs)
        assert EncoderOutputKeys.RGBD.value in output

    def test_temporal_reshaping_produces_correct_shape(
        self,
        light_geometric_encoder_factory: Callable[..., LightGeometricEncoder],
        rgbd_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 2
        time_steps = 3
        encoder = light_geometric_encoder_factory()
        inputs = rgbd_input_factory(
            batch_size=batch_size,
            time_steps=time_steps,
        )
        output = encoder(inputs)
        features = output[EncoderOutputKeys.RGBD.value]
        assert features.shape == (batch_size, time_steps, encoder.output_dim)


class TestLightGeometricEncoderIntegration:

    @pytest.mark.integration
    def test_forward_pass(
        self,
        rgbd_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 2
        encoder = LightGeometricEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            embedding_dimension=32,
            num_heads=2,
            ffn_dimension=64,
            pooling_method=PoolingMethod.AVERAGE.value,
        )
        inputs = rgbd_input_factory(batch_size=batch_size)
        output = encoder(inputs)
        features = output[EncoderOutputKeys.RGBD.value]
        assert features.ndim == 2
        assert features.shape[0] == batch_size
        assert features.shape[1] == encoder.output_dim

    @pytest.mark.integration
    @pytest.mark.parametrize("time_steps", [None, 2])
    def test_temporal_reshaping(
        self,
        rgbd_input_factory: Callable[..., dict[str, torch.Tensor]],
        time_steps: int | None,
    ):
        batch_size = 2
        encoder = LightGeometricEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            embedding_dimension=32,
            num_heads=2,
            ffn_dimension=64,
            pooling_method=PoolingMethod.AVERAGE.value,
        )
        inputs = rgbd_input_factory(
            batch_size=batch_size,
            time_steps=time_steps,
        )
        output = encoder(inputs)
        features = output[EncoderOutputKeys.RGBD.value]
        if time_steps is not None:
            assert features.shape == (batch_size, time_steps, encoder.output_dim)
        else:
            assert features.shape == (batch_size, encoder.output_dim)
