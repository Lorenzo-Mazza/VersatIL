from collections.abc import Callable

import numpy as np
import pytest
import torch

from versatil.models.layers.constants import AttentionDecompositionMode
from versatil.models.layers.geometric_attention.depth_decay import DepthAwareDecayMask
from versatil.models.layers.geometric_attention.geometric_attention import (
    GeometricSelfAttention,
)
from versatil.models.layers.geometric_attention.geometric_attention_encoder import (
    GeometricAttentionEncoderBlock,
)
from versatil.models.layers.geometric_attention.geometric_bias import (
    GeometricAttentionBias,
)
from versatil.models.layers.geometric_attention.spatial_decay import SpatialDecayMask


@pytest.fixture
def depth_map_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for depth maps of shape (B, 1, H, W) with positive values."""

    def factory(
        batch_size: int = 2,
        height: int = 4,
        width: int = 6,
    ) -> torch.Tensor:
        data = rng.uniform(low=0.1, high=5.0, size=(batch_size, 1, height, width))
        return torch.from_numpy(data.astype(np.float32))

    return factory


@pytest.fixture
def spatial_decay_factory() -> Callable[..., SpatialDecayMask]:
    """Factory for SpatialDecayMask instances."""

    def factory(
        num_heads: int = 4,
        initial_decay: float = 5.0,
        decay_range: float = 3.0,
    ) -> SpatialDecayMask:
        return SpatialDecayMask(
            num_heads=num_heads,
            initial_decay=initial_decay,
            decay_range=decay_range,
        )

    return factory


@pytest.fixture
def depth_decay_factory() -> Callable[..., DepthAwareDecayMask]:
    """Factory for DepthAwareDecayMask instances."""

    def factory(
        num_heads: int = 4,
    ) -> DepthAwareDecayMask:
        return DepthAwareDecayMask(num_heads=num_heads)

    return factory


@pytest.fixture
def geometric_bias_factory() -> Callable[..., GeometricAttentionBias]:
    """Factory for GeometricAttentionBias instances."""

    def factory(
        embedding_dimension: int = 32,
        num_heads: int = 4,
        initial_decay: float = 5.0,
        decay_range: float = 3.0,
        base_frequency: float = 10000.0,
    ) -> GeometricAttentionBias:
        return GeometricAttentionBias(
            embedding_dimension=embedding_dimension,
            num_heads=num_heads,
            initial_decay=initial_decay,
            decay_range=decay_range,
            base_frequency=base_frequency,
        )

    return factory


@pytest.fixture
def geometric_attention_factory() -> Callable[..., GeometricSelfAttention]:
    """Factory for GeometricSelfAttention instances."""

    def factory(
        embedding_dimension: int = 32,
        num_heads: int = 4,
        value_dimension_factor: int = 1,
        decomposition_mode: str = AttentionDecompositionMode.FULL.value,
        initial_decay: float = 5.0,
        decay_range: float = 3.0,
        depthwise_convolution_kernel_size: int = 5,
        depthwise_convolution_padding: int = 2,
    ) -> GeometricSelfAttention:
        return GeometricSelfAttention(
            embedding_dimension=embedding_dimension,
            num_heads=num_heads,
            value_dimension_factor=value_dimension_factor,
            decomposition_mode=decomposition_mode,
            initial_decay=initial_decay,
            decay_range=decay_range,
            depthwise_convolution_kernel_size=depthwise_convolution_kernel_size,
            depthwise_convolution_padding=depthwise_convolution_padding,
        )

    return factory


@pytest.fixture
def encoder_block_factory() -> Callable[..., GeometricAttentionEncoderBlock]:
    """Factory for GeometricAttentionEncoderBlock instances."""

    def factory(
        decomposition_mode: AttentionDecompositionMode = AttentionDecompositionMode.FULL,
        embedding_dimension: int = 32,
        num_heads: int = 4,
        ffn_dimension: int = 64,
        drop_path_rate: float = 0.0,
        use_layer_scale: bool = False,
        layer_scale_init_value: float = 1e-5,
        initial_decay: float = 2.0,
        decay_range: float = 4.0,
        value_dimension_factor: int = 1,
    ) -> GeometricAttentionEncoderBlock:
        return GeometricAttentionEncoderBlock(
            decomposition_mode=decomposition_mode,
            embedding_dimension=embedding_dimension,
            num_heads=num_heads,
            ffn_dimension=ffn_dimension,
            drop_path_rate=drop_path_rate,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            initial_decay=initial_decay,
            decay_range=decay_range,
            value_dimension_factor=value_dimension_factor,
        )

    return factory
