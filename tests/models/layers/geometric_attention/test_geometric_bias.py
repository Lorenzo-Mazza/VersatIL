"""Tests for versatil.models.layers.geometric_attention.geometric_bias module."""

import pytest
import torch

from versatil.models.layers.constants import AttentionDecompositionMode


class TestGeometricAttentionBiasConfiguration:
    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    @pytest.mark.parametrize("num_heads", [4, 8])
    def test_stores_configuration(
        self, geometric_bias_factory, embedding_dimension, num_heads
    ):
        bias = geometric_bias_factory(
            embedding_dimension=embedding_dimension,
            num_heads=num_heads,
        )
        assert bias.embedding_dimension == embedding_dimension
        assert bias.num_heads == num_heads
        assert bias.bias_weights.shape == (2, 1, 1, 1)
        assert bias.bias_weights.requires_grad is True
        assert bias.spatial_decay.num_heads == num_heads
        assert bias.depth_decay.num_heads == num_heads
        assert bias.rotary_encoding.embedding_dimension == embedding_dimension
        assert bias.rotary_encoding.num_heads == num_heads

    def test_bias_weights_initialized_to_ones(self, geometric_bias_factory):
        bias = geometric_bias_factory()
        assert torch.allclose(bias.bias_weights, torch.ones(2, 1, 1, 1))


class TestGeometricBiasForwardFull:
    def test_returns_rotation_components_and_single_bias_mask(
        self, geometric_bias_factory, depth_map_factory
    ):
        embedding_dimension = 32
        num_heads = 4
        bias = geometric_bias_factory(
            embedding_dimension=embedding_dimension, num_heads=num_heads
        )
        height, width = 4, 6
        depth_map = depth_map_factory(batch_size=2, height=height, width=width)
        (sine, cosine), bias_masks = bias(
            height=height,
            width=width,
            depth_map=depth_map,
            device=depth_map.device,
            decomposition_mode=AttentionDecompositionMode.FULL.value,
        )
        head_dimension = embedding_dimension // num_heads
        assert sine.shape == (height, width, head_dimension)
        assert cosine.shape == (height, width, head_dimension)
        assert len(bias_masks) == 1

    @pytest.mark.parametrize(
        "num_heads, height, width",
        [(4, 3, 5), (8, 4, 4)],
    )
    def test_full_bias_mask_shape(
        self, geometric_bias_factory, depth_map_factory, num_heads, height, width
    ):
        bias = geometric_bias_factory(num_heads=num_heads)
        batch_size = 2
        depth_map = depth_map_factory(batch_size=batch_size, height=height, width=width)
        (sine, cosine), bias_masks = bias(
            height=height,
            width=width,
            depth_map=depth_map,
            device=depth_map.device,
            decomposition_mode=AttentionDecompositionMode.FULL.value,
        )
        sequence_length = height * width
        assert bias_masks[0].shape == (
            batch_size,
            num_heads,
            sequence_length,
            sequence_length,
        )

    def test_bias_weights_control_spatial_vs_depth_contribution(
        self, geometric_bias_factory, depth_map_factory
    ):
        bias = geometric_bias_factory(num_heads=4)
        depth_map = depth_map_factory(batch_size=1, height=3, width=3)

        _, baseline_masks = bias(
            height=3,
            width=3,
            depth_map=depth_map,
            device=depth_map.device,
        )
        baseline = baseline_masks[0].clone()

        # Zero out depth contribution weight
        with torch.no_grad():
            bias.bias_weights[1] = 0.0

        _, spatial_only_masks = bias(
            height=3,
            width=3,
            depth_map=depth_map,
            device=depth_map.device,
        )
        spatial_only = spatial_only_masks[0]

        # Baseline includes both spatial + depth; spatial_only has just spatial
        assert not torch.allclose(baseline, spatial_only, atol=1e-6)

    def test_zero_spatial_weight_removes_spatial_component(
        self, geometric_bias_factory, depth_map_factory
    ):
        bias = geometric_bias_factory(num_heads=4)
        depth_map = depth_map_factory(batch_size=1, height=3, width=3)

        _, full_masks = bias(
            height=3,
            width=3,
            depth_map=depth_map,
            device=depth_map.device,
        )
        full_bias = full_masks[0].clone()

        # Zero out spatial weight
        with torch.no_grad():
            bias.bias_weights[0] = 0.0

        _, depth_only_masks = bias(
            height=3,
            width=3,
            depth_map=depth_map,
            device=depth_map.device,
        )
        depth_only = depth_only_masks[0]

        assert not torch.allclose(full_bias, depth_only, atol=1e-6)

    def test_zero_both_weights_produces_zero_bias(
        self, geometric_bias_factory, depth_map_factory
    ):
        bias = geometric_bias_factory(num_heads=4)
        depth_map = depth_map_factory(batch_size=1, height=3, width=3)

        with torch.no_grad():
            bias.bias_weights[0] = 0.0
            bias.bias_weights[1] = 0.0

        _, zero_masks = bias(
            height=3,
            width=3,
            depth_map=depth_map,
            device=depth_map.device,
        )
        assert torch.allclose(zero_masks[0], torch.zeros_like(zero_masks[0]))

    def test_uniform_depth_full_bias_equals_spatial_only(self, geometric_bias_factory):
        # With uniform depth, depth differences are zero, so combined = spatial_weight * spatial
        bias = geometric_bias_factory(num_heads=4)
        height, width = 3, 3
        uniform_depth = torch.ones(1, 1, height, width)

        _, combined_masks = bias(
            height=height,
            width=width,
            depth_map=uniform_depth,
            device=uniform_depth.device,
        )

        # Zero out depth weight and compare
        with torch.no_grad():
            bias.bias_weights[1] = 0.0

        _, spatial_only_masks = bias(
            height=height,
            width=width,
            depth_map=uniform_depth,
            device=uniform_depth.device,
        )

        # With uniform depth, depth_mask is all zeros, so combined should equal spatial-only
        assert torch.allclose(combined_masks[0], spatial_only_masks[0], atol=1e-6)


class TestGeometricBiasForwardSeparable:
    def test_returns_two_bias_masks(self, geometric_bias_factory, depth_map_factory):
        bias = geometric_bias_factory()
        depth_map = depth_map_factory(batch_size=2, height=4, width=6)
        (sine, cosine), bias_masks = bias(
            height=4,
            width=6,
            depth_map=depth_map,
            device=depth_map.device,
            decomposition_mode=AttentionDecompositionMode.SEPARABLE.value,
        )
        assert len(bias_masks) == 2

    @pytest.mark.parametrize(
        "batch_size, num_heads, height, width",
        [(2, 4, 3, 5), (1, 8, 4, 4)],
    )
    def test_separable_bias_mask_shapes(
        self,
        geometric_bias_factory,
        depth_map_factory,
        batch_size,
        num_heads,
        height,
        width,
    ):
        bias = geometric_bias_factory(num_heads=num_heads)
        depth_map = depth_map_factory(batch_size=batch_size, height=height, width=width)
        (sine, cosine), (height_bias, width_bias) = bias(
            height=height,
            width=width,
            depth_map=depth_map,
            device=depth_map.device,
            decomposition_mode=AttentionDecompositionMode.SEPARABLE.value,
        )
        assert height_bias.shape == (batch_size, num_heads, width, height, height)
        assert width_bias.shape == (batch_size, num_heads, height, width, width)

    def test_separable_zero_both_weights_produces_zero_bias(
        self, geometric_bias_factory, depth_map_factory
    ):
        bias = geometric_bias_factory(num_heads=4)
        depth_map = depth_map_factory(batch_size=1, height=3, width=4)

        with torch.no_grad():
            bias.bias_weights[0] = 0.0
            bias.bias_weights[1] = 0.0

        _, (height_bias, width_bias) = bias(
            height=3,
            width=4,
            depth_map=depth_map,
            device=depth_map.device,
            decomposition_mode=AttentionDecompositionMode.SEPARABLE.value,
        )
        assert torch.allclose(height_bias, torch.zeros_like(height_bias))
        assert torch.allclose(width_bias, torch.zeros_like(width_bias))
