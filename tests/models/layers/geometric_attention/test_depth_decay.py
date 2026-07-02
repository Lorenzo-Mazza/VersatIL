"""Tests for versatil.models.layers.geometric_attention.depth_decay module."""

from collections.abc import Callable

import numpy as np
import pytest
import torch

from versatil.models.layers.constants import AttentionDecompositionMode, Axis
from versatil.models.layers.geometric_attention.depth_decay import DepthAwareDecayMask


@pytest.fixture
def uniform_depth_map_factory() -> Callable[..., torch.Tensor]:
    """Factory for depth maps with uniform depth (no discontinuities)."""

    def factory(
        batch_size: int = 2,
        height: int = 4,
        width: int = 6,
        depth_value: float = 1.0,
    ) -> torch.Tensor:
        return torch.full((batch_size, 1, height, width), depth_value)

    return factory


@pytest.fixture
def discontinuous_depth_map_factory() -> Callable[..., torch.Tensor]:
    """Factory for depth maps with a sharp depth edge at the width midpoint."""

    def factory(
        batch_size: int = 2,
        height: int = 4,
        width: int = 6,
        near_depth: float = 1.0,
        far_depth: float = 5.0,
    ) -> torch.Tensor:
        depth_map = torch.full((batch_size, 1, height, width), near_depth)
        half_width = width // 2
        depth_map[:, :, :, half_width:] = far_depth
        return depth_map

    return factory


@pytest.fixture
def height_varying_depth_map_factory() -> Callable[..., torch.Tensor]:
    """Factory for depth maps with a sharp depth edge at the height midpoint."""

    def factory(
        batch_size: int = 1,
        height: int = 4,
        width: int = 4,
        near_depth: float = 1.0,
        far_depth: float = 5.0,
    ) -> torch.Tensor:
        depth_map = torch.full((batch_size, 1, height, width), near_depth)
        half_height = height // 2
        depth_map[:, :, half_height:, :] = far_depth
        return depth_map

    return factory


@pytest.fixture
def decay_rates_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for per-head negative decay rate tensors."""

    def factory(
        num_heads: int = 4,
    ) -> torch.Tensor:
        rates = (
            -torch.abs(
                torch.from_numpy(rng.standard_normal((num_heads,)).astype(np.float32))
            )
            - 0.1
        )
        return rates

    return factory


class TestDepthAwareDecayMaskConfiguration:
    @pytest.mark.parametrize("num_heads", [2, 8])
    def test_stores_configuration(self, depth_decay_factory, num_heads):
        mask = depth_decay_factory(num_heads=num_heads)
        assert mask.num_heads == num_heads


class TestDepthDifferenceMatrix:
    def test_self_difference_is_zero_with_nonuniform_depth(self, depth_map_factory):
        # Non-uniform depth: diagonal should still be zero (position vs itself)
        depth_map = depth_map_factory(batch_size=2, height=3, width=3)
        differences = DepthAwareDecayMask.compute_depth_difference_matrix(
            depth_map=depth_map, height=3, width=3
        )
        for batch_index in range(2):
            diagonal = torch.diagonal(differences[batch_index])
            assert torch.allclose(diagonal, torch.zeros_like(diagonal))

    def test_difference_is_symmetric(self, depth_map_factory):
        depth_map = depth_map_factory(batch_size=2, height=4, width=4)
        differences = DepthAwareDecayMask.compute_depth_difference_matrix(
            depth_map=depth_map, height=4, width=4
        )
        for batch_index in range(2):
            assert torch.allclose(
                differences[batch_index],
                differences[batch_index].T,
                atol=1e-5,
            )

    def test_uniform_depth_produces_zero_differences(self, uniform_depth_map_factory):
        depth_map = uniform_depth_map_factory(
            batch_size=2, height=4, width=4, depth_value=3.0
        )
        differences = DepthAwareDecayMask.compute_depth_difference_matrix(
            depth_map=depth_map, height=4, width=4
        )
        assert torch.allclose(differences, torch.zeros_like(differences))

    def test_discontinuity_produces_expected_difference_values(
        self, discontinuous_depth_map_factory
    ):
        near_depth = 1.0
        far_depth = 5.0
        depth_map = discontinuous_depth_map_factory(
            batch_size=1,
            height=4,
            width=4,
            near_depth=near_depth,
            far_depth=far_depth,
        )
        differences = DepthAwareDecayMask.compute_depth_difference_matrix(
            depth_map=depth_map, height=4, width=4
        )
        expected_cross_boundary_diff = abs(far_depth - near_depth)
        # Position (0,0) flat=0 at near_depth, position (0,3) flat=3 at far_depth
        assert torch.allclose(
            differences[0, 0, 3],
            torch.tensor(expected_cross_boundary_diff),
            atol=1e-5,
        )
        # Same-side positions: (0,0) flat=0 and (0,1) flat=1 both at near_depth
        assert torch.allclose(
            differences[0, 0, 1],
            torch.tensor(0.0),
            atol=1e-5,
        )

    @pytest.mark.parametrize(
        "target_height, target_width",
        [(2, 2), (8, 8)],
    )
    def test_interpolation_to_target_size(
        self, depth_map_factory, target_height, target_width
    ):
        depth_map = depth_map_factory(batch_size=2, height=4, width=6)
        differences = DepthAwareDecayMask.compute_depth_difference_matrix(
            depth_map=depth_map, height=target_height, width=target_width
        )
        sequence_length = target_height * target_width
        assert differences.shape == (2, sequence_length, sequence_length)


class TestDepthDifference1D:
    def test_height_axis_detects_height_varying_depth(
        self, height_varying_depth_map_factory
    ):
        # Depth varies along height → height-axis differences should be nonzero
        depth_map = height_varying_depth_map_factory(
            batch_size=1, height=4, width=4, near_depth=1.0, far_depth=5.0
        )
        differences = DepthAwareDecayMask.compute_1d_depth_difference_matrix(
            depth_map=depth_map, axis=Axis.HEIGHT.value, height=4, width=4
        )
        # Height axis after transpose: shape (B, W, H, H)
        # Row 0 (near) vs row 2 (far) should have nonzero difference
        assert differences[0, 0, 0, 2].item() > 0.0

    def test_width_axis_is_zero_for_height_varying_depth(
        self, height_varying_depth_map_factory
    ):
        # Depth varies along height only → width-axis differences should be zero
        depth_map = height_varying_depth_map_factory(
            batch_size=1, height=4, width=4, near_depth=1.0, far_depth=5.0
        )
        differences = DepthAwareDecayMask.compute_1d_depth_difference_matrix(
            depth_map=depth_map, axis=Axis.WIDTH.value, height=4, width=4
        )
        assert torch.allclose(differences, torch.zeros_like(differences))

    def test_width_axis_detects_width_discontinuity_at_boundary(
        self, discontinuous_depth_map_factory
    ):
        near_depth = 1.0
        far_depth = 5.0
        depth_map = discontinuous_depth_map_factory(
            batch_size=1, height=2, width=4, near_depth=near_depth, far_depth=far_depth
        )
        differences = DepthAwareDecayMask.compute_1d_depth_difference_matrix(
            depth_map=depth_map, axis=Axis.WIDTH.value, height=2, width=4
        )
        expected_boundary_diff = abs(far_depth - near_depth)
        # Shape: (B, H, W, W) = (1, 2, 4, 4)
        # Column 1 (near) vs column 2 (far) should have expected difference
        assert torch.allclose(
            differences[0, 0, 1, 2],
            torch.tensor(expected_boundary_diff),
            atol=1e-5,
        )
        # Same-side: column 0 vs column 1 (both near) should be zero
        assert torch.allclose(
            differences[0, 0, 0, 1],
            torch.tensor(0.0),
            atol=1e-5,
        )

    def test_height_axis_is_zero_for_width_varying_depth(
        self, discontinuous_depth_map_factory
    ):
        # Depth varies along width only → height-axis differences should be zero
        depth_map = discontinuous_depth_map_factory(
            batch_size=1, height=4, width=4, near_depth=1.0, far_depth=5.0
        )
        differences = DepthAwareDecayMask.compute_1d_depth_difference_matrix(
            depth_map=depth_map, axis=Axis.HEIGHT.value, height=4, width=4
        )
        assert torch.allclose(differences, torch.zeros_like(differences))

    @pytest.mark.parametrize("axis", [Axis.HEIGHT.value, Axis.WIDTH.value])
    def test_output_shape_per_axis(self, depth_map_factory, axis):
        height, width = 4, 6
        depth_map = depth_map_factory(batch_size=2, height=height, width=width)
        differences = DepthAwareDecayMask.compute_1d_depth_difference_matrix(
            depth_map=depth_map, axis=axis, height=height, width=width
        )
        if axis == Axis.HEIGHT.value:
            # transpose(-2,-1) → (B, 1, W, H), squeeze → (B, W, H, H)
            assert differences.shape == (2, width, height, height)
        else:
            # no transpose → (B, 1, H, W), squeeze → (B, H, W, W)
            assert differences.shape == (2, height, width, width)


class TestDepthDecayForwardFull:
    def test_output_is_single_element_tuple(
        self, depth_decay_factory, depth_map_factory, decay_rates_factory
    ):
        mask = depth_decay_factory(num_heads=4)
        depth_map = depth_map_factory(batch_size=2, height=4, width=6)
        decay_rates = decay_rates_factory(num_heads=4)
        result = mask(
            depth_map=depth_map,
            height=4,
            width=6,
            decay_rates=decay_rates,
            decomposition_mode=AttentionDecompositionMode.FULL.value,
        )
        assert len(result) == 1

    @pytest.mark.parametrize(
        "batch_size, num_heads, height, width",
        [(2, 4, 3, 5), (1, 8, 4, 4)],
    )
    def test_full_mask_shape(
        self,
        depth_decay_factory,
        depth_map_factory,
        decay_rates_factory,
        batch_size,
        num_heads,
        height,
        width,
    ):
        mask = depth_decay_factory(num_heads=num_heads)
        depth_map = depth_map_factory(batch_size=batch_size, height=height, width=width)
        decay_rates = decay_rates_factory(num_heads=num_heads)
        (result,) = mask(
            depth_map=depth_map,
            height=height,
            width=width,
            decay_rates=decay_rates,
            decomposition_mode=AttentionDecompositionMode.FULL.value,
        )
        sequence_length = height * width
        assert result.shape == (batch_size, num_heads, sequence_length, sequence_length)

    def test_uniform_depth_produces_zero_mask(
        self, depth_decay_factory, uniform_depth_map_factory, decay_rates_factory
    ):
        mask = depth_decay_factory(num_heads=4)
        depth_map = uniform_depth_map_factory(batch_size=2, height=4, width=4)
        decay_rates = decay_rates_factory(num_heads=4)
        (result,) = mask(
            depth_map=depth_map,
            height=4,
            width=4,
            decay_rates=decay_rates,
            decomposition_mode=AttentionDecompositionMode.FULL.value,
        )
        assert torch.allclose(result, torch.zeros_like(result))

    def test_depth_discontinuity_produces_larger_penalty_than_smooth_region(
        self, depth_decay_factory, decay_rates_factory
    ):
        batch_size = 1
        height, width = 4, 4
        depth_map = torch.ones(batch_size, 1, height, width)
        depth_map[:, :, :, width // 2 :] = 5.0

        mask = depth_decay_factory(num_heads=4)
        decay_rates = decay_rates_factory(num_heads=4)
        (result,) = mask(
            depth_map=depth_map,
            height=height,
            width=width,
            decay_rates=decay_rates,
            decomposition_mode=AttentionDecompositionMode.FULL.value,
        )
        # flat index: (0,0)=0, (0,1)=1, (0,2)=2
        # Same side (0,0)-(0,1): both at 1.0, diff=0
        # Cross boundary (0,1)-(0,2): 1.0 vs 5.0, diff=4
        # mask = diff * decay_rate (negative), so |cross| > |same|
        same_side = result[0, :, 0, 1].abs()
        cross_boundary = result[0, :, 1, 2].abs()
        assert (cross_boundary > same_side).all()

    def test_zero_decay_rates_produce_zero_mask(
        self, depth_decay_factory, discontinuous_depth_map_factory
    ):
        mask = depth_decay_factory(num_heads=4)
        depth_map = discontinuous_depth_map_factory(
            batch_size=1, height=4, width=4, near_depth=1.0, far_depth=5.0
        )
        zero_rates = torch.zeros(4)
        (result,) = mask(
            depth_map=depth_map,
            height=4,
            width=4,
            decay_rates=zero_rates,
            decomposition_mode=AttentionDecompositionMode.FULL.value,
        )
        assert torch.allclose(result, torch.zeros_like(result))

    def test_mask_magnitude_scales_with_decay_rate(
        self, depth_decay_factory, discontinuous_depth_map_factory
    ):
        mask = depth_decay_factory(num_heads=2)
        depth_map = discontinuous_depth_map_factory(
            batch_size=1, height=4, width=4, near_depth=1.0, far_depth=5.0
        )
        # Head 0 has small rate, head 1 has large rate
        small_rate = -0.5
        large_rate = -2.0
        decay_rates = torch.tensor([small_rate, large_rate])
        (result,) = mask(
            depth_map=depth_map,
            height=4,
            width=4,
            decay_rates=decay_rates,
            decomposition_mode=AttentionDecompositionMode.FULL.value,
        )
        # Cross-boundary pair: flat 1 vs flat 2 (diff=4.0)
        head_0_value = result[0, 0, 1, 2].abs()
        head_1_value = result[0, 1, 1, 2].abs()
        assert head_1_value > head_0_value

    def test_separable_discontinuity_produces_nonzero_width_mask(
        self, depth_decay_factory, discontinuous_depth_map_factory, decay_rates_factory
    ):
        mask = depth_decay_factory(num_heads=4)
        depth_map = discontinuous_depth_map_factory(
            batch_size=1, height=3, width=4, near_depth=1.0, far_depth=5.0
        )
        decay_rates = decay_rates_factory(num_heads=4)
        height_mask, width_mask = mask(
            depth_map=depth_map,
            height=3,
            width=4,
            decay_rates=decay_rates,
            decomposition_mode=AttentionDecompositionMode.SEPARABLE.value,
        )
        # Width-varying discontinuity → width mask nonzero, height mask zero
        assert width_mask.abs().max().item() > 0.0
        assert torch.allclose(height_mask, torch.zeros_like(height_mask))


class TestDepthDecayForwardSeparable:
    def test_depth_map_resized_to_target_grid(
        self, depth_decay_factory, depth_map_factory, decay_rates_factory
    ):
        mask = depth_decay_factory(num_heads=2)
        depth_map = depth_map_factory(batch_size=1, height=32, width=32)
        decay_rates = decay_rates_factory(num_heads=2)
        height_mask, width_mask = mask(
            depth_map=depth_map,
            height=8,
            width=8,
            decay_rates=decay_rates,
            decomposition_mode=AttentionDecompositionMode.SEPARABLE.value,
        )
        assert height_mask.shape == (1, 2, 8, 8, 8)
        assert width_mask.shape == (1, 2, 8, 8, 8)

    def test_output_is_two_element_tuple(
        self, depth_decay_factory, depth_map_factory, decay_rates_factory
    ):
        mask = depth_decay_factory(num_heads=4)
        depth_map = depth_map_factory(batch_size=2, height=4, width=6)
        decay_rates = decay_rates_factory(num_heads=4)
        result = mask(
            depth_map=depth_map,
            height=4,
            width=6,
            decay_rates=decay_rates,
            decomposition_mode=AttentionDecompositionMode.SEPARABLE.value,
        )
        assert len(result) == 2

    @pytest.mark.parametrize(
        "batch_size, num_heads, height, width",
        [(2, 4, 3, 5), (1, 6, 4, 4)],
    )
    def test_separable_mask_shapes(
        self,
        depth_decay_factory,
        depth_map_factory,
        decay_rates_factory,
        batch_size,
        num_heads,
        height,
        width,
    ):
        mask = depth_decay_factory(num_heads=num_heads)
        depth_map = depth_map_factory(batch_size=batch_size, height=height, width=width)
        decay_rates = decay_rates_factory(num_heads=num_heads)
        height_mask, width_mask = mask(
            depth_map=depth_map,
            height=height,
            width=width,
            decay_rates=decay_rates,
            decomposition_mode=AttentionDecompositionMode.SEPARABLE.value,
        )
        assert height_mask.shape == (batch_size, num_heads, width, height, height)
        assert width_mask.shape == (batch_size, num_heads, height, width, width)

    def test_uniform_depth_produces_zero_separable_masks(
        self, depth_decay_factory, uniform_depth_map_factory, decay_rates_factory
    ):
        mask = depth_decay_factory(num_heads=4)
        depth_map = uniform_depth_map_factory(batch_size=2, height=3, width=5)
        decay_rates = decay_rates_factory(num_heads=4)
        height_mask, width_mask = mask(
            depth_map=depth_map,
            height=3,
            width=5,
            decay_rates=decay_rates,
            decomposition_mode=AttentionDecompositionMode.SEPARABLE.value,
        )
        assert torch.allclose(height_mask, torch.zeros_like(height_mask))
        assert torch.allclose(width_mask, torch.zeros_like(width_mask))
