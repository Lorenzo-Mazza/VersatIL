"""Tests for versatil.models.layers.geometric_attention.spatial_decay module."""
import math
from collections.abc import Callable

import pytest
import torch

from versatil.models.layers.constants import AttentionDecompositionMode
from versatil.models.layers.geometric_attention.spatial_decay import SpatialDecayMask


@pytest.fixture
def distance_matrix_factory() -> Callable[..., torch.Tensor]:
    def factory(
        num_heads: int = 4,
        height: int = 3,
        width: int = 3,
    ) -> torch.Tensor:
        mask = SpatialDecayMask(num_heads=num_heads)
        return mask.compute_2d_distance_matrix(height=height, width=width)

    return factory


@pytest.fixture
def distance_matrix_1d_factory() -> Callable[..., torch.Tensor]:
    def factory(
        num_heads: int = 4,
        length: int = 5,
    ) -> torch.Tensor:
        mask = SpatialDecayMask(num_heads=num_heads)
        return mask.compute_1d_distance_matrix(length=length)

    return factory


class TestSpatialDecayMaskConfiguration:

    @pytest.mark.parametrize("num_heads", [2, 8])
    @pytest.mark.parametrize("initial_decay", [3.0, 7.0])
    @pytest.mark.parametrize("decay_range", [1.0, 5.0])
    def test_stores_configuration(
        self, spatial_decay_factory, num_heads, initial_decay, decay_range
    ):
        mask = spatial_decay_factory(
            num_heads=num_heads,
            initial_decay=initial_decay,
            decay_range=decay_range,
        )
        assert mask.num_heads == num_heads
        assert mask.decay_rates.shape == (num_heads,)


class TestPerHeadDecayComputation:

    def test_decay_rates_are_negative(self, spatial_decay_factory):
        mask = spatial_decay_factory(num_heads=4, initial_decay=5.0, decay_range=3.0)
        assert (mask.decay_rates < 0).all()

    def test_decay_rates_are_monotonically_increasing(self, spatial_decay_factory):
        # Higher head index gets larger decay_offset, so 2^(-initial-offset) is smaller,
        # making log(1 - smaller_value) closer to zero (less negative).
        mask = spatial_decay_factory(num_heads=8, initial_decay=5.0, decay_range=3.0)
        for head_index in range(7):
            assert mask.decay_rates[head_index] < mask.decay_rates[head_index + 1]

    @pytest.mark.parametrize("num_heads", [2, 6])
    def test_different_heads_have_different_decay_rates(
        self, spatial_decay_factory, num_heads
    ):
        mask = spatial_decay_factory(num_heads=num_heads)
        unique_rates = mask.decay_rates.unique()
        assert unique_rates.numel() == num_heads

    def test_decay_rate_exact_values_for_two_heads(self, spatial_decay_factory):
        # For num_heads=2, initial_decay=5.0, decay_range=3.0:
        # head 0: offset = 3.0 * 0 / 2 = 0.0 => log(1 - 2^(-5.0))
        # head 1: offset = 3.0 * 1 / 2 = 1.5 => log(1 - 2^(-6.5))
        mask = spatial_decay_factory(
            num_heads=2, initial_decay=5.0, decay_range=3.0
        )
        expected_head_0 = math.log(1 - 2 ** (-5.0))
        expected_head_1 = math.log(1 - 2 ** (-6.5))
        assert torch.allclose(
            mask.decay_rates,
            torch.tensor([expected_head_0, expected_head_1]),
            atol=1e-6,
        )


class TestDistanceMatrix2D:

    def test_self_distance_is_zero(self, distance_matrix_factory):
        distances = distance_matrix_factory(height=4, width=5)
        diagonal = torch.diagonal(distances)
        assert (diagonal == 0).all()

    def test_distance_is_symmetric(self, distance_matrix_factory):
        distances = distance_matrix_factory(height=3, width=4)
        assert torch.allclose(distances, distances.T)

    def test_distance_uses_manhattan_metric(self, distance_matrix_factory):
        # For a 2x2 grid, positions are (0,0),(0,1),(1,0),(1,1)
        distances = distance_matrix_factory(height=2, width=2)
        # Position 0=(0,0), Position 3=(1,1) => |0-1| + |0-1| = 2
        assert distances[0, 3].item() == 2.0
        # Position 0=(0,0), Position 1=(0,1) => distance 1
        assert distances[0, 1].item() == 1.0
        # Position 0=(0,0), Position 2=(1,0) => distance 1
        assert distances[0, 2].item() == 1.0

    def test_full_distance_matrix_for_2x3_grid(self, distance_matrix_factory):
        # 2x3 grid positions: (0,0)=0, (0,1)=1, (0,2)=2, (1,0)=3, (1,1)=4, (1,2)=5
        distances = distance_matrix_factory(height=2, width=3)
        assert distances.shape == (6, 6)
        # (0,0) to (1,2): |0-1| + |0-2| = 3
        assert distances[0, 5].item() == 3.0
        # (0,1) to (1,0): |0-1| + |1-0| = 2
        assert distances[1, 3].item() == 2.0
        # (0,2) to (1,2): |0-1| + |2-2| = 1
        assert distances[2, 5].item() == 1.0

    @pytest.mark.parametrize(
        "height, width",
        [(3, 4), (5, 5), (2, 7)],
    )
    def test_distance_matrix_shape(self, distance_matrix_factory, height, width):
        distances = distance_matrix_factory(height=height, width=width)
        sequence_length = height * width
        assert distances.shape == (sequence_length, sequence_length)


class TestDistanceMatrix1D:

    def test_self_distance_is_zero(self, distance_matrix_1d_factory):
        distances = distance_matrix_1d_factory(length=5)
        diagonal = torch.diagonal(distances)
        assert (diagonal == 0).all()

    def test_distance_is_symmetric(self, distance_matrix_1d_factory):
        distances = distance_matrix_1d_factory(length=6)
        assert torch.allclose(distances, distances.T)

    def test_adjacent_distance_is_one(self, distance_matrix_1d_factory):
        distances = distance_matrix_1d_factory(length=4)
        for index in range(3):
            assert distances[index, index + 1].item() == 1.0

    def test_distance_equals_absolute_index_difference(
        self, distance_matrix_1d_factory
    ):
        distances = distance_matrix_1d_factory(length=5)
        assert distances[0, 4].item() == 4.0
        assert distances[1, 3].item() == 2.0
        assert distances[2, 0].item() == 2.0
        assert distances[3, 0].item() == 3.0

    def test_1d_distance_shape(self, distance_matrix_1d_factory):
        distances = distance_matrix_1d_factory(length=7)
        assert distances.shape == (7, 7)


class TestSpatialDecayForwardFull:

    def test_output_is_single_element_tuple(self, spatial_decay_factory):
        mask = spatial_decay_factory(num_heads=4)
        result = mask(height=3, width=4)
        assert len(result) == 1

    @pytest.mark.parametrize(
        "num_heads, height, width",
        [(2, 3, 4), (8, 5, 5)],
    )
    def test_full_mask_shape(self, spatial_decay_factory, num_heads, height, width):
        mask = spatial_decay_factory(num_heads=num_heads)
        (result,) = mask(
            height=height,
            width=width,
            decomposition_mode=AttentionDecompositionMode.FULL.value,
        )
        sequence_length = height * width
        assert result.shape == (num_heads, sequence_length, sequence_length)

    def test_closer_positions_have_higher_mask_values(self, spatial_decay_factory):
        # Since decay_rates are negative, mask = distance * decay_rate.
        # Closer positions have smaller distance => less negative => higher value.
        mask = spatial_decay_factory(num_heads=4)
        (result,) = mask(height=3, width=3)
        # Position 0 is (0,0), position 1 is (0,1) distance=1, position 8 is (2,2) distance=4
        for head_index in range(4):
            adjacent_value = result[head_index, 0, 1].item()
            far_value = result[head_index, 0, 8].item()
            assert adjacent_value > far_value

    def test_diagonal_is_zero(self, spatial_decay_factory):
        # Self-attention position has distance 0, so mask value = 0 * decay = 0
        mask = spatial_decay_factory(num_heads=4)
        (result,) = mask(height=3, width=3)
        for head_index in range(4):
            diagonal = torch.diagonal(result[head_index])
            assert torch.allclose(diagonal, torch.zeros_like(diagonal))

    def test_all_nondiagonal_values_are_negative(self, spatial_decay_factory):
        # distance > 0 and decay_rates < 0, so product is negative
        mask = spatial_decay_factory(num_heads=4)
        (result,) = mask(height=3, width=3)
        sequence_length = 9
        off_diagonal_mask = ~torch.eye(sequence_length, dtype=torch.bool)
        for head_index in range(4):
            off_diagonal_values = result[head_index][off_diagonal_mask]
            assert (off_diagonal_values < 0).all()

    def test_mask_value_equals_distance_times_decay_rate(self, spatial_decay_factory):
        # For a 2x2 grid with known decay rates, verify exact mask values
        mask = spatial_decay_factory(
            num_heads=2, initial_decay=5.0, decay_range=3.0
        )
        (result,) = mask(height=2, width=2)
        decay_rate_head_0 = mask.decay_rates[0].item()
        # Position (0,0) to (1,1): Manhattan distance = 2
        expected_value = 2.0 * decay_rate_head_0
        assert abs(result[0, 0, 3].item() - expected_value) < 1e-6
        # Position (0,0) to (0,1): Manhattan distance = 1
        expected_value_adjacent = 1.0 * decay_rate_head_0
        assert abs(result[0, 0, 1].item() - expected_value_adjacent) < 1e-6

    def test_early_heads_decay_faster_than_later_heads(self, spatial_decay_factory):
        # Early heads have more negative decay => same distance produces more negative mask
        mask = spatial_decay_factory(num_heads=4)
        (result,) = mask(height=3, width=3)
        # For a fixed position pair at distance > 0, head 0 should be more negative than head 3
        assert result[0, 0, 8].item() < result[3, 0, 8].item()


class TestSpatialDecayForwardSeparable:

    def test_output_is_two_element_tuple(self, spatial_decay_factory):
        mask = spatial_decay_factory(num_heads=4)
        result = mask(
            height=3,
            width=4,
            decomposition_mode=AttentionDecompositionMode.SEPARABLE.value,
        )
        assert len(result) == 2

    @pytest.mark.parametrize(
        "num_heads, height, width",
        [(2, 3, 5), (6, 4, 4)],
    )
    def test_separable_mask_shapes(
        self, spatial_decay_factory, num_heads, height, width
    ):
        mask = spatial_decay_factory(num_heads=num_heads)
        height_mask, width_mask = mask(
            height=height,
            width=width,
            decomposition_mode=AttentionDecompositionMode.SEPARABLE.value,
        )
        assert height_mask.shape == (num_heads, height, height)
        assert width_mask.shape == (num_heads, width, width)

    def test_separable_diagonal_is_zero(self, spatial_decay_factory):
        mask = spatial_decay_factory(num_heads=4)
        height_mask, width_mask = mask(
            height=3,
            width=4,
            decomposition_mode=AttentionDecompositionMode.SEPARABLE.value,
        )
        for head_index in range(4):
            assert torch.allclose(
                torch.diagonal(height_mask[head_index]),
                torch.zeros(3),
            )
            assert torch.allclose(
                torch.diagonal(width_mask[head_index]),
                torch.zeros(4),
            )

    def test_separable_closer_positions_have_higher_values(
        self, spatial_decay_factory
    ):
        mask = spatial_decay_factory(num_heads=4)
        height_mask, width_mask = mask(
            height=5,
            width=5,
            decomposition_mode=AttentionDecompositionMode.SEPARABLE.value,
        )
        for head_index in range(4):
            # Adjacent (distance=1) should have higher value than far (distance=4)
            assert height_mask[head_index, 0, 1] > height_mask[head_index, 0, 4]
            assert width_mask[head_index, 0, 1] > width_mask[head_index, 0, 4]

    def test_separable_mask_value_equals_1d_distance_times_decay_rate(
        self, spatial_decay_factory
    ):
        mask = spatial_decay_factory(
            num_heads=2, initial_decay=5.0, decay_range=3.0
        )
        height_mask, width_mask = mask(
            height=4,
            width=3,
            decomposition_mode=AttentionDecompositionMode.SEPARABLE.value,
        )
        decay_rate_head_0 = mask.decay_rates[0].item()
        # Height mask: distance between row 0 and row 3 = 3
        expected = 3.0 * decay_rate_head_0
        assert abs(height_mask[0, 0, 3].item() - expected) < 1e-6
        # Width mask: distance between col 0 and col 2 = 2
        expected_w = 2.0 * decay_rate_head_0
        assert abs(width_mask[0, 0, 2].item() - expected_w) < 1e-6
