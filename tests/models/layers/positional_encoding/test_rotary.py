"""Tests for versatil.models.layers.positional_encoding.rotary module."""
import re
from collections.abc import Callable

import numpy as np
import pytest
import torch

from versatil.models.layers.positional_encoding.rotary import (
    RotaryPositionalEncoding,
    RotaryPositionalEncoding1D,
    RotaryPositionalEncoding2D,
)


@pytest.fixture
def rotary_factory() -> Callable[..., RotaryPositionalEncoding1D]:
    """Factory for RotaryPositionalEncoding1D instances."""
    def factory(
        embedding_dimension: int = 64,
        num_heads: int = 4,
        base_frequency: float = 10000.0,
        learnable_frequencies: bool = False,
    ) -> RotaryPositionalEncoding1D:
        return RotaryPositionalEncoding1D(
            embedding_dimension=embedding_dimension,
            num_heads=num_heads,
            base_frequency=base_frequency,
            learnable_frequencies=learnable_frequencies,
        )
    return factory


@pytest.fixture
def rotary_2d_factory() -> Callable[..., RotaryPositionalEncoding2D]:
    """Factory for RotaryPositionalEncoding2D instances."""
    def factory(
        embedding_dimension: int = 128,
        num_heads: int = 4,
        base_frequency: float = 10000.0,
        learnable_frequencies: bool = False,
    ) -> RotaryPositionalEncoding2D:
        return RotaryPositionalEncoding2D(
            embedding_dimension=embedding_dimension,
            num_heads=num_heads,
            base_frequency=base_frequency,
            learnable_frequencies=learnable_frequencies,
        )
    return factory


@pytest.fixture
def rotation_input_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for tensors used with apply_rotation."""
    def factory(
        batch_size: int = 2,
        num_heads: int = 4,
        seq_len: int = 8,
        head_dim: int = 16,
    ) -> torch.Tensor:
        shape = (batch_size, num_heads, seq_len, head_dim)
        return torch.from_numpy(
            rng.standard_normal(shape).astype(np.float32)
        )
    return factory


class TestRotaryPositionalEncoding:

    @pytest.mark.parametrize("embedding_dimension", [64, 128])
    @pytest.mark.parametrize("num_heads", [4, 8])
    @pytest.mark.parametrize("base_frequency", [10000.0, 5000.0])
    def test_stores_configuration(
        self,
        embedding_dimension: int,
        num_heads: int,
        base_frequency: float,
    ):
        module = RotaryPositionalEncoding1D(
            embedding_dimension=embedding_dimension,
            num_heads=num_heads,
            base_frequency=base_frequency,
        )
        assert module.embedding_dimension == embedding_dimension
        assert module.num_heads == num_heads
        assert module.head_dimension == embedding_dimension // num_heads

    def test_odd_head_dimension_raises_value_error(self):
        with pytest.raises(
            ValueError,
            match=re.escape("head_dimension must be even for rotary encoding"),
        ):
            RotaryPositionalEncoding1D(
                embedding_dimension=48,
                num_heads=5,
            )

    @pytest.mark.parametrize("embedding_dimension, num_heads", [
        (64, 4),
        (128, 8),
    ])
    def test_frequencies_shape(
        self,
        rotary_factory: Callable[..., RotaryPositionalEncoding1D],
        embedding_dimension: int,
        num_heads: int,
    ):
        module = rotary_factory(
            embedding_dimension=embedding_dimension,
            num_heads=num_heads,
        )
        head_dim = embedding_dimension // num_heads
        assert module.frequencies.shape == (head_dim,)

    @pytest.mark.parametrize("learnable_frequencies", [True, False])
    def test_learnable_frequencies_controls_requires_grad(
        self,
        rotary_factory: Callable[..., RotaryPositionalEncoding1D],
        learnable_frequencies: bool,
    ):
        module = rotary_factory(learnable_frequencies=learnable_frequencies)
        assert module.frequencies.requires_grad == learnable_frequencies

    @pytest.mark.parametrize("dimension", [8, 16])
    def test_compute_frequencies_output_shape(self, dimension: int):
        frequencies = RotaryPositionalEncoding._compute_frequencies(
            dimension=dimension, base_frequency=10000.0,
        )
        assert frequencies.shape == (dimension,)

    def test_compute_frequencies_first_element_is_one(self):
        frequencies = RotaryPositionalEncoding._compute_frequencies(
            dimension=8, base_frequency=10000.0,
        )
        # First exponent is 0, so 1/base^0 = 1.0; repeat_interleave means [0] and [1] are both 1.0
        assert frequencies[0].item() == pytest.approx(1.0)
        assert frequencies[1].item() == pytest.approx(1.0)

    def test_compute_frequencies_are_monotonically_decreasing_per_pair(self):
        frequencies = RotaryPositionalEncoding._compute_frequencies(
            dimension=16, base_frequency=10000.0,
        )
        # After repeat_interleave, pairs share the same value; across pairs, values decrease
        pair_values = frequencies[0::2]
        for i in range(len(pair_values) - 1):
            assert pair_values[i] > pair_values[i + 1]


class TestRotaryApplyRotation:

    @pytest.mark.parametrize("seq_len, head_dim", [
        (8, 16),
        (12, 32),
    ])
    def test_output_shape_matches_input(
        self,
        rotation_input_factory: Callable[..., torch.Tensor],
        seq_len: int,
        head_dim: int,
    ):
        batch_size = 2
        num_heads = 4
        tensor = rotation_input_factory(
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
        )
        sine = torch.zeros(seq_len, head_dim)
        cosine = torch.ones(seq_len, head_dim)
        result = RotaryPositionalEncoding.apply_rotation(
            tensor=tensor, sine=sine, cosine=cosine,
        )
        assert result.shape == tensor.shape

    def test_identity_rotation_preserves_input(
        self,
        rotation_input_factory: Callable[..., torch.Tensor],
    ):
        tensor = rotation_input_factory(
            batch_size=2, num_heads=4, seq_len=8, head_dim=16,
        )
        sine = torch.zeros(8, 16)
        cosine = torch.ones(8, 16)
        result = RotaryPositionalEncoding.apply_rotation(
            tensor=tensor, sine=sine, cosine=cosine,
        )
        assert torch.allclose(result, tensor, atol=1e-6)

    def test_rotation_changes_values(
        self,
        rotation_input_factory: Callable[..., torch.Tensor],
        rng: np.random.Generator,
    ):
        tensor = rotation_input_factory(
            batch_size=2, num_heads=4, seq_len=8, head_dim=16,
        )
        angles = torch.from_numpy(
            rng.standard_normal((8, 16)).astype(np.float32)
        )
        sine = torch.sin(angles)
        cosine = torch.cos(angles)
        result = RotaryPositionalEncoding.apply_rotation(
            tensor=tensor, sine=sine, cosine=cosine,
        )
        assert not torch.allclose(result, tensor)


class TestRotaryPositionalEncoding1D:

    @pytest.mark.parametrize("seq_len", [8, 16])
    @pytest.mark.parametrize("embedding_dimension, num_heads", [
        (64, 4),
        (128, 8),
    ])
    def test_compute_rotation_components_shape(
        self,
        rotary_factory: Callable[..., RotaryPositionalEncoding1D],
        seq_len: int,
        embedding_dimension: int,
        num_heads: int,
    ):
        module = rotary_factory(
            embedding_dimension=embedding_dimension,
            num_heads=num_heads,
        )
        head_dim = embedding_dimension // num_heads
        sine, cosine = module.compute_rotation_components(seq_len=seq_len)
        assert sine.shape == (seq_len, head_dim)
        assert cosine.shape == (seq_len, head_dim)

    def test_compute_rotation_components_first_position_sine_is_zero(
        self,
        rotary_factory: Callable[..., RotaryPositionalEncoding1D],
    ):
        module = rotary_factory(embedding_dimension=64, num_heads=4)
        sine, cosine = module.compute_rotation_components(seq_len=8)
        # Position 0 -> angles = 0 * freq = 0 -> sin(0) = 0, cos(0) = 1
        assert torch.allclose(sine[0], torch.zeros_like(sine[0]), atol=1e-6)
        assert torch.allclose(cosine[0], torch.ones_like(cosine[0]), atol=1e-6)


class TestRotaryPositionalEncoding2D:

    @pytest.mark.parametrize("embedding_dimension", [128, 256])
    @pytest.mark.parametrize("num_heads", [4, 8])
    def test_stores_half_head_dim(
        self,
        embedding_dimension: int,
        num_heads: int,
    ):
        module = RotaryPositionalEncoding2D(
            embedding_dimension=embedding_dimension,
            num_heads=num_heads,
        )
        expected_half = (embedding_dimension // num_heads) // 2
        assert module.half_head_dim == expected_half

    def test_odd_half_head_dim_raises_value_error(self):
        # embedding_dimension=24, num_heads=4 -> head_dim=6, half_head_dim=3
        # 3 % 2 != 0, so it should raise
        with pytest.raises(
            ValueError,
            match=re.escape("half_head_dimension must be even for 2D rotary encoding"),
        ):
            RotaryPositionalEncoding2D(
                embedding_dimension=24,
                num_heads=4,
            )

    @pytest.mark.parametrize("height, width", [
        (4, 6),
        (8, 8),
    ])
    def test_compute_rotation_components_shape(
        self,
        rotary_2d_factory: Callable[..., RotaryPositionalEncoding2D],
        height: int,
        width: int,
    ):
        module = rotary_2d_factory(embedding_dimension=128, num_heads=4)
        head_dim = 128 // 4
        sine, cosine = module.compute_rotation_components(
            height=height, width=width,
        )
        assert sine.shape == (height, width, head_dim)
        assert cosine.shape == (height, width, head_dim)

    def test_frequencies_shape_is_full_head_dim(
        self,
        rotary_2d_factory: Callable[..., RotaryPositionalEncoding2D],
    ):
        module = rotary_2d_factory(embedding_dimension=128, num_heads=4)
        head_dim = 128 // 4
        assert module.frequencies.shape == (head_dim,)
