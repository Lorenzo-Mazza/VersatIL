"""Tests for versatil.models.layers.positional_encoding.sinusoidal module."""
import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise

import pytest
import torch

from versatil.models.layers.positional_encoding.base import (
    DenominatorMode,
    OrderingMode,
    PositionSource,
)
from versatil.models.layers.positional_encoding.sinusoidal import (
    SinusoidalPositionalEncoding1D,
    SinusoidalPositionalEncoding2D,
)


@pytest.fixture
def sinusoidal_2d_factory() -> Callable[..., SinusoidalPositionalEncoding2D]:
    """Factory for SinusoidalPositionalEncoding2D instances."""
    def factory(
        embedding_dimension: int = 64,
        temperature: float = 10000.0,
        normalize: bool = False,
        scale: float | None = None,
    ) -> SinusoidalPositionalEncoding2D:
        return SinusoidalPositionalEncoding2D(
            embedding_dimension=embedding_dimension,
            temperature=temperature,
            normalize=normalize,
            scale=scale,
        )
    return factory


class TestSinusoidalPositionalEncoding1DInit:

    def test_odd_embedding_dimension_raises(self):
        with pytest.raises(
            ValueError,
            match=re.escape("embedding_dimension must be even"),
        ):
            SinusoidalPositionalEncoding1D(embedding_dimension=63)

    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    @pytest.mark.parametrize("temperature", [10000.0, 5000.0])
    @pytest.mark.parametrize("ordering_mode", [
        OrderingMode.INTERLEAVE_SIN_COS.value,
        OrderingMode.CAT_COS_SIN.value,
    ])
    def test_stores_configuration(
        self,
        sinusoidal_1d_factory: Callable[..., SinusoidalPositionalEncoding1D],
        embedding_dimension: int,
        temperature: float,
        ordering_mode: str,
    ):
        module = sinusoidal_1d_factory(
            embedding_dimension=embedding_dimension,
            temperature=temperature,
        )
        assert module.embedding_dimension == embedding_dimension
        assert module.temperature == temperature

    def test_has_precomputed_buffer_when_precomputed_true(
        self,
        sinusoidal_1d_factory: Callable[..., SinusoidalPositionalEncoding1D],
    ):
        module = sinusoidal_1d_factory(
            embedding_dimension=64,
            precompute_encodings=True,
            maximum_length=100,
            position_source=PositionSource.TENSOR_INDICES.value,
        )
        assert hasattr(module, "precomputed_encodings")
        assert module.precomputed_encodings is not None

    def test_no_precomputed_buffer_when_precomputed_false(
        self,
        sinusoidal_1d_factory: Callable[..., SinusoidalPositionalEncoding1D],
    ):
        module = sinusoidal_1d_factory(
            embedding_dimension=64,
            precompute_encodings=False,
            position_source=PositionSource.TENSOR_INDICES.value,
        )
        assert not hasattr(module, "precomputed_encodings")

    @pytest.mark.parametrize("denominator_mode, expectation", [
        (DenominatorMode.HALF.value, does_not_raise()),
        (DenominatorMode.HALF_MINUS_ONE.value, does_not_raise()),
        ("invalid_mode", pytest.raises(
            ValueError,
            match=re.escape("Invalid denominator_mode: invalid_mode"),
        )),
    ])
    def test_denominator_mode_validation(
        self,
        denominator_mode: str,
        expectation: does_not_raise,
    ):
        with expectation:
            SinusoidalPositionalEncoding1D(
                embedding_dimension=64,
                denominator_mode=denominator_mode,
            )

    @pytest.mark.parametrize("learnable_frequencies", [True, False])
    def test_learnable_frequencies_controls_requires_grad(
        self,
        sinusoidal_1d_factory: Callable[..., SinusoidalPositionalEncoding1D],
        learnable_frequencies: bool,
    ):
        module = sinusoidal_1d_factory(
            embedding_dimension=64,
            learnable_frequencies=learnable_frequencies,
        )
        assert module.frequencies.requires_grad == learnable_frequencies


class TestSinusoidalPositionalEncoding1DForward:

    @pytest.mark.parametrize("batch_size, seq_len, embedding_dimension", [
        (2, 10, 64),
        (4, 20, 128),
    ])
    def test_output_shape_tensor_indices(
        self,
        sinusoidal_1d_factory: Callable[..., SinusoidalPositionalEncoding1D],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
        seq_len: int,
        embedding_dimension: int,
    ):
        module = sinusoidal_1d_factory(
            embedding_dimension=embedding_dimension,
            position_source=PositionSource.TENSOR_INDICES.value,
        )
        tensor = sequence_tensor_factory(
            batch_size=batch_size,
            seq_len=seq_len,
            embedding_dimension=embedding_dimension,
        )
        output = module(tensor)
        assert output.shape == (batch_size, seq_len, embedding_dimension)

    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_output_shape_scalar(
        self,
        sinusoidal_1d_factory: Callable[..., SinusoidalPositionalEncoding1D],
        scalar_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        embedding_dimension = 64
        module = sinusoidal_1d_factory(
            embedding_dimension=embedding_dimension,
            position_source=PositionSource.SCALAR.value,
            precompute_encodings=False,
        )
        tensor = scalar_tensor_factory(batch_size=batch_size)
        output = module(tensor)
        assert output.shape == (batch_size, embedding_dimension)

    def test_different_positions_produce_different_encodings(
        self,
        sinusoidal_1d_factory: Callable[..., SinusoidalPositionalEncoding1D],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        module = sinusoidal_1d_factory(embedding_dimension=64)
        tensor = sequence_tensor_factory(
            batch_size=1, seq_len=10, embedding_dimension=64,
        )
        output = module(tensor)
        encoding_pos_0 = output[0, 0]
        encoding_pos_5 = output[0, 5]
        assert not torch.allclose(encoding_pos_0, encoding_pos_5)

    def test_precomputed_matches_non_precomputed(
        self,
        sinusoidal_1d_factory: Callable[..., SinusoidalPositionalEncoding1D],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 64
        seq_len = 10
        precomputed = sinusoidal_1d_factory(
            embedding_dimension=embedding_dimension,
            precompute_encodings=True,
            maximum_length=100,
        )
        non_precomputed = sinusoidal_1d_factory(
            embedding_dimension=embedding_dimension,
            precompute_encodings=False,
        )
        tensor = sequence_tensor_factory(
            batch_size=2, seq_len=seq_len, embedding_dimension=embedding_dimension,
        )
        output_precomputed = precomputed(tensor)
        output_non_precomputed = non_precomputed(tensor)
        assert torch.allclose(output_precomputed, output_non_precomputed, atol=1e-6)


class TestCreateEncodingTable:

    @pytest.mark.parametrize("number_of_positions, embedding_dimension", [
        (10, 64),
        (50, 128),
    ])
    def test_output_shape(
        self,
        number_of_positions: int,
        embedding_dimension: int,
    ):
        table = SinusoidalPositionalEncoding1D.create_encoding_table(
            number_of_positions=number_of_positions,
            embedding_dimension=embedding_dimension,
        )
        assert table.shape == (1, number_of_positions, embedding_dimension)

    def test_encoding_table_values_are_bounded(self):
        table = SinusoidalPositionalEncoding1D.create_encoding_table(
            number_of_positions=20,
            embedding_dimension=64,
        )
        # Sinusoidal values are bounded in [-1, 1]
        assert table.min() >= -1.0
        assert table.max() <= 1.0


class TestSinusoidalPositionalEncoding2D:

    def test_odd_embedding_dimension_raises(self):
        with pytest.raises(
            ValueError,
            match=re.escape("embedding_dimension must be even"),
        ):
            SinusoidalPositionalEncoding2D(embedding_dimension=63)

    @pytest.mark.parametrize("batch_size, embedding_dimension, height, width", [
        (2, 64, 8, 8),
        (4, 128, 4, 6),
    ])
    def test_output_shape(
        self,
        sinusoidal_2d_factory: Callable[..., SinusoidalPositionalEncoding2D],
        spatial_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
        embedding_dimension: int,
        height: int,
        width: int,
    ):
        module = sinusoidal_2d_factory(embedding_dimension=embedding_dimension)
        tensor = spatial_tensor_factory(
            batch_size=batch_size,
            channels=embedding_dimension,
            height=height,
            width=width,
        )
        output = module(tensor)
        assert output.shape == (batch_size, embedding_dimension, height, width)

    @pytest.mark.parametrize("normalize", [True, False])
    def test_stores_normalize(
        self,
        sinusoidal_2d_factory: Callable[..., SinusoidalPositionalEncoding2D],
        normalize: bool,
    ):
        module = sinusoidal_2d_factory(
            embedding_dimension=64,
            normalize=normalize,
        )
        assert module.normalize == normalize

    def test_different_spatial_positions_produce_different_encodings(
        self,
        sinusoidal_2d_factory: Callable[..., SinusoidalPositionalEncoding2D],
        spatial_tensor_factory: Callable[..., torch.Tensor],
    ):
        module = sinusoidal_2d_factory(embedding_dimension=64)
        tensor = spatial_tensor_factory(
            batch_size=1, channels=64, height=4, width=4,
        )
        output = module(tensor)
        encoding_00 = output[0, :, 0, 0]
        encoding_11 = output[0, :, 1, 1]
        assert not torch.allclose(encoding_00, encoding_11)
