"""Tests for versatil.models.layers.positional_encoding.base module."""

import re
from collections.abc import Callable

import pytest
import torch

from versatil.models.layers.positional_encoding.base import (
    DenominatorMode,
    OrderingMode,
    PositionSource,
    add_positional_encoding,
)
from versatil.models.layers.positional_encoding.sinusoidal import (
    SinusoidalPositionalEncoding1D,
)


class TestPositionSourceEnum:
    def test_has_tensor_indices_member(self):
        assert PositionSource.TENSOR_INDICES.value == "tensor_indices"

    def test_has_scalar_member(self):
        assert PositionSource.SCALAR.value == "scalar"

    def test_has_grid_2d_member(self):
        assert PositionSource.GRID_2D.value == "grid_2d"

    def test_has_exactly_three_members(self):
        assert len(PositionSource) == 3


class TestDenominatorModeEnum:
    def test_has_half_member(self):
        assert DenominatorMode.HALF.value == "half"

    def test_has_half_minus_one_member(self):
        assert DenominatorMode.HALF_MINUS_ONE.value == "half_minus_one"

    def test_has_exactly_two_members(self):
        assert len(DenominatorMode) == 2


class TestOrderingModeEnum:
    def test_has_interleave_sin_cos_member(self):
        assert OrderingMode.INTERLEAVE_SIN_COS.value == "interleave_sin_cos"

    def test_has_cat_cos_sin_member(self):
        assert OrderingMode.CAT_COS_SIN.value == "cat_cos_sin"

    def test_has_exactly_two_members(self):
        assert len(OrderingMode) == 2


class TestAddPositionalEncoding:
    def test_returns_source_unchanged_when_encoding_is_none(
        self,
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        source = sequence_tensor_factory(
            batch_size=2, sequence_length=10, embedding_dimension=64
        )
        result = add_positional_encoding(
            source=source,
            positional_encoding=None,
        )
        assert torch.equal(result, source)

    def test_adds_encoding_when_provided(
        self,
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        source = sequence_tensor_factory(
            batch_size=2,
            sequence_length=10,
            embedding_dimension=64,
        )
        encoding = sequence_tensor_factory(
            batch_size=2,
            sequence_length=10,
            embedding_dimension=64,
        )
        result = add_positional_encoding(
            source=source,
            positional_encoding=encoding,
        )
        expected = source + encoding
        assert torch.allclose(result, expected, atol=1e-7)

    def test_result_shape_matches_source(
        self,
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        source = sequence_tensor_factory(
            batch_size=4,
            sequence_length=20,
            embedding_dimension=128,
        )
        encoding = sequence_tensor_factory(
            batch_size=4,
            sequence_length=20,
            embedding_dimension=128,
        )
        result = add_positional_encoding(
            source=source,
            positional_encoding=encoding,
        )
        assert result.shape == source.shape


class TestPositionalEncoding1DInit:
    def test_precompute_with_none_maximum_length_raises(self):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "maximum_length must be set when precompute_encodings=True"
            ),
        ):
            SinusoidalPositionalEncoding1D(
                embedding_dimension=64,
                position_source=PositionSource.TENSOR_INDICES.value,
                precompute_encodings=True,
                maximum_length=None,
            )

    def test_precompute_false_with_none_maximum_length_succeeds(self):
        module = SinusoidalPositionalEncoding1D(
            embedding_dimension=64,
            position_source=PositionSource.TENSOR_INDICES.value,
            precompute_encodings=False,
            maximum_length=None,
        )
        assert module.maximum_length is None

    def test_scalar_source_with_none_maximum_length_succeeds(self):
        module = SinusoidalPositionalEncoding1D(
            embedding_dimension=64,
            position_source=PositionSource.SCALAR.value,
            precompute_encodings=True,
            maximum_length=None,
        )
        assert module.maximum_length is None


class TestPositionalEncoding1DForward:
    def test_tensor_indices_with_precompute(
        self,
        sinusoidal_1d_factory: Callable[..., SinusoidalPositionalEncoding1D],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 64
        sequence_length = 10
        batch_size = 2
        module = sinusoidal_1d_factory(
            embedding_dimension=embedding_dimension,
            position_source=PositionSource.TENSOR_INDICES.value,
            precompute_encodings=True,
            maximum_length=100,
        )
        tensor = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=embedding_dimension,
        )
        output = module(tensor)
        assert output.shape == (batch_size, sequence_length, embedding_dimension)

    def test_tensor_indices_without_precompute(
        self,
        sinusoidal_1d_factory: Callable[..., SinusoidalPositionalEncoding1D],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 64
        sequence_length = 10
        batch_size = 2
        module = sinusoidal_1d_factory(
            embedding_dimension=embedding_dimension,
            position_source=PositionSource.TENSOR_INDICES.value,
            precompute_encodings=False,
        )
        tensor = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=embedding_dimension,
        )
        output = module(tensor)
        assert output.shape == (batch_size, sequence_length, embedding_dimension)

    def test_scalar_path(
        self,
        sinusoidal_1d_factory: Callable[..., SinusoidalPositionalEncoding1D],
        scalar_tensor_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 64
        batch_size = 2
        module = sinusoidal_1d_factory(
            embedding_dimension=embedding_dimension,
            position_source=PositionSource.SCALAR.value,
            precompute_encodings=False,
        )
        tensor = scalar_tensor_factory(batch_size=batch_size)
        output = module(tensor)
        assert output.shape == (batch_size, embedding_dimension)

    def test_invalid_position_source_raises(
        self,
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        invalid_source = "invalid_source"
        module = SinusoidalPositionalEncoding1D(
            embedding_dimension=64,
            position_source=invalid_source,
            precompute_encodings=False,
        )
        tensor = sequence_tensor_factory(
            batch_size=2,
            sequence_length=10,
            embedding_dimension=64,
        )
        with pytest.raises(
            ValueError,
            match=re.escape(f"Unsupported position_source for 1D: {invalid_source}"),
        ):
            module(tensor)

    def test_precomputed_encodings_are_expanded_to_batch(
        self,
        sinusoidal_1d_factory: Callable[..., SinusoidalPositionalEncoding1D],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 64
        module = sinusoidal_1d_factory(
            embedding_dimension=embedding_dimension,
            position_source=PositionSource.TENSOR_INDICES.value,
            precompute_encodings=True,
            maximum_length=100,
        )
        tensor = sequence_tensor_factory(
            batch_size=3,
            sequence_length=10,
            embedding_dimension=embedding_dimension,
        )
        output = module(tensor)
        # All batch elements should have the same positional encoding
        assert torch.allclose(output[0], output[1], atol=1e-7)
        assert torch.allclose(output[1], output[2], atol=1e-7)

    @pytest.mark.parametrize("precompute_encodings", [True, False])
    def test_offset_matches_full_sequence_slice(
        self,
        sinusoidal_1d_factory: Callable[..., SinusoidalPositionalEncoding1D],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        precompute_encodings: bool,
    ):
        embedding_dimension = 64
        full_sequence_length = 20
        sub_sequence_length = 5
        offset = 10
        module = sinusoidal_1d_factory(
            embedding_dimension=embedding_dimension,
            position_source=PositionSource.TENSOR_INDICES.value,
            precompute_encodings=precompute_encodings,
            maximum_length=100,
        )
        full_tensor = sequence_tensor_factory(
            batch_size=2,
            sequence_length=full_sequence_length,
            embedding_dimension=embedding_dimension,
        )
        sub_tensor = sequence_tensor_factory(
            batch_size=2,
            sequence_length=sub_sequence_length,
            embedding_dimension=embedding_dimension,
        )
        full_output = module(full_tensor)
        offset_output = module(sub_tensor, offset=offset)
        # Encodings with offset should match the corresponding slice from full sequence
        assert torch.allclose(
            full_output[:, offset : offset + sub_sequence_length, :],
            offset_output,
            atol=1e-6,
        )

    def test_offset_zero_is_default_behavior(
        self,
        sinusoidal_1d_factory: Callable[..., SinusoidalPositionalEncoding1D],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 64
        module = sinusoidal_1d_factory(
            embedding_dimension=embedding_dimension,
            position_source=PositionSource.TENSOR_INDICES.value,
            precompute_encodings=True,
            maximum_length=100,
        )
        tensor = sequence_tensor_factory(
            batch_size=2,
            sequence_length=10,
            embedding_dimension=embedding_dimension,
        )
        output_default = module(tensor)
        output_explicit = module(tensor, offset=0)
        assert torch.equal(output_default, output_explicit)

    def test_precomputed_sequence_exceeding_maximum_length_raises(
        self,
        sinusoidal_1d_factory: Callable[..., SinusoidalPositionalEncoding1D],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 64
        maximum_length = 16
        sequence_length = maximum_length + 4
        module = sinusoidal_1d_factory(
            embedding_dimension=embedding_dimension,
            position_source=PositionSource.TENSOR_INDICES.value,
            precompute_encodings=True,
            maximum_length=maximum_length,
        )
        tensor = sequence_tensor_factory(
            batch_size=2,
            sequence_length=sequence_length,
            embedding_dimension=embedding_dimension,
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Requested positions [0, {sequence_length}) exceed "
                f"precomputed maximum_length {maximum_length}. "
                f"Increase maximum_length."
            ),
        ):
            module(tensor)

    def test_precomputed_offset_pushes_past_maximum_length_raises(
        self,
        sinusoidal_1d_factory: Callable[..., SinusoidalPositionalEncoding1D],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 64
        maximum_length = 16
        sequence_length = 4
        offset = 14
        module = sinusoidal_1d_factory(
            embedding_dimension=embedding_dimension,
            position_source=PositionSource.TENSOR_INDICES.value,
            precompute_encodings=True,
            maximum_length=maximum_length,
        )
        tensor = sequence_tensor_factory(
            batch_size=2,
            sequence_length=sequence_length,
            embedding_dimension=embedding_dimension,
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Requested positions [{offset}, {offset + sequence_length}) exceed "
                f"precomputed maximum_length {maximum_length}. "
                f"Increase maximum_length."
            ),
        ):
            module(tensor, offset=offset)
