"""Tests for versatil.models.layers.positional_encoding.learned module."""

import re
from collections.abc import Callable

import pytest
import torch

from versatil.models.layers.positional_encoding.base import PositionSource
from versatil.models.layers.positional_encoding.learned import (
    LearnedPositionalEncoding1D,
    LearnedPositionalEncoding2D,
)


@pytest.fixture
def learned_1d_factory() -> Callable[..., LearnedPositionalEncoding1D]:
    """Factory for LearnedPositionalEncoding1D instances."""

    def factory(
        embedding_dimension: int = 64,
        position_source: str = PositionSource.TENSOR_INDICES.value,
        maximum_length: int = 100,
    ) -> LearnedPositionalEncoding1D:
        return LearnedPositionalEncoding1D(
            embedding_dimension=embedding_dimension,
            position_source=position_source,
            maximum_length=maximum_length,
        )

    return factory


@pytest.fixture
def learned_2d_factory() -> Callable[..., LearnedPositionalEncoding2D]:
    """Factory for LearnedPositionalEncoding2D instances."""

    def factory(
        embedding_dimension: int = 64,
        max_height: int = 50,
        max_width: int = 50,
    ) -> LearnedPositionalEncoding2D:
        return LearnedPositionalEncoding2D(
            embedding_dimension=embedding_dimension,
            max_height=max_height,
            max_width=max_width,
        )

    return factory


class TestLearnedPositionalEncoding1D:
    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    @pytest.mark.parametrize("maximum_length", [50, 100])
    def test_stores_configuration(
        self,
        embedding_dimension: int,
        maximum_length: int,
    ):
        module = LearnedPositionalEncoding1D(
            embedding_dimension=embedding_dimension,
            maximum_length=maximum_length,
        )
        assert module.embedding_dimension == embedding_dimension
        assert module.maximum_length == maximum_length

    def test_maximum_length_none_raises(self):
        with pytest.raises(
            ValueError,
            match=re.escape("maximum_length must be provided for 1D learned encoding"),
        ):
            LearnedPositionalEncoding1D(
                embedding_dimension=64,
                maximum_length=None,
            )

    def test_learned_encoding_dimensions(
        self,
        learned_1d_factory: Callable[..., LearnedPositionalEncoding1D],
    ):
        module = learned_1d_factory(
            embedding_dimension=64,
            maximum_length=100,
        )
        assert module.learned_encoding.num_embeddings == 100
        assert module.learned_encoding.embedding_dim == 64
        # Verify the encoding is learnable (has requires_grad=True weights)
        assert module.learned_encoding.weight.requires_grad is True

    @pytest.mark.parametrize(
        "batch_size, sequence_length, embedding_dimension",
        [
            (2, 10, 64),
            (4, 20, 128),
        ],
    )
    def test_output_shape_tensor_indices(
        self,
        learned_1d_factory: Callable[..., LearnedPositionalEncoding1D],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
        sequence_length: int,
        embedding_dimension: int,
    ):
        module = learned_1d_factory(
            embedding_dimension=embedding_dimension,
            maximum_length=max(sequence_length, 100),
        )
        tensor = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=embedding_dimension,
        )
        output = module(tensor)
        assert output.shape == (batch_size, sequence_length, embedding_dimension)

    def test_different_positions_produce_different_encodings(
        self,
        learned_1d_factory: Callable[..., LearnedPositionalEncoding1D],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        module = learned_1d_factory(embedding_dimension=64, maximum_length=100)
        tensor = sequence_tensor_factory(
            batch_size=1,
            sequence_length=10,
            embedding_dimension=64,
        )
        output = module(tensor)
        encoding_pos_0 = output[0, 0]
        encoding_pos_5 = output[0, 5]
        # Learned embeddings are randomly initialized so positions should differ
        assert not torch.equal(encoding_pos_0, encoding_pos_5)

    def test_output_shape_scalar(
        self,
        learned_1d_factory: Callable[..., LearnedPositionalEncoding1D],
        scalar_tensor_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 64
        batch_size = 2
        module = learned_1d_factory(
            embedding_dimension=embedding_dimension,
            position_source=PositionSource.SCALAR.value,
            maximum_length=100,
        )
        tensor = scalar_tensor_factory(batch_size=batch_size)
        output = module(tensor)
        assert output.shape == (batch_size, embedding_dimension)


class TestLearnedPositionalEncoding2D:
    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    @pytest.mark.parametrize(
        "max_height, max_width",
        [
            (10, 10),
            (50, 50),
        ],
    )
    def test_stores_configuration(
        self,
        embedding_dimension: int,
        max_height: int,
        max_width: int,
    ):
        module = LearnedPositionalEncoding2D(
            embedding_dimension=embedding_dimension,
            max_height=max_height,
            max_width=max_width,
        )
        assert module.embedding_dimension == embedding_dimension

    def test_odd_embedding_dimension_raises(self):
        with pytest.raises(
            ValueError,
            match=re.escape("embedding_dimension must be even for 2D learned encoding"),
        ):
            LearnedPositionalEncoding2D(
                embedding_dimension=63,
                max_height=50,
                max_width=50,
            )

    def test_none_max_height_raises(self):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "max_height and max_width must be provided for 2D learned encoding"
            ),
        ):
            LearnedPositionalEncoding2D(
                embedding_dimension=64,
                max_height=None,
                max_width=50,
            )

    def test_none_max_width_raises(self):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "max_height and max_width must be provided for 2D learned encoding"
            ),
        ):
            LearnedPositionalEncoding2D(
                embedding_dimension=64,
                max_height=50,
                max_width=None,
            )

    def test_row_and_col_encoding_dimensions(
        self,
        learned_2d_factory: Callable[..., LearnedPositionalEncoding2D],
    ):
        module = learned_2d_factory(
            embedding_dimension=64,
            max_height=10,
            max_width=20,
        )
        half_dim = 32
        assert module.row_encoding.num_embeddings == 10
        assert module.row_encoding.embedding_dim == half_dim
        assert module.col_encoding.num_embeddings == 20
        assert module.col_encoding.embedding_dim == half_dim
        # Verify encodings are learnable
        assert module.row_encoding.weight.requires_grad is True
        assert module.col_encoding.weight.requires_grad is True

    @pytest.mark.parametrize(
        "batch_size, embedding_dimension, height, width",
        [
            (2, 64, 8, 8),
            (4, 128, 4, 6),
        ],
    )
    def test_output_shape(
        self,
        learned_2d_factory: Callable[..., LearnedPositionalEncoding2D],
        nchw_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
        embedding_dimension: int,
        height: int,
        width: int,
    ):
        module = learned_2d_factory(
            embedding_dimension=embedding_dimension,
            max_height=max(height, 50),
            max_width=max(width, 50),
        )
        tensor = nchw_tensor_factory(
            batch_size=batch_size,
            channels=embedding_dimension,
            height=height,
            width=width,
        )
        output = module(tensor)
        assert output.shape == (batch_size, embedding_dimension, height, width)
