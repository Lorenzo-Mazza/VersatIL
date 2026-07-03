"""Tests for versatil.models.layers.transformer.positional_encoding module."""

import re
from collections.abc import Callable

import numpy as np
import pytest
import torch

from versatil.models.layers.constants import PositionalEncodingType
from versatil.models.layers.positional_encoding.rotary import (
    RotaryPositionalEncoding1D,
)
from versatil.models.layers.positional_encoding.sinusoidal import (
    SinusoidalPositionalEncoding1D,
)
from versatil.models.layers.transformer.positional_encoding import (
    apply_rope_positional_encoding,
    create_positional_encoding,
)


@pytest.fixture
def rope_encoding() -> RotaryPositionalEncoding1D:
    return RotaryPositionalEncoding1D(
        embedding_dimension=32,
        number_of_heads=4,
    )


@pytest.fixture
def query_key_factory(
    rng: np.random.Generator,
) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    def factory(
        batch_size: int = 2,
        number_of_heads: int = 4,
        sequence_length: int = 5,
        head_dimension: int = 8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        queries = torch.from_numpy(
            rng.standard_normal(
                (batch_size, number_of_heads, sequence_length, head_dimension)
            ).astype(np.float32)
        )
        keys = torch.from_numpy(
            rng.standard_normal(
                (batch_size, number_of_heads, sequence_length, head_dimension)
            ).astype(np.float32)
        )
        return queries, keys

    return factory


class TestCreatePositionalEncoding:
    def test_sinusoidal_produces_additive_encoding(self):
        encoding = create_positional_encoding(
            encoding_type=PositionalEncodingType.SINUSOIDAL.value,
            embedding_dimension=32,
            maximum_sequence_length=128,
        )
        # Sinusoidal produces position-dependent additive encoding
        input_tensor = torch.zeros(1, 5, 32)
        output = encoding(input_tensor)
        assert output.shape == (1, 5, 32)
        # Different positions produce different encodings
        assert not torch.equal(output[0, 0], output[0, 1])

    def test_learned_produces_additive_encoding(self):
        encoding = create_positional_encoding(
            encoding_type=PositionalEncodingType.LEARNED.value,
            embedding_dimension=32,
            maximum_sequence_length=128,
        )
        # Learned encoding produces additive positional encoding
        input_tensor = torch.zeros(1, 5, 32)
        output = encoding(input_tensor)
        assert output.shape == (1, 5, 32)

    def test_rope_produces_rotation_components(self):
        encoding = create_positional_encoding(
            encoding_type=PositionalEncodingType.ROPE.value,
            embedding_dimension=32,
            maximum_sequence_length=128,
            number_of_heads=4,
        )
        # RoPE produces sine/cosine rotation components
        sine, cosine = encoding.compute_rotation_components(seq_len=5)
        assert sine.shape == (5, 8)
        assert cosine.shape == (5, 8)

    def test_rope_without_num_heads_raises(self):
        with pytest.raises(
            ValueError,
            match=re.escape("number_of_heads is required for RoPE positional encoding"),
        ):
            create_positional_encoding(
                encoding_type=PositionalEncodingType.ROPE.value,
                embedding_dimension=32,
                maximum_sequence_length=128,
                number_of_heads=None,
            )

    def test_unsupported_type_raises(self):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Unsupported positional encoding type: invalid_type. "
                f"Must be one of {[e.value for e in PositionalEncodingType]}."
            ),
        ):
            create_positional_encoding(
                encoding_type="invalid_type",
                embedding_dimension=32,
                maximum_sequence_length=128,
            )

    def test_rope_different_base_frequency_produces_different_rotations(self):
        encoding_a = create_positional_encoding(
            encoding_type=PositionalEncodingType.ROPE.value,
            embedding_dimension=32,
            maximum_sequence_length=128,
            number_of_heads=4,
            base_frequency=5000.0,
        )
        encoding_b = create_positional_encoding(
            encoding_type=PositionalEncodingType.ROPE.value,
            embedding_dimension=32,
            maximum_sequence_length=128,
            number_of_heads=4,
            base_frequency=10000.0,
        )
        sine_a, _ = encoding_a.compute_rotation_components(seq_len=5)
        sine_b, _ = encoding_b.compute_rotation_components(seq_len=5)
        assert not torch.equal(sine_a, sine_b)

    def test_rope_learnable_frequencies_included_in_parameters(self):
        encoding = create_positional_encoding(
            encoding_type=PositionalEncodingType.ROPE.value,
            embedding_dimension=32,
            maximum_sequence_length=128,
            number_of_heads=4,
            learnable_frequencies=True,
        )
        # Learnable frequencies appear in the parameter list for optimizer
        parameter_names = [name for name, _ in encoding.named_parameters()]
        assert "frequencies" in parameter_names

    def test_rope_non_learnable_frequencies_excluded_from_gradient(self):
        encoding = create_positional_encoding(
            encoding_type=PositionalEncodingType.ROPE.value,
            embedding_dimension=32,
            maximum_sequence_length=128,
            number_of_heads=4,
            learnable_frequencies=False,
        )
        # Non-learnable: gradient computation is disabled
        assert encoding.frequencies.requires_grad is False


class TestApplyRopePositionalEncoding:
    def test_output_shape_preserved(
        self,
        rope_encoding: RotaryPositionalEncoding1D,
        query_key_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    ):
        queries, keys = query_key_factory(
            batch_size=2, number_of_heads=4, sequence_length=5, head_dimension=8
        )
        rotated_queries, rotated_keys = apply_rope_positional_encoding(
            queries=queries,
            keys=keys,
            positional_encoding=rope_encoding,
        )
        assert rotated_queries.shape == queries.shape
        assert rotated_keys.shape == keys.shape

    def test_rope_modifies_values(
        self,
        rope_encoding: RotaryPositionalEncoding1D,
        query_key_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    ):
        queries, keys = query_key_factory()
        rotated_queries, rotated_keys = apply_rope_positional_encoding(
            queries=queries,
            keys=keys,
            positional_encoding=rope_encoding,
        )
        assert not torch.equal(rotated_queries, queries)
        assert not torch.equal(rotated_keys, keys)

    def test_cache_position_offsets_rotation(
        self,
        rope_encoding: RotaryPositionalEncoding1D,
        query_key_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    ):
        queries, keys = query_key_factory(sequence_length=1)
        result_at_zero, _ = apply_rope_positional_encoding(
            queries=queries,
            keys=keys,
            positional_encoding=rope_encoding,
            cache_position=0,
        )
        result_at_three, _ = apply_rope_positional_encoding(
            queries=queries,
            keys=keys,
            positional_encoding=rope_encoding,
            cache_position=3,
        )
        # Same input at different positions should produce different rotated output
        assert not torch.equal(result_at_zero, result_at_three)

    def test_non_rope_encoding_returns_unchanged(
        self,
        query_key_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    ):
        queries, keys = query_key_factory()
        sinusoidal = SinusoidalPositionalEncoding1D(
            embedding_dimension=32,
            maximum_sequence_length=128,
        )
        rotated_queries, rotated_keys = apply_rope_positional_encoding(
            queries=queries,
            keys=keys,
            positional_encoding=sinusoidal,
        )
        assert torch.equal(rotated_queries, queries)
        assert torch.equal(rotated_keys, keys)
