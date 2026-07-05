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
    PeriodInterpolationPositionalEncoding1D,
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
        mlp_hidden_dimensions: list[int] | None = None,
        mlp_activation: type | None = None,
    ) -> SinusoidalPositionalEncoding2D:
        return SinusoidalPositionalEncoding2D(
            embedding_dimension=embedding_dimension,
            temperature=temperature,
            normalize=normalize,
            scale=scale,
            mlp_hidden_dimensions=mlp_hidden_dimensions,
            mlp_activation=mlp_activation,
        )

    return factory


class TestSinusoidalPositionalEncoding1DInit:
    def test_odd_embedding_dimension_raises(self):
        with pytest.raises(
            ValueError,
            match=re.escape("embedding_dimension must be even"),
        ):
            SinusoidalPositionalEncoding1D(embedding_dimension=63)

    def test_half_minus_one_denominator_requires_positive_denominator(self):
        embedding_dimension = 2
        denominator_mode = DenominatorMode.HALF_MINUS_ONE.value
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"denominator must be positive for embedding_dimension "
                f"{embedding_dimension} and denominator_mode {denominator_mode}."
            ),
        ):
            SinusoidalPositionalEncoding1D(
                embedding_dimension=embedding_dimension,
                denominator_mode=denominator_mode,
            )

    def test_non_positive_temperature_raises(self):
        temperature = 0.0
        with pytest.raises(
            ValueError,
            match=re.escape(f"temperature must be positive, got {temperature}."),
        ):
            SinusoidalPositionalEncoding1D(
                embedding_dimension=64,
                temperature=temperature,
            )

    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    @pytest.mark.parametrize("temperature", [10000.0, 5000.0])
    @pytest.mark.parametrize(
        "ordering_mode",
        [
            OrderingMode.INTERLEAVE_SIN_COS.value,
            OrderingMode.CAT_COS_SIN.value,
        ],
    )
    def test_stores_configuration(
        self,
        embedding_dimension: int,
        temperature: float,
        ordering_mode: str,
    ):
        module = SinusoidalPositionalEncoding1D(
            embedding_dimension=embedding_dimension,
            temperature=temperature,
            ordering_mode=ordering_mode,
        )
        assert module.embedding_dimension == embedding_dimension
        assert module.temperature == temperature
        assert module.ordering_mode == ordering_mode

    def test_precomputed_buffer_has_correct_shape(
        self,
        sinusoidal_1d_factory: Callable[..., SinusoidalPositionalEncoding1D],
    ):
        embedding_dimension = 64
        maximum_sequence_length = 100
        module = sinusoidal_1d_factory(
            embedding_dimension=embedding_dimension,
            precompute_encodings=True,
            maximum_sequence_length=maximum_sequence_length,
            position_source=PositionSource.TENSOR_INDICES.value,
        )
        # Buffer should be (1, maximum_sequence_length, embedding_dimension)
        assert module.precomputed_encodings.shape == (
            1,
            maximum_sequence_length,
            embedding_dimension,
        )

    def test_no_precomputed_buffer_when_precomputed_false(
        self,
        sinusoidal_1d_factory: Callable[..., SinusoidalPositionalEncoding1D],
    ):
        module = sinusoidal_1d_factory(
            embedding_dimension=64,
            precompute_encodings=False,
            position_source=PositionSource.TENSOR_INDICES.value,
        )
        # When precompute_encodings is False, accessing the buffer should raise
        buffers = dict(module.named_buffers())
        assert "precomputed_encodings" not in buffers

    @pytest.mark.parametrize(
        "denominator_mode, expectation",
        [
            (DenominatorMode.HALF.value, does_not_raise()),
            (DenominatorMode.HALF_MINUS_ONE.value, does_not_raise()),
            (
                "invalid_mode",
                pytest.raises(
                    ValueError,
                    match=re.escape("Invalid denominator_mode: invalid_mode"),
                ),
            ),
        ],
    )
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

    def test_learnable_frequencies_disable_precomputed_buffer(
        self,
        sinusoidal_1d_factory: Callable[..., SinusoidalPositionalEncoding1D],
    ):
        module = sinusoidal_1d_factory(
            embedding_dimension=64,
            learnable_frequencies=True,
            precompute_encodings=True,
            maximum_sequence_length=100,
        )
        buffers = dict(module.named_buffers())
        assert module.precompute_encodings is False
        assert "precomputed_encodings" not in buffers


class TestSinusoidalPositionalEncoding1DForward:
    @pytest.mark.parametrize(
        "batch_size, sequence_length, embedding_dimension",
        [
            (2, 10, 64),
            (4, 20, 128),
        ],
    )
    def test_output_shape_tensor_indices(
        self,
        sinusoidal_1d_factory: Callable[..., SinusoidalPositionalEncoding1D],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
        sequence_length: int,
        embedding_dimension: int,
    ):
        module = sinusoidal_1d_factory(
            embedding_dimension=embedding_dimension,
            position_source=PositionSource.TENSOR_INDICES.value,
        )
        tensor = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=embedding_dimension,
        )
        output = module(tensor)
        assert output.shape == (batch_size, sequence_length, embedding_dimension)

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
            batch_size=1,
            sequence_length=10,
            embedding_dimension=64,
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
        sequence_length = 10
        precomputed = sinusoidal_1d_factory(
            embedding_dimension=embedding_dimension,
            precompute_encodings=True,
            maximum_sequence_length=100,
        )
        non_precomputed = sinusoidal_1d_factory(
            embedding_dimension=embedding_dimension,
            precompute_encodings=False,
        )
        tensor = sequence_tensor_factory(
            batch_size=2,
            sequence_length=sequence_length,
            embedding_dimension=embedding_dimension,
        )
        output_precomputed = precomputed(tensor)
        output_non_precomputed = non_precomputed(tensor)
        assert torch.allclose(output_precomputed, output_non_precomputed, atol=1e-6)

    def test_learnable_frequencies_receive_gradient_when_precompute_requested(
        self,
        sinusoidal_1d_factory: Callable[..., SinusoidalPositionalEncoding1D],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 64
        module = sinusoidal_1d_factory(
            embedding_dimension=embedding_dimension,
            learnable_frequencies=True,
            precompute_encodings=True,
            maximum_sequence_length=100,
        )
        tensor = sequence_tensor_factory(
            batch_size=2,
            sequence_length=8,
            embedding_dimension=embedding_dimension,
        )
        output = module(tensor)
        output[:, 1:, 0].sum().backward()
        assert module.frequencies.grad is not None
        assert torch.any(module.frequencies.grad != 0.0)


class TestCreateEncodingTable:
    @pytest.mark.parametrize(
        "number_of_positions, embedding_dimension",
        [
            (10, 64),
            (50, 128),
        ],
    )
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

    def test_embedding_dimension_not_divisible_by_four_raises(self):
        with pytest.raises(
            ValueError,
            match=re.escape("embedding_dimension must be divisible by 4"),
        ):
            SinusoidalPositionalEncoding2D(embedding_dimension=6)

    def test_non_positive_temperature_raises(self):
        temperature = 0.0
        with pytest.raises(
            ValueError,
            match=re.escape(f"temperature must be positive, got {temperature}."),
        ):
            SinusoidalPositionalEncoding2D(
                embedding_dimension=64,
                temperature=temperature,
            )

    @pytest.mark.parametrize(
        "batch_size, embedding_dimension, height, width",
        [
            (2, 64, 8, 8),
            (4, 128, 4, 6),
        ],
    )
    def test_output_shape(
        self,
        sinusoidal_2d_factory: Callable[..., SinusoidalPositionalEncoding2D],
        nchw_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
        embedding_dimension: int,
        height: int,
        width: int,
    ):
        module = sinusoidal_2d_factory(embedding_dimension=embedding_dimension)
        tensor = nchw_tensor_factory(
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
        nchw_tensor_factory: Callable[..., torch.Tensor],
    ):
        module = sinusoidal_2d_factory(embedding_dimension=64)
        tensor = nchw_tensor_factory(
            batch_size=1,
            channels=64,
            height=4,
            width=4,
        )
        output = module(tensor)
        encoding_00 = output[0, :, 0, 0]
        encoding_11 = output[0, :, 1, 1]
        assert not torch.allclose(encoding_00, encoding_11)

    def test_normalize_changes_encoding_values(
        self,
        sinusoidal_2d_factory: Callable[..., SinusoidalPositionalEncoding2D],
        nchw_tensor_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 64
        height, width = 8, 8
        normalized = sinusoidal_2d_factory(
            embedding_dimension=embedding_dimension,
            normalize=True,
        )
        unnormalized = sinusoidal_2d_factory(
            embedding_dimension=embedding_dimension,
            normalize=False,
        )
        tensor = nchw_tensor_factory(
            batch_size=1,
            channels=embedding_dimension,
            height=height,
            width=width,
        )
        output_normalized = normalized(tensor)
        output_unnormalized = unnormalized(tensor)
        assert not torch.allclose(output_normalized, output_unnormalized, atol=1e-5)

    def test_normalize_produces_scale_invariant_encodings(
        self,
        sinusoidal_2d_factory: Callable[..., SinusoidalPositionalEncoding2D],
        nchw_tensor_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 64
        module = sinusoidal_2d_factory(
            embedding_dimension=embedding_dimension,
            normalize=True,
        )
        small_tensor = nchw_tensor_factory(
            batch_size=1,
            channels=embedding_dimension,
            height=4,
            width=4,
        )
        large_tensor = nchw_tensor_factory(
            batch_size=1,
            channels=embedding_dimension,
            height=8,
            width=8,
        )
        output_small = module(small_tensor)
        output_large = module(large_tensor)
        # The last spatial position should have similar encoding since both are
        # normalized to the same scale range
        assert torch.allclose(
            output_small[0, :, -1, -1],
            output_large[0, :, -1, -1],
            atol=1e-4,
        )


class TestSinusoidalPositionalEncoding1DInvalidOrderingMode:
    def test_invalid_ordering_mode_raises_on_compute(self):
        invalid_ordering = "invalid_ordering"
        module = SinusoidalPositionalEncoding1D(
            embedding_dimension=64,
            ordering_mode=OrderingMode.INTERLEAVE_SIN_COS.value,
            precompute_encodings=False,
        )
        # Bypass __init__ validation by setting ordering_mode after construction
        module.ordering_mode = invalid_ordering
        input_values = torch.arange(10).float()
        with pytest.raises(
            ValueError,
            match=re.escape(f"Invalid ordering mode: {invalid_ordering}"),
        ):
            module._compute_encodings(input_values)


class TestSinusoidalPositionalEncoding1DMlpPostProcessing:
    def test_mlp_changes_output_dimension(
        self,
        sinusoidal_1d_factory: Callable[..., SinusoidalPositionalEncoding1D],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 64
        mlp_output_dimension = 128
        batch_size = 2
        sequence_length = 10
        module = sinusoidal_1d_factory(
            embedding_dimension=embedding_dimension,
            precompute_encodings=False,
            mlp_hidden_dimensions=[mlp_output_dimension],
        )
        tensor = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=embedding_dimension,
        )
        output = module(tensor)
        assert output.shape == (batch_size, sequence_length, mlp_output_dimension)

    def test_mlp_produces_different_output_than_no_mlp(
        self,
        sinusoidal_1d_factory: Callable[..., SinusoidalPositionalEncoding1D],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 64
        batch_size = 2
        sequence_length = 10
        module_without_mlp = sinusoidal_1d_factory(
            embedding_dimension=embedding_dimension,
            precompute_encodings=False,
        )
        module_with_mlp = sinusoidal_1d_factory(
            embedding_dimension=embedding_dimension,
            precompute_encodings=False,
            mlp_hidden_dimensions=[embedding_dimension],
        )
        tensor = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=embedding_dimension,
        )
        output_without_mlp = module_without_mlp(tensor)
        output_with_mlp = module_with_mlp(tensor)
        assert not torch.allclose(output_without_mlp, output_with_mlp, atol=1e-6)

    def test_mlp_with_scalar_source(
        self,
        sinusoidal_1d_factory: Callable[..., SinusoidalPositionalEncoding1D],
        scalar_tensor_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 64
        mlp_output_dimension = 32
        batch_size = 2
        module = sinusoidal_1d_factory(
            embedding_dimension=embedding_dimension,
            position_source=PositionSource.SCALAR.value,
            precompute_encodings=False,
            mlp_hidden_dimensions=[mlp_output_dimension],
        )
        tensor = scalar_tensor_factory(batch_size=batch_size)
        output = module(tensor)
        assert output.shape == (batch_size, mlp_output_dimension)

    def test_no_mlp_network_when_mlp_hidden_dimensions_is_none(
        self,
        sinusoidal_1d_factory: Callable[..., SinusoidalPositionalEncoding1D],
    ):
        module = sinusoidal_1d_factory(
            embedding_dimension=64,
            mlp_hidden_dimensions=None,
        )
        assert module.mlp_network is None


class TestSinusoidalPositionalEncoding2DMlpPostProcessing:
    def test_mlp_changes_2d_output_dimension(
        self,
        sinusoidal_2d_factory: Callable[..., SinusoidalPositionalEncoding2D],
        nchw_tensor_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 64
        mlp_output_dimension = 128
        batch_size = 2
        height = 4
        width = 4
        module = sinusoidal_2d_factory(
            embedding_dimension=embedding_dimension,
            mlp_hidden_dimensions=[mlp_output_dimension],
        )
        tensor = nchw_tensor_factory(
            batch_size=batch_size,
            channels=embedding_dimension,
            height=height,
            width=width,
        )
        output = module(tensor)
        assert output.shape == (batch_size, mlp_output_dimension, height, width)

    def test_2d_mlp_produces_different_output_than_no_mlp(
        self,
        sinusoidal_2d_factory: Callable[..., SinusoidalPositionalEncoding2D],
        nchw_tensor_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 64
        batch_size = 2
        height = 4
        width = 4
        module_without_mlp = sinusoidal_2d_factory(
            embedding_dimension=embedding_dimension,
        )
        module_with_mlp = sinusoidal_2d_factory(
            embedding_dimension=embedding_dimension,
            mlp_hidden_dimensions=[embedding_dimension],
        )
        tensor = nchw_tensor_factory(
            batch_size=batch_size,
            channels=embedding_dimension,
            height=height,
            width=width,
        )
        output_without_mlp = module_without_mlp(tensor)
        output_with_mlp = module_with_mlp(tensor)
        assert not torch.allclose(output_without_mlp, output_with_mlp, atol=1e-6)


class TestPeriodInterpolationPositionalEncoding1D:
    def test_odd_dimension_raises(self):
        with pytest.raises(
            ValueError,
            match=re.escape("embedding_dimension must be even"),
        ):
            PeriodInterpolationPositionalEncoding1D(embedding_dimension=7)

    def test_non_positive_min_period_raises(self):
        min_period = 0.0
        with pytest.raises(
            ValueError,
            match=re.escape(f"min_period must be positive, got {min_period}."),
        ):
            PeriodInterpolationPositionalEncoding1D(
                embedding_dimension=32,
                min_period=min_period,
            )

    def test_non_positive_max_period_raises(self):
        max_period = 0.0
        with pytest.raises(
            ValueError,
            match=re.escape(f"max_period must be positive, got {max_period}."),
        ):
            PeriodInterpolationPositionalEncoding1D(
                embedding_dimension=32,
                max_period=max_period,
            )

    def test_max_period_less_than_min_period_raises(self):
        min_period = 4.0
        max_period = 1.0
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"max_period must be greater than or equal to min_period, "
                f"got max_period={max_period} and min_period={min_period}."
            ),
        ):
            PeriodInterpolationPositionalEncoding1D(
                embedding_dimension=32,
                min_period=min_period,
                max_period=max_period,
            )

    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    def test_output_shape_for_scalar_input(self, embedding_dimension: int):
        module = PeriodInterpolationPositionalEncoding1D(
            embedding_dimension=embedding_dimension,
            position_source=PositionSource.SCALAR.value,
        )
        timestep = torch.tensor([0.5, 0.8])
        output = module(timestep)
        assert output.shape == (2, embedding_dimension)

    def test_different_timesteps_produce_different_encodings(self):
        module = PeriodInterpolationPositionalEncoding1D(
            embedding_dimension=32,
            position_source=PositionSource.SCALAR.value,
        )
        output_a = module(torch.tensor([0.1]))
        output_b = module(torch.tensor([0.9]))
        assert not torch.allclose(output_a, output_b)

    def test_frequencies_shape(self):
        module = PeriodInterpolationPositionalEncoding1D(
            embedding_dimension=32,
        )
        assert module.frequencies.shape == (16,)

    def test_frequencies_are_monotonically_decreasing(self):
        module = PeriodInterpolationPositionalEncoding1D(
            embedding_dimension=32,
            min_period=4e-3,
            max_period=4.0,
        )
        for i in range(len(module.frequencies) - 1):
            assert module.frequencies[i] > module.frequencies[i + 1]
