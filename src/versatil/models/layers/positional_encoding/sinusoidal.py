"""Sinusoidal positional encoding implementations."""

import math
from collections.abc import Callable

import torch
import torch.nn as nn

from versatil.models.layers.positional_encoding.base import (
    DenominatorMode,
    OrderingMode,
    PositionalEncoding1D,
    PositionalEncoding2D,
    PositionSource,
)


class SinusoidalPositionalEncoding1D(PositionalEncoding1D):
    """Sinusoidal positional encoding for 1D."""

    def __init__(
        self,
        embedding_dimension: int,
        denominator_mode: str = DenominatorMode.HALF.value,
        ordering_mode: str = OrderingMode.INTERLEAVE_SIN_COS.value,
        learnable_frequencies: bool = False,
        temperature: float = 10000.0,
        position_source: str = PositionSource.TENSOR_INDICES.value,
        precompute_encodings: bool = True,
        maximum_length: int | None = 5000,
        mlp_hidden_dimensions: list[int] | None = None,
        mlp_activation: Callable | None = nn.SiLU,
    ):
        """Initialize a 1D sinusoidal positional encoding module.

        Args:
            embedding_dimension: Output embedding dimension.
            denominator_mode: Frequency denominator convention.
            ordering_mode: Sine/cosine channel ordering convention.
            learnable_frequencies: Whether frequency bands are trainable.
            temperature: Base temperature for geometric frequency spacing.
            position_source: Source used to derive positions.
            precompute_encodings: Whether to cache tensor-index encodings.
            maximum_length: Maximum length for cached tensor-index encodings.
            mlp_hidden_dimensions: Optional post-encoding MLP dimensions.
            mlp_activation: Optional post-encoding MLP activation.

        Raises:
            ValueError: If dimensions or frequency settings are invalid.
        """
        if embedding_dimension % 2 != 0:
            raise ValueError("embedding_dimension must be even")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}.")

        self.ordering_mode = ordering_mode
        self.temperature = temperature

        half_dimension = embedding_dimension // 2
        if denominator_mode == DenominatorMode.HALF.value:
            denominator = half_dimension
        elif denominator_mode == DenominatorMode.HALF_MINUS_ONE.value:
            denominator = half_dimension - 1
        else:
            raise ValueError(f"Invalid denominator_mode: {denominator_mode}")
        if denominator <= 0:
            raise ValueError(
                f"denominator must be positive for embedding_dimension "
                f"{embedding_dimension} and denominator_mode {denominator_mode}."
            )

        log_scale = math.log(self.temperature) / denominator
        frequencies = torch.exp(torch.arange(half_dimension) * -log_scale).float()
        self._temp_frequencies = frequencies
        self._learnable_frequencies = learnable_frequencies
        if learnable_frequencies and precompute_encodings:
            precompute_encodings = False
        super().__init__(
            embedding_dimension=embedding_dimension,
            position_source=position_source,
            precompute_encodings=precompute_encodings,
            maximum_length=maximum_length,
            mlp_hidden_dimensions=mlp_hidden_dimensions,
            mlp_activation=mlp_activation,
        )
        self.register_parameter(
            "frequencies",
            nn.Parameter(
                self._temp_frequencies, requires_grad=self._learnable_frequencies
            ),
        )
        del self._temp_frequencies
        del self._learnable_frequencies

    @classmethod
    def create_encoding_table(
        cls,
        number_of_positions: int,
        embedding_dimension: int,
        temperature: float = 10000.0,
    ) -> torch.Tensor:
        """Create a sinusoidal encoding table for transformer inputs.

        This is a convenience method that generates a standard positional encoding
        table from a number of positions and embedding dimension.

        Args:
            number_of_positions: Number of positions to encode.
            embedding_dimension: Dimension of positional encodings.
            temperature: Temperature parameter (default 10000 from "Attention Is All You Need").

        Returns:
            Encoding table of shape (1, number_of_positions, embedding_dimension).
        """
        encoder = cls(
            embedding_dimension=embedding_dimension,
            temperature=temperature,
            precompute_encodings=True,
            maximum_length=number_of_positions,
            mlp_hidden_dimensions=None,
        )
        dummy_input = torch.zeros(1, number_of_positions, embedding_dimension)
        encoding_table: torch.Tensor = encoder(dummy_input)
        return encoding_table

    def _compute_encodings(self, input_values: torch.Tensor) -> torch.Tensor:
        if input_values.dim() == 0:
            input_values = input_values.unsqueeze(0)
        # Use temporary tensor during init, otherwise use registered parameter
        if hasattr(self, "frequencies"):
            frequencies = self.frequencies
        else:
            frequencies = self._temp_frequencies
        scaled_values = (
            input_values[:, None] * frequencies[None]
            if input_values.dim() == 1
            else input_values * frequencies
        )
        sine_values = torch.sin(scaled_values)
        cosine_values = torch.cos(scaled_values)
        if self.ordering_mode == OrderingMode.INTERLEAVE_SIN_COS.value:
            encodings = torch.zeros(
                *scaled_values.shape[:-1] or [1],
                self.embedding_dimension,
                device=input_values.device,
            )
            encodings[..., 0::2] = sine_values
            encodings[..., 1::2] = cosine_values
        elif self.ordering_mode == OrderingMode.CAT_COS_SIN.value:
            encodings = torch.cat((cosine_values, sine_values), dim=-1)
        else:
            raise ValueError(f"Invalid ordering mode: {self.ordering_mode}")
        return encodings


class PeriodInterpolationPositionalEncoding1D(PositionalEncoding1D):
    """Sinusoidal encoding with geometric period interpolation.

    Computes frequencies by logarithmically interpolating between a minimum
    and maximum period, giving direct control over the sensitivity range.
    Suited for encoding continuous scalar values like normalized timesteps.

    Frequency formula::

        fraction = linspace(0, 1, dim // 2)
        period = min_period * (max_period / min_period) ^ fraction
        freq = 2π / period
    """

    def __init__(
        self,
        embedding_dimension: int,
        min_period: float = 4e-3,
        max_period: float = 4.0,
        position_source: str = PositionSource.SCALAR.value,
        mlp_hidden_dimensions: list[int] | None = None,
        mlp_activation: Callable | None = nn.SiLU,
    ):
        """Initialize scalar sinusoidal encoding with interpolated periods.

        Args:
            embedding_dimension: Output embedding dimension.
            min_period: Smallest encoded period.
            max_period: Largest encoded period.
            position_source: Source used to derive positions.
            mlp_hidden_dimensions: Optional post-encoding MLP dimensions.
            mlp_activation: Optional post-encoding MLP activation.

        Raises:
            ValueError: If dimensions or periods are invalid.
        """
        if embedding_dimension % 2 != 0:
            raise ValueError("embedding_dimension must be even")
        if min_period <= 0.0:
            raise ValueError(f"min_period must be positive, got {min_period}.")
        if max_period <= 0.0:
            raise ValueError(f"max_period must be positive, got {max_period}.")
        if max_period < min_period:
            raise ValueError(
                f"max_period must be greater than or equal to min_period, "
                f"got max_period={max_period} and min_period={min_period}."
            )

        self.min_period = min_period
        self.max_period = max_period

        half_dimension = embedding_dimension // 2
        fraction = torch.linspace(0.0, 1.0, half_dimension, dtype=torch.float64)
        periods = min_period * (max_period / min_period) ** fraction
        frequencies = (2.0 * math.pi / periods).float()
        self._temp_frequencies = frequencies

        super().__init__(
            embedding_dimension=embedding_dimension,
            position_source=position_source,
            precompute_encodings=False,
            maximum_length=None,
            mlp_hidden_dimensions=mlp_hidden_dimensions,
            mlp_activation=mlp_activation,
        )
        self.register_buffer("frequencies", self._temp_frequencies)
        del self._temp_frequencies

    def _compute_encodings(self, input_values: torch.Tensor) -> torch.Tensor:
        if input_values.dim() == 0:
            input_values = input_values.unsqueeze(0)
        scaled_values = (
            input_values[:, None] * self.frequencies[None]
            if input_values.dim() == 1
            else input_values * self.frequencies
        )
        return torch.cat([torch.sin(scaled_values), torch.cos(scaled_values)], dim=-1)


class SinusoidalPositionalEncoding2D(PositionalEncoding2D):
    """Sinusoidal positional encoding for 2D."""

    def __init__(
        self,
        embedding_dimension: int,
        temperature: float = 10000.0,
        normalize: bool = False,
        scale: float | None = None,
        mlp_hidden_dimensions: list[int] | None = None,
        mlp_activation: Callable | None = nn.SiLU,
    ):
        """Initialize a 2D sinusoidal positional encoding module.

        Args:
            embedding_dimension: Output channel dimension.
            temperature: Base temperature for frequency spacing.
            normalize: Whether to normalize coordinates to ``scale``.
            scale: Coordinate scale used when ``normalize`` is true.
            mlp_hidden_dimensions: Optional post-encoding MLP dimensions.
            mlp_activation: Optional post-encoding MLP activation.

        Raises:
            ValueError: If dimensions or frequency settings are invalid.
        """
        if embedding_dimension % 2 != 0:
            raise ValueError("embedding_dimension must be even")
        if embedding_dimension % 4 != 0:
            raise ValueError("embedding_dimension must be divisible by 4")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}.")

        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale if scale is not None else 2 * math.pi

        super().__init__(
            embedding_dimension=embedding_dimension,
            mlp_hidden_dimensions=mlp_hidden_dimensions,
            mlp_activation=mlp_activation,
        )

    def _compute_encodings(self, input_values: torch.Tensor) -> torch.Tensor:
        height, width = input_values.shape
        device = input_values.device
        not_mask = torch.ones((height, width), dtype=torch.float32, device=device)
        y_encoding = torch.cumsum(not_mask, dim=0)
        x_encoding = torch.cumsum(not_mask, dim=1)
        if self.normalize:
            eps = 1e-6
            y_encoding = y_encoding / (y_encoding[-1:, :] + eps) * self.scale
            x_encoding = x_encoding / (x_encoding[:, -1:] + eps) * self.scale
        num_pos_feats = self.embedding_dimension // 2
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / num_pos_feats)
        pos_y = y_encoding[:, :, None] / dim_t
        pos_x = x_encoding[:, :, None] / dim_t
        pos_y = torch.stack(
            (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos_x = torch.stack(
            (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2)
        return pos.permute(2, 0, 1)
