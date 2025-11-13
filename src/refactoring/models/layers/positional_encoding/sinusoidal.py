import math
from collections.abc import Callable

import torch
import torch.nn as nn

from refactoring.models.layers.positional_encoding.base import (
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
        if embedding_dimension % 2 != 0:
            raise ValueError("embedding_dimension must be even")

        self.ordering_mode = ordering_mode
        self.temperature = temperature

        half_dimension = embedding_dimension // 2
        if denominator_mode == DenominatorMode.HALF.value:
            denominator = half_dimension
        elif denominator_mode == DenominatorMode.HALF_MINUS_ONE.value:
            denominator = half_dimension - 1
        else:
            raise ValueError(f"Invalid denominator_mode: {denominator_mode}")

        log_scale = math.log(self.temperature) / denominator
        frequencies = torch.exp(torch.arange(half_dimension) * -log_scale).float()
        self._temp_frequencies = frequencies
        self._learnable_frequencies = learnable_frequencies
        super().__init__(
            embedding_dimension=embedding_dimension,
            position_source=position_source,
            precompute_encodings=precompute_encodings,
            maximum_length=maximum_length,
            mlp_hidden_dimensions=mlp_hidden_dimensions,
            mlp_activation=mlp_activation,
        )
        self.register_parameter("frequencies", nn.Parameter(self._temp_frequencies, requires_grad=self._learnable_frequencies))
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
        if hasattr(self, 'frequencies'):
            frequencies = self.frequencies
        else:
            frequencies = self._temp_frequencies
        scaled_values = input_values[:, None] * frequencies[None] if input_values.dim() == 1 else input_values * frequencies
        sine_values = torch.sin(scaled_values)
        cosine_values = torch.cos(scaled_values)
        if self.ordering_mode == OrderingMode.INTERLEAVE_SIN_COS.value:
            encodings = torch.zeros(*scaled_values.shape[:-1] or [1], self.embedding_dimension, device=input_values.device)
            encodings[..., 0::2] = sine_values
            encodings[..., 1::2] = cosine_values
        elif self.ordering_mode == OrderingMode.CAT_COS_SIN.value:
            encodings = torch.cat((cosine_values, sine_values), dim=-1)
        else:
            raise ValueError(f"Invalid ordering mode: {self.ordering_mode}")
        return encodings


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
        if embedding_dimension % 2 != 0:
            raise ValueError("embedding_dimension must be even")

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
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2)
        return pos.permute(2, 0, 1)

