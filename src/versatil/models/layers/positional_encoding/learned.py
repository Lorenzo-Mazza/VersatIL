from collections.abc import Callable

import torch
import torch.nn as nn

from versatil.models.layers.positional_encoding.base import (
    PositionalEncoding1D,
    PositionalEncoding2D,
    PositionSource,
)


class LearnedPositionalEncoding1D(PositionalEncoding1D):
    """Learned positional encoding for 1D."""

    def __init__(
        self,
        embedding_dimension: int,
        position_source: str = PositionSource.TENSOR_INDICES.value,
        maximum_sequence_length: int = 5000,
        mlp_hidden_dimensions: list[int] | None = None,
        mlp_activation: Callable | None = nn.SiLU,
    ):
        if maximum_sequence_length is None:
            raise ValueError(
                "maximum_sequence_length must be provided for 1D learned encoding"
            )
        super().__init__(
            embedding_dimension=embedding_dimension,
            position_source=position_source,
            precompute_encodings=False,
            maximum_sequence_length=maximum_sequence_length,
            mlp_hidden_dimensions=mlp_hidden_dimensions,
            mlp_activation=mlp_activation,
        )
        self.learned_encoding = nn.Embedding(
            maximum_sequence_length, embedding_dimension
        )

    def _compute_encodings(self, input_values: torch.Tensor) -> torch.Tensor:
        if self.maximum_sequence_length is None:
            raise RuntimeError(
                "maximum_sequence_length must be set for learned encoding"
            )
        input_values = input_values.long()
        if (
            self.position_source == PositionSource.TENSOR_INDICES.value
            and input_values.numel() > 0
            # torch.export/compile cannot trace the data-dependent Python int
            # conversion below. Side effect: compiled/exported graphs skip this
            # eager range check and clamp invalid positions instead of raising.
            and not torch.compiler.is_compiling()
        ):
            max_position = int(input_values.max())
            min_position = int(input_values.min())
            if min_position < 0 or max_position >= self.maximum_sequence_length:
                raise ValueError(
                    f"Position indices [{min_position}, {max_position}] out of range "
                    f"[0, {self.maximum_sequence_length - 1}]. Increase maximum_sequence_length."
                )
        else:
            input_values = input_values.clamp(0, self.maximum_sequence_length - 1)
        result: torch.Tensor = self.learned_encoding(input_values)
        return result


class LearnedPositionalEncoding2D(PositionalEncoding2D):
    """Learned positional encoding for 2D."""

    def __init__(
        self,
        embedding_dimension: int,
        max_height: int = 50,
        max_width: int = 50,
        mlp_hidden_dimensions: list[int] | None = None,
        mlp_activation: Callable | None = nn.SiLU,
    ):
        if embedding_dimension % 2 != 0:
            raise ValueError("embedding_dimension must be even for 2D learned encoding")
        if max_height is None or max_width is None:
            raise ValueError(
                "max_height and max_width must be provided for 2D learned encoding"
            )
        half_dim = embedding_dimension // 2

        super().__init__(
            embedding_dimension=embedding_dimension,
            mlp_hidden_dimensions=mlp_hidden_dimensions,
            mlp_activation=mlp_activation,
        )
        self.row_encoding = nn.Embedding(max_height, half_dim)
        self.col_encoding = nn.Embedding(max_width, half_dim)

    def _compute_encodings(self, input_values: torch.Tensor) -> torch.Tensor:
        height, width = input_values.shape
        rows = torch.arange(height, device=input_values.device)
        cols = torch.arange(width, device=input_values.device)
        y_enc = (
            self.row_encoding(rows).unsqueeze(1).repeat(1, width, 1)
        )  # [height, width, half_dim]
        x_enc = (
            self.col_encoding(cols).unsqueeze(0).repeat(height, 1, 1)
        )  # [height, width, half_dim]
        encodings = torch.cat(
            [y_enc, x_enc], dim=-1
        )  # [height, width, embedding_dimension]
        return encodings.permute(2, 0, 1)  # [embedding_dimension, height, width]
