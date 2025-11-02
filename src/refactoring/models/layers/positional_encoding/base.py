import abc
import enum
from abc import abstractmethod
from collections.abc import Callable

import torch
import torch.nn as nn

from refactoring.models.layers.mlp import MLP


class PositionSource(enum.Enum):
    TENSOR_INDICES = 'tensor_indices'  # For encoding positions in a sequence
    SCALAR = 'scalar'  # For encoding continuous scalar values like timesteps
    GRID_2D = 'grid_2d'  # For encoding 2D grid positions, e.g., images


class DenominatorMode(enum.Enum):
    HALF = 'half'  # Original Vaswani et al. formulation
    HALF_MINUS_ONE = 'half_minus_one'  # DDPM formulation


class OrderingMode(enum.Enum):
    INTERLEAVE_SIN_COS = 'interleave_sin_cos'  # Original Vaswani et al. formulation
    CAT_COS_SIN = 'cat_cos_sin'  # DDPM formulation


class PositionalEncoding(abc.ABC, nn.Module):
    """Base class for positional encoding with optional precomputing and MLP learnable layer."""
    def __init__(
        self,
        embedding_dimension: int,
        precompute_encodings: bool = True,
        maximum_length: int | None = 5000,
        mlp_hidden_dimensions: list[int] | None = None,
        mlp_activation: Callable | None = nn.SiLU,
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.maximum_length = maximum_length if precompute_encodings else None

        self.mlp_network = None  # An extra learnable MLP layer after positional encoding.
        if mlp_hidden_dimensions:
            # Use nn.SiLU as default if mlp_activation is None
            activation = mlp_activation if mlp_activation is not None else nn.SiLU
            self.mlp_network = MLP(input_dim=embedding_dimension, hidden_dims=mlp_hidden_dimensions, activation_function=activation)

    @abstractmethod
    def _compute_encodings(self, input_values: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement _compute_encodings")

    @abstractmethod
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward")


class PositionalEncoding1D(PositionalEncoding, abc.ABC):
    """Base class for 1D positional encodings."""
    def __init__(
        self,
        embedding_dimension: int,
        position_source: str = PositionSource.TENSOR_INDICES.value,
        precompute_encodings: bool = True,
        maximum_length: int | None = 5000,
        mlp_hidden_dimensions: list[int] | None = None,
        mlp_activation: Callable | None = nn.SiLU,
    ):
        self.position_source = position_source
        super().__init__(
            embedding_dimension=embedding_dimension,
            precompute_encodings=precompute_encodings,
            maximum_length=maximum_length,
            mlp_hidden_dimensions=mlp_hidden_dimensions,
            mlp_activation=mlp_activation,
        )
        if precompute_encodings and self.position_source == PositionSource.TENSOR_INDICES.value:
            if self.maximum_length is None:
                raise ValueError("maximum_length must be set when precompute_encodings=True")
            precomputed_encodings = self._compute_encodings(torch.arange(self.maximum_length).float())
            self.register_buffer("precomputed_encodings", precomputed_encodings.unsqueeze(1))  # [maximum_length, 1, embedding_dimension]

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        encodings: torch.Tensor
        if self.position_source == PositionSource.TENSOR_INDICES.value:
            seq_len = input_tensor.size(0)
            if self.precomputed_encodings is not None:
                encodings = self.precomputed_encodings[:seq_len].clone().detach()
            else:
                encodings = self._compute_encodings(torch.arange(seq_len).to(input_tensor.device))
            encodings = encodings.repeat(1, input_tensor.size(1), 1)  # [sequence_length, batch_size, embedding_dimension]
        elif self.position_source == PositionSource.SCALAR.value:
            encodings = self._compute_encodings(input_tensor)  # [batch_size, embedding_dimension]
        else:
            raise ValueError(f"Unsupported position_source for 1D: {self.position_source}")
        if self.mlp_network:
            encodings_mlp: torch.Tensor = self.mlp_network(encodings)
            return encodings_mlp
        return encodings


class PositionalEncoding2D(PositionalEncoding, abc.ABC):
    """Base class for 2D positional encodings."""
    def __init__(
        self,
        embedding_dimension: int,
        mlp_hidden_dimensions: list[int] | None = None,
        mlp_activation: Callable | None = nn.SiLU,
    ):
        super().__init__(
            embedding_dimension=embedding_dimension,
            precompute_encodings=False,  # No precompute for variable 2D shapes
            mlp_hidden_dimensions=mlp_hidden_dimensions,
            mlp_activation=mlp_activation,
        )


    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = input_tensor.shape

        encodings = self._compute_encodings(torch.empty(height, width).to(input_tensor.device))  # [embedding_dimension, height, width]
        encodings = encodings.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [batch_size, embedding_dimension, height, width]
        if self.mlp_network:
            # Reshape for MLP: [batch_size, embedding_dimension, height, width] -> [batch_size * height * width, embedding_dimension]
            encodings = encodings.permute(0, 2, 3, 1).reshape(-1, self.embedding_dimension)
            encodings = self.mlp_network(encodings)
            # Reshape back: [batch_size, embedding_dimension, height, width]
            encodings = encodings.reshape(batch_size, height, width, self.embedding_dimension).permute(0, 3, 1, 2)
        return encodings



def add_positional_encoding(source: torch.Tensor, positional_encoding: torch.Tensor | None = None) -> torch.Tensor:
    """Adds positional encoding to the tensor if provided.

    Args:
        source: Input tensor.
        positional_encoding: Positional encoding tensor to add (optional).

    Returns:
        Tensor with positional encoding added if provided, otherwise the original tensor.
    """
    return source if positional_encoding is None else source + positional_encoding
