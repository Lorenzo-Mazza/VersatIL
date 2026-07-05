from collections.abc import Callable

import torch
import torch.nn as nn

from versatil.models.layers.gated_linear_unit import GatedLinearUnit


class MLP(nn.Module):
    """Multi-layer perceptron with configurable hidden sizes, activation, and dropout."""

    def __init__(
        self,
        input_dimension: int,
        hidden_dimensions: list[int] | None = None,
        output_dim: int | None = None,
        activation_function: Callable = nn.GELU,
        dropout: float = 0.0,
    ):
        """Multi-layer Perceptron (MLP) module.

        Args:
            input_dimension: Input feature dimension
            hidden_dimensions: List of hidden layer dimensions
            output_dim: Output feature dimension
            activation_function: Activation function class callable
            dropout: Dropout rate between layers
        """
        super().__init__()
        hidden_dimensions = hidden_dimensions if hidden_dimensions is not None else []
        layers: list[nn.Module] = []
        prev_dim = input_dimension
        for hidden_dimension in hidden_dimensions:
            if issubclass(activation_function, GatedLinearUnit):
                layers.append(
                    activation_function(
                        input_dimension=prev_dim, hidden_dimension=hidden_dimension
                    )
                )
            else:
                layers.append(nn.Linear(prev_dim, hidden_dimension))
                layers.append(activation_function())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dimension
        if output_dim is not None:
            layers.append(nn.Linear(prev_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP."""
        result: torch.Tensor = self.layers(x)
        return result
