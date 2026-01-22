from collections.abc import Callable

import torch
import torch.nn as nn

from versatil.models.layers.swiglu import SwiGLU


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        output_dim: int | None = None,
        activation_function: Callable = nn.GELU,
        dropout: float = 0.0,
    ):
        """Multi-layer Perceptron (MLP) module.

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output feature dimension
            activation_function: Activation function class callable
            dropout: Dropout rate between layers
        """
        super().__init__()
        hidden_dims = hidden_dims if hidden_dims is not None else []
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            if issubclass(activation_function, SwiGLU):
                layers.append(
                    activation_function(input_dim=prev_dim, hidden_dim=hidden_dim)
                )
            else:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(activation_function())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        if output_dim is not None:
            layers.append(nn.Linear(prev_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP."""
        result: torch.Tensor = self.layers(x)
        return result
