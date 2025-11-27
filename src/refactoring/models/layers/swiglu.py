"""Swish-Gated Linear Unit activation.

From Shazeer (2020): https://arxiv.org/abs/2002.05202

SwiGLU is a gated activation function that has shown strong performance
in large language models and transformers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit activation.

    SwiGLU(x) = Swish(xW) ⊗ (xV) where Swish(x) = x * sigmoid(x)

    This is a gated activation that has shown strong performance in transformers.
    Note: This effectively doubles the parameter count of the feedforward layer
    compared to a simple linear + activation.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension (output will be this size)
        bias: Whether to use bias in linear layers
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        bias: bool = False,
    ):
        """Initialize SwiGLU module.

        Args:
            input_dim: Input dimension
            hidden_dim: Output hidden dimension
            bias: Whether to include bias terms in linear layers
        """
        super().__init__()
        self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.value_proj = nn.Linear(input_dim, hidden_dim, bias=bias)
        self._init_weights()


    def _init_weights(self):
        """Variance-preserving initialization for SwiGLU."""
        nn.init.kaiming_uniform_(self.gate_proj.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.value_proj.weight, mode='fan_in', nonlinearity='linear')


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU activation.

        Args:
            x: Input tensor (..., input_dim)

        Returns:
            Activated tensor (..., hidden_dim)
        """
        gate = F.silu(self.gate_proj(x))  # Swish = SiLU in PyTorch
        value = self.value_proj(x)
        return gate * value