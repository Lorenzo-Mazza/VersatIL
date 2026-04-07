"""Swish-Gated Linear Unit activation.

From Shazeer (2020): https://arxiv.org/abs/2002.05202

SwiGLU is a gated activation function that has shown strong performance
in large language models and transformers.
"""

import torch
import torch.nn as nn


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit with configurable activation.

    GLU(x) = act(xW_gate) ⊗ (xW_value)

    SwiGLU uses SiLU, GeGLU uses GELU. The gating doubles the parameter count
    compared to a simple linear + activation.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden dimension (output will be this size).
        bias: Whether to use bias in linear layers.
        gate_activation: Activation applied to the gate projection.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        bias: bool = False,
        gate_activation: nn.Module | None = None,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.value_proj = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.gate_activation = gate_activation or nn.SiLU()
        self._init_weights()

    def _init_weights(self):
        """Variance-preserving initialization."""
        nn.init.kaiming_uniform_(
            self.gate_proj.weight, mode="fan_in", nonlinearity="relu"
        )
        nn.init.kaiming_uniform_(
            self.value_proj.weight, mode="fan_in", nonlinearity="linear"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_activation(self.gate_proj(x))
        value = self.value_proj(x)
        return gate * value


class SwiGLU(GatedLinearUnit):
    """Swish-Gated Linear Unit: GLU with SiLU (Swish) activation."""

    def __init__(self, input_dim: int, hidden_dim: int, bias: bool = False):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            bias=bias,
            gate_activation=nn.SiLU(),
        )


class GeGLU(GatedLinearUnit):
    """GELU-Gated Linear Unit: GLU with GELU(approximate='tanh') activation."""

    def __init__(self, input_dim: int, hidden_dim: int, bias: bool = False):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            bias=bias,
            gate_activation=nn.GELU(approximate="tanh"),
        )
