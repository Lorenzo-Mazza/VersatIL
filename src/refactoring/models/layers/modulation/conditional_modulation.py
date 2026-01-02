"""Inspired from FiLM,  https://arxiv.org/pdf/2212.09748 and https://github.com/sudeepdasari/dit-policy"""
from typing import Literal

import torch
import torch.nn as nn

from refactoring.models.layers.activation import ActivationFunction


class ConditionalModulation(nn.Module):
    """Conditional modulation layer.

    Supports FiLM (for CNNs), adaLN (for transformers), and variants.
    """

    def __init__(
        self,
        condition_dim: int,
        feature_dim: int,
        use_shift: bool = True,
        activation: str = ActivationFunction.SILU.value,
        init_strategy: Literal["identity", "xavier", "zero"] = "identity",
    ):
        """
        Args:
            condition_dim: Dimension of conditioning vector
            feature_dim: Dimension of features to modulate
            use_shift: Whether to include shift (beta) or just scale (gamma)
            activation: Activation function to apply to condition before modulation.
            init_strategy: Weight initialization strategy
        """
        super().__init__()

        self.use_shift = use_shift
        self.init_strategy = init_strategy
        if activation == ActivationFunction.SWIGLU.value:
            self.scale_linear = ActivationFunction(activation).to_torch_activation()(
                input_dim=condition_dim, hidden_dim=feature_dim
            )
        else:
            self.scale_linear = nn.Sequential(
                ActivationFunction(activation).to_torch_activation()(),
                nn.Linear(condition_dim, feature_dim),
            )
        if use_shift:
            self.shift_linear = nn.Linear(condition_dim, feature_dim)
        self.init_parameters()

    def init_parameters(self):
        """Initialize weights based on strategy."""
        scale_linears = [
            m for m in self.scale_linear.modules() if isinstance(m, nn.Linear)
        ]
        if self.init_strategy == "identity":
            for layer in scale_linears:
                nn.init.constant_(layer.weight, 0)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 1)
            if self.use_shift:
                nn.init.constant_(self.shift_linear.weight, 0)
                nn.init.constant_(self.shift_linear.bias, 0)
        elif self.init_strategy == "xavier":
            for layer in scale_linears:
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            if self.use_shift:
                nn.init.xavier_uniform_(self.shift_linear.weight)
                nn.init.zeros_(self.shift_linear.bias)
        elif self.init_strategy == "zero":
            for layer in scale_linears:
                nn.init.zeros_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            if self.use_shift:
                nn.init.zeros_(self.shift_linear.weight)
                nn.init.zeros_(self.shift_linear.bias)

        else:
            raise ValueError(f"Unknown init_strategy: {self.init_strategy}")

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features to modulate
               - CNN: (B, C, H, W)
               - Transformer: (S, B, D) or (B, S, D)
            condition: Conditioning vector (B, condition_dim)

        Returns:
            Modulated features (same shape as x)
        """
        gamma = self.scale_linear(condition)
        beta = None

        if x.dim() == 4:
            gamma = gamma.view(x.size(0), x.size(1), 1, 1)
            if self.use_shift:
                beta = self.shift_linear(condition)
                beta = beta.view(x.size(0), x.size(1), 1, 1)

        elif x.dim() == 3:

            if x.size(0) == condition.size(0):
                # x is (B, C, T) for Conv1D or (B, S, D) for Transformer
                gamma = gamma.unsqueeze(-1)  # (B, feature_dim) -> (B, feature_dim, 1)
                if self.use_shift:
                    beta = self.shift_linear(condition).unsqueeze(-1)
            elif x.size(1) == condition.size(0):
                # Tensor with sequence-first: x is (S, B, D), condition is (B, condition_dim)
                # gamma is (B, D) -> need (1, B, D) to broadcast over sequence
                gamma = gamma[None]  # (B, D) -> (1, B, D)
                if self.use_shift:
                    beta = self.shift_linear(condition)[None]
            else:
                raise ValueError(
                    f"Cannot match batch dimension: x.shape={x.shape}, condition.shape={condition.shape}. "
                    f"Expected x.size(0) or x.size(1) to equal condition.size(0)={condition.size(0)}"
                )

        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

        if self.use_shift:
            result: torch.Tensor = gamma * x + beta
            return result
        else:
            result_no_shift: torch.Tensor = gamma * x
            return result_no_shift
