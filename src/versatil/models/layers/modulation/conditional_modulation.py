"""Conditional feature modulation via learned affine transform.

Computes y = x * (1 + gamma) + beta, where gamma (scale) and beta (shift)
are projected from a conditioning vector. Optionally produces a gate for
residual connections (AdaLN-Zero).

References:
    FiLM: https://arxiv.org/pdf/2212.09748
"""

from typing import Literal

import torch
import torch.nn as nn

from versatil.models.layers.activation import ActivationFunction


class ConditionalModulation(nn.Module):
    """Conditional modulation layer.

    Supports FiLM (for CNNs), adaLN (for transformers), and variants.
    """

    def __init__(
        self,
        condition_dim: int,
        feature_dim: int,
        use_shift: bool = True,
        use_gate: bool = False,
        activation: str = ActivationFunction.SILU.value,
        init_strategy: Literal["zero", "xavier"] = "zero",
    ):
        """
        Args:
            condition_dim: Dimension of conditioning vector
            feature_dim: Dimension of features to modulate
            use_shift: Whether to include shift (beta) or just scale (gamma)
            use_gate: Whether to include gate output
            activation: Activation function to apply to condition before modulation.
            init_strategy: Weight initialization strategy
        """
        super().__init__()
        self.use_shift = use_shift
        self.use_gate = use_gate
        self.init_strategy = init_strategy
        self.feature_dim = feature_dim
        self.output_dim = feature_dim
        if use_shift:
            self.output_dim += feature_dim
        if use_gate:
            self.output_dim += feature_dim
        activation_enum = ActivationFunction(activation)
        if activation_enum.is_gated:
            self.projection = activation_enum.to_torch_activation()(
                input_dim=condition_dim, hidden_dim=self.output_dim
            )
        else:
            self.projection = nn.Sequential(
                activation_enum.to_torch_activation()(),
                nn.Linear(condition_dim, self.output_dim),
            )
        self.init_parameters()

    def init_parameters(self):
        """Initialize weights based on strategy."""
        linear_layers = [
            m for m in self.projection.modules() if isinstance(m, nn.Linear)
        ]
        if self.init_strategy == "zero":
            for layer in linear_layers:
                layer._is_modulation_layer = True
                nn.init.constant_(layer.weight, 0)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        elif self.init_strategy == "xavier":
            for layer in linear_layers:
                layer._is_modulation_layer = True
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        else:
            raise ValueError(f"Unknown init_strategy: {self.init_strategy}")

    def forward(
        self, x: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Features to modulate
                - CNN: (B, C, H, W)
                - Transformer: (S, B, D) or (B, S, D)
            condition: Conditioning vector (B, condition_dim)

        Returns:
            Tuple of (modulated features, gate). Gate is a learned tensor
            when use_gate=True, or ones(1) when use_gate=False.
        """
        projected_condition = self.projection(condition)
        chunks = projected_condition.split(self.feature_dim, dim=-1)
        gamma = chunks[0]
        current_chunk_index = 1
        beta = None
        if self.use_shift:
            beta = chunks[current_chunk_index]
            current_chunk_index += 1
        gate = torch.ones(1, dtype=x.dtype, device=x.device)
        if self.use_gate:
            gate = chunks[current_chunk_index]

        if x.dim() == 4:
            gamma = gamma.view(x.size(0), x.size(1), 1, 1)
            if beta is not None:
                beta = beta.view(x.size(0), x.size(1), 1, 1)
            if self.use_gate:
                gate = gate.view(x.size(0), x.size(1), 1, 1)
        elif x.dim() == 3:
            if x.size(0) == condition.size(0):  # Batch size in dim 0
                if (
                    x.size(1) == self.feature_dim
                ):  # Conv1D format: (B, C, T) - channels in dim 1
                    gamma = gamma.unsqueeze(2)  # (B, C) -> (B, C, 1)
                    if beta is not None:
                        beta = beta.unsqueeze(2)
                    if self.use_gate:
                        gate = gate.unsqueeze(2)
                else:  # Transformer format: (B, S, D) - features in dim 2
                    gamma = gamma.unsqueeze(1)  # (B, D) -> (B, 1, D)
                    if beta is not None:
                        beta = beta.unsqueeze(1)
                    if self.use_gate:
                        gate = gate.unsqueeze(1)
            elif x.size(1) == condition.size(0):
                gamma = gamma.unsqueeze(0)
                if beta is not None:
                    beta = beta.unsqueeze(0)
                if self.use_gate:
                    gate = gate.unsqueeze(0)
            else:
                raise ValueError(
                    f"Cannot match batch dimension: x.shape={x.shape}, condition.shape={condition.shape}. "
                    f"Expected x.size(0) or x.size(1) to equal condition.size(0)={condition.size(0)}"
                )
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        result = x * (1 + gamma)
        if beta is not None:
            result = result + beta
        return result, gate
