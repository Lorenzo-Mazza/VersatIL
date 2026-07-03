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
from versatil.models.layers.gated_linear_unit import GatedLinearUnit


class ConditionalModulation(nn.Module):
    """Conditional modulation layer.

    Supports FiLM (for CNNs), adaLN (for transformers), and variants.
    """

    def __init__(
        self,
        conditioning_dimension: int,
        feature_dim: int,
        use_shift: bool = True,
        use_gate: bool = False,
        activation: str = ActivationFunction.SILU.value,
        init_strategy: Literal["zero", "xavier"] = "zero",
        feature_axis: int = -1,
    ):
        """Initialize conditional modulation.

        Args:
            conditioning_dimension: Dimension of conditioning vector.
            feature_dim: Dimension of features to modulate.
            use_shift: Whether to include shift (beta) or just scale (gamma).
            use_gate: Whether to include gate output.
            activation: Activation function to apply to condition before modulation.
            init_strategy: Weight initialization strategy.
            feature_axis: Feature axis for 3D tensors. Use ``-1`` for
                transformer layout ``(B, S, D)``, and ``1`` for Conv1D
                layout ``(B, C, T)``.

        Raises:
            ValueError: If ``feature_axis`` is not supported.
        """
        super().__init__()
        if feature_axis not in {-1, 1}:
            raise ValueError(
                f"feature_axis must be one of [-1, 1], got {feature_axis}."
            )
        self.use_shift = use_shift
        self.use_gate = use_gate
        self.init_strategy = init_strategy
        self.feature_dim = feature_dim
        self.feature_axis = feature_axis
        self.output_dim = feature_dim
        if use_shift:
            self.output_dim += feature_dim
        if use_gate:
            self.output_dim += feature_dim
        activation_enum = ActivationFunction(activation)
        if activation_enum.is_gated:
            self.projection = activation_enum.to_torch_activation()(
                input_dimension=conditioning_dimension, hidden_dimension=self.output_dim
            )
        else:
            self.projection = nn.Sequential(
                activation_enum.to_torch_activation()(),
                nn.Linear(conditioning_dimension, self.output_dim),
            )
        self.init_parameters()

    def init_parameters(self) -> None:
        """Initialize projection weights from the configured strategy.

        Raises:
            ValueError: If ``init_strategy`` is not supported.
        """
        linear_layers = [
            m for m in self.projection.modules() if isinstance(m, nn.Linear)
        ]
        for layer in linear_layers:
            layer._is_modulation_layer = True
        if self.init_strategy == "zero":
            if isinstance(self.projection, GatedLinearUnit):
                # Zeroing both GLU branches makes the product's gradient
                # identically zero, freezing the modulation forever. Zeroing
                # only the value branch keeps the initial output at zero while
                # gradients still flow through the gate.
                zero_layers = [self.projection.value_proj]
            else:
                zero_layers = linear_layers
            for layer in zero_layers:
                nn.init.constant_(layer.weight, 0)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        elif self.init_strategy == "xavier":
            for layer in linear_layers:
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        else:
            raise ValueError(f"Unknown init_strategy: {self.init_strategy}")

    def forward(
        self, x: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply conditional modulation.

        Args:
            x: Features to modulate.
                - CNN: (B, C, H, W)
                - Transformer: (B, S, D)
                - Conv1D: (B, C, T) when ``feature_axis=1``
            condition: Conditioning vector (B, conditioning_dimension).

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
            if x.size(0) != condition.size(0):
                raise ValueError(
                    f"Cannot match batch dimension: x.shape={x.shape}, "
                    f"condition.shape={condition.shape}. Expected x.size(0) "
                    f"to equal condition.size(0)={condition.size(0)}."
                )
            if self.feature_axis == 1:
                if x.size(1) != self.feature_dim:
                    raise ValueError(
                        f"Expected x.size(1) to equal feature_dim={self.feature_dim}, "
                        f"got x.shape={x.shape}."
                    )
                gamma = gamma.unsqueeze(2)  # (B, C) -> (B, C, 1)
                if beta is not None:
                    beta = beta.unsqueeze(2)
                if self.use_gate:
                    gate = gate.unsqueeze(2)
            else:
                if x.size(2) != self.feature_dim:
                    raise ValueError(
                        f"Expected x.size(2) to equal feature_dim={self.feature_dim}, "
                        f"got x.shape={x.shape}."
                    )
                gamma = gamma.unsqueeze(1)  # (B, D) -> (B, 1, D)
                if beta is not None:
                    beta = beta.unsqueeze(1)
                if self.use_gate:
                    gate = gate.unsqueeze(1)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        result = x * (1 + gamma)
        if beta is not None:
            result = result + beta
        return result, gate
