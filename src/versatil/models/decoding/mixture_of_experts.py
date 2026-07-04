"""Base Mixture of Experts module with shared gating and routing logic."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from versatil.models.decoding.constants import DecoderOutputKey, MoERoutingType
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.mlp import MLP


class BaseMixtureOfExperts(nn.Module):
    """Base class for Mixture of Experts with shared gating and routing logic.

    Handles:
    - Gating network creation and management
    - Temperature-scaled routing weight computation
    - Expert specialization analysis
    - Routing strategies (soft, top-k)
    """

    def __init__(
        self,
        num_experts: int,
        device: str,
        gating_input_dim: int | None = None,
        gating_activation_function: str = ActivationFunction.RELU.value,
        gating_hidden_dims: list[int] | None = None,
        routing_type: str = MoERoutingType.SOFT.value,
        top_k: int = 2,
        temperature: float = 1.0,
        learnable_temperature: bool = False,
        gating_dropout: float = 0.1,
        gating_normalization: bool = True,
    ):
        """Initialize Mixture of Experts routing logic.

        Args:
            num_experts: Number of expert models
            device: Device to place gating network on
            gating_input_dim: Input dimension for gating network (None for external routing)
            gating_activation_function: Activation function for gating network
            gating_hidden_dims: Hidden layer dimensions for gating network
            routing_type: Routing strategy ("soft" or "top_k")
            top_k: Number of experts to use for top-k routing
            temperature: Temperature for softmax scaling of routing weights
            learnable_temperature: Whether temperature should be a learnable parameter
            gating_dropout: Dropout rate in gating network
            gating_normalization: Whether to normalize inputs to gating network
        """
        nn.Module.__init__(self)
        if num_experts == 0:
            raise ValueError("Must provide at least one expert")
        valid_routing_types = [e.value for e in MoERoutingType]
        if routing_type not in valid_routing_types:
            raise ValueError(
                f"Invalid routing_type: {routing_type}. Expected one of {valid_routing_types}"
            )

        if num_experts < 1:
            raise ValueError(f"num_experts must be positive, got {num_experts}.")
        if not 1 <= top_k <= num_experts:
            raise ValueError(
                f"top_k must be in [1, num_experts={num_experts}], got {top_k}."
            )
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}.")
        self.num_experts = num_experts
        self.routing_type = routing_type
        self.top_k = top_k
        self.has_gating_network = gating_input_dim is not None
        self.gating_network: nn.Sequential | None
        if self.has_gating_network:
            self.gating_network = self._build_gating_network(
                input_dimension=gating_input_dim,
                hidden_dimensions=gating_hidden_dims,
                activation=gating_activation_function,
                dropout=gating_dropout,
                normalization=gating_normalization,
                device=device,
            )
        else:
            self.gating_network = None
        if learnable_temperature:
            self.temperature = nn.Parameter(
                torch.tensor(temperature, dtype=torch.float32), requires_grad=True
            )
        else:
            self.register_buffer(
                "temperature", torch.tensor(temperature, dtype=torch.float32)
            )

    def _build_gating_network(
        self,
        input_dimension: int,
        hidden_dimensions: list[int] | None,
        activation: str,
        dropout: float,
        normalization: bool,
        device: str,
    ) -> nn.Sequential:
        """Build gating network for computing routing weights.

        Args:
            input_dimension: Input feature dimension
            hidden_dimensions: List of hidden layer dimensions (defaults to [input_dimension // 2])
            activation: Activation function name for gating network
            dropout: Dropout rate between layers
            normalization: Whether to apply layer normalization before MLP
            device: Device to place the network on

        Returns:
            Sequential module containing normalization (optional) and MLP
        """
        if hidden_dimensions is None or len(hidden_dimensions) == 0:
            hidden_dimensions = [input_dimension // 2]
        layers: list[nn.Module] = []
        if normalization:
            layers.append(nn.LayerNorm(input_dimension))
        mlp = MLP(
            input_dimension=input_dimension,
            hidden_dimensions=hidden_dimensions,
            output_dim=self.num_experts,
            activation_function=ActivationFunction(activation).to_torch_activation(),
            dropout=dropout,
        )
        layers.append(mlp)
        return nn.Sequential(*layers).to(device)

    def compute_routing_weights(self, features: torch.Tensor) -> torch.Tensor:
        """Compute routing weights from input or external source.

        Args:
            features: Input tensor for gating network

        Returns:
            Normalized routing weights (B, [horizon,] num_experts)

        """
        logits = self.gating_network(features) if self.has_gating_network else features
        logits = logits / self.temperature
        return F.softmax(logits, dim=-1)

    def get_expert_specialization(
        self,
        gating_feature: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Analyze expert usage patterns.

        Args:
            gating_feature: Input gating feature

        Returns:
            Dictionary with expert_usage, routing_entropy, top_expert_confidence

        Note:
            This method uses torch.no_grad() for efficiency but does not modify
            the model's training state. Caller should set model to eval mode if needed.
        """
        with torch.no_grad():
            weights = self.compute_routing_weights(gating_feature)
            expert_usage = weights.mean(dim=tuple(range(weights.ndim - 1)))
            entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()
            top_expert_confidence = weights.max(dim=-1)[0].mean()
        return {
            DecoderOutputKey.EXPERT_USAGE.value: expert_usage,
            DecoderOutputKey.ROUTING_ENTROPY.value: entropy,
            DecoderOutputKey.TOP_EXPERT_CONFIDENCE.value: top_expert_confidence,
        }

    def _apply_routing(
        self,
        expert_outputs: list[torch.Tensor],
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Apply routing strategy to combine expert outputs.

        Stacks expert outputs along dimension 1 and applies the configured routing
        strategy (soft or top-k) to produce a weighted combination.

        Args:
            expert_outputs: List of expert output tensors, each with shape
                (B, ...) where ... can be any additional dimensions
            weights: Routing weights with shape (B, num_experts) or (B, horizon, num_experts)

        Returns:
            Combined output tensor with expert dimension removed
        """
        stacked = torch.stack(expert_outputs, dim=1)
        if self.routing_type == MoERoutingType.SOFT.value:
            return self._combine_soft(stacked, weights)
        elif self.routing_type == MoERoutingType.TOP_K.value:
            return self._combine_topk(stacked, weights)
        else:
            raise ValueError(f"Unknown routing type: {self.routing_type}")

    def _combine_soft(
        self, stacked_predictions: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Soft routing: weighted combination of all experts.

        Computes a weighted sum of all expert outputs using the routing weights.
        All experts contribute to the final output proportional to their weights.

        Args:
            stacked_predictions: Stacked expert outputs with shape (B, num_experts, ...)
            weights: Routing weights with shape (B, num_experts) or (B, horizon, num_experts)

        Returns:
            Weighted combination of expert outputs with expert dimension summed out
        """
        if weights.ndim == 3:
            weights = weights.transpose(1, 2)

        expanded_weights = weights
        for _ in range(stacked_predictions.ndim - weights.ndim):
            expanded_weights = expanded_weights.unsqueeze(-1)
        return (stacked_predictions * expanded_weights).sum(dim=1)

    def _combine_topk(
        self, stacked_predictions: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Top-k routing: only use top-k experts.

        Selects the top-k experts with highest routing weights and combines only
        their outputs. The top-k weights are renormalized to sum to 1.0 before
        combining. This provides sparse routing and can improve efficiency.

        Args:
            stacked_predictions: Stacked expert outputs with shape (B, num_experts, ...)
            weights: Routing weights with shape (B, num_experts) or (B, horizon, num_experts)

        Returns:
            Weighted combination of top-k expert outputs with expert dimension summed out
        """
        if weights.ndim == 3:
            weights = weights.transpose(1, 2)  # (B, E, H)

        top_k_weights, top_k_indices = torch.topk(weights, self.top_k, dim=1)  # (B, k)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=1, keepdim=True) + 1e-8)
        indices_expanded = top_k_indices
        for _ in range(stacked_predictions.ndim - top_k_indices.ndim):
            indices_expanded = indices_expanded.unsqueeze(-1)

        expand_shape = list(
            stacked_predictions.shape
        )  # [B, num_experts, pred_horizon, action_dim]
        expand_shape[1] = self.top_k
        indices_expanded = indices_expanded.expand(
            expand_shape
        )  # (B, k, pred_horizon, action_dim)
        top_k_outputs = torch.gather(
            stacked_predictions, dim=1, index=indices_expanded
        )  # (B, k, pred_horizon, action_dim)
        expanded_weights = top_k_weights
        for _ in range(top_k_outputs.ndim - top_k_weights.ndim):
            expanded_weights = expanded_weights.unsqueeze(-1)  # (B, k, 1, 1)
        return (top_k_outputs * expanded_weights).sum(dim=1)
