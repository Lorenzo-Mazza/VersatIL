"""Base Mixture of Experts module with shared gating and routing logic."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from refactoring.models.decoding.constants import (
    EXPERT_USAGE,
    ROUTING_ENTROPY,
    TOP_EXPERT_CONFIDENCE,
    MoERoutingType,
)
from refactoring.models.layers.activation import ActivationFunction
from refactoring.models.layers.mlp import MLP


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
        gating_feature_key: str | None = None,
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
            gating_feature_key: Key to use for extracting gating features from feature dict.
                If None, uses first available feature. For Variational policies, typically 'latent'.
        """
        super().__init__()

        if num_experts == 0:
            raise ValueError("Must provide at least one expert")

        self.num_experts = num_experts
        self.routing_type = routing_type
        self.top_k = min(top_k, num_experts)
        self.gating_feature_key = gating_feature_key

        valid_routing_types = [e.value for e in MoERoutingType]
        if routing_type not in valid_routing_types:
            raise ValueError(
                f"Invalid routing_type: {routing_type}. Expected one of {valid_routing_types}"
            )

        self.has_gating_network = gating_input_dim is not None
        self.gating_network: nn.Sequential | None
        if self.has_gating_network:
            if gating_input_dim is None:
                raise ValueError("gating_input_dim must be set when has_gating_network is True")
            self.gating_network = self._build_gating_network(
                input_dim=gating_input_dim,
                hidden_dims=gating_hidden_dims,
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
        input_dim: int,
        hidden_dims: list[int] | None,
        activation: str,
        dropout: float,
        normalization: bool,
        device: str,
    ) -> nn.Sequential:
        """Build gating network for computing routing weights.

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions (defaults to [input_dim // 2])
            activation: Activation function name for gating network
            dropout: Dropout rate between layers
            normalization: Whether to apply layer normalization before MLP
            device: Device to place the network on

        Returns:
            Sequential module containing normalization (optional) and MLP
        """
        if hidden_dims is None or len(hidden_dims) == 0:
            hidden_dims = [input_dim // 2]

        layers: list[nn.Module] = []
        if normalization:
            layers.append(nn.LayerNorm(input_dim))

        mlp = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=self.num_experts,
            activation_function=ActivationFunction(activation).to_torch_activation(),
            dropout=dropout,
        )
        layers.append(mlp)

        return nn.Sequential(*layers).to(device)

    def compute_routing_weights(
        self, features: torch.Tensor | dict[str, torch.Tensor], external_weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute routing weights from input or external source.

        Args:
            features: Input tensor or dict of tensors
            external_weights: Optional external routing logits/probabilities

        Returns:
            Normalized routing weights (B, [horizon,] num_experts)

        Raises:
            ValueError: If gating_feature_key is specified but not found in features dict
        """
        if external_weights is not None:
            logits = external_weights
        elif self.has_gating_network:
            if isinstance(features, dict):
                if self.gating_feature_key is not None:
                    if self.gating_feature_key not in features:
                        raise ValueError(
                            f"Gating feature key '{self.gating_feature_key}' not found in features. "
                            f"Available keys: {list(features.keys())}"
                        )
                    features = features[self.gating_feature_key]
                else:
                    # Default behavior: use first available feature
                    features = next(iter(features.values()))
            if self.gating_network is None:
                raise RuntimeError("gating_network must be initialized when has_gating_network is True")
            logits = self.gating_network(features)
        else:
            raise ValueError(
                "Either gating_input_dim must be provided at initialization "
                "or external routing_weights must be passed to forward()"
            )

        logits = logits / self.temperature
        return F.softmax(logits, dim=-1)

    def get_expert_specialization(
        self, features: torch.Tensor, external_routing: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Analyze expert usage patterns.

        Args:
            features: Input features
            external_routing: Optional external routing weights

        Returns:
            Dictionary with expert_usage, routing_entropy, top_expert_confidence

        Note:
            This method uses torch.no_grad() for efficiency but does not modify
            the model's training state. Caller should set model to eval mode if needed.
        """
        with torch.no_grad():
            weights = self.compute_routing_weights(features, external_routing)
            expert_usage = weights.mean(dim=tuple(range(weights.ndim - 1)))
            entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()
            top_expert_confidence = weights.max(dim=-1)[0].mean()

        return {
            EXPERT_USAGE: expert_usage,
            ROUTING_ENTROPY: entropy,
            TOP_EXPERT_CONFIDENCE: top_expert_confidence,
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
            weights = weights.transpose(1, 2)

        top_k_weights, top_k_indices = torch.topk(weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-8)

        indices_expanded = top_k_indices
        for _ in range(stacked_predictions.ndim - top_k_indices.ndim):
            indices_expanded = indices_expanded.unsqueeze(-1)

        expand_shape = list(stacked_predictions.shape)
        expand_shape[1] = self.top_k
        indices_expanded = indices_expanded.expand(expand_shape)

        top_k_outputs = torch.gather(stacked_predictions, dim=1, index=indices_expanded)

        expanded_weights = top_k_weights
        for _ in range(top_k_outputs.ndim - top_k_weights.ndim):
            expanded_weights = expanded_weights.unsqueeze(-1)

        return (top_k_outputs * expanded_weights).sum(dim=1)
