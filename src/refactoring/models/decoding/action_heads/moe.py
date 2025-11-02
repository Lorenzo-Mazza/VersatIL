"""Mixture of Experts (MoE) action head for phase-conditioned or multi-modal action prediction."""

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig

from refactoring.data.constants import ACTION_KEY
from refactoring.models.decoding.action_heads.head import ActionHead
from refactoring.models.decoding.constants import (
    EXPERT_OUTPUTS,
    ROUTING_WEIGHT,
    MoERoutingType,
)
from refactoring.models.decoding.mixture_of_experts import BaseMixtureOfExperts
from refactoring.models.layers.activation import ActivationFunction


class MoEHead(BaseMixtureOfExperts):
    """Mixture of Experts head for action prediction.

    Supports two modes:
    1. Explicit expert list: Pass pre-instantiated experts
    2. Config-based: Pass base_expert_config and num_experts (recommended)

    Example:
        moe = MoEHead(
            base_expert_config=ActionHeadConfig(...),
            num_experts=5,
            output_dim=3,
            gating_input_dim=256
        )
    """

    def __init__(
        self,
        output_dim: int,
        device: str = "cpu",
        experts: list[ActionHead] | None = None,
        base_expert_config: DictConfig | dict | None = None,
        num_experts: int | None = None,
        expert_configs: list[DictConfig | dict] | None = None,
        gating_input_dim: int | None = None,
        gating_activation: str = ActivationFunction.SILU.value,
        gating_hidden_dims: list[int] | None = None,
        routing_type: str = MoERoutingType.SOFT.value,
        top_k: int = 2,
        temperature: float = 1.0,
        learnable_temperature: bool = False,
        gating_dropout: float = 0.1,
        gating_normalization: bool = True,
        gating_feature_key: str | None = None,
    ):
        """Initialize Mixture of Experts action head.

        Args:
            output_dim: Output action dimension (must match all experts)
            device: Device to place the module on
            experts: Optional pre-instantiated expert action heads
            base_expert_config: Base config for creating experts (with num_experts)
            num_experts: Number of experts to create from base_expert_config
            expert_configs: Optional per-expert config overrides
            gating_input_dim: Input dimension for gating network (None for external routing)
            gating_activation: Activation function for gating network
            gating_hidden_dims: Hidden layer dimensions for gating network
            routing_type: Routing strategy ("soft" or "top_k")
            top_k: Number of experts to use for top-k routing
            temperature: Temperature for softmax scaling of routing weights
            learnable_temperature: Whether temperature should be a learnable parameter
            gating_dropout: Dropout rate in gating network
            gating_normalization: Whether to normalize inputs to gating network
            gating_feature_key: Optional feature key for gating network input
        """
        if experts is not None:
            expert_list = experts
            num_experts = len(experts)
        elif base_expert_config is not None and num_experts is not None:
            expert_list = self._create_experts_from_config(
                base_expert_config, num_experts, expert_configs
            )
        else:
            raise ValueError("Must provide either 'experts' or both 'base_expert_config' and 'num_experts'")

        super().__init__(
            num_experts=num_experts,
            device=device,
            gating_input_dim=gating_input_dim,
            gating_activation_function=gating_activation,
            gating_hidden_dims=gating_hidden_dims,
            routing_type=routing_type,
            top_k=top_k,
            temperature=temperature,
            learnable_temperature=learnable_temperature,
            gating_dropout=gating_dropout,
            gating_normalization=gating_normalization,
            gating_feature_key=gating_feature_key,
        )

        self.output_dim = output_dim
        for i, expert in enumerate(expert_list):
            if expert.output_dim != output_dim:
                raise ValueError(f"Expert {i} output_dim={expert.output_dim} does not match expected {output_dim}")
        self.experts = nn.ModuleList(expert_list)

    @staticmethod
    def _create_experts_from_config(
        base_config: DictConfig | dict,
        num_experts: int,
        overrides: list[DictConfig | dict] | None = None,
    ) -> list[ActionHead]:
        """Create expert action heads from configuration.

        Args:
            base_config: Base configuration to use for all experts
            num_experts: Number of expert heads to create
            overrides: Optional list of configs to override base config for specific experts

        Returns:
            List of instantiated ActionHead experts
        """
        experts = []
        for i in range(num_experts):
            config = overrides[i] if overrides and i < len(overrides) else base_config
            expert = instantiate(config)
            experts.append(expert)
        return experts

    def forward(
        self,
        features: torch.Tensor,
        routing_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through mixture of experts.

        Args:
            features: Input features for action prediction
            routing_weights: Optional external routing weights (if not using gating network)

        Returns:
            Dictionary containing:
                - action: Combined action predictions from experts
                - routing_weights: Computed routing weights
                - expert_outputs: Individual expert predictions (stacked)
        """
        weights = self.compute_routing_weights(features, routing_weights)

        expert_outputs = [expert(features) for expert in self.experts]
        expert_outputs_stacked = torch.stack(expert_outputs, dim=-2)

        final_output = self._apply_routing(expert_outputs, weights)

        return {
            ACTION_KEY: final_output,
            ROUTING_WEIGHT: weights,
           EXPERT_OUTPUTS: expert_outputs_stacked,
        }
