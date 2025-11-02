"""Mixture of Experts (MoE) decoder for task/phase-conditioned action prediction."""

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig

from refactoring.configs.task.task import ActionSpace, ObservationSpace
from refactoring.models.decoding.constants import (
    EXPERT_OUTPUTS,
    ROUTING_WEIGHT,
    MoERoutingType,
)
from refactoring.models.decoding.constants import (
    MoERoutingType as MoERoutingTypeEnum,
)
from refactoring.models.decoding.decoders.base import ActionDecoder, DecoderInput
from refactoring.models.decoding.mixture_of_experts import BaseMixtureOfExperts
from refactoring.models.layers.activation import ActivationFunction
from refactoring.models.layers.mlp import MLP


class MoEDecoder(BaseMixtureOfExperts, ActionDecoder):
    """Mixture of Experts decoder for task/phase-conditioned action prediction.

    Supports two modes:
    1. Explicit expert list: Pass pre-instantiated expert decoders
    2. Config-based: Pass base_expert_config and num_experts (recommended)

    Example:
        moe = MoEDecoder(
            base_expert_config=ACTConfig(...),
            num_experts=5,
            decoder_input=decoder_input,
            observation_space=obs_space,
            action_space=action_space,
            action_heads=action_heads,
            device="cuda"
        )
    """

    def __init__(
        self,
        decoder_input: DecoderInput,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        action_heads: dict[str, nn.Module],
        device: str,
        observation_horizon: int = 1,
        prediction_horizon: int = 1,
        expert_decoders: list[ActionDecoder] | None = None,
        base_expert_config: DictConfig | dict | None = None,
        num_experts: int | None = None,
        expert_configs: list[DictConfig | dict] | None = None,
        gating_input_dim: int | None = None,
        gating_hidden_dims: list[int] | None = None,
        routing_type: str = MoERoutingType.SOFT.value,
        top_k: int = 2,
        temperature: float = 1.0,
        learnable_temperature: bool = False,
        gating_dropout: float = 0.1,
        gating_normalization: bool = True,
        gating_feature_key: str | None = None,
    ):
        if expert_decoders is not None:
            expert_list = expert_decoders
            num_experts = len(expert_decoders)
        elif base_expert_config is not None and num_experts is not None:
            expert_list = self._create_experts_from_config(
                base_expert_config,
                num_experts,
                expert_configs,
                decoder_input,
                observation_space,
                action_space,
                action_heads,
                device,
                observation_horizon,
                prediction_horizon,
            )
        else:
            raise ValueError(
                "Must provide either 'expert_decoders' or both 'base_expert_config' and 'num_experts'"
            )

        if num_experts == 0:
            raise ValueError("Must provide at least one expert")

        nn.Module.__init__(self)

        self.decoder_input = decoder_input
        self.action_heads = nn.ModuleDict(action_heads)
        self.observation_space = observation_space
        self.action_space = action_space
        self.observation_horizon = observation_horizon
        self.prediction_horizon = prediction_horizon
        self.device = torch.device(device)
        self.validate_action_heads()

        self.num_experts = num_experts
        self.routing_type = routing_type
        self.top_k = min(top_k, num_experts)
        self.gating_feature_key = gating_feature_key

        valid_routing_types = [e.value for e in MoERoutingTypeEnum]
        if routing_type not in valid_routing_types:
            raise ValueError(
                f"Invalid routing_type: {routing_type}. Expected one of {valid_routing_types}"
            )

        self.has_gating_network = gating_input_dim is not None
        if self.has_gating_network:
            if gating_input_dim is None:
                raise ValueError("gating_input_dim must be set when has_gating_network is True")
            layers: list[nn.Module] = []
            if gating_normalization:
                layers.append(nn.LayerNorm(gating_input_dim))
            if gating_hidden_dims is None:
                gating_hidden_dims = [gating_input_dim // 2]
            mlp = MLP(
                input_dim=gating_input_dim,
                hidden_dims=gating_hidden_dims,
                output_dim=num_experts,
                activation_function=ActivationFunction.RELU.to_torch_activation(),
                dropout=gating_dropout,
            )
            layers.append(mlp)
            self.gating_network = nn.Sequential(*layers).to(device)
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

        for i, expert in enumerate(expert_list):
            if expert.action_dim != self.action_dim:
                raise ValueError(
                    f"Expert {i} action_dim={expert.action_dim} != expected {self.action_dim}"
                )
            if expert.prediction_horizon != self.prediction_horizon:
                raise ValueError(
                    f"Expert {i} prediction_horizon={expert.prediction_horizon} != expected {self.prediction_horizon}"
                )

        self.expert_decoders = nn.ModuleList(expert_list)

    @staticmethod
    def _create_experts_from_config(
        base_config: DictConfig | dict,
        num_experts: int,
        overrides: list[DictConfig | dict] | None,
        decoder_input: DecoderInput,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        action_heads: dict[str, nn.Module],
        device: str,
        observation_horizon: int,
        prediction_horizon: int,
    ) -> list[ActionDecoder]:
        """Create expert decoders from configuration.

        Args:
            base_config: Base decoder configuration to use for all experts
            num_experts: Number of expert decoders to create
            overrides: Optional list of configs to override base config for specific experts
            decoder_input: Decoder input specification
            observation_space: Observation space configuration
            action_space: Action space configuration
            action_heads: Dictionary of action head modules
            device: Device to place decoders on
            observation_horizon: Number of observation timesteps
            prediction_horizon: Number of action timesteps to predict

        Returns:
            List of instantiated ActionDecoder experts
        """
        experts = []
        for i in range(num_experts):
            config = overrides[i] if overrides and i < len(overrides) else base_config

            expert = instantiate(
                config,
                decoder_input=decoder_input,
                observation_space=observation_space,
                action_space=action_space,
                action_heads=action_heads,
                device=device,
                observation_horizon=observation_horizon,
                prediction_horizon=prediction_horizon,
            )
            experts.append(expert)
        return experts

    def forward(  # type: ignore[override]
        self,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
        routing_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        """Forward pass through mixture of expert decoders.

        Args:
            features: Dictionary of input features
            actions: Optional ground-truth actions (for training)
            routing_weights: Optional external routing weights (if not using gating network)

        Returns:
            Dictionary containing:
                - Combined predictions from routed experts (action keys)
                - routing_weights: Computed routing weights
                - expert_outputs: Individual expert prediction dictionaries
        """
        weights = self.compute_routing_weights(features, routing_weights)

        expert_outputs = [expert(features, actions) for expert in self.expert_decoders]

        combined_outputs = self._combine_expert_outputs(expert_outputs, weights)
        combined_outputs[ROUTING_WEIGHT] = weights
        combined_outputs[EXPERT_OUTPUTS] = expert_outputs

        return combined_outputs

    def _combine_expert_outputs(
        self, expert_outputs: list[dict[str, torch.Tensor]], weights: torch.Tensor
    ) -> dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        """Combine expert output dictionaries using routing weights.

        Applies routing to each tensor output key across all experts, producing
        a single combined dictionary with weighted outputs.

        Args:
            expert_outputs: List of expert output dictionaries
            weights: Routing weights for combining experts

        Returns:
            Dictionary with same keys as expert outputs, but with routed values
        """
        combined: dict[str, torch.Tensor | list[dict[str, torch.Tensor]]] = {}
        output_keys = expert_outputs[0].keys()

        for key in output_keys:
            if not isinstance(expert_outputs[0][key], torch.Tensor):
                continue

            expert_tensors = [exp[key] for exp in expert_outputs]
            combined[key] = self._apply_routing(expert_tensors, weights)

        return combined
