"""Mixture of Experts (MoE) decoder for action prediction."""

import copy

import torch
import torch.nn as nn

from versatil.models.decoding.constants import DecoderOutputKey, MoERoutingType
from versatil.models.decoding.decoders.base import ActionDecoder
from versatil.models.decoding.mixture_of_experts import BaseMixtureOfExperts
from versatil.models.layers.activation import ActivationFunction


class MoEDecoder(BaseMixtureOfExperts, ActionDecoder):
    """Mixture of action decoder experts."""

    def __init__(
        self,
        base_expert: ActionDecoder,
        num_experts: int,
        gating_feature_key: str,
        inference_gating_key: str | None = None,
        gating_input_dim: int | None = None,
        gating_hidden_dims: list[int] | None = None,
        gating_activation: str = ActivationFunction.SILU.value,
        routing_type: str = MoERoutingType.SOFT.value,
        top_k: int = 2,
        temperature: float = 1.0,
        learnable_temperature: bool = False,
        gating_dropout: float = 0.1,
        gating_normalization: bool = True,
    ):
        expert_list = self._create_experts_from_config(
            base_expert=base_expert, num_experts=num_experts
        )
        ActionDecoder.__init__(
            self,
            decoder_input=base_expert.decoder_input,
            observation_space=base_expert.observation_space,
            action_space=base_expert.action_space,
            action_heads=dict(base_expert.action_heads),
            device=str(base_expert.device),
            observation_horizon=base_expert.observation_horizon,
            prediction_horizon=base_expert.prediction_horizon,
        )

        BaseMixtureOfExperts.__init__(
            self,
            num_experts=num_experts,
            device=base_expert.device,
            gating_input_dim=gating_input_dim,
            gating_activation_function=gating_activation,
            gating_hidden_dims=gating_hidden_dims,
            routing_type=routing_type,
            top_k=top_k,
            temperature=temperature,
            learnable_temperature=learnable_temperature,
            gating_dropout=gating_dropout,
            gating_normalization=gating_normalization,
        )
        self.expert_decoders = nn.ModuleList(expert_list)
        self.base_expert = base_expert
        self.num_experts = num_experts
        self.gating_feature_key = gating_feature_key
        self.inference_gating_key = (
            inference_gating_key
            if inference_gating_key is not None
            else gating_feature_key
        )
        self.action_keys = list(base_expert.action_heads.keys())
        self.action_heads = base_expert.action_heads

    def get_auxiliary_output_keys(self) -> set[str]:
        """MoE decoder produces routing weights and per-expert outputs."""
        keys = super().get_auxiliary_output_keys()
        keys.add(DecoderOutputKey.ROUTING_WEIGHTS.value)
        keys.add(DecoderOutputKey.EXPERT_OUTPUTS.value)
        return keys

    @staticmethod
    def _create_experts_from_config(
        base_expert: ActionDecoder,
        num_experts: int,
    ) -> list[ActionDecoder]:
        """Create expert decoders from configuration.

        Args:
            base_expert: Base expert decoder instance to clone
            num_experts: Number of expert decoders to create

        Returns:
            List of instantiated ActionDecoder experts
        """
        experts = []
        for _ in range(num_experts):
            # Deep copy creates a completely independent module with separate weights
            expert = copy.deepcopy(base_expert)
            for module in expert.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
            if hasattr(expert, "_init_weights"):
                expert.apply(expert._init_weights)
            experts.append(expert)
        return experts

    def forward(
        self,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        """Forward pass through mixture of expert decoders.

        Args:
            features: Dictionary of input features
            actions: Optional ground-truth actions (for training)

        Returns:
            Dictionary containing:
                - Combined predictions from routed experts (action keys)
                - routing_weights: Computed routing weights
                - expert_outputs: Individual expert prediction dictionaries
        """
        if self.training:
            gating_key = self.gating_feature_key
        else:
            gating_key = self.inference_gating_key
        gating_feature = features[gating_key]  # (B, embedding dimension)
        mixing_probabilities = self.compute_routing_weights(
            gating_feature
        )  # (B, num_experts)
        features.pop(gating_key)
        expert_outputs = [None] * len(self.expert_decoders)
        if torch.cuda.is_available() and features[next(iter(features))].is_cuda:
            streams = [torch.cuda.Stream() for _ in self.expert_decoders]
            for i, (expert, stream) in enumerate(zip(self.expert_decoders, streams)):
                with torch.cuda.stream(stream):
                    expert_outputs[i] = expert(features, actions)
            torch.cuda.synchronize()
        else:
            for i, expert in enumerate(self.expert_decoders):
                expert_outputs[i] = expert(features, actions)
        combined_outputs = self._combine_expert_outputs(
            expert_outputs=expert_outputs, weights=mixing_probabilities
        )
        combined_outputs[DecoderOutputKey.ROUTING_WEIGHTS.value] = mixing_probabilities
        combined_outputs[DecoderOutputKey.EXPERT_OUTPUTS.value] = expert_outputs
        return combined_outputs

    def _combine_expert_outputs(
        self, expert_outputs: list[dict[str, torch.Tensor]], weights: torch.Tensor
    ) -> dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        """Combine expert output dictionaries using routing weights.

        Applies routing to each action key across all experts, producing
        a single combined dictionary with weighted outputs.

        Args:
            expert_outputs: List of expert output dictionaries
            weights: Routing weights for combining experts

        Returns:
            Dictionary with action head keys and aggregated predictions as values.
        """
        combined: dict[str, torch.Tensor | list[dict[str, torch.Tensor]]] = {}
        for key in self.action_keys:
            expert_tensors = [exp[key] for exp in expert_outputs]
            combined[key] = self._apply_routing(expert_tensors, weights)

        return combined
