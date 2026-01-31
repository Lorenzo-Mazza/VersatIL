"""Mixture of Experts (MoE) action head for phase-conditioned or multi-modal action prediction."""

import copy

import torch
import torch.nn as nn

from versatil.data.constants import SampleKey
from versatil.models.decoding.action_heads.head import ActionHead
from versatil.models.decoding.constants import DecoderOutputKey, MoERoutingType
from versatil.models.decoding.mixture_of_experts import BaseMixtureOfExperts
from versatil.models.layers.activation import ActivationFunction


class MoEHead(BaseMixtureOfExperts):
    """Mixture of Experts head for action prediction.

    Supports three initialization modes:
    1. Explicit expert list: Pass pre-instantiated experts
    2. Base expert cloning: Pass base_expert + num_experts (creates experts immediately)
    3. Lazy initialization: Pass only base_expert (num_experts set later via set_num_experts)

    The lazy mode is useful when num_experts needs to be inferred from metadata at runtime,
    such as when PhaseACT infers the number of phases from action_space.

    Note:
        output_dim is set by the decoder through set_output_dim(), based on the action key.
    """

    def __init__(
        self,
        device: str = "cpu",
        experts: list[ActionHead] | None = None,
        base_expert: ActionHead | None = None,
        num_experts: int | None = None,
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
            device: Device to place the module on
            experts: Optional pre-instantiated expert action heads
            base_expert: Single expert instance to clone num_experts times
            num_experts: Number of experts to create from base_expert (optional for lazy init)
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
        if experts is not None and len(experts) > 0:
            super().__init__(
                num_experts=len(experts),
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
            )
            self.experts = nn.ModuleList(experts)
            self._is_initialized = True
            self._base_expert_template = None
            self._lazy_init_params = None
        elif base_expert is not None and num_experts is not None:
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
            )
            expert_list = self._create_experts_from_instance(base_expert, num_experts)
            self.experts = nn.ModuleList([e.to(device) for e in expert_list])
            self._is_initialized = True
            self._base_expert_template = None
            self._lazy_init_params = None
        elif base_expert is not None:
            nn.Module.__init__(
                self
            )  # nn.Module init, defer parent init until set_num_experts()
            self._base_expert_template = base_expert
            self._lazy_init_params = {
                "device": device,
                "gating_input_dim": gating_input_dim,
                "gating_activation_function": gating_activation,
                "gating_hidden_dims": gating_hidden_dims,
                "routing_type": routing_type,
                "top_k": top_k,
                "temperature": temperature,
                "learnable_temperature": learnable_temperature,
                "gating_dropout": gating_dropout,
                "gating_normalization": gating_normalization,
            }
            self.experts = None
            self._is_initialized = False
        else:
            raise ValueError("Must provide 'experts' or 'base_expert'")

        self._output_dim: int | None = None
        self._device = device
        self.gating_feature_key = gating_feature_key

    @property
    def is_initialized(self) -> bool:
        """Check if experts have been created."""
        return self._is_initialized

    def set_num_experts(self, num_experts: int) -> None:
        """Create experts after inferring num_experts from metadata.

        Called by decoders (e.g., PhaseACT) that infer the number of experts
        from action_space metadata at runtime.

        Args:
            num_experts: Number of experts to create

        Raises:
            RuntimeError: If already initialized or no base_expert template stored
        """
        if self._is_initialized:
            raise RuntimeError(
                "MoEHead already initialized. Cannot call set_num_experts twice."
            )
        if self._base_expert_template is None:
            raise RuntimeError("No base_expert template stored. Cannot create experts.")
        if self._lazy_init_params is None:
            raise RuntimeError("No lazy init params stored.")
        base_expert = self._base_expert_template
        lazy_params = self._lazy_init_params
        output_dim = self._output_dim
        device = self._device
        BaseMixtureOfExperts.__init__(
            self,
            num_experts=num_experts,
            **lazy_params,
        )
        expert_list = self._create_experts_from_instance(base_expert, num_experts)
        self.experts = nn.ModuleList([e.to(device) for e in expert_list])
        if output_dim is None:
            raise ValueError(
                "Output dimension is not set for MoE Head. Call set_output_dim() first."
            )
        for expert in self.experts:
            expert.set_output_dim(output_dim)
        self._is_initialized = True
        self._output_dim = output_dim
        self._device = device
        self._base_expert_template = None
        self._lazy_init_params = None

    @property
    def output_dim(self) -> int:
        """Get output dimension. Raises if not set."""
        if self._output_dim is None:
            raise RuntimeError("output_dim not set. Call set_output_dim() first.")
        return self._output_dim

    @output_dim.setter
    def output_dim(self, value: int) -> None:
        self._output_dim = value

    def set_output_dim(self, dim: int) -> None:
        """Set output dimension on this head and all expert heads.

        Called by the decoder based on the action metadata prediction_dimension.
        If in lazy mode (experts not yet created), stores the dim for later use
        when set_num_experts() is called.

        Args:
            dim: Output action dimension
        """
        self._output_dim = dim
        if self._is_initialized and self.experts is not None:
            for expert in self.experts:
                expert.set_output_dim(dim)

    @staticmethod
    def _create_experts_from_instance(
        base_expert: ActionHead,
        num_experts: int,
    ) -> list[ActionHead]:
        """Create expert action heads by deep copying a base instance.

        Args:
            base_expert: Base expert instance to clone
            num_experts: Number of expert heads to create

        Returns:
            List of deep-copied ActionHead experts with independent weights

        Note:
            Uses copy.deepcopy() to ensure each expert has completely independent
            parameters and modules. This works for any ActionHead architecture,
            including complex block compositions.
        """
        experts = []
        for _ in range(num_experts):
            # Deep copy creates a completely independent module with separate weights
            expert = copy.deepcopy(base_expert)
            for module in expert.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
            for module in expert.modules():
                if hasattr(module, "_init_weights"):
                    module._init_weights()
            experts.append(expert)

        return experts

    def forward(
        self,
        features: torch.Tensor,
        gating_feature: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through mixture of experts.

        Args:
            features: Input features for action prediction
            gating_feature: gating feature for combining expert outputs

        Returns:
            Dictionary containing:
                - action: Combined action predictions from experts
                - routing_weights: Computed routing weights
                - expert_outputs: Individual expert predictions (stacked)

        Raises:
            RuntimeError: If MoEHead is not initialized (lazy mode without set_num_experts)
        """
        if not self._is_initialized:
            raise RuntimeError("MoEHead not initialized. Call set_num_experts() first.")
        weights = self.compute_routing_weights(gating_feature)  # (B, num_experts)
        expert_outputs = [expert(features) for expert in self.experts]
        expert_outputs_stacked = torch.stack(expert_outputs, dim=-2)
        final_output = self._apply_routing(expert_outputs, weights)
        return {
            SampleKey.ACTION.value: final_output,
            DecoderOutputKey.ROUTING_WEIGHTS.value: weights,
            DecoderOutputKey.EXPERT_OUTPUTS.value: expert_outputs_stacked,
        }
