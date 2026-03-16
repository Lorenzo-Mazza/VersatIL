"""Phase-conditioned ACT decoder with MoE routing.

Extends the base ACT architecture to support phase-based expert routing.
The phase classifier head produces routing logits that are used to route
position and gripper predictions through phase-specific expert networks.
"""


import torch

from versatil.common.omegaconf_ops import resolve_dict_keys
from versatil.data.constants import ObsKey, SampleKey
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.models.decoding.action_heads import ActionHead
from versatil.models.decoding.action_heads.moe import MoEHead
from versatil.models.decoding.constants import DecoderOutputKey
from versatil.models.decoding.decoders.factory.act import ACT


class PhaseACT(ACT):
    """Phase-conditioned Action Chunking Transformer.

    This decoder extends ACT with phase-based expert routing. It expects:
        - A phase classifier head (output key: 'phase_label')
        - MoE action heads that use phase predictions for routing

    During forward pass:
        1. Phase classifier predicts phase logits/probabilities
        2. MoE action heads use phase logits as routing weights
        3. Each expert specializes in one surgical phase
    """

    def __init__(
        self,
        input_keys: list[str],
        action_space: ActionSpace,
        action_heads: dict[str, ActionHead],
        observation_space: ObservationSpace,
        observation_horizon: int,
        prediction_horizon: int,
        device: str,
        embedding_dimension: int = 256,
        number_of_heads: int = 8,
        feedforward_dimension: int = 512,
        number_of_encoder_layers: int = 6,
        number_of_decoder_layers: int = 6,
        activation: str = "relu",
        dropout_rate: float = 0.1,
        normalize_before: bool = False,
        phase_routing_key: str = ObsKey.PHASE_LABEL.value,
    ):
        """Initialize PhaseACT decoder.

        Args:
            input_keys: List of feature keys expected from encoder pipeline.
            action_space: Action space configuration.
            action_heads: Dictionary of action head modules.
            observation_space: Observation space configuration.
            observation_horizon: Number of observation timesteps.
            prediction_horizon: Number of actions to predict.
            device: Device to run the model on.
            embedding_dimension: Transformer hidden dimension.
            number_of_heads: Number of attention heads.
            feedforward_dimension: Feedforward network dimension.
            number_of_encoder_layers: Number of transformer encoder layers.
            number_of_decoder_layers: Number of transformer decoder layers.
            activation: Activation function name.
            dropout_rate: Dropout probability.
            normalize_before: Use pre-normalization.
            phase_routing_key: Key for the phase classifier head that provides routing weights.
        """
        super().__init__(
            input_keys=input_keys,
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            feedforward_dimension=feedforward_dimension,
            number_of_encoder_layers=number_of_encoder_layers,
            number_of_decoder_layers=number_of_decoder_layers,
            activation=activation,
            dropout_rate=dropout_rate,
            normalize_before=normalize_before,
        )
        self.phase_routing_key = phase_routing_key
        if self.phase_routing_key not in self.action_heads:
            raise ValueError(
                f"PhaseACT requires '{self.phase_routing_key}' head for routing, "
                f"but only found: {list(self.action_heads.keys())}"
            )
        if not any(
            isinstance(self.action_heads[key], MoEHead)
            for key in self.action_heads
            if key != self.phase_routing_key
        ):
            raise ValueError(
                "PhaseACT requires at least one MoE action head for phase-based routing."
            )

        self._initialize_moe_experts()

    def _initialize_moe_experts(self) -> None:
        """Set num_experts on lazy MoE heads from phase metadata."""
        resolved_metadata = resolve_dict_keys(dict(self.action_space.actions_metadata))
        phase_metadata = resolved_metadata[self.phase_routing_key]
        num_phases = phase_metadata.prediction_dimension
        for key, head in self.action_heads.items():
            if isinstance(head, MoEHead) and not head.is_initialized:
                head.set_num_experts(num_phases)

    def _apply_action_heads(
        self, action_embeddings: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Apply action heads with phase-based routing.

        Flow:
            1. Compute phase logits from phase classifier head
            2. Pass phase logits as routing weights to MoE action heads
            3. Non-MoE heads are called normally without routing

        Args:
            action_embeddings: Decoded action embeddings from transformer
                Shape: (B, prediction_horizon, embedding_dimension)

        Returns:
            Dictionary containing:
                - phase_label: Phase predictions (routing weights)
                - Other action predictions (position, gripper, etc.)
                - routing_weights: Computed routing weights (from MoE heads)
                - expert_outputs: Individual expert predictions (from MoE heads)
        """
        predictions = {}

        # Get phase predictions (these become routing weights)
        phase_head = self.action_heads[self.phase_routing_key]
        phase_logits = phase_head(
            action_embeddings
        )  # (B, prediction horizon, num_phases)
        predictions[self.phase_routing_key] = phase_logits
        for action_key, head in self.action_heads.items():
            if action_key == self.phase_routing_key:
                continue  # Already computed above
            # MoE heads receive phase logits as external routing weights
            if isinstance(head, MoEHead):
                output = head(
                    action_embeddings,
                    gating_feature=phase_logits,  # Phase-based routing
                )
                predictions[action_key] = output[SampleKey.ACTION.value]
                predictions[DecoderOutputKey.ROUTING_WEIGHTS.value] = output[
                    DecoderOutputKey.ROUTING_WEIGHTS.value
                ]  # This will be overwritten but is the same for all MoE heads
                predictions[
                    f"{action_key}_{DecoderOutputKey.EXPERT_OUTPUTS.value}"
                ] = output[DecoderOutputKey.EXPERT_OUTPUTS.value]
            else:
                predictions[action_key] = head(action_embeddings)
        return predictions
