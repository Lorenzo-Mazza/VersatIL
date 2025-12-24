"""Phase-conditioned ACT decoder with MoE routing.

Extends the base ACT architecture to support phase-based expert routing.
The phase classifier head produces routing logits that are used to route
position and gripper predictions through phase-specific expert networks.
"""


import torch

from refactoring.data.constants import ACTION_KEY, ObsKey
from refactoring.models.decoding.action_heads.moe import MoEHead
from refactoring.models.decoding.constants import EXPERT_OUTPUTS, ROUTING_WEIGHT
from refactoring.models.decoding.decoders.factory.act import ACT


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

    def __init__(self, *args, phase_routing_key: str = ObsKey.PHASE_LABEL.value, **kwargs):
        """Initialize PhaseACT decoder.

        Args:
            phase_routing_key: Key for the phase classifier head that provides routing weights
            *args, **kwargs: Arguments passed to base ACT decoder
        """
        super().__init__(*args, **kwargs)
        self.phase_routing_key = phase_routing_key
        if self.phase_routing_key not in self.action_heads:
            raise ValueError(
                f"PhaseACT requires '{self.phase_routing_key}' head for routing, "
                f"but only found: {list(self.action_heads.keys())}"
            )
        if any(isinstance(self.action_heads[key], MoEHead) for key in self.action_heads if key != self.phase_routing_key) is False:
            raise ValueError("PhaseACT requires at least one MoE action head for phase-based routing.")

    def _apply_action_heads(self, action_embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
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
        phase_logits = phase_head(action_embeddings)  # (B, prediction horizon, num_phases)
        predictions[self.phase_routing_key] = phase_logits
        for action_key, head in self.action_heads.items():
            if action_key == self.phase_routing_key:
                continue  # Already computed above
            # MoE heads receive phase logits as external routing weights
            if isinstance(head, MoEHead):
                output = head(
                    action_embeddings,
                    gating_feature=phase_logits  # Phase-based routing
                )
                predictions[action_key] = output[ACTION_KEY]
                predictions[ROUTING_WEIGHT] = output[ROUTING_WEIGHT] # This will be overwritten but is the same for all MoE heads
                predictions[f'{action_key}_{EXPERT_OUTPUTS}'] = output[EXPERT_OUTPUTS]
            else:
                predictions[action_key] = head(action_embeddings)
        return predictions
