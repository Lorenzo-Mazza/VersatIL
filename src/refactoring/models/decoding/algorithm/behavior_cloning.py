"""Behavioral Cloning (BC) algorithm for supervised action prediction.

Note: For multi-modal action prediction with latent variables, use:
    VariationalAlgorithm(BehavioralCloning(), VAETransformerEncoder(...))

This provides the same functionality as the old BehavioralCloning with latent_encoder,
but with a cleaner compositional design.
"""


import torch

from refactoring.models.decoding.algorithm.base import DecodingAlgorithm
from refactoring.models.decoding.decoders.base import ActionDecoder


class BehavioralCloning(DecodingAlgorithm):
    """Pure Behavioral Cloning algorithm without variational inference.

    Simplest imitation learning approach: directly predicts actions from observations
    using supervised learning. The network is trained to minimize the difference
    between predicted and ground-truth actions.

    This is a deterministic, uni-modal algorithm. For multi-modal action prediction,
    wrap with VariationalAlgorithm:
        VariationalAlgorithm(BehavioralCloning(), VAETransformerEncoder(...))
    """

    def __init__(self):
        """Initialize Behavioral Cloning algorithm."""
        super().__init__()

    def forward(
        self,
        network: ActionDecoder,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass during training.

        Args:
            network: The action decoder network module
            features: Dictionary of encoded features from the encoding pipeline.
            actions: Optional dictionary of ground truth actions, if architecture requires it (e.g., DETR).

        Returns:
            Decoder output dictionary containing action predictions and any architecture-specific outputs.
            Keys typically include:
                - 'position_action': Predicted position actions (B, T, D_pos)
                - 'orientation_action': Predicted orientation actions if used (B, T, D_ori)
                - 'gripper_action': Predicted gripper actions if used (B, T, D_grip)
                - Additional architecture-specific outputs (e.g., 'is_pad')
        """
        # Direct prediction without latent variables
        return network(features=features, actions=actions)  # type: ignore[no-any-return]

    def predict(
        self,
        network: ActionDecoder,
        features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Inference/prediction pass.

        Args:
            network: The action decoder network module
            features: Dict of encoded features from encoding pipeline

        Returns:
            Decoder output dictionary containing action predictions.
            Keys typically include:
                - 'position_action': Predicted position actions (B, T, D_pos)
                - 'orientation_action': Predicted orientation actions if used (B, T, D_ori)
                - 'gripper_action': Predicted gripper actions if used (B, T, D_grip)
        """
        # Direct prediction without latent variables
        return network(features, actions=None)  # type: ignore[no-any-return]
