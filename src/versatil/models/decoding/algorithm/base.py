"""Base class for action-decoding algorithms (BC, Diffusion, FlowMatching, etc.)."""

import abc
from abc import abstractmethod

import torch
import torch.nn as nn

from versatil.models.decoding.decoders.base import ActionDecoder


def resolve_feature_reference(
    features: dict[str, torch.Tensor],
) -> tuple[int, torch.device, torch.dtype]:
    """Return batch size, device, and dtype for tensors created from features.

    Feature dicts mix encoder outputs with integer token ids and boolean
    padding masks, and their ordering follows decoder configuration, so the
    reference must come from a floating-point feature rather than whichever
    value happens to be first.

    Args:
        features: Encoded features keyed by name.

    Raises:
        ValueError: If the feature dict is empty.
    """
    if len(features) == 0:
        raise ValueError("Cannot infer batch size from an empty feature dict.")
    for value in features.values():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            return value.shape[0], value.device, value.dtype
    first_value = next(iter(features.values()))
    return first_value.shape[0], first_value.device, torch.float32


class DecodingAlgorithm(nn.Module, abc.ABC):
    """Base class for decoding algorithms.

    Algorithms define the learning paradigm (behavioral cloning, diffusion, flow matching, etc.).
    Pure algorithms should not contain latent variable logic - use VariationalAlgorithm wrapper
    to add variational inference capabilities to any algorithm.

    Examples:
        # Pure algorithm (deterministic)
        algorithm = BehavioralCloning()

        # Algorithm with variational inference
        algorithm = VariationalAlgorithm(
            base_algorithm=BehavioralCloning(),
            posterior_encoder=VAETransformerEncoder(...),
            prior=GaussianPrior(...)  # or DiffusionPrior(...) for learned prior
        )
    """

    def __init__(self):
        """Initialize decoding algorithm."""
        super().__init__()

    @abstractmethod
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
            actions: Optional dictionary of ground truth actions, if architecture requires it, e.g. ACT.

        Returns:
            Decoder output dictionary containing predictions and any algorithm-specific outputs.
        """
        raise NotImplementedError

    def get_auxiliary_output_keys(self) -> set[str]:
        """Get keys for auxiliary outputs this algorithm adds to the decoder output.

        Override in subclasses that inject additional outputs (e.g. latent variables).

        Returns:
            Set of auxiliary output key strings.
        """
        return set()

    @property
    def predicts_in_action_space(self) -> bool:
        """Whether the network output lives in the action space.

        When True (e.g. Behavioral Cloning), the network directly
        predicts actions and classification losses like BCE are valid.
        When False (e.g. Flow Matching, Diffusion with epsilon/velocity),
        the network predicts in a derived space (velocity field, noise)
        and classification losses on those outputs are meaningless.
        """
        return True

    def get_targets(
        self,
        algorithm_output: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        ground_truth_actions: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Return the correct regression targets for the loss.

        The loss module is algorithm-agnostic: it computes error between
        predictions and targets. This method lets each algorithm specify
        what those targets are.

        Default returns the ground-truth actions (correct for Behavioral
        Cloning). Generative algorithms override to return their
        algorithm-specific targets (velocity field, noise, etc.).

        Args:
            algorithm_output: Full output dict from ``forward()``.
            ground_truth_actions: Ground-truth actions from the data batch.

        Returns:
            Per-key target tensors the loss should compare predictions against.
        """
        return ground_truth_actions

    @abstractmethod
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
            Decoder output dictionary containing predictions and any algorithm-specific outputs.
        """
        raise NotImplementedError
