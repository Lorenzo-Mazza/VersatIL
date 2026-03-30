from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from versatil.common.omegaconf_ops import resolve_dict_keys
from versatil.data.constants import SampleKey
from versatil.data.normalization.normalizer import LinearNormalizer
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.data.tokenization import ActionTokenizer, Tokenizer
from versatil.models.decoding.action_heads.moe import MoEHead
from versatil.models.decoding.constants import DecoderOutputKey
from versatil.models.feature_meta import FeatureMetadata


@dataclass
class DecoderInput:
    """Structured input specification for decoder architectures."""

    keys: list[str]  # feature keys required by the decoder
    #: If specified, the decoder needs at least one input observation key from all these feature types
    #: They have to be `FeatureType` values, i.e. either 'spatial', 'sequential' or 'flat'
    required_types: list[str] = field(default_factory=list)
    #: If specified, the decoder will raise an error at init time, if the input key belongs to the specified feature types.
    raises_for_types: list[str] = field(default_factory=list)
    #: Requires actions during decoding
    requires_actions: bool = False
    #: Requires VLM backbone layers for interleaved decoding (Pi0/SmolVLA)
    requires_vlm_backbone: bool = False
    # For conditional decoders
    conditioning_key: str | None = None
    conditioning_required: list[str] = field(default_factory=list)
    conditioning_one_of_groups: list[list[str]] = field(default_factory=list)

    def __post_init__(self):
        """Post-initialization to ensure feature keys are consistent."""
        if self.conditioning_key:
            conditioning_set = {self.conditioning_key}
            missing_conditioning = set(self.conditioning_required) - conditioning_set
            if missing_conditioning:
                raise ValueError(
                    f"Missing required conditioning for decoder input: {missing_conditioning}"
                )
            for group in self.conditioning_one_of_groups:
                matches = conditioning_set.intersection(group)
                if len(matches) != 1:
                    raise ValueError(
                        f"Exactly one from {group} required for decoder input conditioning"
                    )

    def validate_feature_types(
        self,
        available_features: dict[str, FeatureMetadata],
    ):
        """Validate that required feature types are available at instantiation time.

        Args:
            available_features: Dict mapping feature names to their metadata.

        Raises:
            ValueError: If a required type is missing or a rejected type is present.
        """
        for expected_type in self.required_types:
            matched = any(
                available_features[key].feature_type == expected_type
                for key in self.keys
                if key in available_features
            )
            if not matched:
                raise ValueError(
                    f"Decoder requires at least one feature of type '{expected_type}', "
                    f"but none found among: {self.keys}. "
                    f"Available: {available_features}"
                )
        for rejected_type in self.raises_for_types:
            for key, meta in available_features.items():
                if meta.feature_type == rejected_type:
                    raise ValueError(
                        f"Decoder cannot accept {rejected_type} features, "
                        f"but '{key}' is {rejected_type}."
                    )


class ActionDecoder(nn.Module, ABC):
    """Abstract base class for Neural Network architectures used for action decoding.

    Attributes:
        supports_tokenized_actions: Whether this decoder architecture supports discrete tokenized actions.

            Note: This is separate from algorithm support - both decoder AND algorithm must support
            tokenization for it to work. Set this to True only for specialized autoregressive decoders
            designed for discrete action tokens.
    """

    supports_tokenized_actions: bool = False

    def __init__(
        self,
        decoder_input: DecoderInput,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        action_heads: dict,
        device: str,
        observation_horizon: int,
        prediction_horizon: int,
    ):
        super().__init__()
        self.decoder_input = decoder_input
        resolved_heads = resolve_dict_keys(action_heads)
        self.action_heads = nn.ModuleDict(resolved_heads)
        self.observation_space = observation_space
        self.action_space = action_space
        self.observation_horizon = observation_horizon
        self.prediction_horizon = prediction_horizon
        self.device = torch.device(device)
        self._set_action_head_dimensions()
        self.validate_action_heads()
        self.normalizer: LinearNormalizer = LinearNormalizer()
        self.tokenizer: ActionTokenizer | None = None

    def _set_action_head_dimensions(self) -> None:
        """Set output dimensions on action heads from action_space.

        Each action head's output_dim is set based on the corresponding
        action_space.actions_metadata[key].prediction_dimension.

        Raises:
            ValueError: If an action head key is not found in action_space.actions_metadata
                (only for non-tokenized decoders)
        """
        if self.supports_tokenized_actions:
            # Use placeholder dimension - set_tokenizer() will set real dimension
            for head in self.action_heads.values():
                head.set_output_dim(1)
            return
        predicted_metadata = {
            k: v
            for k, v in self.action_space.actions_metadata.items()
            if v.requires_prediction_head
        }
        for key, head in self.action_heads.items():
            if key not in predicted_metadata:
                raise ValueError(
                    f"Action head '{key}' not found in action_space.actions_metadata. "
                    f"Available keys: {list(predicted_metadata.keys())}"
                )
            dim = predicted_metadata[key].prediction_dimension
            head.set_output_dim(dim)

    def set_tokenizer(self, tokenizer: Tokenizer | None = None):
        """Set tokenizer for discrete action tokenization.

        This method is called by Policy.set_tokenizer() to pass the tokenizer
        to the decoder. Only decoders with supports_tokenized_actions=True should
        use this tokenizer in their forward/predict methods.

        Args:
            tokenizer: Tokenizer instance from data pipeline (can be None)
        """
        if not self.supports_tokenized_actions:
            self.tokenizer = None
            return
        if tokenizer is None:
            raise ValueError(
                "Tokenizer must be provided for tokenized action decoders."
            )
        self.tokenizer = tokenizer.action_tokenizer

    def set_normalizer(self, normalizer: "LinearNormalizer") -> None:
        """Set normalizer for data-dependent initialization.

        Args:
            normalizer: LinearNormalizer instance with loaded state.
        """
        self.normalizer = normalizer

    def enable_encoder_cache(self) -> None:
        """No-op. Override in decoders that support encoder caching."""

    def disable_encoder_cache(self) -> None:
        """No-op. Override in decoders that support encoder caching."""

    @abstractmethod
    def forward(
        self,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the Neural Network architecture for action decoding."""
        raise NotImplementedError(
            "Subclasses of ActionDecoder must implement the forward pass."
        )

    @property
    def has_history(self) -> bool:
        """Whether the architecture processes temporal sequences."""
        return self.observation_horizon > 1

    def get_auxiliary_output_keys(self) -> set[str]:
        """Get keys for auxiliary outputs this decoder produces beyond action predictions.

        The base implementation includes tokenized action keys and MoE routing weights
        when applicable. Subclasses override to add decoder-specific auxiliary keys.

        Returns:
            Set of auxiliary output key strings.
        """
        auxiliary_keys: set[str] = set()
        if self.supports_tokenized_actions:
            auxiliary_keys.add(SampleKey.TOKENIZED_ACTIONS.value)
        if any(isinstance(head, MoEHead) for head in self.action_heads.values()):
            auxiliary_keys.add(DecoderOutputKey.ROUTING_WEIGHTS.value)
        return auxiliary_keys

    @property
    def action_dim(self) -> int:
        """Get the total action dimension."""
        return self.action_space.get_total_action_dim()

    @property
    def use_gripper_actions(self) -> bool:
        """Whether the architecture uses gripper actions."""
        return self.action_space.has_gripper_actions

    @property
    def gripper_dim(self) -> int | None:
        """Get the gripper dimension if used."""
        return self.action_space.gripper_dim if self.use_gripper_actions else None

    @property
    def use_orientation_actions(self) -> bool:
        """Whether the architecture uses orientation actions."""
        return self.action_space.has_orientation_actions

    @property
    def orientation_dim(self) -> int | None:
        """Get the orientation dimension if used."""
        return (
            self.action_space.orientation_dim if self.use_orientation_actions else None
        )

    @property
    def position_dim(self) -> int | None:
        """Get the position dimension if used."""
        return (
            self.action_space.position_dim
            if self.action_space.has_position_actions
            else None
        )

    def validate_action_heads(self):
        """Validate that action heads match the action space configuration.

        Ensures that:
        1. Required action modalities have corresponding heads
        2. Head output dimensions match action space dimensions
        3. No extra heads are defined for non-existent actions

        Raises:
            ValueError: If validation fails
        """
        if self.supports_tokenized_actions:
            return

        configured_heads = set(self.action_heads.keys())
        required_heads = {}
        for key, meta in self.action_space.actions_metadata.items():
            if meta.requires_prediction_head:
                required_heads[key] = meta.prediction_dimension

        required_keys = set(required_heads.keys())
        missing_heads = required_keys - configured_heads
        if missing_heads:
            raise ValueError(
                f"Action space requires heads for {missing_heads}, but they are not configured. "
                f"Configured heads: {configured_heads}"
            )
        extra_heads = configured_heads - required_keys
        if extra_heads:
            raise ValueError(
                f"Action heads defined for {extra_heads}, but these actions are not in the action space. "
                f"Required heads: {required_keys}"
            )
        for action_key, expected_dim in required_heads.items():
            head = self.action_heads[action_key]
            actual_dim = head.output_dim
            if actual_dim != expected_dim:
                raise ValueError(
                    f"Action head '{action_key}' has output_dim={actual_dim}, "
                    f"but action space requires dim={expected_dim}"
                )
