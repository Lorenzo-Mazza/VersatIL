
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from refactoring.data.task import ObservationSpace, ActionSpace
from refactoring.data.tokenization import Tokenizer, ActionTokenizer
from refactoring.models.decoding.constants import FeatureType


@dataclass
class DecoderInput:
    """Structured input specification for decoder architectures."""
    keys: list[str] # feature keys required by the decoder
    #: If specified, the decoder needs at least one input observation key from all these feature types
    #: They have to be `FeatureType` values, i.e. either 'spatial', 'sequential' or 'flat'
    required_types: list[str] = field(default_factory=list)
    #: If specified, the decoder will raise an error at init time, if the input key belongs to the specified feature types.
    raises_for_types: list[str] = field(default_factory=list)
    #: Requires actions during decoding
    requires_actions: bool = False
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
                raise ValueError(f"Missing required conditioning for decoder input: {missing_conditioning}")
            for group in self.conditioning_one_of_groups:
                matches = conditioning_set.intersection(group)
                if len(matches) != 1:
                    raise ValueError(f"Exactly one from {group} required for decoder input conditioning")


    def validate_feature_types(
        self,
        available_features_to_dims: dict[str, int | tuple[int, ...]],
    ):
        """Validate that the required input features to the decoder architecture are available at instantiation time.

        Note: this validation avoids throwing errors at runtime, optimizing experiment runs.

        Args:
            available_features_to_dims: Dict mapping feature names to their dimensions

        Raises:
            ValueError: If validation fails
        """
        for expected_type in self.required_types:
            matched = False
            for feature_name in self.keys:
                feature_dim = available_features_to_dims[feature_name]
                is_spatial = isinstance(feature_dim, tuple) and len(feature_dim) == 3
                is_sequential = isinstance(feature_dim, tuple) and len(feature_dim) == 2
                is_flat = isinstance(feature_dim, int) or (isinstance(feature_dim, tuple) and len(feature_dim) == 1)
                if (expected_type == FeatureType.SPATIAL.value and is_spatial
                        or expected_type == FeatureType.SEQUENTIAL.value and is_sequential
                        or expected_type == FeatureType.FLAT.value and is_flat):
                    matched = True
            if not matched:
                raise ValueError(
                    f"Decoder architecture requires at least one input feature of type '{expected_type}', "
                    f"but none were found among the provided features: {self.keys}."
                    f" Available features and dimensions: {available_features_to_dims.items()}"
                )
        for feature_type in self.raises_for_types:
            for key in available_features_to_dims:
                feature_dim = available_features_to_dims[key]
                is_spatial = isinstance(feature_dim, tuple) and len(feature_dim) == 3
                if feature_type == FeatureType.SPATIAL.value and is_spatial:
                    raise ValueError("Decoder architecture cannot accept spatial features as input.")
                is_sequential = isinstance(feature_dim, tuple) and len(feature_dim) == 2
                if feature_type == FeatureType.SEQUENTIAL.value and is_sequential:
                    raise ValueError("Decoder architecture cannot accept sequential features as input.")
                is_flat = isinstance(feature_dim, int) or (isinstance(feature_dim, tuple) and len(feature_dim) == 1)
                if feature_type == FeatureType.FLAT.value and is_flat:
                    raise ValueError("Decoder architecture cannot accept flat features as input.")


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
        self.action_heads = nn.ModuleDict(action_heads)
        self.observation_space = observation_space
        self.action_space = action_space
        self.observation_horizon = observation_horizon
        self.prediction_horizon = prediction_horizon
        self.device = torch.device(device)
        self.validate_action_heads()
        self.tokenizer: ActionTokenizer | None = None

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
            raise ValueError("Tokenizer must be provided for tokenized action decoders.")
        self.tokenizer = tokenizer.action_tokenizer

    @abstractmethod
    def forward(self,
                features: dict[str, torch.Tensor],
                actions: dict[str, torch.Tensor] | None = None
                ) -> dict[str, torch.Tensor]:
        """Forward pass of the Neural Network architecture for action decoding."""
        raise NotImplementedError("Subclasses of ActionDecoder must implement the forward pass.")

    @property
    def has_history(self) -> bool:
        """Whether the architecture processes temporal sequences."""
        return self.observation_horizon > 1


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
        return self.action_space.orientation_dim if self.use_orientation_actions else None

    @property
    def position_dim(self) -> int | None:
        """Get the position dimension if used."""
        return self.action_space.position_dim if self.use_position_actions else None

    def validate_action_heads(self):
        """Validate that action heads match the action space configuration.

        Ensures that:
        1. Required action modalities have corresponding heads
        2. Head output dimensions match action space dimensions
        3. No extra heads are defined for non-existent actions

        Raises:
            ValueError: If validation fails
        """
        # Skip validation for tokenized action decoders
        if self.supports_tokenized_actions:
            return

        configured_heads = set(self.action_heads.keys())
        required_heads = {}
        for key, meta in self.action_space.actions_metadata.items():
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
