"""Base contracts for action decoders."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from versatil.common.module_attr_mixin import ModuleAttrMixin
from versatil.common.omegaconf_ops import resolve_dict_keys
from versatil.data.constants import SampleKey
from versatil.data.normalization.normalizer import LinearNormalizer
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.data.tokenization import ActionTokenizer, Tokenizer
from versatil.models.decoding.action_heads.base import BaseActionHead
from versatil.models.decoding.action_heads.moe import MoEHead
from versatil.models.decoding.constants import ActionHeadLayout, DecoderOutputKey
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
    #: Requires normalized/tokenized observation tensors in the feature dict.
    #: VLA decoders use this to build image/language prefix tokens internally.
    needs_raw_observations: bool = False
    # For conditional decoders
    conditioning_key: str | None = None
    conditioning_required: list[str] = field(default_factory=list)
    conditioning_one_of_groups: list[list[str]] = field(default_factory=list)

    def __post_init__(self) -> None:
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
    ) -> None:
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
            for key in self.keys:
                if (
                    key in available_features
                    and available_features[key].feature_type == rejected_type
                ):
                    raise ValueError(
                        f"Decoder cannot accept {rejected_type} features, "
                        f"but '{key}' is {rejected_type}."
                    )


class ActionDecoder(ModuleAttrMixin, ABC):
    """Abstract base class for neural network action decoders."""

    requires_tokenized_actions: bool = False
    action_head_layout: ActionHeadLayout = ActionHeadLayout.COMPONENT

    def __init__(
        self,
        decoder_input: DecoderInput,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        action_heads: dict[str, BaseActionHead],
        device: str,
        observation_horizon: int,
        prediction_horizon: int,
    ) -> None:
        """Initialize common action decoder state and action heads."""
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
        """Set output dimensions according to the decoder action-head layout."""
        match self.action_head_layout:
            case ActionHeadLayout.NONE:
                return
            case ActionHeadLayout.COMPONENT:
                self._set_component_action_head_dimensions()
            case ActionHeadLayout.JOINT:
                self._single_action_head().set_output_dim(self.action_dim)
            case ActionHeadLayout.VOCABULARY:
                self._vocabulary_action_head().set_output_dim(1)

    def _set_component_action_head_dimensions(self) -> None:
        """Set one action-head output dimension per predicted action component."""
        predicted_dimensions = self.action_space.predicted_action_dimensions
        for key, head in self.action_heads.items():
            if key not in predicted_dimensions:
                raise ValueError(
                    f"Action head '{key}' is not a predicted action-space key. "
                    f"Predicted keys: {list(predicted_dimensions.keys())}."
                )
            head.set_output_dim(predicted_dimensions[key])

    def _single_action_head(self) -> BaseActionHead:
        """Return the only configured action head."""
        if len(self.action_heads) != 1:
            raise ValueError(
                f"{type(self).__name__} with action_head_layout="
                f"{ActionHeadLayout.JOINT.value} expects exactly one action head, "
                f"got {list(self.action_heads.keys())}."
            )
        return next(iter(self.action_heads.values()))

    def _vocabulary_action_head(self) -> BaseActionHead:
        """Return the token-vocabulary action head."""
        configured_heads = set(self.action_heads.keys())
        required_heads = {DecoderOutputKey.ACTION_LOGITS.value}
        if configured_heads != required_heads:
            raise ValueError(
                f"{type(self).__name__} with action_head_layout="
                f"{ActionHeadLayout.VOCABULARY.value} expects action_heads keys "
                f"{required_heads}, got {configured_heads}."
            )
        return self.action_heads[DecoderOutputKey.ACTION_LOGITS.value]

    def set_tokenizer(self, tokenizer: Tokenizer | None = None) -> None:
        """Set tokenizer for decoders trained on tokenized actions.

        This method is called by Policy.set_tokenizer() to pass the tokenizer
        to the decoder. Continuous decoders ignore it.

        Args:
            tokenizer: Tokenizer instance from data pipeline (can be None)
        """
        if not self.requires_tokenized_actions:
            self.tokenizer = None
            return
        if tokenizer is None or tokenizer.action_tokenizer is None:
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
        if self.requires_tokenized_actions:
            auxiliary_keys.add(SampleKey.TOKENIZED_ACTIONS.value)
        if any(isinstance(head, MoEHead) for head in self.action_heads.values()):
            auxiliary_keys.add(DecoderOutputKey.ROUTING_WEIGHTS.value)
        return auxiliary_keys

    def get_loss_output_keys(self) -> set[str]:
        """Return decoder output keys that can be supervised as actions."""
        match self.action_head_layout:
            case ActionHeadLayout.COMPONENT | ActionHeadLayout.VOCABULARY:
                return set(self.action_heads.keys())
            case ActionHeadLayout.JOINT:
                return set(self.action_space.predicted_action_keys)
            case ActionHeadLayout.NONE:
                if self.requires_tokenized_actions:
                    return set()
                return set(self.action_space.predicted_action_keys)

    def get_prediction_output_keys(self) -> list[str]:
        """Return predicted action keys in action-space metadata order.

        The order is the canonical action-key ordering: exported policies and
        compressed-artifact metadata index output tensors by it.
        """
        return list(self.action_space.predicted_action_keys)

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

    def validate_action_heads(self) -> None:
        """Validate that configured heads match this decoder's head layout."""
        match self.action_head_layout:
            case ActionHeadLayout.NONE:
                self._validate_no_action_heads()
            case ActionHeadLayout.COMPONENT:
                self._validate_component_action_heads()
            case ActionHeadLayout.JOINT:
                self._validate_joint_action_head()
            case ActionHeadLayout.VOCABULARY:
                self._validate_vocabulary_action_head()

    def _validate_no_action_heads(self) -> None:
        """Validate that a decoder does not receive action heads."""
        if self.action_heads:
            raise ValueError(
                f"{type(self).__name__} uses action_head_layout="
                f"{ActionHeadLayout.NONE.value}, so action_heads must be empty. "
                f"Got {list(self.action_heads.keys())}."
            )

    def _validate_component_action_heads(self) -> None:
        """Validate one configured action head per predicted component."""
        configured_heads = set(self.action_heads.keys())
        required_dimensions = self.action_space.predicted_action_dimensions
        required_heads = set(required_dimensions.keys())
        missing_heads = required_heads - configured_heads
        if missing_heads:
            raise ValueError(
                f"Action space requires action heads for {missing_heads}, "
                f"but configured heads are {configured_heads}."
            )
        extra_heads = configured_heads - required_heads
        if extra_heads:
            raise ValueError(
                f"Action heads are configured for {extra_heads}, but predicted "
                f"action-space keys are {required_heads}."
            )
        for action_key, expected_dimension in required_dimensions.items():
            actual_dimension = self.action_heads[action_key].output_dim
            if actual_dimension != expected_dimension:
                raise ValueError(
                    f"Action head '{action_key}' has output_dim={actual_dimension}, "
                    f"but action space requires dim={expected_dimension}."
                )

    def _validate_joint_action_head(self) -> None:
        """Validate one head that predicts the full continuous action vector."""
        joint_action_head = self._single_action_head()
        if joint_action_head.output_dim != self.action_dim:
            raise ValueError(
                f"{type(self).__name__} joint action head output_dim must equal "
                f"total action dimension {self.action_dim}, got "
                f"{joint_action_head.output_dim}."
            )

    def _validate_vocabulary_action_head(self) -> None:
        """Validate the token-vocabulary action head placeholder."""
        vocabulary_head = self._vocabulary_action_head()
        if vocabulary_head.output_dim != 1:
            raise ValueError(
                f"{type(self).__name__} vocabulary action head output_dim must "
                f"be initialized to 1 before tokenizer binding, got "
                f"{vocabulary_head.output_dim}."
            )
