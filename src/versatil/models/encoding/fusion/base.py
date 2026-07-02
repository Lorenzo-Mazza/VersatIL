"""Fusion modules for combining multi-modal features."""

import abc
from dataclasses import dataclass

import torch
import torch.nn as nn

from versatil.models.feature_meta import FeatureMetadata, FeatureType


@dataclass
class FusionInput:
    """Structured input specification for fusion modules."""

    input_features: list[str]
    required_count: int = 1  # Minimum number of features required
    max_count: int | None = (
        None  # Maximum number of features allowed (None = unlimited)
    )


class FusionModule(nn.Module):
    """Base fusion module with validation."""

    def __init__(
        self,
        input_specification: FusionInput,
        output_name: str,
    ):
        super().__init__()
        self.input_specification = input_specification
        self.output_name = output_name
        self._initialized = False

    @property
    def input_features(self) -> list[str]:
        """Get list of input feature names."""
        return self.input_specification.input_features

    @input_features.setter
    def input_features(self, features: list[str]):
        """Set input features (used by pipeline during resolution)."""
        self.input_specification.input_features = features

    @abc.abstractmethod
    def get_output_specification(self) -> FeatureMetadata:
        """Get structured output specification."""
        raise NotImplementedError

    def setup(self, feature_registry: dict[str, FeatureMetadata]):
        """Setup layers once feature metadata is known.

        Note:
            Called once by the encoding pipeline after feature metadata is available.
            Allows fusion modules to be created without knowing input dimensions ahead of time.

        Args:
            feature_registry: Dict mapping available feature names to their metadata.
        """
        if self._initialized:
            return
        self._setup_layers(feature_registry)
        self._initialized = True

    @abc.abstractmethod
    def _setup_layers(self, feature_registry: dict[str, FeatureMetadata]):
        """Build layers once input feature metadata is known."""
        raise NotImplementedError("Must implement _setup_layers in subclass.")

    @abc.abstractmethod
    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Fuse a list of feature tensors into one tensor."""
        raise NotImplementedError


class SequentialFusion(FusionModule, abc.ABC):
    """Base class for fusion modules that project features to a shared dimension."""

    def __init__(
        self,
        input_features: list[str],
        output_name: str,
        hidden_dim: int,
    ):
        """
        Args:
            input_features: List of feature names to fuse.
            output_name: Name of the output fused feature.
            hidden_dim: Dimension to project each input feature to before fusion.
        """
        input_specification = FusionInput(input_features=input_features)
        super().__init__(
            input_specification=input_specification, output_name=output_name
        )
        self.projections: nn.ModuleList | None = None
        self.hidden_dim = hidden_dim
        self._output_feature_type: str | None = None
        self._output_sequence_length: int | None = None

    def _setup_layers(self, feature_registry: dict[str, FeatureMetadata]):
        """Build projection layers for each input feature.

        Raises:
            ValueError: If inputs mix flat and sequential feature types, mix
                sequential lengths, or include spatial features. Mixed inputs
                would otherwise fail with an opaque shape error at the first
                forward pass.
        """
        input_dims: list[int] = []
        feature_types: dict[str, str] = {}
        sequence_lengths: dict[str, int] = {}
        for feat_name in self.input_features:
            metadata = feature_registry[feat_name]
            if metadata.feature_type == FeatureType.SPATIAL.value:
                raise ValueError(
                    f"SequentialFusion requires flat or sequential features, "
                    f"but '{feat_name}' is spatial with dimension {metadata.dimension}. "
                    f"Use SpatialFusion for spatial features."
                )
            feature_types[feat_name] = metadata.feature_type
            if metadata.feature_type == FeatureType.SEQUENTIAL.value:
                sequence_lengths[feat_name] = metadata.dimension[0]
            input_dims.append(metadata.dimension[-1])
        distinct_types = set(feature_types.values())
        if len(distinct_types) > 1:
            raise ValueError(
                f"{type(self).__name__} '{self.output_name}' requires all input "
                f"features to share one feature type, got {feature_types}."
            )
        distinct_lengths = set(sequence_lengths.values())
        if len(distinct_lengths) > 1:
            raise ValueError(
                f"{type(self).__name__} '{self.output_name}' requires sequential "
                f"inputs with equal sequence lengths, got {sequence_lengths}."
            )
        self._output_feature_type = distinct_types.pop()
        self._output_sequence_length = (
            distinct_lengths.pop() if distinct_lengths else None
        )
        self.projections = nn.ModuleList(
            [nn.Linear(dim, self.hidden_dim) for dim in input_dims]
        )
