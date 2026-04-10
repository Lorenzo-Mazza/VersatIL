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

    def _setup_layers(self, feature_registry: dict[str, FeatureMetadata]):
        """Build projection layers for each input feature."""
        input_dims: list[int] = []
        has_sequential = False
        has_flat = False
        for feat_name in self.input_features:
            metadata = feature_registry[feat_name]
            if metadata.feature_type == FeatureType.SPATIAL.value:
                raise ValueError(
                    f"SequentialFusion requires flat or sequential features, "
                    f"but '{feat_name}' is spatial with dimension {metadata.dimension}. "
                    f"Use SpatialFusion for spatial features."
                )
            if metadata.feature_type == FeatureType.SEQUENTIAL.value:
                has_sequential = True
            else:
                has_flat = True
            input_dims.append(metadata.dimension[-1])
        if has_sequential and has_flat:
            raise ValueError(
                f"SequentialFusion cannot mix flat and sequential features. "
                f"All inputs must be the same type. "
                f"Input features: {self.input_features}"
            )
        self._output_feature_type = (
            FeatureType.SEQUENTIAL.value if has_sequential else FeatureType.FLAT.value
        )
        self.projections = nn.ModuleList(
            [nn.Linear(dim, self.hidden_dim) for dim in input_dims]
        )
