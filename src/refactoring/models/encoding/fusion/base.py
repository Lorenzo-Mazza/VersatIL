"""Fusion modules for combining multi-modal features."""
from abc import abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn

from refactoring.models.encoding.fusion.constants import FeatureType

"""Structured specifications for fusion modules."""


@dataclass
class FusionInput:
    """Structured input specification for fusion modules."""
    input_features: list[str]
    required_count: int = 1  # Minimum number of features required
    max_count: int | None = None  # Maximum number of features allowed (None = unlimited)
    feature_type: str = FeatureType.ANY.value  # Type of features expected


@dataclass
class FusionOutput:
    """Structured output specification for fusion modules."""
    output_name: str
    output_dim: int | tuple[int, ...]
    @property
    def is_spatial(self) -> bool:
        """Check if output has spatial dimensions."""
        return isinstance(self.output_dim, tuple) and len(self.output_dim) == 3
    @property
    def is_sequence(self) -> bool:
        """Check if output has sequence dimensions."""
        return isinstance(self.output_dim, tuple) and len(self.output_dim) == 2
    @property
    def is_flat(self) -> bool:
        """Check if output is flat (no spatial or sequence dimensions)."""
        if isinstance(self.output_dim, int):
            return True
        return len(self.output_dim) == 1


class FusionModule(nn.Module):
    """Base fusion module with validation."""
    def __init__(self,
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


    @abstractmethod
    def get_output_specification(self) -> FusionOutput:
        """Get structured output specification."""
        raise NotImplementedError


    def get_output_dim(self) -> int | tuple[int, ...]:
        """Get output dimension for backward compatibility."""
        return self.get_output_specification().output_dim


    def setup(self, feature_keys_to_dims: dict[str, int | tuple]):
        """Setup layers once feature dimensions are known.

        Note:
            This method is called once by the encoding pipeline after feature dimensions are known. This allows the user to
            create fusion modules without knowing the input feature dimensions ahead of time.

        Args:
            feature_keys_to_dims: Dict mapping available feature names to their dimensions
        """
        if self._initialized:
            return
        self._validate_feature_types(feature_keys_to_dims)
        self._setup_layers(feature_keys_to_dims)
        self._initialized = True


    def _validate_feature_types(self, feature_keys_to_dims: dict[str, int | tuple]):
        """Validate that input features match expected type.

        Args:
            feature_keys_to_dims: Dict mapping feature names to their dimensions

        Raises:
            ValueError: If feature types don't match constraints
        """
        if self.input_specification.feature_type == FeatureType.ANY.value:
            return
        for feat_name in self.input_features:
            dim = feature_keys_to_dims[feat_name]
            is_spatial = isinstance(dim, tuple) and len(dim) == 3
            if self.input_specification.feature_type == FeatureType.SPATIAL.value and not is_spatial:
                raise ValueError(
                    f"Feature '{feat_name}' has dimension {dim}, "
                    f"but {self.feature_type.value} fusion requires spatial features (C, H, W)"
                )
            elif self.input_specification.feature_type == FeatureType.SEQUENTIAL.value and is_spatial:
                raise ValueError(
                    f"Feature '{feat_name}' has spatial dimension {dim}, "
                    f"but {self.input_specification.feature_type} fusion requires sequential/flat features"
                )


    @abstractmethod
    def _setup_layers(self, feature_keys_to_dims: dict[str, int | tuple]):
        """Build layers once input feature dimensions are known."""
        raise NotImplementedError("Must implement _setup_layers in subclass.")



    @abstractmethod
    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


