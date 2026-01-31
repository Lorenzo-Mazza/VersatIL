"""Fusion modules for combining multi-modal features."""
import abc
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class FusionInput:
    """Structured input specification for fusion modules."""

    input_features: list[str]
    required_count: int = 1  # Minimum number of features required
    max_count: int | None = (
        None  # Maximum number of features allowed (None = unlimited)
    )


@dataclass
class FusionOutput:
    """Structured output specification for fusion modules."""

    output_name: str
    output_dim: int | tuple[int, ...]


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
        self._setup_layers(feature_keys_to_dims)
        self._initialized = True

    @abc.abstractmethod
    def _setup_layers(self, feature_keys_to_dims: dict[str, int | tuple]):
        """Build layers once input feature dimensions are known."""
        raise NotImplementedError("Must implement _setup_layers in subclass.")

    @abc.abstractmethod
    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class SequentialFusion(FusionModule, abc.ABC):
    """Base class for fusion modules that project features to a shared dimension"""

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

    def _setup_layers(self, feature_keys_to_dims: dict[str, int | tuple]):
        """Build projection layers for each input feature."""
        input_dims_raw = [feature_keys_to_dims[feat] for feat in self.input_features]
        input_dims: list[int] = []
        for feat_name, dim in zip(self.input_features, input_dims_raw):
            if isinstance(dim, tuple):
                if len(dim) > 2:
                    raise ValueError(
                        f"SequentialFusion requires flat or sequential dimensions, but '{feat_name}' has dimension {dim}. "
                        f"Use SpatialFusion for spatial features."
                    )
                proj_dim = dim[-1]
            else:
                proj_dim = dim
            input_dims.append(proj_dim)
        self.projections = nn.ModuleList(
            [nn.Linear(dim, self.hidden_dim) for dim in input_dims]
        )
