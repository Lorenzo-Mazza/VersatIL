"""Modular feature projection utilities for decoders.

This module provides utilities for projecting features to a common dimension
before decoding. While projection layers can handle dimension mismatches,
it's recommended to use fusion modules in the encoding pipeline for better
semantic feature combination.
"""
import warnings

import torch
import torch.nn as nn


class FeatureProjection(nn.Module):
    """Projects features to a common embedding dimension.

    This utility handles features with mismatched dimensions by projecting
    each to a common embedding dimension. It supports both flat features
    (B, D), sequential features (B, T, D), and spatial features (B, C, H, W).

    Warning:
        While this module handles dimension mismatches gracefully, it's
        recommended to use fusion modules in the EncodingPipeline for better
        semantic feature combination. This projection is a fallback mechanism.

    Example:
        >>> # In a decoder
        >>> feature_projection = FeatureProjection(
        ...     embedding_dim=256,
        ...     warn_on_projection=True
        ... )
        >>>
        >>> # Caller filters which features to project
        >>> flat_features = {
        ...     "language": torch.randn(8, 64),      # 64-dim
        ...     "proprio": torch.randn(8, 128),      # 128-dim
        ... }
        >>> projected = feature_projection(flat_features)
        >>> projected["language"].shape  # (8, 256)
        >>> projected["proprio"].shape   # (8, 256)
    """

    def __init__(
        self,
        embedding_dim: int,
        warn_on_projection: bool = True,
        raise_on_mismatch: bool = False,
    ):
        """Initialize feature projection module.

        Args:
            embedding_dim: Target embedding dimension for all features
            warn_on_projection: Whether to warn when projecting features
            raise_on_mismatch: If True, raise error instead of projecting
                (forces user to handle in encoding pipeline)
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.warn_on_projection = warn_on_projection
        self.raise_on_mismatch = raise_on_mismatch

        # Lazy initialization of projection layers
        self.flat_projections = nn.ModuleDict()
        self.spatial_projections = nn.ModuleDict()
        self._warned_features: set[str] = set()  # Track warned features

        # Dummy buffer to track the module's device without relying on parameters (which may not exist yet in lazy init).
        # This ensures lazy-created layers are initialized on the correct device, preventing mismatches in multi-GPU or distributed setups.
        self.register_buffer('_device_tracker', torch.zeros(1))

    def _get_feature_dim(self, feature: torch.Tensor) -> int | tuple[int, ...]:
        """Get feature dimension from tensor shape."""
        if len(feature.shape) == 2:  # Flat (B, D)
            return feature.shape[-1]
        elif len(feature.shape) == 4:  # Spatial (B, C, H, W)
            return (feature.shape[1], feature.shape[2], feature.shape[3])
        elif len(feature.shape) == 3:  # Sequential (B, T, D)
            return feature.shape[-1]
        else:
            raise ValueError(
                f"Unsupported feature shape: {feature.shape}. "
                f"Expected (B, D) for flat, (B, C, H, W) for spatial, or (B, T, D) for sequential."
            )

    def _create_projection_layer(
        self,
        feature_name: str,
        feature: torch.Tensor,
    ) -> nn.Module:
        """Create appropriate projection layer for feature.

        Args:
            feature_name: Name of the feature
            feature: Feature tensor

        Returns:
            Projection layer (Linear for flat, Conv2d for spatial, or Identity)
        """
        feature_dim = self._get_feature_dim(feature)

        # Flat features (B, D) or Sequential (B, T, D)
        if isinstance(feature_dim, int):
            if feature_dim == self.embedding_dim:
                return nn.Identity()

            if self.raise_on_mismatch:
                raise ValueError(
                    f"Feature '{feature_name}' has dimension {feature_dim}, "
                    f"expected {self.embedding_dim}. Consider using a fusion module "
                    f"in the EncodingPipeline to handle dimension mismatches."
                )

            # Warn once per feature
            if self.warn_on_projection and feature_name not in self._warned_features:
                warnings.warn(
                    f"Feature '{feature_name}' has dimension {feature_dim}, projecting to "
                    f"{self.embedding_dim}. Consider using a fusion module in the EncodingPipeline "
                    f"for better semantic feature combination.",
                    UserWarning,
                    stacklevel=2
                )
                self._warned_features.add(feature_name)

            layer: nn.Module = nn.Linear(feature_dim, self.embedding_dim)
            # Move to same device as this module
            return layer.to(self._device_tracker.device)

        # Spatial features (B, C, H, W)
        else:
            channels = feature_dim[0]
            if channels == self.embedding_dim:
                return nn.Identity()

            if self.raise_on_mismatch:
                raise ValueError(
                    f"Spatial feature '{feature_name}' has {channels} channels, "
                    f"expected {self.embedding_dim}. Consider using a SpatialProjectionFusion "
                    f"module in the EncodingPipeline."
                )

            if self.warn_on_projection and feature_name not in self._warned_features:
                warnings.warn(
                    f"Spatial feature '{feature_name}' has {channels} channels, projecting to "
                    f"{self.embedding_dim}. Consider using a SpatialProjectionFusion module "
                    f"in the EncodingPipeline for better control.",
                    UserWarning,
                    stacklevel=2
                )
                self._warned_features.add(feature_name)

            layer = nn.Conv2d(channels, self.embedding_dim, kernel_size=1)
            # Move to same device as this module
            return layer.to(self._device_tracker.device)

    def forward(
        self,
        features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Project features to common embedding dimension.

        The caller controls which features to project by filtering the
        dictionary before passing it to this method.

        Args:
            features: Dictionary of features to project

        Returns:
            Dictionary of projected features with same keys
        """
        projected = {}

        for feature_name, feature in features.items():
            # Create projection layer if needed (lazy initialization)
            is_flat = len(feature.shape) in [2, 3]
            projection_dict = self.flat_projections if is_flat else self.spatial_projections

            if feature_name not in projection_dict:
                projection_dict[feature_name] = self._create_projection_layer(
                    feature_name, feature
                )

            # Apply projection
            projected[feature_name] = projection_dict[feature_name](feature)

        return projected

    def project_and_concatenate(
        self,
        features: dict[str, torch.Tensor],
        concatenation_dimension: int = -1,
    ) -> torch.Tensor:
        """Project features and concatenate them.

        The caller controls which features to include by filtering the
        dictionary before passing it to this method.

        Args:
            features: Dictionary of features to project and concatenate
            concatenation_dimension: Dimension to concatenate along (default: -1)

        Returns:
            Concatenated projected features
        """
        projected = self.forward(features)

        # Sort for deterministic order
        sorted_features = [projected[key] for key in sorted(projected.keys())]

        if len(sorted_features) == 0:
            raise ValueError("No features to concatenate")
        elif len(sorted_features) == 1:
            return sorted_features[0]
        else:
            return torch.cat(sorted_features, dim=concatenation_dimension)


class SpatialFeatureConcatenator(nn.Module):
    """Concatenates spatial features with automatic dimension matching.

    This module is specifically designed for concatenating spatial features
    (e.g., multi-camera setups) with mismatched channel dimensions.

    Example:
        >>> concatenator = SpatialFeatureConcatenator(
        ...     target_channels=256,
        ...     concat_dim=3  # Width concatenation for multi-camera
        ... )
        >>> features = {
        ...     "rgb_left": torch.randn(8, 2048, 7, 7),   # ResNet50
        ...     "rgb_right": torch.randn(8, 2048, 7, 7),  # ResNet50
        ...     "depth": torch.randn(8, 512, 7, 7)        # Lighter CNN
        ... }
        >>> concatenated = concatenator(features)
        >>> concatenated.shape  # (8, 256, 7, 21) - width tripled
    """

    def __init__(
        self,
        target_channels: int,
        concat_dim: int = 3,
        warn_on_projection: bool = True,
    ):
        """Initialize spatial feature concatenator.

        Args:
            target_channels: Target number of channels for all features
            concat_dim: Dimension to concatenate along (3 for width, 2 for height)
            warn_on_projection: Whether to warn when projecting channels
        """
        super().__init__()
        self.target_channels = target_channels
        self.concat_dim = concat_dim
        self.warn_on_projection = warn_on_projection
        self.projections = nn.ModuleDict()
        self._warned_features: set[str] = set()

        # Register a dummy buffer to track device
        self.register_buffer('_device_tracker', torch.zeros(1))

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate spatial features with automatic projection.

        Args:
            features: Dictionary of spatial features (B, C, H, W)

        Returns:
            Concatenated features (B, target_channels, H, W_total)
        """
        projected = []

        for feature_name in sorted(features.keys()):
            feature = features[feature_name]

            if len(feature.shape) != 4:
                raise ValueError(
                    f"Feature '{feature_name}' must be spatial (B, C, H, W), "
                    f"got shape {feature.shape}"
                )

            B, C, H, W = feature.shape

            # Create projection if needed
            if feature_name not in self.projections:
                if self.target_channels != C:
                    if self.warn_on_projection and feature_name not in self._warned_features:
                        warnings.warn(
                            f"Spatial feature '{feature_name}' has {C} channels, "
                            f"projecting to {self.target_channels}. Consider using "
                            f"SpatialProjectionFusion in EncodingPipeline.",
                            UserWarning,
                            stacklevel=2
                        )
                        self._warned_features.add(feature_name)

                    layer = nn.Conv2d(C, self.target_channels, kernel_size=1)
                    # Move to same device as this module
                    self.projections[feature_name] = layer.to(self._device_tracker.device)
                else:
                    self.projections[feature_name] = nn.Identity()

            projected.append(self.projections[feature_name](feature))

        # Concatenate along specified dimension
        return torch.cat(projected, dim=self.concat_dim)
