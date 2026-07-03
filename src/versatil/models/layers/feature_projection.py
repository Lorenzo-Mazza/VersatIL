"""Feature projection torch.nn.Module.

Across the module, the following abbreviations are used:
- B: Batch size
- T: Temporal length (if applicable)
- C: Channels/ Original Feature dimension
- H: Height (for spatial features)
- W: Width (for spatial features)
- Emb: Embedding dimension
"""

import logging

import torch
import torch.nn as nn

from versatil.common.module_attr_mixin import ModuleAttrMixin


class FeatureProjection(ModuleAttrMixin):
    """Projects features to a common embedding dimension.

    It supports both flat features (B, C), sequential features (B, T, C), and spatial features
     (B, Optional[T], C, H, W).

    This module uses lazy initialization - projection layers are created on first forward pass.
    To support loading checkpoints, it overrides _load_from_state_dict to create layers
    dynamically from the state dict.

    Example:
        ```
        feature_projection = FeatureProjection(embedding_dimension=256)
        >>> flat_features = {
        ...     "language": torch.randn(8, 64),
        ...     "proprio": torch.randn(8, 128),
        ... }
        >>> projected = feature_projection(flat_features)
        >>> projected["language"].shape  # (8, 256)
        >>> projected["proprio"].shape   # (8, 256)
        ```
    """

    def __init__(
        self,
        embedding_dimension: int,
        has_time_dim: bool = False,
    ):
        """Initialize feature projection module.

        Args:
            embedding_dimension: Target embedding dimension for all features
            has_time_dim: Whether features may have a time dimension (default: False)
        """
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.has_time_dim = has_time_dim
        # These two dicts are doing exactly the same mathematical operation. But using linear projections
        # for spatial features would require transposing the tensors,  so we use conv2d instead.
        self.linear_projections = nn.ModuleDict()
        self.spatial_projections = nn.ModuleDict()

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list,
        unexpected_keys: list,
        error_msgs: list,
    ) -> None:
        """Load state dict with dynamic layer creation for lazy-initialized projections.

        This method creates projection layers on-the-fly from checkpoint weights,
        enabling proper loading even when layers don't exist yet due to lazy initialization.
        """
        logging.info(
            msg=f"Feature Projection._load_from_state_dict called with prefix='{prefix}'"
        )
        logging.info(msg=f"  state_dict keys: {list(state_dict.keys())[:5]}...")
        linear_prefix = prefix + "linear_projections."
        spatial_prefix = prefix + "spatial_projections."
        linear_features: dict[str, dict[str, torch.Tensor]] = {}
        spatial_features: dict[str, dict[str, torch.Tensor]] = {}
        for key, value in state_dict.items():
            if key.startswith(linear_prefix):
                suffix = key[len(linear_prefix) :]
                parts = suffix.split(".")
                if len(parts) == 2:
                    feature_name, param_name = parts
                    if feature_name not in linear_features:
                        linear_features[feature_name] = {}
                    linear_features[feature_name][param_name] = value
            elif key.startswith(spatial_prefix):
                suffix = key[len(spatial_prefix) :]
                parts = suffix.split(".")
                if len(parts) == 2:
                    feature_name, param_name = parts
                    if feature_name not in spatial_features:
                        spatial_features[feature_name] = {}
                    spatial_features[feature_name][param_name] = value
        device = self.device
        dtype = self.dtype
        for feature_name, params in linear_features.items():
            if feature_name not in self.linear_projections and "weight" in params:
                weight = params["weight"]
                out_features, in_features = weight.shape
                self.linear_projections[feature_name] = nn.Linear(
                    in_features, out_features, device=device, dtype=dtype
                )

        for feature_name, params in spatial_features.items():
            if feature_name not in self.spatial_projections and "weight" in params:
                weight = params["weight"]
                out_channels, in_channels, _, _ = weight.shape
                self.spatial_projections[feature_name] = nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, device=device, dtype=dtype
                )

        # Now parent can load weights into the newly created layers
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _create_projection_layer(
        self,
        feature: torch.Tensor,
    ) -> nn.Module:
        """Create projection layer for feature."""
        if len(feature.shape) < 4:  # flat (B, C) or sequential (B, T, C)
            channel_dim = feature.shape[-1]
            if channel_dim == self.embedding_dimension:
                return nn.Identity()
            layer: nn.Module = nn.Linear(channel_dim, self.embedding_dimension)
            return layer.to(device=self.device, dtype=self.dtype)
        else:
            if len(feature.shape) == 4:  # spatial (B, C, H, W)
                channel_dim = feature.shape[1]
            else:
                raise ValueError(f"Unsupported feature shape: {feature.shape}")
            if channel_dim == self.embedding_dimension:
                return nn.Identity()
            layer = nn.Conv2d(channel_dim, self.embedding_dimension, kernel_size=1)
            return layer.to(device=self.device, dtype=self.dtype)

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
            Dictionary of projected features with shape (B, Emb) or (B, T, Emb) or (B, Optional[T], Emb, H, W)
        """
        projected = {}
        for feature_name, feature in features.items():
            B, T = None, None
            if feature.ndim == 5:
                B, T, _, _, _ = feature.shape
                feature = feature.reshape(B * T, *feature.shape[2:])  # (B*T, C, H, W)
                is_spatial = True
            elif self.has_time_dim and feature.ndim == 4:
                B, T, _, _ = feature.shape
                feature = feature.reshape(B * T, *feature.shape[2:])
                is_spatial = False
            else:
                is_spatial = len(feature.shape) > 3
            projection_dict = (
                self.spatial_projections if is_spatial else self.linear_projections
            )
            if feature_name not in projection_dict:
                projection_dict[feature_name] = self._create_projection_layer(feature)
            feature_projection = projection_dict[feature_name](feature)
            # Restore whenever the input was flattened: a 5D input carries a
            # time dimension regardless of the has_time_dim flag, and folding
            # it into the batch would leak B*T downstream.
            if B is not None and T is not None:
                feature_projection = feature_projection.reshape(
                    B, T, *feature_projection.shape[1:]
                )
            projected[feature_name] = feature_projection
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
        sorted_tensors = [projected[key] for key in sorted(projected.keys())]
        if len(sorted_tensors) == 0:
            raise ValueError("No features to concatenate")
        elif len(sorted_tensors) == 1:
            return sorted_tensors[0]
        else:
            # Ensure tensors have the same shape except for concatenation dimension
            base_shape = list(sorted_tensors[0].shape)
            for tensor in sorted_tensors[1:]:
                for dim in range(len(base_shape)):
                    if dim == concatenation_dimension:
                        continue
                    if tensor.shape[dim] != base_shape[dim]:
                        raise ValueError(
                            f"Feature shapes do not match for concatenation: "
                            f"{base_shape} vs {tensor.shape} at dim {dim}"
                        )
            return torch.cat(sorted_tensors, dim=concatenation_dimension)
