"""Feature projection torch.nn.Module.

Across the module, the following abbreviations are used:
- B: Batch size
- T: Temporal length (if applicable)
- C: Channels/ Original Feature dimension
- H: Height (for spatial features)
- W: Width (for spatial features)
- Emb: Embedding dimension
"""
import torch
import torch.nn as nn


class FeatureProjection(nn.Module):
    """Projects features to a common embedding dimension.

    It supports both flat features (B, C), sequential features (B, T, C), and spatial features
     (B, Optional[T], C, H, W).

    Example:
        >>> feature_projection = FeatureProjection(embedding_dim=256)
        >>> flat_features = {
        ...     "language": torch.randn(8, 64),
        ...     "proprio": torch.randn(8, 128),
        ... }
        >>> projected = feature_projection(flat_features)
        >>> projected["language"].shape  # (8, 256)
        >>> projected["proprio"].shape   # (8, 256)
    """

    def __init__(
        self,
        embedding_dim: int,
    ):
        """Initialize feature projection module.

        Args:
            embedding_dim: Target embedding dimension for all features
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        # These two dicts are doing exactly the same mathematical operation. But using linear projections
        # for spatial features would require transposing the tensors,  so we use conv2d instead.
        self.linear_projections = nn.ModuleDict()
        self.spatial_projections = nn.ModuleDict()
        # Dummy buffer to track the module's device without relying on parameters (which may not exist yet in lazy init).
        # This ensures lazy-created layers are initialized on the correct device, preventing mismatches in multi-GPU or distributed setups.
        self.register_buffer('_device_tracker', torch.zeros(1))


    def _create_projection_layer(
        self,
        feature: torch.Tensor,
    ) -> nn.Module:
        """Create projection layer for feature."""
        if len(feature.shape) < 4:  # flat (B, C) or sequential (B, T, C)
            channel_dim = feature.shape[-1]
            if channel_dim == self.embedding_dim:
                return nn.Identity()
            layer: nn.Module = nn.Linear(channel_dim, self.embedding_dim)
            return layer.to(self._device_tracker.device)
        else:
            if len(feature.shape) == 4:  # spatial (B, C, H, W)
                channel_dim = feature.shape[1]
            else:
                raise ValueError(f"Unsupported feature shape: {feature.shape}")
            if channel_dim == self.embedding_dim:
                return nn.Identity()
            layer = nn.Conv2d(channel_dim, self.embedding_dim, kernel_size=1)
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
            Dictionary of projected features with shape (B, Emb) or (B, T, Emb) or (B, Optional[T], Emb, H, W)
        """
        projected = {}
        for feature_name, feature in features.items():
            has_time = False
            B, T = None, None
            if feature.ndim == 5:
                has_time = True
                B, T, _, _, _= feature.shape
                feature = feature.reshape(B * T, *feature.shape[2:]) #(B*T, C, H, W)
            is_spatial = len(feature.shape) > 3
            projection_dict = self.spatial_projections if is_spatial else self.linear_projections
            if feature_name not in projection_dict:
                projection_dict[feature_name] = self._create_projection_layer(feature)
            feature_projection = projection_dict[feature_name](feature)
            if has_time:
                feature_projection = feature_projection.reshape(B, T, *feature_projection.shape[1:])
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


