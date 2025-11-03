
import torch
from torch import nn

from refactoring.models.encoding.fusion.base import (
    FusionInput,
    FusionModule,
    FusionOutput,
)
from refactoring.models.encoding.fusion.constants import FeatureType


class SpatialFusion(FusionModule):
    """Preserves spatial structure while combining feature maps.
    Only works with features that have spatial dimensions (H, W).
    """
    def __init__(
            self,
            input_features: list[str],
            output_name: str,
            hidden_dim: int,
    ):
        input_specification = FusionInput(input_features=input_features, feature_type=FeatureType.SPATIAL.value)
        super().__init__(input_specification=input_specification, output_name=output_name)
        self.hidden_dim = hidden_dim


    def _setup_layers(self, feature_dims: dict[str, int | tuple]):
        """Build projection layers and validate spatial dimensions.

        Args:
            feature_dims: Dict mapping feature names to (C, H, W) tuples

        Raises:
            ValueError: If spatial dimensions don't match across features
        """
        input_dims_raw = [feature_dims[feat] for feat in self.input_features]
        # Validate all features have same spatial dimensions
        # Assert all dimensions are tuples (spatial features)
        input_dims: list[tuple] = []
        for dim in input_dims_raw:
            if isinstance(dim, int):
                raise ValueError(f"Expected spatial features (tuple dimensions), got flat feature with dim={dim}")
            input_dims.append(dim)
        # Type narrowing: now mypy knows input_dims contains only tuples
        spatial_dims = [dim[1:] for dim in input_dims]  # Extract (H, W)
        if not all(s == spatial_dims[0] for s in spatial_dims):
            raise ValueError(
                f"All input features must have same spatial dimensions. "
                f"Got: {dict(zip(self.input_features, spatial_dims))}"
            )
        self.spatial_dims = spatial_dims[0]
        self.projections = nn.ModuleList([ # Create 1x1 convolution projections to common dimension
            nn.Conv2d(in_dim[0], self.hidden_dim, kernel_size=1)
            for in_dim in input_dims
        ])


    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of feature maps [B, C_i, H_i, W_i] or [B, T, C_i, H_i, W_i]

        Returns:
            Fused feature map [B, hidden_dim * num_features, H, W] or [B, T, hidden_dim * num_features, H, W]
        """
        has_time = features[0].dim() == 5
        if has_time:
            B, T = features[0].shape[:2]
            features = [feat.reshape(B * T, *feat.shape[2:]) for feat in features]

        projected = []
        for feat, proj in zip(features, self.projections):
            proj_feat = proj(feat)  # [B, hidden_dim, H, W] or [B*T, hidden_dim, H, W]
            projected.append(proj_feat)
        fused = torch.cat(projected, dim=1)  # [B, hidden_dim * num_features, H, W] or [B*T, hidden_dim * num_features, H, W]

        if has_time:
            fused = fused.reshape(B, T, *fused.shape[1:])  # [B, T, hidden_dim * num_features, H, W]

        return fused


    def get_output_specification(self) -> FusionOutput:
        """Get output specification."""
        output_dim = (self.hidden_dim * len(self.input_features), *self.spatial_dims)
        return FusionOutput(output_name=self.output_name, output_dim=output_dim,)






