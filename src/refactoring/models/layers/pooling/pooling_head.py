"""Pooling strategies for encoder feature extraction."""
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from refactoring.models.encoding.encoders.constants import PoolingMethod
from refactoring.models.layers import SpatialSoftmax


class PoolingHead(nn.Module, ABC):
    """Abstract base class for pooling operations."""


    @abstractmethod
    def get_output_dim(self, input_channels: int) -> int | tuple[int, ...]:
        """Return output dimension after pooling.

        Args:
            input_channels: Number of input feature channels

        Returns:
            Output dimension (int for 1D, tuple for spatial dimensions)
        """
        raise NotImplementedError


    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply pooling to features.

        Args:
            features: Input features of shape (B, C, H, W)

        Returns:
            Pooled features
        """
        raise NotImplementedError


class SpatialSoftmaxPooling(PoolingHead):
    """Spatial softmax pooling for features."""


    def __init__(self, spatial_height: int, spatial_width: int, channels: int):
        super().__init__()
        self.spatial_softmax = SpatialSoftmax(spatial_height, spatial_width, channels)
        self.channels = channels


    def get_output_dim(self, input_channels: int) -> int:
        return input_channels * 2


    def forward(self, features: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.spatial_softmax(features)
        return result


class GlobalAveragePooling(PoolingHead):
    """Global average pooling for features."""


    def get_output_dim(self, input_channels: int) -> int:
        return input_channels


    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features.mean(dim=[2, 3])


class IdentityPooling(PoolingHead):
    """No pooling - returns features unchanged."""


    def __init__(self, spatial_height: int, spatial_width: int, channels: int):
        super().__init__()
        self.spatial_height = spatial_height
        self.spatial_width = spatial_width
        self.channels = channels


    def get_output_dim(self, input_channels: int) -> tuple[int, int, int]:
        return input_channels, self.spatial_height, self.spatial_width


    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features


def create_pooling_head(
        pooling_method: str,
        feature_channels: int,
        spatial_height: int,
        spatial_width: int,
) -> PoolingHead:
    """Factory function to create pooling heads.

    Args:
        pooling_method: Pooling method from PoolingMethod enum
        feature_channels: Number of feature channels
        spatial_height: Spatial height of feature map
        spatial_width: Spatial width of feature map

    Returns:
        Configured pooling head

    Raises:
        ValueError: If pooling_method is not supported
    """
    if pooling_method == PoolingMethod.SPATIAL_SOFTMAX.value:
        return SpatialSoftmaxPooling(spatial_height, spatial_width, feature_channels)
    elif pooling_method == PoolingMethod.GLOBAL_AVERAGE.value:
        return GlobalAveragePooling()
    elif pooling_method == PoolingMethod.NONE.value:
        return IdentityPooling(spatial_height, spatial_width, feature_channels)
    else:
        raise ValueError(
            f"Unsupported pooling method: {pooling_method}. "
            f"Supported: {[e.value for e in PoolingMethod]}"
        )
