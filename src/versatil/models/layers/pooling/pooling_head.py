"""Pooling strategies for encoder feature extraction."""
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from versatil.models.encoding.encoders.constants import PoolingMethod
from versatil.models.layers import SpatialSoftmax, LearnedAggregation


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
    """Spatial softmax pooling on feature maps."""
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

class MaxPooling(PoolingHead):
    """Global max pooling for features."""

    def get_output_dim(self, input_channels: int) -> int:
        return input_channels

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.amax(features, dim=[2, 3])


class IdentityPooling(PoolingHead):
    """No pooling - returns features unchanged."""


    def __init__(self, channels: int):
        super().__init__()
        self.spatial_height = -1 # Unknown at initialization
        self.spatial_width = -1 # Unknown at initialization
        self.channels = channels


    def get_output_dim(self, input_channels: int) -> tuple[int, int, int]:
        return input_channels, self.spatial_height, self.spatial_width


    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features


class LearnedAggregationPooling(PoolingHead):
    """Learned aggregation of feature maps through attention."""
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.pooling_head = LearnedAggregation(ni=channels)

    def get_output_dim(self, input_channels: int) -> int:
        return self.channels

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.pooling_head(features)



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
    match pooling_method:
        case PoolingMethod.SPATIAL_SOFTMAX.value:
            return SpatialSoftmaxPooling(spatial_height, spatial_width, feature_channels)
        case PoolingMethod.AVERAGE.value:
            return GlobalAveragePooling()
        case PoolingMethod.MAX.value | PoolingMethod.DEFAULT.value:
            return MaxPooling()
        case PoolingMethod.NONE.value:
            return IdentityPooling(feature_channels)
        case PoolingMethod.LEARNED_AGGREGATION.value:
            return LearnedAggregationPooling(feature_channels)
        case _:
            raise ValueError(
                f"Unsupported pooling method: {pooling_method}.Supported: {[e.value for e in PoolingMethod]}" )
