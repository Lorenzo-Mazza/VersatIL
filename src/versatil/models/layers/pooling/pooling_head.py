"""Pooling strategies for spatial feature maps and token sequences."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from versatil.models.encoding.encoders.constants import PoolingMethod
from versatil.models.layers.pooling.attention_pooling import LearnedAggregation
from versatil.models.layers.pooling.spatial_softmax import SpatialSoftmax


class PoolingHead(nn.Module, ABC):
    """Abstract base class for pooling operations on spatial feature maps or token sequences.

    Args:
        input_dimension: Feature vector size, i.e. channel count for spatial
            feature maps (B, C, H, W), or hidden dimension for token
            sequences (B, S, D).
    """

    def __init__(self, input_dimension: int):
        super().__init__()
        self.input_dimension = input_dimension

    @property
    @abstractmethod
    def output_dim(self) -> int | tuple[int, ...]:
        """Output dimension after pooling."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply pooling to input features."""
        raise NotImplementedError


class SpatialSoftmaxPooling(PoolingHead):
    """Spatial softmax pooling on feature maps."""

    def __init__(self, input_dimension: int, spatial_height: int, spatial_width: int):
        """Initialize spatial softmax pooling.

        Args:
            input_dimension: Number of feature channels.
            spatial_height: Height of the feature map.
            spatial_width: Width of the feature map.
        """
        super().__init__(input_dimension=input_dimension)
        self.spatial_softmax = SpatialSoftmax(
            spatial_height, spatial_width, input_dimension
        )

    @property
    def output_dim(self) -> int:
        return self.input_dimension * 2

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.spatial_softmax(features)
        return result


class GlobalAveragePooling(PoolingHead):
    """Global average pooling over spatial dimensions."""

    @property
    def output_dim(self) -> int:
        return self.input_dimension

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features.mean(dim=[2, 3])


class MaxPooling(PoolingHead):
    """Global max pooling over spatial dimensions."""

    @property
    def output_dim(self) -> int:
        return self.input_dimension

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.amax(features, dim=[2, 3])


class SpatialIdentityPooling(PoolingHead):
    """No pooling — returns spatial feature maps unchanged.

    ``output_dim`` returns ``(C, -1, -1)`` where ``-1`` indicates dynamic
    spatial dimensions resolved at forward time from the actual feature map.
    """

    @property
    def output_dim(self) -> tuple[int, int, int]:
        return self.input_dimension, -1, -1

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features


class SpatialLearnedAggregationPooling(PoolingHead):
    """Learned aggregation of spatial feature maps through attention."""

    def __init__(self, input_dimension: int):
        super().__init__(input_dimension=input_dimension)
        self.learned_aggregation = LearnedAggregation(feature_dimension=input_dimension)

    @property
    def output_dim(self) -> int:
        return self.input_dimension

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.learned_aggregation(features)


class TokenPoolingHead(PoolingHead):
    """Pooling head for token sequences (B, S, D).

    Reduces a sequence of token embeddings to a single vector (B, D) via
    CLS token selection, mean pooling, or learned aggregation. With
    ``pooling_method=NONE``, returns the sequence with prefix tokens stripped.

    Args:
        input_dimension: Hidden dimension of the token embeddings.
        pooling_method: Pooling strategy from PoolingMethod enum.
        sequence_length: Fixed sequence length for NONE output dim (-1 for variable).
        num_prefix_tokens: Number of prefix tokens (CLS, registers) to exclude
            from AVERAGE, LEARNED_AGGREGATION, and NONE pooling. The first
            prefix token is still used for DEFAULT (CLS) pooling.
    """

    def __init__(
        self,
        input_dimension: int,
        pooling_method: str,
        sequence_length: int = -1,
        num_prefix_tokens: int = 0,
    ):
        super().__init__(input_dimension=input_dimension)
        self.pooling_method = pooling_method
        self.sequence_length = sequence_length
        self.num_prefix_tokens = num_prefix_tokens
        self.learned_aggregation: LearnedAggregation | None = None
        if pooling_method == PoolingMethod.LEARNED_AGGREGATION.value:
            self.learned_aggregation = LearnedAggregation(
                feature_dimension=input_dimension
            )

    @property
    def output_dim(self) -> int | tuple[int, int]:
        if self.pooling_method == PoolingMethod.NONE.value:
            sequence_length = (
                self.sequence_length - self.num_prefix_tokens
                if self.sequence_length != -1
                else -1
            )
            return sequence_length, self.input_dimension
        return self.input_dimension

    def _slice_padding_mask(
        self,
        padding_mask: torch.Tensor | None,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """Align an optional padding mask with pooled non-prefix tokens."""
        if padding_mask is None:
            return None
        expected_shape = hidden_states.shape[:2]
        if padding_mask.shape != expected_shape:
            raise ValueError(
                f"padding_mask must have shape {expected_shape}, got {padding_mask.shape}."
            )
        return padding_mask[:, self.num_prefix_tokens :].to(
            device=hidden_states.device,
            dtype=torch.bool,
        )

    def _masked_average(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Average valid tokens while ignoring padded positions."""
        if padding_mask is None:
            return tokens.mean(dim=1)
        valid_token_mask = ~padding_mask
        weights = valid_token_mask.unsqueeze(-1).to(dtype=tokens.dtype)
        summed_tokens = (tokens * weights).sum(dim=1)
        token_counts = weights.sum(dim=1).clamp(min=1)
        return summed_tokens / token_counts

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool token sequence.

        Args:
            hidden_states: Token embeddings of shape (B, S, D).
            padding_mask: Optional padding mask of shape (B, S), where True
                means padded. Used by pooling methods that aggregate tokens.

        Returns:
            Pooled features of shape (B, D) or (B, S', D) for NONE.
        """
        start = self.num_prefix_tokens
        match self.pooling_method:
            case PoolingMethod.DEFAULT.value:
                return hidden_states[:, 0]  # CLS token
            case PoolingMethod.AVERAGE.value:
                token_padding_mask = self._slice_padding_mask(
                    padding_mask=padding_mask,
                    hidden_states=hidden_states,
                )
                return self._masked_average(
                    tokens=hidden_states[:, start:],
                    padding_mask=token_padding_mask,
                )
            case PoolingMethod.LEARNED_AGGREGATION.value:
                token_padding_mask = self._slice_padding_mask(
                    padding_mask=padding_mask,
                    hidden_states=hidden_states,
                )
                return self.learned_aggregation(
                    hidden_states[:, start:],
                    padding_mask=token_padding_mask,
                )
            case PoolingMethod.NONE.value:
                return hidden_states[:, start:]
            case _:
                raise ValueError(
                    f"Unsupported token pooling method: {self.pooling_method}. "
                    f"Supported: {[e.value for e in PoolingMethod]}"
                )


def create_spatial_pooling_head(
    pooling_method: str,
    input_dimension: int,
    spatial_height: int,
    spatial_width: int,
) -> PoolingHead:
    """Create a pooling head for spatial feature maps (B, C, H, W).

    Args:
        pooling_method: Pooling strategy from PoolingMethod enum.
        input_dimension: Number of feature channels.
        spatial_height: Height of the feature map.
        spatial_width: Width of the feature map.

    Returns:
        Configured spatial pooling head.

    Raises:
        ValueError: If pooling_method is not supported for spatial features.
    """
    match pooling_method:
        case PoolingMethod.SPATIAL_SOFTMAX.value:
            return SpatialSoftmaxPooling(
                input_dimension=input_dimension,
                spatial_height=spatial_height,
                spatial_width=spatial_width,
            )
        case PoolingMethod.AVERAGE.value:
            return GlobalAveragePooling(input_dimension=input_dimension)
        case PoolingMethod.MAX.value | PoolingMethod.DEFAULT.value:
            return MaxPooling(input_dimension=input_dimension)
        case PoolingMethod.NONE.value:
            return SpatialIdentityPooling(input_dimension=input_dimension)
        case PoolingMethod.LEARNED_AGGREGATION.value:
            return SpatialLearnedAggregationPooling(input_dimension=input_dimension)
        case _:
            raise ValueError(
                f"Unsupported spatial pooling method: {pooling_method}. "
                f"Supported: {[e.value for e in PoolingMethod]}"
            )


def create_token_pooling_head(
    pooling_method: str,
    input_dimension: int,
    sequence_length: int = -1,
    num_prefix_tokens: int = 0,
) -> TokenPoolingHead:
    """Create a pooling head for token sequences (B, S, D).

    Args:
        pooling_method: Pooling strategy from PoolingMethod enum.
        input_dimension: Hidden dimension of the token embeddings.
        sequence_length: Fixed sequence length for NONE output dim (-1 for variable).
        num_prefix_tokens: Number of prefix tokens (CLS, registers) to strip.

    Returns:
        Configured token pooling head.
    """
    return TokenPoolingHead(
        input_dimension=input_dimension,
        pooling_method=pooling_method,
        sequence_length=sequence_length,
        num_prefix_tokens=num_prefix_tokens,
    )
