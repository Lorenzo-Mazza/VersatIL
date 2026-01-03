import torch
from torch import nn

from refactoring.models.layers import RotaryPositionalEncoding2D
from refactoring.models.layers.constants import AttentionDecompositionMode
from refactoring.models.layers.geometric_attention.depth_decay import (
    DepthAwareDecayMask,
)
from refactoring.models.layers.geometric_attention.spatial_decay import SpatialDecayMask


class GeometricAttentionBias(nn.Module):
    """Combines spatial and depth-aware attention biases.

    Generates geometry-aware attention priors that:
    1. Decay with spatial distance
    2. Penalize depth discontinuities
    3. Allow learnable weighting between the two
    """

    def __init__(
        self,
        embedding_dimension: int,
        num_heads: int,
        initial_decay: float = 5.0,
        decay_range: float = 3.0,
        base_frequency: float = 10000.0,
    ):
        """Initializes geometric attention bias generator.

        Args:
            embedding_dimension: Embedding dimension.
            num_heads: Number of attention heads.
            initial_decay: Initial decay rate for spatial distance.
            decay_range: Range of decay rates across heads.
            base_frequency: Base frequency for rotary encoding.
        """
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.num_heads = num_heads
        self.rotary_encoding = RotaryPositionalEncoding2D(
            embedding_dimension=embedding_dimension,
            num_heads=num_heads,
            base_frequency=base_frequency,
            learnable_frequencies=False,
        )

        self.spatial_decay = SpatialDecayMask(
            num_heads=num_heads, initial_decay=initial_decay, decay_range=decay_range
        )

        self.depth_decay = DepthAwareDecayMask(num_heads=num_heads)

        self.bias_weights = nn.Parameter(torch.ones(2, 1, 1, 1), requires_grad=True)

    def forward(
        self,
        height: int,
        width: int,
        depth_map: torch.Tensor,
        device: torch.device,
        decomposition_mode: str = AttentionDecompositionMode.FULL.value,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, ...]]:
        """Generates complete geometric attention bias.

        Args:
            height: Grid height.
            width: Grid width.
            depth_map: Depth map of shape (B, 1, H, W).
            device: Computation device.
            decomposition_mode: Full or separable attention.

        Returns:
            Tuple of ((sine, cosine), bias_masks) where:
            - sine, cosine: Rotary encoding components
            - bias_masks: Either single mask or (height_mask, width_mask)
        """
        sine, cosine = self.rotary_encoding.compute_rotation_components(
            height=height, width=width
        )

        spatial_masks = self.spatial_decay(
            height=height, width=width, decomposition_mode=decomposition_mode
        )

        depth_masks = self.depth_decay(
            depth_map=depth_map,
            height=height,
            width=width,
            decay_rates=self.spatial_decay.decay_rates,
            decomposition_mode=decomposition_mode,
        )

        combined_bias: tuple[torch.Tensor, torch.Tensor] | torch.Tensor
        if decomposition_mode == AttentionDecompositionMode.SEPARABLE.value:
            height_bias = (
                self.bias_weights[0] * spatial_masks[0].unsqueeze(0).unsqueeze(2)
                + self.bias_weights[1] * depth_masks[0]
            )
            width_bias = (
                self.bias_weights[0] * spatial_masks[1].unsqueeze(0).unsqueeze(2)
                + self.bias_weights[1] * depth_masks[1]
            )
            combined_bias = (height_bias, width_bias)
        else:
            combined_bias = (
                self.bias_weights[0] * spatial_masks[0]
                + self.bias_weights[1] * depth_masks[0],
            )  # type: ignore[assignment]

        return (sine, cosine), combined_bias  # type: ignore[return-value]
