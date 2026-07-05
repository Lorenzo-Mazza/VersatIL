import torch
from torch import nn

from versatil.models.layers.constants import AttentionDecompositionMode
from versatil.models.layers.geometric_attention.depth_decay import (
    DepthAwareDecayMask,
)
from versatil.models.layers.geometric_attention.spatial_decay import SpatialDecayMask
from versatil.models.layers.positional_encoding.rotary import (
    RasterRotaryPositionalEncoding2D,
    RotaryPositionalEncoding2D,
)


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
        number_of_heads: int,
        initial_decay: float = 5.0,
        decay_range: float = 3.0,
        base_frequency: float = 10000.0,
        use_raster_positions: bool = False,
    ):
        """Initializes geometric attention bias generator.

        Args:
            embedding_dimension: Embedding dimension.
            number_of_heads: Number of attention heads.
            initial_decay: Initial decay rate for spatial distance.
            decay_range: Range of decay rates across heads.
            base_frequency: Base frequency for rotary encoding.
            use_raster_positions: Whether to rotate by flattened raster grid
                positions (the DFormerv2 reference convention) instead of
                axis-split 2D positions. Required for pretrained DFormerv2
                checkpoints.
        """
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        rotary_class = (
            RasterRotaryPositionalEncoding2D
            if use_raster_positions
            else RotaryPositionalEncoding2D
        )
        self.rotary_encoding = rotary_class(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            base_frequency=base_frequency,
            learnable_frequencies=False,
        )

        self.spatial_decay = SpatialDecayMask(
            number_of_heads=number_of_heads,
            initial_decay=initial_decay,
            decay_range=decay_range,
        )

        self.depth_decay = DepthAwareDecayMask(number_of_heads=number_of_heads)

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
            )

        return (sine, cosine), combined_bias
