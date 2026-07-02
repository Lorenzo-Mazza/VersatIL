import torch
from torch import nn
from torch.nn import functional as F

from versatil.models.layers.constants import AttentionDecompositionMode, Axis


class DepthAwareDecayMask(nn.Module):
    """Generates attention decay based on depth discontinuities.

    Reduces attention across object boundaries by penalizing
    depth differences between positions.
    """

    def __init__(self, num_heads: int):
        """Initializes depth-aware decay mask.

        Args:
            num_heads: Number of attention heads.
        """
        super().__init__()
        self.num_heads = num_heads

    @staticmethod
    def compute_depth_difference_matrix(
        depth_map: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """Computes pairwise depth differences.

        Args:
            depth_map: Depth values of shape (B, 1, H, W).
            height: Target height.
            width: Target width.

        Returns:
            Depth difference matrix of shape (B, H*W, H*W).
        """
        depth_map = F.interpolate(
            depth_map, size=(height, width), mode="bilinear", align_corners=False
        )

        batch_size = depth_map.shape[0]
        depth_flat = depth_map.reshape(batch_size, height * width, 1)
        depth_differences = depth_flat[:, :, None, :] - depth_flat[:, None, :, :]
        depth_differences = depth_differences.abs().sum(dim=-1)
        return depth_differences

    @staticmethod
    def compute_1d_depth_difference_matrix(
        depth_map: torch.Tensor, axis: str, height: int, width: int
    ) -> torch.Tensor:
        """Computes depth differences along one axis.

        Args:
            depth_map: Depth values of shape (B, 1, H, W).
            axis: Either 'height' or 'width'.
            height: Target height.
            width: Target width.

        Returns:
            Depth differences of shape (B, secondary_length, primary_length, primary_length).
        """
        depth_map = F.interpolate(
            depth_map, size=(height, width), mode="bilinear", align_corners=False
        )
        if axis == Axis.HEIGHT.value:
            depth_map = depth_map.transpose(-2, -1)
        depth_differences = depth_map[:, :, :, :, None] - depth_map[:, :, :, None, :]
        depth_differences = depth_differences.abs()
        return depth_differences.squeeze(1)

    def forward(
        self,
        depth_map: torch.Tensor,
        height: int,
        width: int,
        decay_rates: torch.Tensor,
        decomposition_mode: str = AttentionDecompositionMode.FULL.value,
    ) -> tuple[torch.Tensor, ...]:
        """Generates depth-aware decay mask(s).

        Args:
            depth_map: Depth map of shape (B, 1, H, W).
            height: Target height.
            width: Target width.
            decay_rates: Per-head decay rates of shape (num_heads,).
            decomposition_mode: Whether to generate full or separable masks.

        Returns:
            If FULL: Single mask of shape (B, num_heads, H*W, H*W).
            If SEPARABLE: Tuple of (height_mask, width_mask).
        """
        if decomposition_mode == AttentionDecompositionMode.SEPARABLE.value:
            height_depth_diffs = self.compute_1d_depth_difference_matrix(
                depth_map, axis=Axis.HEIGHT.value, height=height, width=width
            )
            width_depth_diffs = self.compute_1d_depth_difference_matrix(
                depth_map, axis=Axis.WIDTH.value, height=height, width=width
            )

            height_mask = (
                height_depth_diffs.unsqueeze(1) * decay_rates[None, :, None, None, None]
            )
            width_mask = (
                width_depth_diffs.unsqueeze(1) * decay_rates[None, :, None, None, None]
            )

            return height_mask, width_mask
        else:
            depth_diffs = self.compute_depth_difference_matrix(depth_map, height, width)
            mask = depth_diffs.unsqueeze(1) * decay_rates[None, :, None, None]
            return (mask,)
