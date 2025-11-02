
import torch
from torch import nn

from refactoring.models.layers.constants import AttentionDecompositionMode


class SpatialDecayMask(nn.Module):
    """Generates attention decay based on spatial distance between positions.

    Farther positions receive exponentially decaying attention weights,
    with per-head decay rates allowing different receptive fields.
    """


    def __init__(
            self,
            num_heads: int,
            initial_decay: float = 5.0,
            decay_range: float = 3.0
    ):
        """Initializes spatial decay mask generator.

        Args:
            num_heads: Number of attention heads.
            initial_decay: Initial decay rate.
            decay_range: Range of decay rates across heads.
        """
        super().__init__()
        self.num_heads = num_heads

        decay_rates = self._compute_per_head_decay(
            num_heads=num_heads,
            initial_decay=initial_decay,
            decay_range=decay_range
        )
        self.register_buffer("decay_rates", decay_rates)


    @staticmethod
    def _compute_per_head_decay(
            num_heads: int,
            initial_decay: float,
            decay_range: float
    ) -> torch.Tensor:
        """Computes per-head decay rates.

        Different heads learn different spatial ranges:
        - Some heads attend locally (large decay)
        - Some heads attend globally (small decay)

        Args:
            num_heads: Number of attention heads.
            initial_decay: Starting decay value.
            decay_range: Range of decay across heads.

        Returns:
            Decay rates of shape (num_heads,).
        """
        head_indices = torch.arange(num_heads, dtype=torch.float)
        decay_offsets = decay_range * head_indices / num_heads
        decay_rates = torch.log(1 - 2 ** (-initial_decay - decay_offsets))
        return decay_rates


    def compute_2d_distance_matrix(
            self,
            height: int,
            width: int
    ) -> torch.Tensor:
        """Computes pairwise Manhattan distances for 2D grid.

        Args:
            height: Grid height.
            width: Grid width.

        Returns:
            Distance matrix of shape (H*W, H*W).
        """
        height_indices = torch.arange(height, device=self.decay_rates.device)
        width_indices = torch.arange(width, device=self.decay_rates.device)

        grid = torch.stack(torch.meshgrid(height_indices, width_indices, indexing='ij'), dim=-1)
        grid_flat = grid.reshape(height * width, 2)

        distance_matrix = grid_flat[:, None, :] - grid_flat[None, :, :]
        distance_matrix = distance_matrix.abs().sum(dim=-1)

        return distance_matrix


    def compute_1d_distance_matrix(self, length: int) -> torch.Tensor:
        """Computes pairwise distances for 1D sequence.

        Args:
            length: Sequence length.

        Returns:
            Distance matrix of shape (length, length).
        """
        indices = torch.arange(length, device=self.decay_rates.device)
        distance_matrix = (indices[:, None] - indices[None, :]).abs()
        return distance_matrix


    def forward(
            self,
            height: int,
            width: int,
            decomposition_mode: str = AttentionDecompositionMode.FULL.value
    ) -> tuple[torch.Tensor, ...]:
        """Generates spatial decay mask(s).

        Args:
            height: Grid height.
            width: Grid width.
            decomposition_mode: Whether to generate full or separable masks.

        Returns:
            If FULL: Single mask of shape (num_heads, H*W, H*W).
            If SEPARABLE: Tuple of (height_mask, width_mask) each (num_heads, H, H) and (num_heads, W, W).
        """
        if decomposition_mode == AttentionDecompositionMode.SEPARABLE.value:
            height_distances = self.compute_1d_distance_matrix(height)
            width_distances = self.compute_1d_distance_matrix(width)

            height_mask = height_distances * self.decay_rates[:, None, None]
            width_mask = width_distances * self.decay_rates[:, None, None]

            return height_mask, width_mask
        else:
            distances = self.compute_2d_distance_matrix(height, width)
            mask = distances * self.decay_rates[:, None, None]
            return (mask,)
