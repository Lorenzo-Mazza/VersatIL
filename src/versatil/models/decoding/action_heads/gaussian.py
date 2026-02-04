"""GaussianHead predicts Gaussian distribution parameters (mean, logvar)."""

import torch
from torch import nn

from versatil.models.decoding.action_heads.blocks import ActionHeadBlock
from versatil.models.decoding.action_heads import BaseActionHead
from versatil.models.decoding.constants import DecoderOutputKey


class GaussianHead(BaseActionHead):
    """Action head that outputs Gaussian distribution parameters (mean, logvar)."""

    def __init__(
        self,
        input_dim: int,
        blocks: list[ActionHeadBlock] | None = None,
        min_logvar: float = -10.0,
        max_logvar: float = 4.0,
    ):
        """Initialize Gaussian head.

        Args:
            input_dim: Input embedding dimension from decoder.
            blocks: Blocks to apply before output projection.
            min_logvar: Minimum value for logvar clamping.
            max_logvar: Maximum value for logvar clamping.
        """
        super().__init__(input_dim=input_dim, blocks=blocks)
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar
        self._logvar_proj: nn.Linear | None = None

    def set_output_dim(self, dim: int) -> None:
        """Create both mean and logvar projections.

        Args:
            dim: Output action dimension.
        """
        super().set_output_dim(dim)
        hidden_dim = self._get_hidden_dim()
        self._logvar_proj = nn.Linear(hidden_dim, dim)

    def forward(self, action_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass returning mean and clamped logvar.

        Args:
            action_embedding: (B, T, embedding_dim)

        Returns:
            Dict with "mean" and "logvar" keys.
        """
        if self.output_proj is None or self._logvar_proj is None:
            raise RuntimeError("output_dim not set. Call set_output_dim() first.")
        action_embedding = self._apply_blocks(action_embedding)
        mean = self.output_proj(action_embedding)
        logvar = self._logvar_proj(action_embedding)
        logvar = logvar.clamp(min=self.min_logvar, max=self.max_logvar)
        return {
            DecoderOutputKey.MEAN.value: mean,
            DecoderOutputKey.LOGVAR.value: logvar,
        }