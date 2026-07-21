"""GaussianHead predicts Gaussian distribution parameters (mean, logvar)."""

import torch
from torch import nn

from versatil.models.decoding.action_heads import BaseActionHead
from versatil.models.decoding.action_heads.blocks import ActionHeadBlock
from versatil.models.decoding.constants import DecoderOutputKey


class GaussianHead(BaseActionHead):
    """Action head that outputs Gaussian distribution parameters (mean, logvar)."""

    def __init__(
        self,
        input_dimension: int,
        blocks: list[ActionHeadBlock] | None = None,
        min_logvar: float = -10.0,
        max_logvar: float = 4.0,
    ) -> None:
        """Initialize Gaussian head.

        Args:
            input_dimension: Input embedding dimension from decoder.
            blocks: Blocks to apply before output projection.
            min_logvar: Minimum value for logvar clamping.
            max_logvar: Maximum value for logvar clamping.
        """
        super().__init__(input_dimension=input_dimension, blocks=blocks)
        self.register_buffer(
            "_min_logvar",
            torch.tensor(min_logvar, dtype=torch.float32),
        )
        self.register_buffer(
            "_max_logvar",
            torch.tensor(max_logvar, dtype=torch.float32),
        )
        self._logvar_proj: nn.Linear | None = None
        self.temporal_bias: nn.Parameter | None = None

    @property
    def min_logvar(self) -> float:
        """Return the lower log-variance clamp."""
        return float(self._min_logvar.item())

    @min_logvar.setter
    def min_logvar(self, value: float) -> None:
        self._min_logvar.fill_(value)

    @property
    def max_logvar(self) -> float:
        """Return the upper log-variance clamp."""
        return float(self._max_logvar.item())

    @max_logvar.setter
    def max_logvar(self, value: float) -> None:
        self._max_logvar.fill_(value)

    def set_output_dim(self, dim: int) -> None:
        """Create both mean and logvar projections.

        Args:
            dim: Output action dimension.
        """
        super().set_output_dim(dim)
        hidden_dimension = self._get_hidden_dim()
        self._logvar_proj = nn.Linear(hidden_dimension, dim)

    def enable_temporal_bias(self, horizon: int) -> None:
        """Create a zero-initialized per-timestep bias added to the mean.

        Gives each timestep an independent mean offset, so a bias-only
        initialization can hold a full trajectory instead of a constant.

        Args:
            horizon: Number of timesteps the head is applied to per forward.

        Raises:
            RuntimeError: If set_output_dim() has not been called yet.
        """
        if self._output_dim is None:
            raise RuntimeError("output_dim not set. Call set_output_dim() first.")
        self.temporal_bias = nn.Parameter(torch.zeros(horizon, self._output_dim))

    def forward(self, action_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass returning mean and clamped logvar.

        Args:
            action_embedding: (B, T, embedding_dimension)

        Returns:
            Dict with "mean" and "logvar" keys.
        """
        if self.output_proj is None or self._logvar_proj is None:
            raise RuntimeError("output_dim not set. Call set_output_dim() first.")
        action_embedding = self._apply_blocks(action_embedding)
        mean = self.output_proj(action_embedding)
        if self.temporal_bias is not None:
            mean = mean + self.temporal_bias
        logvar = self._logvar_proj(action_embedding)
        logvar = torch.maximum(logvar, self._min_logvar.to(logvar))
        logvar = torch.minimum(logvar, self._max_logvar.to(logvar))
        return {
            DecoderOutputKey.MEAN.value: mean,
            DecoderOutputKey.LOGVAR.value: logvar,
        }
