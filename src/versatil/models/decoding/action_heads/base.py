"""Base contract for configurable action heads."""

from abc import ABC, abstractmethod

import torch
from torch import nn

from versatil.models.decoding.action_heads.blocks import ActionHeadBlock


class BaseActionHead(ABC, nn.Module):
    """Abstract base class for action heads with block-based processing and output projection.

    The output dimension is set lazily via set_output_dim() because action heads
    are instantiated from config with only the embedding dimension known. The output
    dimension depends on the ActionSpace (resolved by ActionDecoder during policy
    assembly) or the tokenizer vocabulary size (for tokenized decoders).

    Subclasses must implement forward() with their specific return type.
    """

    def __init__(
        self,
        input_dimension: int,
        blocks: list[ActionHeadBlock] | None = None,
    ) -> None:
        """Initialize base action head.

        Args:
            input_dimension: Input embedding dimension from decoder.
            blocks: Blocks to apply before output projection.
        """
        super().__init__()
        self.input_dimension = input_dimension
        self._output_dim: int | None = None
        if blocks is None:
            blocks = []
        self.blocks = nn.ModuleList(blocks)
        self.output_proj: nn.Linear | None = None

    @property
    def output_dim(self) -> int:
        """Get output dimension. Raises if not set."""
        if self._output_dim is None:
            raise RuntimeError("output_dim not set. Call set_output_dim() first.")
        return self._output_dim

    @output_dim.setter
    def output_dim(self, value: int) -> None:
        """Set the head output dimension."""
        self._output_dim = value

    def _get_hidden_dim(self) -> int:
        """Get output dimension of the last block, or input_dimension if no blocks."""
        return (
            self.input_dimension
            if len(self.blocks) == 0
            else self.blocks[-1].output_dim
        )

    def set_output_dim(self, dim: int) -> None:
        """Set output dimension and create output projection layer.

        Args:
            dim: Output action dimension.
        """
        self._output_dim = dim
        hidden_dimension = self._get_hidden_dim()
        self.output_proj = nn.Linear(hidden_dimension, dim)

    def _apply_blocks(self, action_embedding: torch.Tensor) -> torch.Tensor:
        """Apply all blocks to the input embedding."""
        for block in self.blocks:
            action_embedding = block(action_embedding)
        return action_embedding

    @abstractmethod
    def forward(
        self,
        action_embedding: torch.Tensor,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Forward pass. Subclasses define return type."""
        raise NotImplementedError
