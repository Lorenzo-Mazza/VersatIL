"""ActionHead class for composing blocks into action prediction heads."""

import torch
import torch.nn as nn

from versatil.models.decoding.action_heads import ActionHeadBlock


class ActionHead(nn.Module):
    """Composable action head from sequence of blocks.

    Converts decoder embeddings (B, horizon, embedding_dimension) to action predictions
    (B, horizon, action_dim). Action heads are composed of a sequence of blocks
    followed by a final linear projection to the action dimension.

    Note:
        output_dim is not set at initialization. The decoder must call set_output_dim()
        to set the output dimension based on action_space.actions_metadata[key].prediction_dimension.
    """

    def __init__(
        self,
        input_dim: int,
        blocks: list[ActionHeadBlock] | None = None,
    ):
        """Initialize action head.

        Args:
            input_dim: Input embedding dimension from decoder
            blocks: List of blocks to apply (if None, uses simple linear projection)
        """
        super().__init__()
        self.input_dim = input_dim
        self._output_dim: int | None = None
        if blocks is None:
            blocks = []
        self.blocks = nn.ModuleList(blocks)
        self.output_proj: nn.Linear | None = None  # Created by set_output_dim

    @property
    def output_dim(self) -> int:
        """Get output dimension. Raises if not set."""
        if self._output_dim is None:
            raise RuntimeError("output_dim not set. Call set_output_dim() first.")
        return self._output_dim

    @output_dim.setter
    def output_dim(self, value: int) -> None:
        self._output_dim = value

    def set_output_dim(self, dim: int) -> None:
        """Set output dimension and create output projection layer.

        Called by the decoder based on action_space.actions_metadata[key].prediction_dimension.

        Args:
            dim: Output action dimension
        """
        self._output_dim = dim
        hidden_dim = (
            self.input_dim if len(self.blocks) == 0 else self.blocks[-1].output_dim
        )
        self.output_proj = nn.Linear(hidden_dim, dim)

    def forward(self, action_embedding: torch.Tensor) -> torch.Tensor:
        """Convert embeddings to actions.

        Args:
            action_embedding: Decoder embeddings (B,prediction horizon, embedding_dimension) or (B, embedding_dimension)

        Returns:
            Action predictions (B, prediction horizon, action_dim) or (B, action_dim)

        Raises:
            RuntimeError: If set_output_dim() has not been called
        """
        if self.output_proj is None:
            raise RuntimeError("output_dim not set. Call set_output_dim() first.")
        for block in self.blocks:
            action_embedding = block(action_embedding)
        result: torch.Tensor = self.output_proj(action_embedding)
        return result
