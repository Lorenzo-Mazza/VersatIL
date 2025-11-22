"""ActionHead class for composing blocks into action prediction heads."""

import torch
import torch.nn as nn

from refactoring.models.decoding.action_heads import ActionHeadBlock

class ActionHead(nn.Module):
    """Composable action head from sequence of blocks.

    Converts decoder embeddings (B, horizon, embedding_dimension) to action predictions
    (B, horizon, action_dim). Action heads are composed of a sequence of blocks
    followed by a final linear projection to the action dimension.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        blocks: list[ActionHeadBlock] | None = None,
    ):
        """Initialize action head.

        Args:
            input_dim: Input embedding dimension from decoder
            output_dim: Output action dimension
            blocks: List of blocks to apply (if None, uses simple linear projection)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        if blocks is None:
            blocks = []
        self.blocks = nn.ModuleList(blocks)
        hidden_dim = input_dim
        if len(blocks) > 0:
            hidden_dim = blocks[-1].output_dim
        self.output_proj = nn.Linear(hidden_dim, output_dim)


    def forward(self, action_embedding: torch.Tensor) -> torch.Tensor:
        """Convert embeddings to actions.

        Args:
            action_embedding: Decoder embeddings (B,prediction horizon, embedding_dimension) or (B, embedding_dimension)

        Returns:
            Action predictions (B, prediction horizon, action_dim) or (B, action_dim)
        """
        for block in self.blocks:
            action_embedding = block(action_embedding)
        result: torch.Tensor = self.output_proj(action_embedding)
        return result


