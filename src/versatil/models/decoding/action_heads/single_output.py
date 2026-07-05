"""ActionHead classes for composing blocks into action prediction heads."""

import torch

from versatil.models.decoding.action_heads.base import BaseActionHead


class ActionHead(BaseActionHead):
    """Single-output action head returning one tensor.

    Converts decoder embeddings (B, horizon, embedding_dimension) to action predictions
    (B, horizon, action_dim) via blocks and a final linear projection.
    """

    def forward(self, action_embedding: torch.Tensor) -> torch.Tensor:
        """Convert embeddings to actions.

        Args:
            action_embedding: (B, prediction_horizon, embedding_dimension) or (B, embedding_dimension)

        Returns:
            Action predictions with same batch/horizon dims, last dim is action_dim.
        """
        if self.output_proj is None:
            raise RuntimeError("output_dim not set. Call set_output_dim() first.")
        action_embedding = self._apply_blocks(action_embedding)
        return self.output_proj(action_embedding)
