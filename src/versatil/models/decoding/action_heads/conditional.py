"""Conditional action heads."""

import torch

from versatil.models.decoding.action_heads.base import BaseActionHead
from versatil.models.decoding.action_heads.blocks import ConditionalActionHeadBlock


class ConditionalActionHead(BaseActionHead):
    """Action head whose blocks receive a conditioning vector."""

    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        blocks: list[ConditionalActionHeadBlock] | None = None,
    ) -> None:
        """Initialize the conditional action head.

        Args:
            input_dim: Input action-token embedding dimension.
            condition_dim: Conditioning vector dimension.
            blocks: Conditional blocks applied before the output projection.
        """
        super().__init__(input_dim=input_dim, blocks=blocks)
        self.condition_dim = condition_dim

    def forward(
        self,
        action_embedding: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Project conditioned action embeddings to action predictions.

        Args:
            action_embedding: Action-token embeddings with shape
                ``(B, prediction_horizon, input_dim)``.
            condition: Conditioning tensor with shape ``(B, condition_dim)``.

        Returns:
            Action predictions with shape
            ``(B, prediction_horizon, output_dim)``.
        """
        if self.output_proj is None:
            raise RuntimeError("output_dim not set. Call set_output_dim() first.")
        for block in self.blocks:
            action_embedding = block(action_embedding, condition)
        return self.output_proj(action_embedding)
