"""Conditional action heads."""

import torch

from versatil.models.decoding.action_heads.base import BaseActionHead
from versatil.models.decoding.action_heads.blocks import ConditionalActionHeadBlock


class ConditionalActionHead(BaseActionHead):
    """Action head whose blocks receive a conditioning vector."""

    def __init__(
        self,
        input_dimension: int,
        conditioning_dimension: int,
        blocks: list[ConditionalActionHeadBlock] | None = None,
    ) -> None:
        """Initialize the conditional action head.

        Args:
            input_dimension: Input action-token embedding dimension.
            conditioning_dimension: Conditioning vector dimension.
            blocks: Conditional blocks applied before the output projection.
        """
        super().__init__(input_dimension=input_dimension, blocks=blocks)
        self.conditioning_dimension = conditioning_dimension

    def forward(
        self,
        action_embedding: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Project conditioned action embeddings to action predictions.

        Args:
            action_embedding: Action-token embeddings with shape
                ``(B, prediction_horizon, input_dimension)``.
            condition: Conditioning tensor with shape ``(B, conditioning_dimension)``.

        Returns:
            Action predictions with shape
            ``(B, prediction_horizon, output_dim)``.
        """
        if self.output_proj is None:
            raise RuntimeError("output_dim not set. Call set_output_dim() first.")
        for block in self.blocks:
            action_embedding = block(action_embedding, condition)
        return self.output_proj(action_embedding)
