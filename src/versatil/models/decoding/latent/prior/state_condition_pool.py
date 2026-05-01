"""Create observation-only state vectors for conditional latent losses."""

import torch
import torch.nn.functional as F
from torch import nn


class StateConditionPool(nn.Module):
    """Build the state vector used by conditional latent regularizers.

    The learned prior sees only observations, so its input tokens are the
    right place to define ``s`` for losses that compare ``q(z|s)`` with
    ``p(z|s)``. This module averages those observation tokens into one vector
    per batch element while respecting padding masks.

    Callers should pass only observation tokens. In the current priors, the
    CLS token is appended after the observation tokens and is intentionally
    removed before this module is called. Posterior/action tokens should never
    be included here, otherwise the "state" coordinate would leak action
    information into the conditional matching loss.
    """

    def __init__(self, embedding_dimension: int) -> None:
        super().__init__()
        if embedding_dimension <= 0:
            raise ValueError(
                f"embedding_dimension must be positive, got {embedding_dimension}."
            )
        self.embedding_dimension = embedding_dimension

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one normalized observation vector per batch element.

        Args:
            tokens: Observation token sequence with shape
                ``(batch, sequence, embedding_dimension)``.
            padding_mask: Optional boolean mask with shape ``(batch, sequence)``.
                ``True`` means that token is padding and should not contribute
                to the average.

        Returns:
            A tensor with shape ``(batch, embedding_dimension)``. This tensor
            can be concatenated with prior and posterior latent samples so
            MMD/OT compare ``(state, latent)`` pairs instead of latents alone.
        """
        if tokens.ndim != 3:
            raise ValueError(
                f"tokens must have shape (batch, sequence, embedding), got {tokens.shape}."
            )
        if tokens.shape[-1] != self.embedding_dimension:
            raise ValueError(
                f"tokens embedding dimension must be {self.embedding_dimension}, got {tokens.shape[-1]}."
            )
        if tokens.shape[1] == 0:
            raise ValueError("StateConditionPool requires at least one token to pool.")

        if padding_mask is None:
            pooled = tokens.mean(dim=1)
        else:
            if padding_mask.shape != tokens.shape[:2]:
                raise ValueError(
                    f"padding_mask must have shape {tokens.shape[:2]}, got {padding_mask.shape}."
                )
            valid_tokens = (~padding_mask).to(dtype=tokens.dtype)
            denominator = valid_tokens.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = (tokens * valid_tokens.unsqueeze(-1)).sum(dim=1) / denominator

        return F.layer_norm(pooled, normalized_shape=(self.embedding_dimension,))
