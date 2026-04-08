"""Wrapper that adapts plain norms to the conditioned normalization interface."""

import torch
import torch.nn as nn


class UnconditionedNorm(nn.Module):
    """Wraps a plain norm (LayerNorm, RMSNorm) to return (normed, gate).

    Ignores the conditioning argument and returns gate = ones(1).
    This allows transformer blocks to call all normalizations uniformly.
    """

    def __init__(self, norm: nn.Module):
        super().__init__()
        self.norm = norm

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normed = self.norm(x)
        gate = torch.ones(1, dtype=x.dtype, device=x.device)
        return normed, gate
