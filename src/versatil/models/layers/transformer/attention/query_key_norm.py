"""Query-Key Normalization for attention mechanisms.

Applies RMSNorm to query and key tensors in the attention head dimension
for improved training stability at scale. Used in SD3/MMDiT architectures.

References:
    Esser et al. "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis"
    https://arxiv.org/abs/2403.03206
"""

import torch
import torch.nn as nn

from versatil.models.layers.normalization.rms_norm import RMSNorm


class QueryKeyNorm(nn.Module):
    """Query-Key normalization using RMSNorm.

    Normalizes query and key tensors independently using RMSNorm in the head
    dimension before computing attention. This improves training stability
    especially when scaling to larger models.
    """

    def __init__(
        self,
        head_dimension: int,
        epsilon: float = 1e-6,
    ):
        """Initialize QueryKeyNorm.

        Args:
            head_dimension: Dimension of each attention head.
            epsilon: Small constant for numerical stability.
        """
        super().__init__()
        self.query_norm = RMSNorm(
            head_dimension, epsilon=epsilon, elementwise_affine=True
        )
        self.key_norm = RMSNorm(
            head_dimension, epsilon=epsilon, elementwise_affine=True
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RMSNorm to query and key tensors.

        Args:
            query: Query tensor (B, number_of_heads, sequence_length, head_dimension).
            key: Key tensor (B, number_of_heads, sequence_length, head_dimension).

        Returns:
            Tuple of normalized (query, key) tensors with same shapes.
        """
        return self.query_norm(query), self.key_norm(key)
