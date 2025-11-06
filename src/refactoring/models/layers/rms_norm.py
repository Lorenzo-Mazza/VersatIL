"""Root Mean Square Layer Normalization.

From Zhang et al. (2019): https://arxiv.org/abs/1910.07467

RMSNorm normalizes using only the root mean square statistic,
without centering (no mean subtraction). This is more efficient
than LayerNorm and works well in practice for LLM-scale models.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Args:
        normalized_shape: Input shape from an expected input of size [*, normalized_shape]
        eps: A value added to the denominator for numerical stability
        elementwise_affine: If True, learns affine scaling parameters
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
    ):
        """Initialize RMSNorm.

        Args:
            normalized_shape: Feature dimension to normalize
            eps: Small constant for numerical stability
            elementwise_affine: Whether to learn scaling parameters
        """
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.register_parameter('weight', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor (..., normalized_shape)

        Returns:
            Normalized tensor of same shape
        """
        # Compute RMS: sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms

        if self.elementwise_affine:
            x_normed = x_normed * self.weight

        return x_normed