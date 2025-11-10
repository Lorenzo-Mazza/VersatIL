"""Normalization layers for GPT transformer."""

import torch.nn as nn

from refactoring.models.layers.constants import NormalizationType
from refactoring.models.layers.rms_norm import RMSNorm


def create_normalization_layer(
    normalization_type: str,
    dimension: int,
    epsilon: float = 1e-6,
) -> nn.Module:
    """Factory function to create normalization layer.

    Args:
        normalization_type: Type of normalization (use NormalizationType enum values)
        dimension: Feature dimension
        epsilon: Small constant for numerical stability

    Returns:
        Normalization layer (LayerNorm or RMSNorm)

    Raises:
        ValueError: If normalization_type is not supported
    """
    if normalization_type == NormalizationType.LAYER_NORM.value:
        return nn.LayerNorm(dimension, eps=epsilon)
    elif normalization_type == NormalizationType.RMS_NORM.value:
        return RMSNorm(dimension, eps=epsilon)
    else:
        raise ValueError(
            f"Unsupported normalization type: {normalization_type}. "
            f"Must be one of {[e.value for e in NormalizationType]}."
        )