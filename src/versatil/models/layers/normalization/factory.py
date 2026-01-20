"""Normalization layers factory method."""
import torch.nn as nn

from versatil.models.layers import FrozenBatchNorm2d
from versatil.models.layers.normalization.ada_norm import AdaNorm
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.normalization.rms_norm import RMSNorm


def create_normalization_layer(
    normalization_type: str,
    dimension: int,
    epsilon: float = 1e-6,
    condition_dim: int | None = None,
) -> nn.Module:
    """Factory function to create normalization layer.

    Args:
        normalization_type: Type of normalization (use NormalizationType enum values)
        dimension: Feature dimension
        epsilon: Small constant for numerical stability
        condition_dim: If provided, returns adaptive version (norm → ConditionalModulation)

    Returns:
        Normalization layer (LayerNorm or RMSNorm)

    Raises:
        ValueError: If normalization_type is not supported
    """
    if normalization_type in (
        NormalizationType.ADALN.value,
        NormalizationType.ADARMS.value,
    ):
        if condition_dim is None:
            raise ValueError("condition_dim is required for ada_ln / ada_rms")
        if normalization_type == NormalizationType.ADALN.value:
            base_norm = nn.LayerNorm(dimension, eps=epsilon)
        else:
            base_norm = RMSNorm(dimension, eps=epsilon)
        return AdaNorm(base_norm, condition_dim=condition_dim, feature_dim=dimension)
    elif normalization_type == NormalizationType.LAYER_NORM.value:
        return nn.LayerNorm(dimension, eps=epsilon)
    elif normalization_type == NormalizationType.RMS_NORM.value:
        return RMSNorm(dimension, eps=epsilon)
    elif normalization_type == NormalizationType.FROZEN_BATCHNORM2D.value:
        return FrozenBatchNorm2d(dimension)
    else:
        raise ValueError(
            f"Unsupported normalization type: {normalization_type}. "
            f"Must be one of {[e.value for e in NormalizationType]}."
        )
