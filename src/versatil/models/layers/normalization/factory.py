"""Normalization layers factory method."""

from typing import Literal

import torch.nn as nn

from versatil.models.layers import FrozenBatchNorm2d
from versatil.models.layers.normalization.ada_norm import AdaNorm
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.normalization.rms_norm import RMSNorm
from versatil.models.layers.normalization.typedefs import BlockNormalization
from versatil.models.layers.normalization.unconditioned_norm import UnconditionedNorm


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
            base_norm = nn.LayerNorm(dimension, eps=epsilon, elementwise_affine=False)
        else:
            base_norm = RMSNorm(dimension, eps=epsilon, elementwise_affine=False)
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


def create_block_normalization(
    normalization_type: str,
    dimension: int,
    epsilon: float = 1e-6,
    condition_dim: int | None = None,
    use_gating: bool = False,
    init_strategy: Literal["zero", "xavier"] = "zero",
) -> BlockNormalization:
    """Create a normalization conforming to the block interface: (x, condition) -> (normed, gate).

    Args:
        normalization_type: Type of normalization (use NormalizationType enum values).
        dimension: Feature dimension.
        epsilon: Small constant for numerical stability.
        condition_dim: Conditioning dimension. Required for adaptive types.
        use_gating: Whether to produce a learned gate (AdaLN-Zero).
            Only applies to adaptive normalization types.
        init_strategy: Initialization strategy for modulation weights.

    Returns:
        AdaNorm for adaptive types, UnconditionedNorm for plain types.

    Raises:
        ValueError: If normalization_type is not supported, condition_dim
            is missing for adaptive normalization types or provided for non-adaptive types.
    """
    valid_values = [e.value for e in NormalizationType]
    if normalization_type not in valid_values:
        raise ValueError(
            f"Unsupported normalization type: {normalization_type}. "
            f"Must be one of {valid_values}."
        )
    norm_enum = NormalizationType(normalization_type)
    if norm_enum.is_adaptive:
        if condition_dim is None:
            raise ValueError(
                f"condition_dim is required for adaptive normalization type "
                f"{normalization_type}"
            )
        match norm_enum.value:
            case NormalizationType.ADALN.value:
                base_norm = nn.LayerNorm(
                    dimension, eps=epsilon, elementwise_affine=False
                )
            case NormalizationType.ADARMS.value:
                base_norm = RMSNorm(dimension, eps=epsilon, elementwise_affine=False)
            case _:
                raise ValueError(
                    f"Unsupported adaptive normalization type: {normalization_type}"
                )
        return AdaNorm(
            base_norm=base_norm,
            condition_dim=condition_dim,
            feature_dim=dimension,
            use_gate=use_gating,
            init_strategy=init_strategy,
        )
    else:
        if condition_dim is not None:
            raise ValueError(
                f"condition_dim should not be provided for non-adaptive normalization type "
                f"{normalization_type}"
            )
        match norm_enum.value:
            case NormalizationType.LAYER_NORM.value:
                base_norm = nn.LayerNorm(dimension, eps=epsilon)
            case NormalizationType.RMS_NORM.value:
                base_norm = RMSNorm(dimension, eps=epsilon)
            case _:
                raise ValueError(
                    f"Unsupported normalization type for blocks: {normalization_type}. "
                    f"Use {NormalizationType.LAYER_NORM.value} or "
                    f"{NormalizationType.RMS_NORM.value}."
                )
        return UnconditionedNorm(base_norm)
