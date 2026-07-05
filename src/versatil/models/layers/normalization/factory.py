"""Normalization layers factory method."""

from typing import Literal

import torch.nn as nn

from versatil.models.layers.normalization.ada_norm import AdaNorm
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.normalization.rms_norm import RMSNorm
from versatil.models.layers.normalization.typedefs import BlockNormalization
from versatil.models.layers.normalization.unconditioned_norm import UnconditionedNorm


def _create_base_norm(
    normalization_type: str,
    dimension: int,
    epsilon: float,
    elementwise_affine: bool = True,
) -> nn.Module:
    """Create the base normalization module.

    Args:
        normalization_type: Type of normalization (use NormalizationType enum values).
        dimension: Feature dimension.
        epsilon: Small constant for numerical stability.
        elementwise_affine: Whether to include learnable affine parameters.

    Returns:
        Normalization base module.

    Raises:
        ValueError: If normalization_type is not supported.
    """
    match normalization_type:
        case NormalizationType.LAYER_NORM.value:
            return nn.LayerNorm(
                dimension, eps=epsilon, elementwise_affine=elementwise_affine
            )
        case NormalizationType.RMS_NORM.value:
            return RMSNorm(
                dimension, epsilon=epsilon, elementwise_affine=elementwise_affine
            )
        case _:
            raise ValueError(
                f"Unsupported normalization type: {normalization_type}. "
                f"Must be one of {[e.value for e in NormalizationType]}."
            )


def create_normalization_layer(
    normalization_type: str,
    dimension: int,
    epsilon: float = 1e-6,
    conditioning_dimension: int | None = None,
) -> nn.Module:
    """Create a normalization layer, optionally wrapped with adaptive conditioning.

    When ``conditioning_dimension`` is provided, returns an AdaNorm that wraps the base
    normalization with a learned modulation. Otherwise returns a plain norm.

    Args:
        normalization_type: Base normalization type (use NormalizationType enum values).
        dimension: Feature dimension.
        epsilon: Small constant for numerical stability.
        conditioning_dimension: Conditioning dimension. When set, wraps the base norm
            in AdaNorm for adaptive modulation.

    Returns:
        Plain normalization layer or AdaNorm.

    Raises:
        ValueError: If normalization_type is not supported.
    """
    if conditioning_dimension is not None:
        base_norm = _create_base_norm(
            normalization_type=normalization_type,
            dimension=dimension,
            epsilon=epsilon,
            elementwise_affine=False,
        )
        return AdaNorm(
            base_norm=base_norm,
            conditioning_dimension=conditioning_dimension,
            feature_dim=dimension,
        )
    return _create_base_norm(
        normalization_type=normalization_type,
        dimension=dimension,
        epsilon=epsilon,
    )


def create_block_normalization(
    normalization_type: str,
    dimension: int,
    epsilon: float = 1e-6,
    conditioning_dimension: int | None = None,
    use_gating: bool = False,
    init_strategy: Literal["zero", "xavier"] = "zero",
) -> BlockNormalization:
    """Create normalization for transformer blocks: ``(x, condition) -> (normed, gate)``.

    When ``conditioning_dimension`` is provided, returns an AdaNorm with learned
    modulation (and optional gating for AdaLN-Zero). Otherwise returns
    an UnconditionedNorm that wraps a plain normalization layer.

    Args:
        normalization_type: Base normalization type (use NormalizationType enum values).
        dimension: Feature dimension.
        epsilon: Small constant for numerical stability.
        conditioning_dimension: Conditioning dimension. When set, creates AdaNorm.
        use_gating: Whether to produce a learned gate (AdaLN-Zero).
            Only applies when conditioning_dimension is set.
        init_strategy: Initialization strategy for modulation weights.

    Returns:
        AdaNorm when conditioned, UnconditionedNorm when not.

    Raises:
        ValueError: If normalization_type is not supported.
    """
    if conditioning_dimension is not None:
        base_norm = _create_base_norm(
            normalization_type=normalization_type,
            dimension=dimension,
            epsilon=epsilon,
            elementwise_affine=False,
        )
        return AdaNorm(
            base_norm=base_norm,
            conditioning_dimension=conditioning_dimension,
            feature_dim=dimension,
            use_gate=use_gating,
            init_strategy=init_strategy,
        )
    base_norm = _create_base_norm(
        normalization_type=normalization_type,
        dimension=dimension,
        epsilon=epsilon,
    )
    return UnconditionedNorm(base_norm)
