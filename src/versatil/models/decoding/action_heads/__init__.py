"""Modular action prediction heads for decoding architectures.

This package provides composable building blocks for action prediction heads:
- Base classes for extensibility
- Individual block modules (MLP, Attention, Residual)
- ActionHead class for composition
- MoEHead for phase-conditioned or multi-modal prediction
- Factory functions for common patterns
"""

from versatil.models.decoding.action_heads.base import BaseActionHead
from versatil.models.decoding.action_heads.blocks import (
    ActionHeadBlock,
    AdaNormBlock,
    AttentionBlock,
    ConditionalActionHeadBlock,
    LayerNormBlock,
    MLPBlock,
    ResidualBlock,
)
from versatil.models.decoding.action_heads.conditional import ConditionalActionHead
from versatil.models.decoding.action_heads.gaussian import (
    GaussianHead,
)
from versatil.models.decoding.action_heads.moe import MoEHead
from versatil.models.decoding.action_heads.single_output import (
    ActionHead,
)

__all__ = [
    "ActionHeadBlock",
    "ConditionalActionHeadBlock",
    "MLPBlock",
    "LayerNormBlock",
    "AttentionBlock",
    "ResidualBlock",
    "AdaNormBlock",
    "BaseActionHead",
    "ActionHead",
    "ConditionalActionHead",
    "GaussianHead",
    "MoEHead",
]
