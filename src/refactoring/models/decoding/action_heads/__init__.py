"""Modular action prediction heads for decoding architectures.

This package provides composable building blocks for action prediction heads:
- Base classes for extensibility
- Individual block modules (MLP, Attention, Residual)
- ActionHead class for composition
- MoEHead for phase-conditioned or multi-modal prediction
- Factory functions for common patterns
"""
from refactoring.models.decoding.action_heads.blocks import (
    ActionHeadBlock,
    AttentionBlock,
    MLPBlock,
    ResidualBlock,
)
from refactoring.models.decoding.action_heads.head import (
    ActionHead,
)
from refactoring.models.decoding.action_heads.moe import MoEHead

__all__ = [
    "ActionHeadBlock",
    "MLPBlock",
    "AttentionBlock",
    "ResidualBlock",
    "ActionHead",
    "MoEHead",
]
