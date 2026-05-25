"""Configuration classes for modular action heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING

from versatil.models.decoding.constants import MoERoutingType

type ActionHeadBlockConfigDict = dict[str, Any]


@dataclass
class ActionHeadBlockConfig:
    """Base configuration for action head blocks."""

    _target_: str = MISSING


@dataclass
class LayerNormBlockConfig(ActionHeadBlockConfig):
    """Configuration for layer-normalization action-head block."""

    _target_: str = "versatil.models.decoding.action_heads.LayerNormBlock"
    input_dim: int = MISSING


@dataclass
class MLPBlockConfig(ActionHeadBlockConfig):
    """Configuration for MLP block in action head."""

    _target_: str = "versatil.models.decoding.action_heads.MLPBlock"
    input_dim: int = MISSING  # Set by parent head
    hidden_dims: list[int] | None = None
    output_dim: int | None = None  # None = keep input_dim
    activation: str = "silu"
    dropout: float = 0.1
    normalization: bool = True


@dataclass
class AttentionBlockConfig(ActionHeadBlockConfig):
    """Configuration for attention block in action head."""

    _target_: str = "versatil.models.decoding.action_heads.AttentionBlock"
    embedding_dimension: int = MISSING
    num_heads: int = 8
    dropout: float = 0.1
    normalization: bool = True


@dataclass
class ResidualBlockConfig(ActionHeadBlockConfig):
    """Configuration for residual wrapper block."""

    _target_: str = "versatil.models.decoding.action_heads.ResidualBlock"
    block: ActionHeadBlockConfigDict = MISSING
    dropout: float = 0.1


@dataclass
class AdaNormBlockConfig(ActionHeadBlockConfig):
    """Configuration for adaptive normalization action-head block."""

    _target_: str = "versatil.models.decoding.action_heads.AdaNormBlock"
    input_dim: int = MISSING
    condition_dim: int = MISSING
    activation: str = "silu"


@dataclass
class ActionHeadConfig:
    """Configuration for a single action head.

    Note:
        output dimension is set by the decoder based on the action key.
    """

    _target_: str = "versatil.models.decoding.action_heads.ActionHead"
    input_dim: int = MISSING  # Set from decoder embedding_dimension
    blocks: list[ActionHeadBlockConfigDict] | None = None


@dataclass
class ConditionalActionHeadConfig:
    """Configuration for a conditioned action head."""

    _target_: str = "versatil.models.decoding.action_heads.ConditionalActionHead"
    input_dim: int = MISSING
    condition_dim: int = MISSING
    blocks: list[ActionHeadBlockConfigDict] | None = None


@dataclass
class GaussianHeadConfig:
    """Configuration for GaussianHead that outputs mean and logvar."""

    _target_: str = "versatil.models.decoding.action_heads.GaussianHead"
    input_dim: int = MISSING
    blocks: list[ActionHeadBlockConfigDict] | None = None
    min_logvar: float = -10.0
    max_logvar: float = 4.0


@dataclass
class MixtureOfExpertsHeadConfig:
    """Configuration for Mixture of Experts action head.

    Supports two modes:
    1. Explicit experts: Pass list of ActionHeadConfig
    2. Base expert cloning: Pass base_expert and num_experts (recommended)

    Note:
        base_expert is instantiated by Hydra, then cloned num_experts times
        by MoEHead to create separate expert networks with independent weights.
        output_dim is set by the decoder based on the action key.
    """

    _target_: str = "versatil.models.decoding.action_heads.MoEHead"
    device: str = "${policy.device}"
    experts: list[ActionHeadConfig] | None = None
    base_expert: ActionHeadConfig | None = None
    num_experts: int = MISSING
    gating_input_dim: int | None = None  # None for external routing
    gating_hidden_dims: list[int] | None = None
    routing_type: str = MoERoutingType.SOFT.value
    top_k: int = 2  # For top-k routing
    temperature: float = 1.0
    learnable_temperature: bool = False
    gating_dropout: float = 0.1
    gating_normalization: bool = True
