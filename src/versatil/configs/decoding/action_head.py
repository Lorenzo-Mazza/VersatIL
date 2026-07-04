"""Configuration classes for modular action heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING

from versatil.models.decoding.constants import MoERoutingType


@dataclass
class ActionHeadBlockConfig:
    """Base configuration for action head blocks.

    Attributes:
        _target_: Import path instantiated by Hydra.
    """

    _target_: str = MISSING


@dataclass
class LayerNormBlockConfig(ActionHeadBlockConfig):
    """Configuration for layer-normalization action-head block.

    Attributes:
        _target_: Import path instantiated by Hydra.
        input_dimension: Input and output feature dimension.
    """

    _target_: str = "versatil.models.decoding.action_heads.LayerNormBlock"
    input_dimension: int = MISSING


@dataclass
class MLPBlockConfig(ActionHeadBlockConfig):
    """Configuration for MLP block in action head.

    Attributes:
        _target_: Import path instantiated by Hydra.
        input_dimension: Input dimension.
        hidden_dimensions: List of hidden dimensions.
        output_dim: Output dimension (None to keep same as last hidden).
        activation: Activation function name.
        dropout: Dropout rate.
        normalization: Whether to apply layer normalization before MLP.
    """

    _target_: str = "versatil.models.decoding.action_heads.MLPBlock"
    input_dimension: int = MISSING  # Set by parent head
    hidden_dimensions: list[int] | None = None
    output_dim: int | None = None  # None = keep input_dimension
    activation: str = "silu"
    dropout: float = 0.1
    normalization: bool = True


@dataclass
class AttentionBlockConfig(ActionHeadBlockConfig):
    """Configuration for attention block in action head.

    Attributes:
        _target_: Import path instantiated by Hydra.
        embedding_dimension: Embedding dimension.
        number_of_heads: Number of attention heads.
        dropout: Dropout rate.
        normalization: Whether to apply layer normalization.
    """

    _target_: str = "versatil.models.decoding.action_heads.AttentionBlock"
    embedding_dimension: int = MISSING
    number_of_heads: int = 8
    dropout: float = 0.1
    normalization: bool = True


@dataclass
class ResidualBlockConfig(ActionHeadBlockConfig):
    """Configuration for residual wrapper block.

    Attributes:
        _target_: Import path instantiated by Hydra.
        block: Block to wrap with residual connection.
        dropout: Dropout rate after block.
    """

    _target_: str = "versatil.models.decoding.action_heads.ResidualBlock"
    block: dict[str, Any] = MISSING
    dropout: float = 0.1


@dataclass
class AdaNormBlockConfig(ActionHeadBlockConfig):
    """Configuration for adaptive normalization action-head block.

    Attributes:
        _target_: Import path instantiated by Hydra.
        input_dimension: Action embedding feature dimension.
        conditioning_dimension: Conditioning vector dimension.
        activation: Activation used inside the modulation projection.
    """

    _target_: str = "versatil.models.decoding.action_heads.AdaNormBlock"
    input_dimension: int = MISSING
    conditioning_dimension: int = MISSING
    activation: str = "silu"


@dataclass
class ActionHeadConfig:
    """Configuration for a single action head.

    Note:
        output dimension is set by the decoder based on the action key.

    Attributes:
        _target_: Import path instantiated by Hydra.
        input_dimension: Set from decoder embedding_dimension.
        blocks: Head blocks applied in order.
    """

    _target_: str = "versatil.models.decoding.action_heads.ActionHead"
    input_dimension: int = MISSING  # Set from decoder embedding_dimension
    blocks: list[dict[str, Any]] | None = None


@dataclass
class ConditionalActionHeadConfig:
    """Configuration for a conditioned action head.

    Attributes:
        _target_: Import path instantiated by Hydra.
        input_dimension: Input action-token embedding dimension.
        conditioning_dimension: Conditioning vector dimension.
        blocks: Conditional blocks applied before the output projection.
    """

    _target_: str = "versatil.models.decoding.action_heads.ConditionalActionHead"
    input_dimension: int = MISSING
    conditioning_dimension: int = MISSING
    blocks: list[dict[str, Any]] | None = None


@dataclass
class GaussianHeadConfig:
    """Configuration for GaussianHead that outputs mean and logvar.

    Attributes:
        _target_: Import path instantiated by Hydra.
        input_dimension: Input embedding dimension from decoder.
        blocks: Blocks to apply before output projection.
        min_logvar: Minimum value for logvar clamping.
        max_logvar: Maximum value for logvar clamping.
    """

    _target_: str = "versatil.models.decoding.action_heads.GaussianHead"
    input_dimension: int = MISSING
    blocks: list[dict[str, Any]] | None = None
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

    Attributes:
        _target_: Import path instantiated by Hydra.
        device: Device to place the module on.
        experts: Optional pre-instantiated expert action heads.
        base_expert: Single expert instance to clone num_experts times.
        num_experts: Number of experts to create from base_expert (optional for lazy
            init).
        gating_input_dim: Input dimension for gating network (None for external
            routing).
        gating_hidden_dims: Hidden layer dimensions for gating network.
        routing_type: Routing strategy ("soft" or "top_k").
        top_k: Number of experts to use for top-k routing.
        temperature: Temperature for softmax scaling of routing weights.
        learnable_temperature: Whether temperature should be a learnable parameter.
        gating_dropout: Dropout rate in gating network.
        gating_normalization: Whether to normalize inputs to gating network.
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
