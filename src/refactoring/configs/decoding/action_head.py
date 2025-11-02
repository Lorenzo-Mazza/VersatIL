"""Configuration classes for modular action heads."""
from dataclasses import dataclass, field

from omegaconf import MISSING

from refactoring.models.decoding.constants import MoERoutingType


@dataclass
class ActionHeadBlockConfig:
    """Base configuration for action head blocks."""
    _target_: str = MISSING


@dataclass
class MLPBlockConfig(ActionHeadBlockConfig):
    """Configuration for MLP block in action head."""
    _target_: str = "refactoring.models.decoding.action_heads.MLPBlock"
    input_dim: int = MISSING  # Set by parent head
    hidden_dims: list[int] | None = None
    output_dim: int | None = None  # None = keep input_dim
    activation: str = "silu"
    dropout: float = 0.1
    normalization: bool = True


@dataclass
class AttentionBlockConfig(ActionHeadBlockConfig):
    """Configuration for attention block in action head."""
    _target_: str = "refactoring.models.decoding.action_heads.AttentionBlock"
    embed_dim: int = MISSING  # Set by parent head
    num_heads: int = 8
    dropout: float = 0.1
    normalization: bool = True


@dataclass
class ResidualBlockConfig(ActionHeadBlockConfig):
    """Configuration for residual wrapper block."""
    _target_: str = "refactoring.models.decoding.action_heads.ResidualBlock"
    block: ActionHeadBlockConfig = MISSING
    dropout: float = 0.1


@dataclass
class ActionHeadConfig:
    """Configuration for a single action head."""
    _target_: str = "refactoring.models.decoding.action_heads.ActionHead"
    input_dim: int = MISSING  # Set from decoder embedding_dimension
    output_dim: int = MISSING  # Set from action_space
    blocks: list[ActionHeadBlockConfig] = field(default_factory=list)


@dataclass
class DefaultActionHeadConfig:
    """Default action head: single MLP layer."""
    _target_: str = "refactoring.models.decoding.action_heads.create_default_action_head"
    input_dim: int = MISSING
    output_dim: int = MISSING
    hidden_dim: int | None = None  # Defaults to input_dim // 2
    activation: str = "silu"
    dropout: float = 0.1


@dataclass
class MLPActionHeadConfig:
    """Multi-layer MLP action head."""
    _target_: str = "refactoring.models.decoding.action_heads.create_mlp_action_head"
    input_dim: int = MISSING
    output_dim: int = MISSING
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64])
    activation: str = "silu"
    dropout: float = 0.1


@dataclass
class AttentionMLPActionHeadConfig:
    """Attention + MLP action head."""
    _target_: str = "refactoring.models.decoding.action_heads.create_attention_mlp_head"
    input_dim: int = MISSING
    output_dim: int = MISSING
    num_heads: int = 8
    mlp_hidden_dim: int | None = None
    activation: str = "silu"
    dropout: float = 0.1


@dataclass
class MixtureOfExpertsHeadConfig:
    """Configuration for Mixture of Experts action head.

    Supports two modes:
    1. Explicit experts: Pass list of ActionHeadConfig
    2. Config-based (recommended): Pass base_expert_config and num_experts

    Example:
        moe_config = MixtureOfExpertsHeadConfig(
            base_expert_config=ActionHeadConfig(input_dim=256, output_dim=3),
            num_experts=5,
            output_dim=3,
            gating_input_dim=256,
            device="cuda"
        )
    """
    _target_: str = "refactoring.models.decoding.action_heads.MoEHead"
    output_dim: int = MISSING
    device: str = "${policy.device}"
    experts: list[ActionHeadConfig] | None = None
    base_expert_config: ActionHeadConfig | None = None
    num_experts: int | None = None
    expert_configs: list[ActionHeadConfig] | None = None
    gating_input_dim: int | None = None
    gating_hidden_dims: list[int] | None = None
    routing_type: str = MoERoutingType.SOFT.value
    top_k: int = 2  # For top-k routing
    temperature: float = 1.0
    learnable_temperature: bool = False
    gating_dropout: float = 0.1
    gating_normalization: bool = True
