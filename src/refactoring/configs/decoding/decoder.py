"""Configuration classes for different action decoder architectures."""
from dataclasses import dataclass

from omegaconf import DictConfig, MISSING

from refactoring.configs.decoding.action_head import ActionHeadConfig
from refactoring.configs.task.task import ActionSpace, ObservationSpace
from refactoring.models.decoding.constants import MoERoutingType
from refactoring.models.layers.activation import ActivationFunction
from refactoring.models.layers.constants import PositionalEncodingType


@dataclass
class DecodingNetworkConfig:
    """Base architecture configuration."""
    _target_: str = MISSING
    action_heads: dict[str, ActionHeadConfig] = MISSING
    input_keys: list[str] = MISSING
    observation_space: "ObservationSpace" = "${policy.observation_space}"  # type: ignore[assignment]
    observation_horizon: int = "${policy.observation_horizon}"  # type: ignore[assignment]
    prediction_horizon: int = "${policy.prediction_horizon}"  # type: ignore[assignment]
    action_space: ActionSpace = "${policy.action_space}"  # type: ignore[assignment]
    device: str = "${policy.device}"



@dataclass
class ACTConfig(DecodingNetworkConfig):
    """Action Chunking Transformer network config.

    Note: Latent action encoding (e.g., VAE) is now handled at the Algorithm level,
    not at the Decoder level. To use a VAE with ACT, configure the latent_encoder
    parameter in the algorithm config (e.g., BehavioralCloningConfig).
    """
    _target_: str = "refactoring.models.decoding.decoders.factory.act.ACT"
    embedding_dimension: int = 512
    number_of_heads: int = 8  # Number of attention heads
    feedforward_dimension: int = 3200
    number_of_encoder_layers: int = 4
    number_of_decoder_layers: int = 7
    activation: str = ActivationFunction.RELU.value
    dropout_rate: float = 0.1
    normalize_before: bool = False


@dataclass
class FASTDecoderConfig(DecodingNetworkConfig):
    """FAST Decoder for tokenized action prediction.

    Reference: https://arxiv.org/abs/2501.09747

    Autoregressive transformer specifically designed for FAST tokenization:
    - Supports variable-length action token sequences
    - Teacher forcing during training
    - Autoregressive generation during inference
    - Cross-attention to visual features using DETR-style feature encoding (like ACT)
    - GPT-like decoder architecture

    Note: Requires tokenizer to be set at runtime via set_tokenizer().
    The vocab_size must match the tokenizer's vocabulary size (default 2048 for pretrained FAST).
    """
    _target_: str = "refactoring.models.decoding.decoders.factory.fast_decoder.FASTDecoder"
    vocab_size: int = 2048  # Pretrained FAST vocabulary size
    max_seq_len: int = 512  # Maximum sequence length for positional encoding
    embedding_dimension: int = 256
    number_of_heads: int = 8  # Number of attention heads
    feedforward_dimension: int = 512
    number_of_encoder_layers: int = 6
    number_of_decoder_layers: int = 6
    activation: str = ActivationFunction.RELU.value
    dropout_rate: float = 0.1
    normalize_before: bool = False
    eos_token_id: int = 1  # End of sequence token
    pad_token_id: int = 0  # Padding token (default 0)
    deterministic: bool = True  # If True, use greedy decoding during inference
    temperature: float = 1.0  # Sampling temperature for stochastic decoding
    learnable_temperature: bool = True  # If True, make temperature a learnable parameter


# TODO: Implement these decoder architectures
# @dataclass
# class UNetConfig(DecodingNetworkConfig):
#     """U-Net architecture configuration."""
#     _target_: str = "refactoring.models.decoding.decoders.unet.UNetArchitecture"
#     down_dims: Tuple[int, int, int] = (256, 512, 1024)
#     kernel_size: int = 5
#     n_groups: int = 8
#     latent_dim: int = 32


# @dataclass
# class DPTransformerConfig(DecodingNetworkConfig):
#     """Diffusion Policy-like Transformer architecture configuration."""
#     _target_: str = "refactoring.models.decoding.decoders.dp_transformer.DPTransformerArchitecture"
#     kernel_size: int = 5
#     n_groups: int = 8
#     latent_dim: int = 32


# @dataclass
# class MLPConfig(DecodingNetworkConfig):
#     """Simple MLP architecture configuration."""
#     _target_: str = "refactoring.models.decoding.decoders.mlp.MLPArchitecture"
#     hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
#     activation: str = "relu"





@dataclass
class FreeTransformerConfig(DecodingNetworkConfig):
    """Free Transformer architecture configuration.

    Based on "The Free Transformer" (Fleuret, 2025) - arXiv:2510.17558
    https://arxiv.org/abs/2510.17558

    The Free Transformer injects learnable discrete latent variables at the middle layer
    of a transformer decoder, enabling conditional generation through variational inference.

    Uses modern architecture components:
    - RMSNorm for pre-normalization
    - SwiGLU for feedforward activation
    - Binary mapper for discrete latent codes (2^latent_bits total codes)

    Note:
        The encoder is only used during training to predict latent codes.
        During inference, latents are sampled from a uniform prior.
    """
    _target_: str = "refactoring.models.decoding.decoders.factory.free_transformer.FreeTransformer"
    embedding_dimension: int = 256
    number_of_heads: int = 8
    feedforward_dimension: int = 1024
    number_of_decoder_layers: int = 6  # Must be even (split at midpoint for latent injection)
    number_of_encoder_layers: int = 1  # Encoder for latent prediction (training only)
    latent_bits: int = 16  # Number of bits for latent codes (2^16 = 65536 codes)
    dropout_rate: float = 0.1
    use_rope: bool = False  # Rotary positional embeddings (not yet fully implemented)
    rope_base: float = 10000.0


@dataclass
class MixtureOfExpertsDecoderConfig(DecodingNetworkConfig):
    """Mixture of Experts (MoE) decoder configuration.

    Supports two modes:
    1. Explicit experts: Pass list of DecodingNetworkConfig
    2. Config-based (recommended): Pass base_expert_config and num_experts

    Example:
        moe_config = MixtureOfExpertsDecoderConfig(
            base_expert_config=ACTConfig(...),
            num_experts=5,
            gating_input_dim=256,
            routing_type=MoERoutingType.SOFT.value,
            gating_feature_key="latent",  # Use latent from VAE for routing
        )

    Note:
        base_expert_config should be typed as DictConfig | dict to prevent
        Hydra from instantiating it prematurely. The MoEDecoder will
        instantiate it num_experts times internally.
    """
    _target_: str = "refactoring.models.decoding.decoders.mixture_of_experts.MoEDecoder"

    base_expert_config: DictConfig | dict | None = None  # Config with _target_, not pre-instantiated
    num_experts: int | None = None
    expert_configs: list[DictConfig | dict] | None = None  # List of configs with _target_

    gating_input_dim: int | None = None
    gating_hidden_dims: list[int] | None = None
    routing_type: str = MoERoutingType.SOFT.value
    top_k: int = 2
    temperature: float = 1.0
    learnable_temperature: bool = False
    gating_dropout: float = 0.1
    gating_normalization: bool = True
    gating_feature_key: str | None = None  # Key to use as input feature for routing
