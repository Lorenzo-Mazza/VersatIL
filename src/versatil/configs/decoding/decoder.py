"""Configuration classes for different action decoder architectures."""
from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

from versatil.configs.decoding.action_head import MixtureOfExpertsHeadConfig
from versatil.configs.data.task import ActionSpaceConfig, ObservationSpaceConfig
from versatil.models.decoding.constants import MoERoutingType, DiTType, GMMInitStrategy
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import (
    AttentionType,
    PositionalEncodingType,
)
from versatil.models.layers.normalization.constants import NormalizationType


@dataclass
class DecodingNetworkConfig:
    """Base architecture configuration."""

    _target_: str = MISSING
    action_heads: dict[
        str, Any
    ] | None = None  # Any means ActionHeadConfig | MixtureOfExpertsHeadConfig, but custom union type aliases don't work well with omegaconf
    input_keys: list[str] = MISSING
    observation_space: ObservationSpaceConfig = "${policy.observation_space}"  # type: ignore[assignment]
    action_space: ActionSpaceConfig = "${policy.action_space}"  # type: ignore[assignment]
    observation_horizon: int = "${policy.observation_horizon}"  # type: ignore[assignment]
    prediction_horizon: int = "${policy.prediction_horizon}"  # type: ignore[assignment]
    device: str = "${policy.device}"


@dataclass
class ACTConfig(DecodingNetworkConfig):
    """Action Chunking Transformer network config.

    Note: Latent action encoding (e.g., VAE) is now handled at the Algorithm level,
    not at the Decoder level. To use a VAE with ACT, configure the latent_encoder
    parameter in the algorithm config (e.g., BehavioralCloningConfig).
    """

    _target_: str = "versatil.models.decoding.decoders.factory.act.ACT"
    embedding_dimension: int = 512
    number_of_heads: int = 8  # Number of attention heads
    feedforward_dimension: int = 3200
    number_of_encoder_layers: int = 4
    number_of_decoder_layers: int = 7
    activation: str = ActivationFunction.RELU.value
    dropout_rate: float = 0.1
    normalize_before: bool = False


@dataclass
class PhaseACTConfig(ACTConfig):
    """Phase-conditioned ACT decoder with MoE routing.

    Extends the base ACT architecture to support phase-based expert routing.
    The phase classifier head produces routing logits that are used to route
    position and gripper predictions through phase-specific expert networks.
    """

    _target_: str = "versatil.models.decoding.decoders.factory.phase_act.PhaseACT"
    phase_routing_key: str = (
        MISSING  # Key for the phase classifier head that provides routing weights
    )


@dataclass
class FASTDETRDecoderConfig(DecodingNetworkConfig):
    """FAST DETR Decoder for tokenized action prediction.

    DETR non-autoregressive transformer that predicts tokenized actions:
    - Cross-attention to visual features using DETR-style feature encoding (like ACT)

    Note: Requires tokenizer to be set at runtime via set_tokenizer().
    """

    _target_: str = (
        "versatil.models.decoding.decoders.factory.fast_detr_decoder.FASTDETRDecoder"
    )
    max_seq_len: int = 512  # Maximum token sequence length to predict
    embedding_dimension: int = 256
    number_of_heads: int = 8  # Number of attention heads
    feedforward_dimension: int = 512
    number_of_encoder_layers: int = 4
    number_of_decoder_layers: int = 7
    activation: str = ActivationFunction.RELU.value
    dropout_rate: float = 0.1
    normalize_before: bool = False
    deterministic: bool = True  # If True, use greedy decoding during inference
    temperature: float = 1.0  # Sampling temperature for stochastic decoding
    learnable_temperature: bool = (
        True  # If True, make temperature a learnable parameter
    )


@dataclass
class FASTGPTDecoderConfig(DecodingNetworkConfig):
    """GPT Decoder for tokenized action prediction.

    Reference: https://arxiv.org/abs/2501.09747

    Pure GPT-style autoregressive decoder (self-attention only, no cross-attention):
    - Concatenates visual/proprioceptive features as prefix tokens
    - Supports variable-length action token sequences
    - Teacher forcing during training
    - Autoregressive generation with KV caching during inference
    - Works with any feature encoder (spatial, sequential, flat)

    Note: Requires tokenizer to be set at runtime via set_tokenizer().
    """

    _target_: str = (
        "versatil.models.decoding.decoders.factory.fast_gpt_decoder.FASTGPTDecoder"
    )
    max_seq_len: int = 512  # Maximum sequence length for GPT (features + action tokens)
    embedding_dimension: int = 256
    number_of_heads: int = 8  # Number of query attention heads
    number_of_key_value_heads: int | None = (
        None  # Number of K/V heads for GQA (None = same as heads = MHA)
    )
    feedforward_dimension: int | None = (
        None  # FFN hidden dimension (default: 4 * embedding_dimension)
    )
    number_of_layers: int = 6
    activation: str = ActivationFunction.SWIGLU.value  # Activation function
    normalization_type: str = NormalizationType.RMS_NORM.value  # Normalization type
    attention_type: str = AttentionType.GROUPED_QUERY.value  # Attention type
    dropout_rate: float = 0.1
    attention_dropout: float = 0.0
    positional_encoding_type: str | None = (
        PositionalEncodingType.ROPE.value
    )  # Type of positional encoding
    temperature: float = 1.0  # Sampling temperature
    learnable_temperature: bool = (
        False  # If True, make temperature a learnable parameter
    )
    deterministic: bool = True  # If True, use greedy decoding during inference


@dataclass
class ActionTransformerConfig(DecodingNetworkConfig):
    """Action Transformer architecture configuration."""

    _target_: str = (
        "versatil.models.decoding.decoders.factory.action_transformer.ActionTransformer"
    )
    embedding_dimension: int = 256
    number_of_heads: int = 8  # Number of query attention heads
    number_of_key_value_heads: int | None = (
        None  # Number of K/V heads for GQA (None = same as heads = MHA)
    )
    feedforward_dimension: int | None = (
        None  # FFN hidden dimension (default: 4 * embedding_dimension)
    )
    number_of_layers: int = 6
    activation: str = ActivationFunction.SWIGLU.value  # Activation function
    normalization_type: str = NormalizationType.RMS_NORM.value  # Normalization type
    attention_type: str = AttentionType.GROUPED_QUERY.value  # Attention type
    dropout_rate: float = 0.1
    attention_dropout: float = 0.0
    positional_encoding_type: str | None = (
        PositionalEncodingType.ROPE.value
    )  # Type of positional encoding


@dataclass
class MixtureOfDensitiesActionTransformerConfig(DecodingNetworkConfig):
    """Mixture of Densities Action Transformer (MODE-ACT) configuration.

    MODE-ACT extends ActionTransformer with mixture density network capabilities
    for multi-modal action prediction. It uses K copies of each action head and
    a gating network to predict mixture weights.
    """

    _target_: str = (
        "versatil.models.decoding.decoders.factory.mode_act.MixtureOfDensitiesActionTransformer"
    )
    embedding_dimension: int = 256
    number_of_heads: int = 8
    number_of_key_value_heads: int | None = None
    feedforward_dimension: int | None = None
    number_of_layers: int = 6
    activation: str = ActivationFunction.SWIGLU.value
    normalization_type: str = NormalizationType.RMS_NORM.value
    attention_type: str = AttentionType.MULTI_HEAD.value
    dropout_rate: float = 0.1
    attention_dropout: float = 0.0
    positional_encoding_type: str | None = PositionalEncodingType.ROPE.value
    num_mixture_components: int = 8
    gating_hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    gating_activation: str = ActivationFunction.SILU.value
    gating_dropout: float = 0.1
    gating_normalization: bool = True
    temperature: float = 1.0
    learnable_temperature: bool = False
    gating_feature_key: str | None = None
    gmm_init_strategy: str = GMMInitStrategy.KMEANS_PLUS_PLUS.value
    deterministic_inference: bool = True


@dataclass
class LACTConfig(DecodingNetworkConfig):
    """Latent Action Transformer (LACT) architecture configuration.

    LACT extends a standard Action Transformer with latent-conditioned decoding.
    It uses a Pix-Art style DiT with AdaLN-Zero modulation on the latent token instead of the timestep
    and cross-attention to encoder features.

    Must be used with a variational algorithm (e.g., VariationalAlgorithm)
    that provides a latent embedding indexed by LatentKey.POSTERIOR_LATENT in the features dictionary.
    """

    _target_: str = "versatil.models.decoding.decoders.factory.lact.LACT"
    latent_dimension: int = MISSING
    embedding_dimension: int = 256
    number_of_heads: int = 8
    number_of_key_value_heads: int | None = None
    feedforward_dimension: int | None = None
    number_of_layers: int = 6
    activation: str = ActivationFunction.SWIGLU.value
    normalization_type: str = NormalizationType.RMS_NORM.value
    attention_type: str = AttentionType.MULTI_HEAD.value
    positional_encoding_type: str | None = (
        PositionalEncodingType.ROPE.value
    )  # Type of positional encoding
    dropout_rate: float = 0.1
    attention_dropout: float = 0.0
    use_gating: bool = True


@dataclass
class FreeTransformerConfig(DecodingNetworkConfig):
    """Free Transformer architecture configuration.

    Based on "The Free Transformer" (Fleuret, 2025) - arXiv:2510.17558
    https://arxiv.org/abs/2510.17558

    The Free Transformer injects learnable discrete latent variables at the middle layer
    of a transformer decoder, enabling conditional generation through variational inference.

    Note:
        The encoder is only used during training to predict latent codes.
        During inference, latents are sampled from a uniform prior.
    """

    _target_: str = "versatil.models.decoding.decoders.factory.free_transformer.FreeTransformerDecoder"
    embedding_dimension: int = 256
    number_of_decoder_layers: int = (
        6  # Must be even (split at midpoint for latent injection)
    )
    number_of_encoder_layers: int = 1  # Encoder for latent prediction (training only)
    latent_bits: int = 16  # Number of bits for latent codes (2^16 = 65536 codes)
    max_seq_len: int = (
        512  # Maximum input and output sequence length (features + action tokens)
    )
    number_of_heads: int = 8  # Number of query attention heads
    number_of_key_value_heads: int | None = (
        None  # Number of K/V heads for GQA (None = same as heads = MHA)
    )
    feedforward_dimension: int | None = (
        None  # FFN hidden dimension (default: 4 * embedding_dimension)
    )
    activation: str = ActivationFunction.SWIGLU.value  # Activation function
    normalization_type: str = NormalizationType.RMS_NORM.value  # Normalization type
    attention_type: str = AttentionType.GROUPED_QUERY.value  # Attention type
    dropout_rate: float = 0.1
    attention_dropout: float = 0.0
    positional_encoding_type: str | None = (
        PositionalEncodingType.ROPE.value
    )  # Type of positional encoding
    temperature: float = 1.0  # Sampling temperature
    learnable_temperature: bool = (
        False  # If True, make temperature a learnable parameter
    )
    deterministic: bool = True  # If True, use greedy decoding during inference
    use_global_latent: bool = (
        True  # If True, use a single global latent code for the entire sequence
    )


@dataclass
class MoEFreeTransformerConfig(FreeTransformerConfig):
    """Free Transformer with Mixture of Experts head  configuration."""

    _target_: str = "versatil.models.decoding.decoders.factory.moe_free_transformer.MoEFreeTransformer"
    action_heads: dict[str, MixtureOfExpertsHeadConfig] = MISSING


@dataclass
class MixtureOfExpertsDecoderConfig(DecodingNetworkConfig):
    """Mixture of Experts (MoE) decoder configuration."""

    _target_: str = "versatil.models.decoding.decoders.mixture_of_experts.MoEDecoder"
    base_expert: Any = MISSING
    num_experts: int = MISSING
    gating_feature_key: str = MISSING
    inference_gating_key: str | None = None  # If None, uses gating_feature_key
    gating_input_dim: int | None = None
    gating_hidden_dims: list[int] = field(default_factory=list)
    routing_type: str = MoERoutingType.SOFT.value
    top_k: int = 2
    temperature: float = 1.0
    learnable_temperature: bool = False
    gating_dropout: float = 0.1
    gating_normalization: bool = True


@dataclass
class DiTBlockActionTransformerConfig(DecodingNetworkConfig):
    """DiTBlock action transformer with pooled conditioning.

    Encoder-decoder architecture that pools encoder output to a single conditioning
    vector.

    Must be used with a denoising algorithm that provides timesteps and noisy actions.
    """

    _target_: str = "versatil.models.decoding.decoders.factory.dit_block_action_transformer.DiTBlockActionTransformer"
    max_sequence_length: int = 1024
    embedding_dimension: int = 512
    timestep_embedding_dimension: int = 256
    number_of_heads: int = 8
    number_of_key_value_heads: int | None = None
    number_of_encoder_layers: int = 6
    number_of_decoder_layers: int = 6
    feedforward_dimension: int = 2048
    activation: str = ActivationFunction.SWIGLU.value
    normalization_type: str = NormalizationType.RMS_NORM.value
    attention_type: str = AttentionType.MULTI_HEAD.value
    dropout_rate: float = 0.1
    attention_dropout: float = 0.0
    positional_encoding_type: str | None = PositionalEncodingType.ROPE.value
    use_gating: bool = True


@dataclass
class DiffusionActionTransformerConfig(DecodingNetworkConfig):
    """Diffusion action transformer for CrossAttentionDiT and MMDiT.

    Decoder-only architecture that operates on unpooled observation tokens.

    Must be used with a denoising algorithm that provides timesteps and noisy actions.
    """

    _target_: str = "versatil.models.decoding.decoders.factory.diffusion_action_transformer.DiffusionActionTransformer"
    diffusion_transformer_type: str = DiTType.CROSS_ATTENTION.value
    max_sequence_length: int = 1024
    embedding_dimension: int = 512
    timestep_embedding_dimension: int = 256
    number_of_heads: int = 8
    number_of_key_value_heads: int | None = None
    number_of_layers: int = 6
    feedforward_dimension: int = 2048
    activation: str = ActivationFunction.SWIGLU.value
    normalization_type: str = NormalizationType.RMS_NORM.value
    attention_type: str = AttentionType.MULTI_HEAD.value
    dropout_rate: float = 0.1
    attention_dropout: float = 0.0
    positional_encoding_type: str | None = PositionalEncodingType.ROPE.value
    use_gating: bool = True


@dataclass
class ConditionalUNetDecoderConfig(DecodingNetworkConfig):
    """Conditional U-Net decoder configuration."""

    _target_: str = "versatil.models.decoding.decoders.factory.conditional_unet_decoder.ConditionalUNetDecoder"
    embedding_dimension: int = 256
    down_dimensions: list[int] = field(default_factory=lambda: [256, 512, 1024])
    kernel_size: int = 5
    num_groups: int = 8
    use_local_conditioning: bool = False
    condition_predict_scale: bool = False
