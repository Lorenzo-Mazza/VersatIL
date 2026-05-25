"""Configuration classes for different action decoder architectures."""

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

from versatil.configs.data.task import ActionSpaceConfig, ObservationSpaceConfig
from versatil.models.decoding.constants import (
    DiTType,
    GMMInitStrategy,
    MixtureSamplingMode,
    MoERoutingType,
    TimeConditioning,
)
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
    action_heads: dict[str, Any] | None = (
        None  # Any means ActionHeadConfig | MixtureOfExpertsHeadConfig, but custom union type aliases don't work well with omegaconf
    )
    input_keys: list[str] = MISSING
    observation_space: ObservationSpaceConfig = "${policy.observation_space}"
    action_space: ActionSpaceConfig = "${policy.action_space}"
    observation_horizon: int = "${policy.observation_horizon}"
    prediction_horizon: int = "${policy.prediction_horizon}"
    device: str = "${policy.device}"


@dataclass
class ACTConfig(DecodingNetworkConfig):
    """Action Chunking Transformer with encoder-decoder architecture and parallel generation.

    Note:
        Ref. https://arxiv.org/abs/2304.13705
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

    Note:
        Cf. https://arxiv.org/abs/2601.21971
        Extends the base ACT architecture to support phase-based expert routing.
        The phase classifier head produces routing logits that are used to route
        action predictions through phase-specific expert networks.
    """

    _target_: str = "versatil.models.decoding.decoders.factory.phase_act.PhaseACT"
    phase_routing_key: str = (
        MISSING  # Key for the phase classifier head that provides routing weights
    )


@dataclass
class GPTActionTransformerConfig(DecodingNetworkConfig):
    """Autoregressive transformer that models a categorical distribution over discrete action tokens.

    Pure GPT-style autoregressive action decoder:
    - Concatenates visual/proprioceptive features as prefix tokens
    - Supports variable-length action token sequences (e.g. for FAST tokenization)
    - Teacher forcing during training
    - Autoregressive generation with KV caching during inference
    """

    _target_: str = "versatil.models.decoding.decoders.factory.gpt_action_transformer.GPTActionTransformer"
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
class AutoregressiveVLAConfig(DecodingNetworkConfig):
    """Autoregressive VLA action-token decoder backed by a VLM."""

    _target_: str = (
        "versatil.models.decoding.decoders.factory.autoregressive_vla."
        "AutoregressiveVLADecoder"
    )
    action_heads: dict[str, Any] = field(default_factory=dict)
    input_keys: list[str] = field(default_factory=list)
    vlm_backbone: Any = MISSING
    max_seq_len: int = 512
    temperature: float = 1.0
    learnable_temperature: bool = False
    deterministic: bool = True
    causal_prefix: bool = False


@dataclass
class OpenVLAOFTConfig(DecodingNetworkConfig):
    """OpenVLA-OFT-style continuous action chunk decoder backed by a VLM."""

    _target_: str = (
        "versatil.models.decoding.decoders.factory.openvla_oft.OpenVLAOFTDecoder"
    )
    input_keys: list[str] = field(default_factory=list)
    vlm_backbone: Any = MISSING
    slots_per_action_dimension: bool = True
    causal_action_slots: bool = True
    min_period: float = 4e-3
    max_period: float = 4.0


@dataclass
class ActionTransformerConfig(DecodingNetworkConfig):
    """Action Decoder-only Transformer architecture with cross-attention to encoded features."""

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

    _target_: str = "versatil.models.decoding.decoders.factory.mode_act.MixtureOfDensitiesActionTransformer"
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
    inference_sampling_mode: str = MixtureSamplingMode.STOCHASTIC_MEAN.value


@dataclass
class LACTConfig(DecodingNetworkConfig):
    """Latent Action Transformer (LACT) architecture configuration.

    LACT extends a standard Action Transformer with latent-conditioned decoding.
    It uses a Pix-Art style DiT with AdaLN-Zero modulation on the latent token
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
class MixtureOfExpertsDecoderConfig(DecodingNetworkConfig):
    """Mixture of Experts (MoE) decoder configuration."""

    _target_: str = "versatil.models.decoding.decoders.moe.MoEDecoder"
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

    Note:
        Ref. https://arxiv.org/abs/2410.10088v1
        It uses an encoder-decoder architecture that pools encoder output to a single
         conditioning vector.
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
    """Diffusion action transformer for CrossAttentionDiT and MultiModal DiT.

    Note:
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
class ConditionalActionUNetConfig(DecodingNetworkConfig):
    """Conditional U-Net action decoder configuration with FiLM conditioning.
    Note:
        Ref. https://diffusion-policy.cs.columbia.edu.
        Must be used with a denoising algorithm that provides timesteps and noisy actions.
    """

    _target_: str = "versatil.models.decoding.decoders.factory.conditional_action_unet.ConditionalActionUNet"
    embedding_dimension: int = 256
    down_dimensions: list[int] = field(default_factory=lambda: [256, 512, 1024])
    kernel_size: int = 5
    num_groups: int = 8
    use_local_conditioning: bool = False
    condition_predict_scale: bool = False


@dataclass
class SmolVLADecoderConfig(DecodingNetworkConfig):
    """SmolVLA Vision Language Action model with interleaved VLM + expert cross-attention.

    Note:
        Ref. https://arxiv.org/abs/2506.01844
        Must be used with a denoising algorithm that provides timesteps and noisy actions.
    """

    _target_: str = "versatil.models.decoding.decoders.factory.smolvla.SmolVLADecoder"
    vlm_backbone: Any = MISSING
    expert_width_multiplier: float = 0.75
    num_expert_layers: int = -1
    num_vlm_layers: int = 16
    self_attention_every_n_layers: int = 2
    proprioceptive_feature_key: str | None = None
    min_period: float = 4e-3
    max_period: float = 4.0
    freeze_vlm: bool = True
    normalization_type: str = NormalizationType.RMS_NORM.value
    activation: str = ActivationFunction.SWIGLU.value
    dropout: float = 0.1


@dataclass
class Pi0DecoderConfig(DecodingNetworkConfig):
    """Pi0/Pi0.5 Vision Language Action model with interleaved VLM + expert joint attention.

    Note:
        Ref. https://arxiv.org/abs/2410.24164, https://arxiv.org/abs/2504.16054
        Must be used with a denoising algorithm that provides timesteps and noisy actions.
    """

    _target_: str = "versatil.models.decoding.decoders.factory.pi0.Pi0Decoder"
    vlm_backbone: Any = MISSING
    expert_hidden_size: int = 1024
    expert_intermediate_size: int = 4096
    expert_number_of_heads: int = 8
    expert_number_of_key_value_heads: int = 1
    expert_number_of_layers: int = 18
    expert_head_dimension: int = 256
    time_conditioning: str = TimeConditioning.CONCAT_MLP.value
    min_period: float = 4e-3
    max_period: float = 4.0
    proprioceptive_feature_key: str | None = None
    normalization_type: str = NormalizationType.RMS_NORM.value
    activation: str = ActivationFunction.GEGLU.value
    dropout: float = 0.0
