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
    """Base architecture configuration.

    Attributes:
        _target_: Import path instantiated by Hydra.
        action_heads: Action head config per predicted action key.
        input_keys: Observation keys consumed as inputs.
        observation_space: Observation space of the task, wired via interpolation.
        action_space: Action space of the task, wired via interpolation.
        observation_horizon: Number of past observation frames consumed.
        prediction_horizon: Number of future actions predicted per chunk.
        device: Torch device for the module.
    """

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

    Attributes:
        _target_: Import path instantiated by Hydra.
        embedding_dimension: Transformer hidden dimension.
        number_of_heads: Number of attention heads.
        feedforward_dimension: Feedforward network dimension.
        number_of_encoder_layers: Number of transformer encoder layers.
        number_of_decoder_layers: Number of transformer decoder layers.
        activation: Activation function name.
        dropout_rate: Dropout probability.
        normalize_before: Use pre-normalization.
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

    Attributes:
        _target_: Import path instantiated by Hydra.
        phase_routing_key: Key for the phase classifier head that provides routing
            weights.
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

    Attributes:
        _target_: Import path instantiated by Hydra.
        max_seq_len: Maximum sequence length for GPT (features + action tokens).
        embedding_dimension: Common embedding dimension to bring input tokens to, also
            Transformer hidden size.
        number_of_heads: Number of query attention heads.
        number_of_key_value_heads: Number of K/V heads for GQA (None = same as heads =
            MHA).
        feedforward_dimension: FFN hidden dimension (default: 4 * embedding_dimension).
        number_of_layers: Number of transformer layers.
        activation: Activation function (swiglu, gelu, relu, silu).
        normalization_type: Normalization type (rmsnorm, layernorm).
        attention_type: Attention type (gqa, mha).
        dropout_rate: Dropout probability.
        attention_dropout: Attention dropout probability.
        positional_encoding_type: Type of positional encoding (sinusoidal, rope, None).
        temperature: Initial temperature for sampling (not used in greedy decoding).
        learnable_temperature: If True, make temperature a learnable parameter.
        deterministic: If True, use greedy decoding during inference.
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
    """Autoregressive VLA action-token decoder backed by a VLM.

    Attributes:
        _target_: Import path instantiated by Hydra.
        action_heads: Must be empty. This decoder predicts action tokens with the VLM
            language vocabulary head.
        input_keys: Must be empty. Raw observation keys are declared by
            ``vlm_backbone.input_specification``.
        vlm_backbone: Generative VLM that builds image-language prefix embeddings and
            exposes the causal language model vocabulary.
        max_seq_len: Maximum prefix plus generated action-token length.
        temperature: Softmax temperature for stochastic inference.
        learnable_temperature: Whether ``temperature`` is optimized as a model
            parameter.
        deterministic: Whether inference uses greedy token selection.
        causal_prefix: Whether to use a standard causal padding mask (OpenVLA) for the
            whole sequence instead of bidirectional prefix attention (Pi0-FAST).
    """

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
    """OpenVLA-OFT-style continuous action chunk decoder backed by a VLM.

    Attributes:
        _target_: Import path instantiated by Hydra.
        input_keys: Must be empty. Raw observation keys are declared by
            ``vlm_backbone.input_specification``.
        vlm_backbone: Generative VLM that builds image-language prefix embeddings and
            exposes the language tower.
        slots_per_action_dimension: When ``True``, each action scalar owns one VLM
            hidden-state slot before the joint action projection. When ``False``, each
            timestep owns one slot.
        causal_action_slots: Whether action slots use causal self-attention.
        min_period: Minimum period for sinusoidal timestep embeddings used by denoising
            algorithms.
        max_period: Maximum period for sinusoidal timestep embeddings used by denoising
            algorithms.
    """

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
    """Action Decoder-only Transformer architecture with cross-attention to encoded features.

    Attributes:
        _target_: Import path instantiated by Hydra.
        embedding_dimension: Embedding dimension of the model tokens.
        number_of_heads: Number of query attention heads.
        number_of_key_value_heads: Key/value head count for grouped-query attention.
        feedforward_dimension: Feedforward layer width.
        number_of_layers: Transformer layer count.
        activation: Activation function.
        normalization_type: Normalization type.
        attention_type: Attention type.
        dropout_rate: Dropout probability.
        attention_dropout: Dropout probability inside attention.
        positional_encoding_type: Self-attention positional encoding: rope, sinusoidal,
            learned, or null.
    """

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

    Attributes:
        _target_: Import path instantiated by Hydra.
        embedding_dimension: Transformer embedding dimension.
        number_of_heads: Number of attention heads.
        number_of_key_value_heads: Number of key-value heads for GQA.
        feedforward_dimension: FFN hidden dimension.
        number_of_layers: Number of decoder layers.
        activation: Activation function.
        normalization_type: Normalization type.
        attention_type: Attention type.
        dropout_rate: Dropout rate.
        attention_dropout: Attention dropout rate.
        positional_encoding_type: Positional encoding type.
        num_mixture_components: Number of mixture components (K).
        gating_hidden_dims: Hidden dimensions for gating MLP.
        gating_activation: Activation for gating MLP.
        gating_dropout: Dropout rate in gating MLP.
        gating_normalization: Whether to normalize gating input.
        temperature: Temperature for softmax scaling.
        learnable_temperature: Whether temperature is learnable.
        gating_feature_key: If set, use this feature for gating instead of mode
            embedding.
        gmm_init_strategy: Strategy for initializing GMM component means.
        inference_sampling_mode: How to sample from the mixture at inference.
            DETERMINISTIC: argmax component, return mean. STOCHASTIC_MEAN: multinomial
            component, return mean (no noise). STOCHASTIC_SAMPLE: multinomial component,
            add Gaussian noise.
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

    Attributes:
        _target_: Import path instantiated by Hydra.
        latent_dimension: Dimension of latent conditioning vector.
        embedding_dimension: Transformer hidden dimension.
        number_of_heads: Number of attention heads.
        number_of_key_value_heads: Number of K/V heads for GQA (None for MHA).
        feedforward_dimension: FFN hidden dimension (default: 4 * embedding_dimension).
        number_of_layers: Number of conditional transformer decoder layers.
        activation: Activation function name.
        normalization_type: Type of adaptive normalization layer.
        attention_type: Type of attention mechanism (multi-head, grouped query, etc.).
        positional_encoding_type: Type of positional encoding.
        dropout_rate: Dropout probability for residual connections.
        attention_dropout: Dropout probability for attention weights.
        use_gating: Whether to use AdaLN-Zero gating on residual connections.
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
    """Mixture of Experts (MoE) decoder configuration.

    Attributes:
        _target_: Import path instantiated by Hydra.
        base_expert: Decoder config replicated per expert.
        num_experts: Number of expert decoders.
        gating_feature_key: Feature key routed to the gating network during training.
        inference_gating_key: If None, uses gating_feature_key.
        gating_input_dim: Gating network input dimension; null derives routing from the
            latent.
        gating_hidden_dims: Gating network hidden layer widths.
        routing_type: Expert routing: soft or top_k.
        top_k: Experts activated per sample with top_k routing.
        temperature: Routing softmax temperature.
        learnable_temperature: Whether the routing temperature is learned.
        gating_dropout: Dropout probability inside the gating network.
        gating_normalization: Normalization layer name inside the gating network.
    """

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

    Attributes:
        _target_: Import path instantiated by Hydra.
        max_sequence_length: Maximum sequence length for input tokens.
        embedding_dimension: Transformer hidden dimension.
        timestep_embedding_dimension: Diffusion timestep embedding dimension.
        number_of_heads: Number of attention heads.
        number_of_key_value_heads: Number of K/V heads for GQA.
        number_of_encoder_layers: Number of transformer encoder layers.
        number_of_decoder_layers: Number of transformer decoder layers.
        feedforward_dimension: Feedforward network dimension.
        activation: Activation function name.
        normalization_type: Normalization type name.
        attention_type: Attention type name (gqa, mha).
        dropout_rate: Dropout probability for residual connections.
        attention_dropout: Dropout probability for attention weights.
        positional_encoding_type: Type of positional encoding.
        use_gating: Whether to use gating in AdaLN-Zero layers.
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

    Attributes:
        _target_: Import path instantiated by Hydra.
        diffusion_transformer_type: Type of Diffusion Transformer architecture.
        max_sequence_length: Maximum sequence length for input tokens.
        embedding_dimension: Transformer hidden dimension.
        timestep_embedding_dimension: Diffusion timestep embedding dimension.
        number_of_heads: Number of attention heads.
        number_of_key_value_heads: Number of K/V heads for GQA.
        number_of_layers: Number of transformer layers.
        feedforward_dimension: Feedforward network dimension.
        activation: Activation function name.
        normalization_type: Normalization type name.
        attention_type: Attention type name (gqa, mha).
        dropout_rate: Dropout probability for residual connections.
        attention_dropout: Dropout probability for attention weights.
        positional_encoding_type: Type of positional encoding.
        use_gating: Whether to use gating in AdaLN-Zero layers.
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

    Attributes:
        _target_: Import path instantiated by Hydra.
        embedding_dimension: Diffusion timestep embedding dimension.
        down_dimensions: List of channel dimensions for downsampling layers.
        kernel_size: Kernel size for convolutions in residual blocks.
        num_groups: Number of groups for group normalization.
        use_local_conditioning: Whether to use local (sequence-aligned) conditioning.
        condition_predict_scale: If True, conditions predict scaling factors in FiLM.
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

    Attributes:
        _target_: Import path instantiated by Hydra.
        vlm_backbone: Generative VLM backbone that builds the raw observation prefix
            with shape ``(B, P, D_vlm)``.
        expert_width_multiplier: Expert hidden size as fraction of VLM hidden size.
        num_expert_layers: Number of expert layers. ``-1`` uses the same count as VLM.
        num_vlm_layers: Number of VLM layers to use (truncates if fewer than available).
        self_attention_every_n_layers: Period for joint self-attention layers. ``0``
            disables joint self-attention (all cross- attention).
        proprioceptive_feature_key: Feature key for proprioceptive state from the
            encoding pipeline. When set, the feature is prepended to the VLM prefix
            before interleaved processing. None disables state prepend.
        min_period: Minimum period for sinusoidal timestep embedding.
        max_period: Maximum period for sinusoidal timestep embedding.
        freeze_vlm: Whether to freeze VLM layer parameters (disable gradients).
        normalization_type: Normalization layer type.
        activation: Activation function for expert feedforward layers.
        dropout: Dropout rate.
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

    Attributes:
        _target_: Import path instantiated by Hydra.
        vlm_backbone: Generative VLM backbone that builds the raw observation prefix
            with shape ``(B, P, D_vlm)``.
        expert_hidden_size: Expert network hidden dimension.
        expert_intermediate_size: Expert feedforward intermediate dimension.
        expert_number_of_heads: Number of attention heads in expert layers.
        expert_number_of_key_value_heads: Number of K/V heads in expert layers.
        expert_number_of_layers: Number of expert layers (must match VLM layers).
        expert_head_dimension: Per-head dimension in expert layers.
        time_conditioning: Timestep conditioning mode (use TimeConditioning enum
            values).
        min_period: Minimum period for sinusoidal timestep embedding.
        max_period: Maximum period for sinusoidal timestep embedding.
        proprioceptive_feature_key: Feature key for proprioceptive state. When set, the
            feature is prepended to the VLM prefix.
        normalization_type: Normalization layer type.
        activation: Activation function for expert feedforward layers.
        dropout: Dropout rate.
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
