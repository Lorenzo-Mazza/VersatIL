import enum


class LatentKey(str, enum.Enum):
    """Enum for latent-related feature keys used in variational models."""

    POSTERIOR_LATENT = "latent"
    POSTERIOR_MU = "mu"
    POSTERIOR_LOGVAR = "logvar"
    PRIOR_MU = "prior_mu"
    PRIOR_LOGVAR = "prior_logvar"
    PRIOR_LATENT = "prior_latent"
    PRIOR_PREDICTION = "prior_prediction"
    PRIOR_TARGET = "prior_target"
    PRIOR_LOG_PROB = "prior_log_prob"


class MoERoutingType(str, enum.Enum):
    """Enum for different Mixture of Experts (MoE) routing strategies."""

    TOP_K = "top_k"
    SOFT = "soft"


class GMMInitStrategy(str, enum.Enum):
    """Initialization strategies for GMM mixture components."""

    KMEANS_PLUS_PLUS = "kmeans_plus_plus"  # K-means++ style spread for means
    UNIFORM = "uniform"  # Simple uniform spread 


class PredictionType(str, enum.Enum):
    """What the diffusion model predicts."""

    EPSILON = "epsilon"  # Predict noise
    SAMPLE = "sample"  # Predict clean sample (x0)
    VELOCITY = "velocity"  # Predict velocity


class BetaSchedule(str, enum.Enum):
    """Beta schedule for diffusion models."""

    LINEAR = "linear"
    SCALED_LINEAR = "scaled_linear"
    SQUAREDCOS_CAP_V2 = "squaredcos_cap_v2"


class VarianceType(str, enum.Enum):
    """Variance type for DDPM scheduler."""

    FIXED_SMALL = "fixed_small"
    FIXED_LARGE = "fixed_large"
    LEARNED = "learned"
    LEARNED_RANGE = "learned_range"


class ODESolver(str, enum.Enum):
    """ODE solver types for flow matching."""

    EULER = "euler"  # First-order Euler method
    HEUN = "heun"  # Second-order Heun's method
    RK4 = "rk4"  # Fourth-order Runge-Kutta
    DOPRI5 = "dopri5"  # Dormand-Prince 5th order adaptive step size


class DenoisingAlgorithm(str, enum.Enum):
    """Algorithm type for denoising-based generative models."""

    DIFFUSION = "diffusion"
    FLOW_MATCHING = "flow_matching"


class DiTType(str, enum.Enum):
    """Types of Diffusion Transformer architectures."""

    CROSS_ATTENTION = "cross_attention"  # PixArt style Cross-Attention DiT
    MMDIT = "mmdit"  # Multimodal Diffusion Transformer (Stable Diffusion 3 style)
    DIT_BLOCK = "dit_block"  # DiT-Block (DiT Policy style)


class DecoderOutputKey(str, enum.Enum):
    """Keys for decoder outputs and intermediate features."""

    TIMESTEP = "timestep"
    TARGET_DIFFUSION = "target_diffusion"
    NOISE = "noise"
    TARGET_VELOCITY = "target_velocity"
    EXPERT_USAGE = "expert_usage"
    ROUTING_ENTROPY = "routing_entropy"
    TOP_EXPERT_CONFIDENCE = "top_expert_confidence"
    ROUTING_WEIGHTS = "routing_weights"
    EXPERT_OUTPUTS = "expert_outputs"
    PHASE_LOGITS = "phase_logits"
    BINARY_LOGITS = "binary_logits"
    LATENT_CODES = "latent_codes"
    ACTION_LOGITS = "action_logits"
    PREDICTED_ACTION_TOKENS = "pred_action_tokens"
    CLASS_TOKEN = "cls_token"
    MEAN = "mean"
    LOGVAR = "logvar"


class FeatureType(str, enum.Enum):
    """Feature types for decoder validation.

    - SPATIAL: (C, H, W) - image features from CNN/ViT
    - SEQUENTIAL: (T, D) - sequence features from transformers
    - FLAT: int or (D,) - pooled/embedded features
    """

    SPATIAL = "spatial"
    SEQUENTIAL = "sequential"
    FLAT = "flat"
