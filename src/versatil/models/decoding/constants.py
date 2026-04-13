import enum


class LatentKey(enum.StrEnum):
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


class MoERoutingType(enum.StrEnum):
    """Enum for different Mixture of Experts (MoE) routing strategies."""

    TOP_K = "top_k"
    SOFT = "soft"


class GMMInitStrategy(enum.StrEnum):
    """Initialization strategies for GMM mixture components."""

    KMEANS_PLUS_PLUS = "kmeans_plus_plus"  # K-means++ style spread for means
    UNIFORM = "uniform"  # Simple uniform spread


class PredictionType(enum.StrEnum):
    """What the diffusion model predicts."""

    EPSILON = "epsilon"  # Predict noise
    SAMPLE = "sample"  # Predict clean sample (x0)
    VELOCITY = "velocity"  # Predict velocity


class BetaSchedule(enum.StrEnum):
    """Beta schedule for diffusion models."""

    LINEAR = "linear"
    SCALED_LINEAR = "scaled_linear"
    SQUAREDCOS_CAP_V2 = "squaredcos_cap_v2"


class VarianceType(enum.StrEnum):
    """Variance type for DDPM scheduler."""

    FIXED_SMALL = "fixed_small"
    FIXED_LARGE = "fixed_large"
    LEARNED = "learned"
    LEARNED_RANGE = "learned_range"


class ODESolver(enum.StrEnum):
    """ODE solver types for flow matching."""

    EULER = "euler"  # First-order Euler method
    HEUN = "heun"  # Second-order Heun's method
    RK4 = "rk4"  # Fourth-order Runge-Kutta
    DOPRI5 = "dopri5"  # Dormand-Prince 5th order adaptive step size


class DenoisingAlgorithm(enum.StrEnum):
    """Algorithm type for denoising-based generative models."""

    DIFFUSION = "diffusion"
    FLOW_MATCHING = "flow_matching"


class MixtureSamplingMode(enum.StrEnum):
    """Sampling strategy for mixture-of-experts inference."""

    DETERMINISTIC = "deterministic"  # argmax component, return mean
    STOCHASTIC_MEAN = "stochastic_mean"  # multinomial component, return mean
    STOCHASTIC_SAMPLE = "stochastic_sample"  # multinomial component, add Gaussian noise


class DiTType(enum.StrEnum):
    """Types of Diffusion Transformer architectures."""

    CROSS_ATTENTION = "cross_attention"  # PixArt style Cross-Attention DiT
    MMDIT = "mmdit"  # Multimodal Diffusion Transformer (Stable Diffusion 3 style)
    DIT_BLOCK = "dit_block"  # DiT-Block (DiT Policy style)


class TimeConditioning(enum.StrEnum):
    """Timestep conditioning strategy for denoising decoders."""

    CONCAT_MLP = "concat_mlp"
    ADANORM = "adanorm"


class DecoderOutputKey(enum.StrEnum):
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
