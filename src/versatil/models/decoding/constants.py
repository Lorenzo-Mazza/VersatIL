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


class MoERoutingType(str, enum.Enum):
    """Enum for different Mixture of Experts (MoE) routing strategies."""

    TOP_K = "top_k"  # Select the top-k experts based on gating scores
    SOFT = "soft"  # Weighted combination of all experts based on gating scores


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
    CROSS_ATTENTION = "cross_attention" # PixArt style Cross-Attention DiT
    MMDIT = "mmdit" # Multimodal Diffusion Transformer (Stable Diffusion 3 style)
    DIT_BLOCK = "dit_block" # DiT-Block (DiT Policy style)

# Feature keys for time/timestep conditioning
TIMESTEP_KEY = (
    "timestep"  # For diffusion or flow processes (discrete or continuous timestep)
)
TARGET_DIFFUSION_KEY = "target_diffusion"  # Target sample (noisy action) for diffusion
NOISE_KEY = "noise"
TARGET_VELOCITY_KEY = "target_velocity"  # Target velocity for flow matching

#: Mixture of Experts keys
EXPERT_USAGE = "expert_usage"
ROUTING_ENTROPY = "routing_entropy"
TOP_EXPERT_CONFIDENCE = "top_expert_confidence"
ROUTING_WEIGHT = "routing_weights"
EXPERT_OUTPUTS = "expert_outputs"

#: Phase keys
PHASE_LOGITS_KEY = "phase_logits"
LATENT_KEY = "latent"  # Latent embedding (z)

# Latent posteriors and priors keys
MU_KEY = "mu"  # VAE latent mean
LOGVAR_KEY = "logvar"  # VAE latent log variance
PRIOR_PREDICTION_KEY = "prior_prediction"  # Prior network prediction
PRIOR_TARGET_KEY = "prior_target"  # Prior network target
PRIOR_MU_KEY = "prior_mu"  # Prior latent mean
PRIOR_LOGVAR_KEY = "prior_logvar"  # Prior latent log variance
PRIOR_LATENT_KEY = "prior_latent"  # Prior latent embedding (z)
PRIOR_LOG_PROB_KEY = "prior_log_prob"  # Prior log probability (for mixture priors)

# Free Transformer keys
BINARY_LOGITS_KEY = "binary_logits"  # Binary mapper logits for KL divergence
LATENT_CODES = "latent_codes"  # Sampled latent z

# Action logits and predicted action tokens keys
ACTION_LOGITS_KEY = "action_logits"
PREDICTED_ACTION_TOKENS_KEY = "pred_action_tokens"

CLASS_TOKEN_KEY = "cls_token"
