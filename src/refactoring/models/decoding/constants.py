import enum


class MoERoutingType(str, enum.Enum):
    """Enum for different Mixture of Experts (MoE) routing strategies."""
    TOP_K = 'top_k'  # Select the top-k experts based on gating scores
    SOFT = 'soft'  # Weighted combination of all experts based on gating scores


class FeatureType(str, enum.Enum):
    """Feature types that the decoder can optionally require."""
    SPATIAL = "spatial"  # Features with (C, H, W) dimensions (besides batch/time)
    SEQUENTIAL = "sequential"  # Features with (T, D) dimensions
    FLAT = "flat"  # Flat features with single dimension


class SchedulerType(str, enum.Enum):
    """Diffusion scheduler types (compatible with diffusers API)."""
    DDIM = "ddim"
    DDPM = "ddpm"


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


# Feature keys for time/timestep conditioning
TIMESTEP_KEY = "timestep"  # For diffusion models (discrete timestep)
TARGET_DIFFUSION_KEY = "target_diffusion"  # Target sample (noisy action) for diffusion
NOISE_KEY = "noise"


# Flow Matching keys
TARGET_VELOCITY_KEY = "target_velocity"  # Target velocity for flow matching
TIME_KEY = "time"  # For flow matching (continuous time in [0, 1])

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

# Free Transformer keys
BINARY_LOGITS_KEY = "binary_logits"  # Binary mapper logits for KL divergence

# This key is used to store state features in the algorithm outputs for computing OT loss.
STATE_FEATURE_KEYS = "state_features"

# Action logits and tokens for tokenizers
ACTION_LOGITS_KEY = "action_logits"
ACTION_TOKENS_KEY = "action_tokens"
ACTION_TOKENS_TARGET_KEY = "action_tokens_target"  # Ground truth token IDs for loss
