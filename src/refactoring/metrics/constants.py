"""Constants for loss computation and metrics tracking."""

from enum import Enum


class MetricKey(str, Enum):
    """Keys for metric tracking."""
    TOTAL_LOSS = "loss"
    MSE_LOSS = "mse_loss"
    L1_LOSS = "l1_loss"
    HUBER_LOSS = "huber_loss"
    BCE_LOSS = "binary_cross_entropy"
    GRIPPER_BCE = "gripper_bce"
    GRIPPER_MSE = "gripper_mse"
    ORIENTATION_MSE = "orientation_mse"
    ORIENTATION_LOSS = "orientation_loss"
    POSITION_LOSS = "position_loss"
    KL_DIVERGENCE = "kl_divergence"
    RAW_KL_DIVERGENCE = "raw_kl_divergence"
    CLAMPED_KL_DIVERGENCE = "clamped_kl"
    LATENT_CODE_USAGE = "latent_usage_ratio"
    POSTERIOR_ENTROPY = "posterior_entropy"
    SINKHORN_LOSS = "sinkhorn_loss"
    LENGTH_LOSS = "length_loss"
    SMOOTHNESS_LOSS = "smoothness_loss"
    PHASE_CROSS_ENTROPY = "phase_cross_entropy"
    PHASE_ENTROPY = "phase_entropy"
    PHASE_ACCURACY = "phase_accuracy"
    PRIOR_DENOISING_LOSS = "prior_denoising_loss"
    DIFFUSION_LOSS = "diffusion_loss"
    FLOW_MATCHING_LOSS = "flow_matching_loss"
    OPTIMAL_TRANSPORT_LOSS = "optimal_transport_loss"
    ACTION_TOKEN_CROSS_ENTROPY = "action_token_cross_entropy"
    TOKEN_ACCURACY = "token_accuracy_ratio"
    PERPLEXITY = "perplexity"


class LossModuleName(str, Enum):
    """Names for loss modules in composite losses."""
    REGRESSION = "regression"
    GRIPPER = "gripper"
    KL = "kl"
    LENGTH = "length"
    SMOOTHNESS = "smoothness"
    PHASE = "phase"
    ACTION = "action"
    DIFFUSION = "diffusion"


class PredictionKey(str, Enum):
    """Keys for model predictions."""

    NOISE_PRED = "noise_pred"
    SAMPLE_PRED = "sample_pred"


class TargetKey(str, Enum):
    """Keys for target values."""
    NOISE = "noise"
    NOISY_SAMPLE = "noisy_sample"


class MetadataKey(str, Enum):
    """Keys for metadata stored in LossOutput."""
    PHASE_LOGITS = "phase_logits"
    PHASE_LABELS = "phase_labels"
