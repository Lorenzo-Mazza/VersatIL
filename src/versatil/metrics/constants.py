"""Constants for loss computation and metrics tracking."""

from enum import StrEnum


class MetricKey(StrEnum):
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
    MMD_LOSS = "mmd_loss"
    BINARY_MMD_LOSS = "binary_mmd_loss"
    LATENT_CODE_USAGE = "latent_usage_ratio"
    POSTERIOR_ENTROPY = "posterior_entropy"
    ENTROPY = "entropy"
    SINKHORN_LOSS = "sinkhorn_loss"
    LENGTH_LOSS = "length_loss"
    SMOOTHNESS_LOSS = "smoothness_loss"
    PHASE_CROSS_ENTROPY = "phase_cross_entropy"
    PHASE_ENTROPY = "phase_entropy"
    PHASE_ACCURACY = "phase_accuracy"
    PRIOR_DENOISING_LOSS = "prior_denoising_loss"
    PRIOR_DENOISING_TARGET_STD = "prior_denoising_target_std"
    PRIOR_DENOISING_NORMALIZED_MSE = "prior_denoising_normalized_mse"
    PRIOR_DENOISING_NORMALIZED_RMSE = "prior_denoising_normalized_rmse"
    DIFFUSION_LOSS = "diffusion_loss"
    FLOW_MATCHING_LOSS = "flow_matching_loss"
    OPTIMAL_TRANSPORT_LOSS = "optimal_transport_loss"
    ACTION_TOKEN_CROSS_ENTROPY = "action_token_cross_entropy"
    TOKEN_ACCURACY = "token_accuracy_ratio"
    PERPLEXITY = "perplexity"
    ACTION_NLL = "action_negative_log_likelihood"
    GRIPPER_NLL = "gripper_negative_log_likelihood"
    GAUSSIAN_MIXTURE_NLL = "gaussian_mixture_nll"
    BERNOULLI_MIXTURE_NLL = "bernoulli_mixture_nll"
    EXPERTS_ENTROPY = "experts_entropy"
    EXPERTS_LOAD_BALANCE = "experts_load_balance"
    HYPERPRIOR_KL_REGULARIZATION = "hyperprior_kl_regularization"
    HYPERPRIOR_MMD_REGULARIZATION = "hyperprior_mmd_regularization"
    COVARIANCE_LOSS = "covariance_loss"
    VARIANCE_LOSS = "variance_loss"
    POSTERIOR_GEOMETRY_MEAN_LOSS = "posterior_geometry_mean_loss"
    POSTERIOR_GEOMETRY_STD_LOSS = "posterior_geometry_std_loss"
    POSTERIOR_GEOMETRY_MAX_STD_LOSS = "posterior_geometry_max_std_loss"
    POSTERIOR_GEOMETRY_COVARIANCE_LOSS = "posterior_geometry_covariance_loss"
    VQ_COMMITMENT_LOSS = "vq_commitment_loss"
    VQ_CODEBOOK_USAGE = "vq_codebook_usage"
    VQ_PRIOR_CROSS_ENTROPY = "vq_prior_cross_entropy"


class LossModuleName(StrEnum):
    """Names for loss modules in composite losses."""

    REGRESSION = "regression"
    GRIPPER = "gripper"
    KL = "kl"
    LENGTH = "length"
    SMOOTHNESS = "smoothness"
    PHASE = "phase"
    ACTION = "action"
    DIFFUSION = "diffusion"


class PredictionKey(StrEnum):
    """Keys for model predictions."""

    NOISE_PRED = "noise_pred"
    SAMPLE_PRED = "sample_pred"


class TargetKey(StrEnum):
    """Keys for target values."""

    NOISE = "noise"
    NOISY_SAMPLE = "noisy_sample"


class MetadataKey(StrEnum):
    """Keys for metadata stored in LossOutput."""

    LATENT_COLOR_LABEL = "latent_color_label"
    PHASE_LOGITS = "phase_logits"
    PHASE_LABEL = "phase_label"
    EXPERT_USAGE = "expert_usage"
    POSTERIOR_Z = "posterior_z"
    POSTERIOR_MU = "posterior_mu"
    POSTERIOR_LOGVAR = "posterior_logvar"
    PRIOR_Z = "prior_z"
    PRIOR_MU = "prior_mu"
    PRIOR_LOGVAR = "prior_logvar"
    HYPERPRIOR_Z = "hyperprior_z"
