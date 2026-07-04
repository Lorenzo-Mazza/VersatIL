"""Latent action posterior and prior network configurations."""

from dataclasses import dataclass

from omegaconf import MISSING

from versatil.configs.data.task import ActionSpaceConfig
from versatil.models.decoding.constants import (
    BetaSchedule,
    DenoisingAlgorithm,
    LatentKey,
    ODESolver,
    PredictionType,
)
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType
from versatil.models.layers.denoising.diffusion_process import SchedulerType
from versatil.models.layers.denoising.timestep_sampling import TimestepSampler
from versatil.models.layers.normalization.constants import NormalizationType


@dataclass
class PosteriorLatentEncoderConfig:
    """Base posterior encoder configuration.

    Attributes:
        _target_: Import path instantiated by Hydra.
        latent_dimension: Dimension of the latent variable.
        device: Torch device for the module.
    """

    _target_: str = MISSING
    latent_dimension: int = MISSING
    device: str = "${policy.device}"


@dataclass
class PriorLatentEncoderConfig:
    """Base latent prior configuration.

    Attributes:
        _target_: Import path instantiated by Hydra.
        latent_dimension: Dimension of latent variable z.
        device: Device to place prior on.
    """

    _target_: str = MISSING
    latent_dimension: int = MISSING
    device: str = "${policy.device}"


@dataclass
class VAETransformerEncoderConfig(PosteriorLatentEncoderConfig):
    """Transformer-based VAE latent action encoder configuration.

    This encoder uses a transformer architecture to encode action sequences into
    a latent space via variational inference.

    Attributes:
        _target_: Import path instantiated by Hydra.
        latent_dimension: Dimension of VAE latent space, i.e. the dimension of the
            output z.
        embedding_dimension: Dimension of the output embedding.
        prediction_horizon: Number of action timesteps.
        observation_horizon: Number of observation timesteps.
        device: Device to place encoder on.
        number_of_heads: Number of attention heads.
        feedforward_dimension: Feedforward network dimension.
        number_of_encoder_layers: Number of transformer encoder layers.
        activation: Activation function name.
        dropout_rate: Dropout probability.
        attention_dropout: Dropout probability inside attention.
        normalization_type: Normalization layer type.
        attention_type: Attention mechanism type (use AttentionType enum values).
        positional_encoding_type: Self-attention positional encoding type.
        exclude_keys: List of keys to exclude from encoding.
        min_logvar: Minimum log variance for avoiding variance collapse.
        deterministic: If True, output deterministic embeddings without
            reparameterization. Use with MMD or OT regularizers instead of KL
            divergence.
        mu_tanh_bound: Optional symmetric bound for posterior mu. When set, applies
            ``bound * tanh(raw_mu / bound)`` before sampling/returning z.
        max_logvar: Optional maximum log variance for avoiding variance explosion.
    """

    _target_: str = "versatil.models.decoding.latent.posterior.transformer_encoder.VAETransformerEncoder"
    latent_dimension: int = MISSING
    embedding_dimension: int = MISSING
    prediction_horizon: int = "${policy.prediction_horizon}"
    observation_horizon: int = "${policy.observation_horizon}"
    device: str = "${policy.device}"
    number_of_heads: int = 8
    feedforward_dimension: int = 512
    number_of_encoder_layers: int = 4
    activation: str = ActivationFunction.SWIGLU.value
    dropout_rate: float = 0.1
    attention_dropout: float = 0.0
    normalization_type: str = NormalizationType.RMS_NORM.value
    attention_type: str = AttentionType.MULTI_HEAD.value
    positional_encoding_type: str | None = None
    exclude_keys: list[str] | None = None
    min_logvar: float | None = None
    deterministic: bool = False
    mu_tanh_bound: float | None = None
    max_logvar: float | None = None


@dataclass
class GaussianPriorConfig(PriorLatentEncoderConfig):
    """Standard Gaussian N(0, I) prior configuration.

    Simple non-learned prior that samples from a standard normal distribution.
    This is the default prior for variational algorithms when no learned prior is specified.


    Attributes:
        _target_: Import path instantiated by Hydra.
        latent_dimension: Dimension of latent variable z.
        infer_constant_prior: ACT-style constant zero latent at inference instead of
            N(0, I) samples.
    """

    _target_: str = "versatil.models.decoding.latent.prior.gaussian_prior.GaussianPrior"
    latent_dimension: int = 32
    # ACT-style constant zero latent at inference instead of N(0, I) samples.
    infer_constant_prior: bool = False


@dataclass
class PriorTransformerEncoderConfig(PriorLatentEncoderConfig):
    """Configuration for the transformer-based prior latent encoder.

    Attributes:
        _target_: Import path instantiated by Hydra.
        latent_dimension: Dimension of the latent variable.
        embedding_dimension: Embedding dimension of the model tokens.
        prediction_horizon: Number of future actions predicted per chunk.
        observation_horizon: Number of past observation frames consumed.
        device: Torch device for the module.
        number_of_heads: Attention head count.
        feedforward_dimension: Feedforward layer width.
        number_of_encoder_layers: Transformer encoder layer count.
        activation: Activation function name.
        dropout_rate: Dropout probability.
        attention_dropout: Dropout probability inside attention.
        normalization_type: Normalization layer type.
        attention_type: Attention implementation name.
        positional_encoding_type: Self-attention positional encoding type.
        exclude_keys: Feature keys excluded from prior conditioning.
        learn_variance: Whether the prior variance is learned instead of fixed.
        min_logvar: Lower clamp for the learned log-variance.
        deterministic: Whether sampling returns the mean instead of drawing noise.
        max_logvar: Upper clamp for the learned log-variance.
    """

    _target_: str = "versatil.models.decoding.latent.prior.transformer_encoder.PriorTransformerEncoder"
    latent_dimension: int = MISSING
    embedding_dimension: int = MISSING
    prediction_horizon: int = "${policy.prediction_horizon}"
    observation_horizon: int = "${policy.observation_horizon}"
    device: str = "${policy.device}"
    number_of_heads: int = 8
    feedforward_dimension: int = 512
    number_of_encoder_layers: int = 4
    activation: str = ActivationFunction.SWIGLU.value
    dropout_rate: float = 0.1
    attention_dropout: float = 0.0
    normalization_type: str = NormalizationType.RMS_NORM.value
    attention_type: str = AttentionType.MULTI_HEAD.value
    positional_encoding_type: str | None = None
    exclude_keys: list[str] | None = None
    learn_variance: bool = True
    min_logvar: float | None = None
    deterministic: bool = False
    max_logvar: float | None = None


@dataclass
class VampPriorConfig(PriorLatentEncoderConfig):
    """VampPrior (Variational Mixture of Posteriors) configuration.

    Reference: "VAE with a VampPrior" (Tomczak & Welling, 2018)

    Attributes:
        _target_: Import path instantiated by Hydra.
        latent_dimension: Dimension of latent variable z.
        num_components: Number of mixture components K.
        action_space: ActionSpace defining the action dimensions.
        prediction_horizon: Number of timesteps in action chunks.
        min_logvar: Optional minimum logvar clamp.
    """

    _target_: str = "versatil.models.decoding.latent.prior.vamp_prior.VampPrior"
    latent_dimension: int = 32
    num_components: int = 50
    action_space: ActionSpaceConfig = "${policy.action_space}"
    prediction_horizon: int = "${policy.prediction_horizon}"
    min_logvar: float | None = None


@dataclass
class DiTPriorConfig(PriorLatentEncoderConfig):
    """DiT-style transformer prior for denoising score matching.

    Uses a non-autoregressive diffusion transformer where noisy latent z is treated
    as a CLS token appended to observation tokens.

    Attributes:
        _target_: Import path instantiated by Hydra.
        latent_dimension: Dimension of latent variable z.
        embedding_dimension: Hidden dimension of the transformer.
        number_of_heads: Number of attention heads.
        number_of_layers: Number of DiT decoder layers.
        feedforward_dimension: Dimension of the feedforward network.
        observation_horizon: Observation history size.
        algorithm_type: Algorithm type ("diffusion" or "flow_matching").
        sigma: Noise level for flow matching (0 = deterministic OT).
        ode_solver: ODE solver for flow matching ("euler", "heun", or "rk4").
        timestep_sampler: Distribution the diffusion timestep is drawn from.
        logit_mean: Mean of the logit-normal timestep sampler.
        logit_std: Standard deviation of the logit-normal timestep sampler.
        beta_alpha: Alpha parameter of the beta timestep sampler.
        beta_beta: Beta parameter of the beta timestep sampler.
        max_timestep: Largest sampled diffusion timestep.
        num_train_timesteps: Number of diffusion timesteps during training.
        num_inference_steps: Number of denoising/integration steps.
        beta_start: Starting beta for noise schedule (diffusion).
        beta_end: Ending beta for noise schedule (diffusion).
        beta_schedule: Type of noise schedule (diffusion).
        scheduler_type: Diffusion scheduler type.
        prediction_type: What diffusion model predicts (epsilon, sample, velocity).
        clip_sample: Whether to clip samples during diffusion.
        variance_type: Variance type for DDPM scheduler.
        dropout: Dropout rate.
        normalization_type: Type of adaptive normalization layer.
        attention_type: Attention implementation name.
        number_of_key_value_heads: Key/value head count for grouped-query attention.
        activation: Activation function name.
        use_gating: Whether to use AdaLN-Zero gating in DiT layers.
        exclude_keys: Keys to exclude from observations.
        prior_target_key: Posterior output key used as denoising target.
        latent_standardization_enabled: Whether to standardize DiT target latents.
        latent_standardization_eps: Numerical epsilon used in latent standardization.
        latent_standardization_max_batches: Maximum train batches to scan when fitting
            latent standardization stats. ``None`` scans the full train loader.
        require_fitted_latent_standardization: Whether missing latent stats should
            raise.
    """

    _target_: str = "versatil.models.decoding.latent.prior.dit_prior.DiTPrior"
    latent_dimension: int = 32
    embedding_dimension: int = 256
    number_of_heads: int = 8
    number_of_layers: int = 4
    feedforward_dimension: int = 1024
    observation_horizon: int = "${policy.observation_horizon}"
    algorithm_type: str = DenoisingAlgorithm.FLOW_MATCHING.value
    sigma: float = 0.0
    ode_solver: str = ODESolver.EULER.value
    timestep_sampler: str = TimestepSampler.BETA.value
    logit_mean: float = 0.0
    logit_std: float = 1.0
    beta_alpha: float = 1.5
    beta_beta: float = 1.0
    max_timestep: float = 0.999
    num_train_timesteps: int = 100
    num_inference_steps: int = 10
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = BetaSchedule.SQUAREDCOS_CAP_V2.value
    scheduler_type: str = SchedulerType.DDIM.value
    prediction_type: str = PredictionType.EPSILON.value
    clip_sample: bool = False
    variance_type: str | None = None
    dropout: float = 0.1
    normalization_type: str = NormalizationType.LAYER_NORM.value
    attention_type: str = AttentionType.MULTI_HEAD.value
    number_of_key_value_heads: int | None = None
    activation: str = ActivationFunction.SILU.value
    use_gating: bool = True
    exclude_keys: list[str] | None = None
    prior_target_key: str = LatentKey.POSTERIOR_MU.value
    latent_standardization_enabled: bool = True
    latent_standardization_eps: float = 1e-6
    latent_standardization_max_batches: int | None = None
    require_fitted_latent_standardization: bool = False


@dataclass
class VQPosteriorEncoderConfig(PosteriorLatentEncoderConfig):
    """VQ posterior encoder configuration.

    Attributes:
        _target_: Import path instantiated by Hydra.
        latent_dimension: Dimension of each codebook vector and the latent space passed
            to the decoder.
        num_codes: Number of codebook entries per residual layer (K).
        num_residual_layers: Number of cascading VQ layers.
        embedding_dimension: Transformer hidden dimension.
        prediction_horizon: Number of action timesteps.
        observation_horizon: Number of observation timesteps.
        device: Device string.
        ema_decay: EMA decay for codebook updates.
        dead_code_threshold: Cluster size below which codes are replaced.
        number_of_heads: Number of attention heads.
        feedforward_dimension: Feedforward network dimension.
        number_of_encoder_layers: Number of transformer encoder layers.
        activation: Activation function name.
        dropout_rate: Dropout probability.
        attention_dropout: Dropout probability inside attention.
        normalization_type: Normalization layer type.
        attention_type: Attention mechanism type (use AttentionType enum values).
        positional_encoding_type: Self-attention positional encoding type.
        exclude_keys: Observation keys to exclude from encoding.
    """

    _target_: str = (
        "versatil.models.decoding.latent.posterior.vq_encoder.VQPosteriorEncoder"
    )
    latent_dimension: int = 8
    num_codes: int = 4
    num_residual_layers: int = 1
    embedding_dimension: int = 64
    prediction_horizon: int = "${policy.prediction_horizon}"
    observation_horizon: int = "${policy.observation_horizon}"
    device: str = "${policy.device}"
    ema_decay: float = 0.99
    dead_code_threshold: float = 1.0
    number_of_heads: int = 4
    feedforward_dimension: int = 128
    number_of_encoder_layers: int = 1
    activation: str = ActivationFunction.SWIGLU.value
    dropout_rate: float = 0.0
    attention_dropout: float = 0.0
    normalization_type: str = NormalizationType.RMS_NORM.value
    attention_type: str = AttentionType.MULTI_HEAD.value
    positional_encoding_type: str | None = None
    exclude_keys: list[str] | None = None


@dataclass
class UniformCodebookPriorConfig(PriorLatentEncoderConfig):
    """Uniform categorical prior over VQ codebook indices.

    Attributes:
        _target_: Import path instantiated by Hydra.
        latent_dimension: Dimension of each codebook vector.
        num_codes: Number of codebook entries per layer (K).
        num_residual_layers: Number of residual VQ layers.
    """

    _target_: str = "versatil.models.decoding.latent.prior.uniform_codebook_prior.UniformCodebookPrior"
    latent_dimension: int = 8
    num_codes: int = 4
    num_residual_layers: int = 1


@dataclass
class CodebookPriorConfig(PriorLatentEncoderConfig):
    """Learned categorical prior over VQ codebook indices.

    Attributes:
        _target_: Import path instantiated by Hydra.
        latent_dimension: Dimension of each codebook vector. Must match the posterior
            encoder's latent dimension.
        num_codes: Number of codebook entries per layer (K).
        num_residual_layers: Number of residual VQ layers.
        embedding_dimension: Transformer hidden dimension.
        observation_horizon: Number of observation timesteps.
        device: Device string.
        number_of_heads: Number of attention heads.
        feedforward_dimension: Feedforward network dimension.
        number_of_encoder_layers: Number of transformer encoder layers.
        activation: Activation function name.
        dropout_rate: Dropout probability.
        attention_dropout: Dropout probability inside attention.
        normalization_type: Normalization layer type.
        attention_type: Attention mechanism type (use AttentionType enum values).
        positional_encoding_type: Self-attention positional encoding type.
        exclude_keys: Observation keys to exclude from encoding.
        temperature: Softmax temperature for sampling. Lower values produce sharper
            categorical distributions.
    """

    _target_: str = "versatil.models.decoding.latent.prior.codebook_prior.CodebookPrior"
    latent_dimension: int = 8
    num_codes: int = 4
    num_residual_layers: int = 1
    embedding_dimension: int = 64
    observation_horizon: int = "${policy.observation_horizon}"
    device: str = "${policy.device}"
    number_of_heads: int = 4
    feedforward_dimension: int = 128
    number_of_encoder_layers: int = 1
    activation: str = ActivationFunction.SWIGLU.value
    dropout_rate: float = 0.0
    attention_dropout: float = 0.0
    normalization_type: str = NormalizationType.RMS_NORM.value
    attention_type: str = AttentionType.MULTI_HEAD.value
    positional_encoding_type: str | None = None
    exclude_keys: list[str] | None = None
    temperature: float = 1.0
