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
    """Base posterior encoder configuration."""

    _target_: str = MISSING
    latent_dimension: int = MISSING
    device: str = "${policy.device}"


@dataclass
class PriorLatentEncoderConfig:
    """Base latent prior configuration."""

    _target_: str = MISSING
    latent_dimension: int = MISSING
    device: str = "${policy.device}"


@dataclass
class VAETransformerEncoderConfig(PosteriorLatentEncoderConfig):
    """Transformer-based VAE latent action encoder configuration.

    This encoder uses a transformer architecture to encode action sequences into
    a latent space via variational inference.
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

    """

    _target_: str = "versatil.models.decoding.latent.prior.gaussian_prior.GaussianPrior"
    latent_dimension: int = 32


@dataclass
class PriorTransformerEncoderConfig(PriorLatentEncoderConfig):
    """Configuration for the transformer-based prior latent encoder."""

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
    """VQ posterior encoder configuration."""

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
    """Uniform categorical prior over VQ codebook indices."""

    _target_: str = "versatil.models.decoding.latent.prior.uniform_codebook_prior.UniformCodebookPrior"
    latent_dimension: int = 8
    num_codes: int = 4
    num_residual_layers: int = 1


@dataclass
class CodebookPriorConfig(PriorLatentEncoderConfig):
    """Learned categorical prior over VQ codebook indices."""

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
