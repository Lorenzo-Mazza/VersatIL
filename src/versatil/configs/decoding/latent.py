"""Latent action posterior and prior network configurations."""
from dataclasses import dataclass

from omegaconf import MISSING

from versatil.configs.data.task import ActionSpaceConfig
from versatil.models.decoding.constants import (
    BetaSchedule,
    DenoisingAlgorithm,
    ODESolver,
    PredictionType,
)
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.diffusion_process import SchedulerType


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
    prediction_horizon: int = "${policy.prediction_horizon}"  # type: ignore[assignment]
    observation_horizon: int = "${policy.observation_horizon}"  # type: ignore[assignment]
    device: str = "${policy.device}"  # type: ignore[assignment]
    number_of_heads: int = 8
    feedforward_dimension: int = 512
    number_of_encoder_layers: int = 4
    activation: str = ActivationFunction.SWIGLU.value
    dropout_rate: float = 0.1
    normalize_before: bool = False
    exclude_keys: list[str] | None = None
    min_logvar: float | None = None  
    
    
@dataclass
class GaussianPriorConfig(PriorLatentEncoderConfig):
    """Standard Gaussian N(0, I) prior configuration.

    Simple non-learned prior that samples from a standard normal distribution.
    This is the default prior for variational algorithms when no learned prior is specified.

    """

    _target_: str = (
        "versatil.models.decoding.latent.prior.gaussian_prior.GaussianPrior"
    )
    latent_dimension: int = 32


@dataclass
class PriorTransformerEncoderConfig(PriorLatentEncoderConfig):
    _target_: str = "versatil.models.decoding.latent.prior.transformer_encoder.PriorTransformerEncoder"
    latent_dimension: int = MISSING
    embedding_dimension: int = MISSING
    prediction_horizon: int = "${policy.prediction_horizon}"  # type: ignore[assignment]
    observation_horizon: int = "${policy.observation_horizon}"  # type: ignore[assignment]
    device: str = "${policy.device}"  # type: ignore[assignment]
    number_of_heads: int = 8
    feedforward_dimension: int = 512
    number_of_encoder_layers: int = 4
    activation: str = ActivationFunction.SWIGLU.value
    dropout_rate: float = 0.1
    normalize_before: bool = False
    exclude_keys: list[str] | None = None
    learn_variance: bool = True
    min_logvar: float | None = None


@dataclass
class VampPriorConfig(PriorLatentEncoderConfig):
    """VampPrior (Variational Mixture of Posteriors) configuration.

    Reference: "VAE with a VampPrior" (Tomczak & Welling, 2018)
    """

    _target_: str = (
        "versatil.models.decoding.latent.prior.vamp_prior.VampPrior"
    )
    latent_dimension: int = 32
    num_components: int = 50
    action_space: ActionSpaceConfig = "${policy.action_space}"  # type: ignore[assignment]
    prediction_horizon: int = "${policy.prediction_horizon}"  # type: ignore[assignment]
    min_logvar: float | None = None


@dataclass
class DenoisingTransformerPriorConfig(PriorLatentEncoderConfig):
    """DiT-style transformer prior for denoising score matching.

    Uses a non-autoregressive transformer where noisy latent z is treated
    as a CLS token appended to observation tokens.
    """

    _target_: str = (
        "versatil.models.decoding.latent.prior.denoising_transformer.DenoisingTransformerPrior"
    )
    latent_dimension: int = 32
    embedding_dimension: int = 256
    number_of_heads: int = 8
    number_of_layers: int = 4
    feedforward_dimension: int = 1024
    observation_horizon: int = "${policy.observation_horizon}"  # type: ignore[assignment]
    algorithm_type: str = DenoisingAlgorithm.FLOW_MATCHING.value
    sigma: float = 0.0
    ode_solver: str = ODESolver.EULER.value
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
    activation: str = ActivationFunction.SILU.value
    use_gating: bool = True
    exclude_keys: list[str] | None = None
