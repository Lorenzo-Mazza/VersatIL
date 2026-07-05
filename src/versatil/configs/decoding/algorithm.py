"""Action-decoding algorithm configurations."""

from dataclasses import dataclass

from omegaconf import MISSING

from versatil.configs.decoding.latent import (
    PosteriorLatentEncoderConfig,
    PriorLatentEncoderConfig,
)
from versatil.models.decoding.constants import (
    BetaSchedule,
    ODESolver,
    PredictionType,
    VarianceType,
)
from versatil.models.layers.denoising.diffusion_process import SchedulerType
from versatil.models.layers.denoising.timestep_sampling import TimestepSampler


@dataclass
class DecodingAlgorithmConfig:
    """Base algorithm configuration.

    Note: For variational algorithms, use VariationalAlgorithmConfig instead
    of setting latent_encoder on individual algorithms.

    Attributes:
        _target_: Import path instantiated by Hydra.
    """

    _target_: str = MISSING


@dataclass
class BehavioralCloningConfig(DecodingAlgorithmConfig):
    """Behavioral Cloning (direct supervised prediction) algorithm configuration.

    This is a pure, deterministic algorithm. For multi-modal action prediction,
    use VariationalAlgorithmConfig with BehavioralCloningConfig as the base_algorithm.
    """

    _target_: str = (
        "versatil.models.decoding.algorithm.behavior_cloning.BehavioralCloning"
    )


@dataclass
class DiffusionConfig(DecodingAlgorithmConfig):
    """Diffusion algorithm configuration.

    Attributes:
        _target_: Import path instantiated by Hydra.
        scheduler_type: Type of diffusion scheduler ("ddpm" or "ddim").
        num_train_timesteps: Number of diffusion steps during training.
        num_inference_steps: Number of denoising steps during inference.
        beta_start: Starting value of noise schedule.
        beta_end: Ending value of noise schedule.
        beta_schedule: Noise schedule type ("linear", "squaredcos_cap_v2", etc.).
        prediction_type: What the network predicts ("epsilon" for noise, "sample" for
            clean actions).
        scheduler_variance_type: Variance type for DDPM scheduler.
        clip_sample: Whether to clip samples to [-1, 1] during inference.
        set_alpha_to_one: Whether to set final alpha to 1.
        steps_offset: Offset for timestep calculation.
    """

    _target_: str = "versatil.models.decoding.algorithm.diffusion.Diffusion"

    scheduler_type: str = SchedulerType.DDIM.value
    num_train_timesteps: int = 100
    num_inference_steps: int = 10
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = BetaSchedule.SQUAREDCOS_CAP_V2.value
    prediction_type: str = PredictionType.EPSILON.value

    scheduler_variance_type: str = VarianceType.FIXED_SMALL.value
    clip_sample: bool = True
    set_alpha_to_one: bool = True
    steps_offset: int = 0


@dataclass
class FlowMatchingConfig(DecodingAlgorithmConfig):
    """Flow Matching algorithm configuration.

    Attributes:
        _target_: Import path instantiated by Hydra.
        sigma: Noise level for conditional flow matching (0 = deterministic OT).
        ode_solver: ODE solver to use ("euler", "heun", or "rk4").
        num_inference_steps: Number of integration steps during inference.
        timestep_sampler: Timestep sampling strategy.
        logit_mean: Mean for logit-normal (shifts mode; 0 centers at t=0.5).
        logit_std: Std for logit-normal (smaller = more concentrated).
        beta_alpha: First shape parameter for Beta distribution (pi0 uses 1.5).
        beta_beta: Second shape parameter for Beta distribution (pi0 uses 1.0).
        max_timestep: Upper bound s for Beta sampling (pi0 uses 0.999).
        reverse_flow_convention: Reverse the time/velocity convention during inference.
            When True, the inference loop remaps t to (1-t) and negates the predicted
            velocity.
    """

    _target_: str = "versatil.models.decoding.algorithm.flow_matching.FlowMatching"
    sigma: float = 0.0
    ode_solver: str = ODESolver.EULER.value
    num_inference_steps: int = 10
    timestep_sampler: str = TimestepSampler.BETA.value
    logit_mean: float = 0.0
    logit_std: float = 1.0
    beta_alpha: float = 1.5
    beta_beta: float = 1.0
    max_timestep: float = 0.999
    reverse_flow_convention: bool = False


@dataclass
class VariationalAlgorithmConfig(DecodingAlgorithmConfig):
    """Compositional variational inference wrapper configuration.

    Wraps any base algorithm with variational latent encoding for multi-modal action prediction.
    This replaces the need for algorithm-specific variational implementations.

    Examples:
        # Behavioral Cloning with VAE + Gaussian prior
        VariationalAlgorithmConfig(
            base_algorithm=BehavioralCloningConfig(),
            posterior_encoder=VAETransformerEncoderConfig(...),
            prior=None  # Auto-creates GaussianPrior
        )

        # Flow Matching with VAE + Diffusion prior (replaces VariationalFlowMatching)
        VariationalAlgorithmConfig(
            base_algorithm=FlowMatchingConfig(sigma=0.0, num_inference_steps=10),
            posterior_encoder=VAETransformerEncoderConfig(...),
            prior=DiTPriorConfig(...)
        )

        # NEW: Diffusion with VAE + learned prior
        VariationalAlgorithmConfig(
            base_algorithm=DiffusionConfig(...),
            posterior_encoder=VAETransformerEncoderConfig(...),
            prior=DiTPriorConfig(...)
        )

    Attributes:
        _target_: Import path instantiated by Hydra.
        base_algorithm: The base decoding algorithm (BC, FlowMatching, Diffusion, etc.)
        posterior_encoder: Latent encoder for posterior q(z|a,s) (typically VAE)
        prior: Latent prior for p(z|s). If None, auto-creates GaussianPrior.
    """

    _target_: str = (
        "versatil.models.decoding.algorithm.variational.VariationalAlgorithm"
    )
    base_algorithm: DecodingAlgorithmConfig = MISSING
    posterior_encoder: PosteriorLatentEncoderConfig = MISSING
    prior: PriorLatentEncoderConfig = MISSING
    sampling_from_prior_probability: float = 0.25
    posterior_decoder_noise_std: float = 0.0
