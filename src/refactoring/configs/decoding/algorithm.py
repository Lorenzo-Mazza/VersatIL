"""Action-decoding algorithm configurations."""
from dataclasses import dataclass

from omegaconf import MISSING

from refactoring.configs.decoding.latent import (
    PosteriorLatentEncoderConfig,
    PriorLatentEncoderConfig,
)
from refactoring.models.decoding.constants import (
    BetaSchedule,
    ODESolver,
    PredictionType,
    VarianceType,
)
from refactoring.models.layers.diffusion_process import SchedulerType


@dataclass
class DecodingAlgorithmConfig:
    """Base algorithm configuration.

    Note: For variational algorithms, use VariationalAlgorithmConfig instead
    of setting latent_encoder on individual algorithms.
    """

    _target_: str = MISSING


@dataclass
class BehavioralCloningConfig(DecodingAlgorithmConfig):
    """Behavioral Cloning (direct supervised prediction) algorithm configuration.

    This is a pure, deterministic algorithm. For multi-modal action prediction,
    use VariationalAlgorithmConfig with BehavioralCloningConfig as the base_algorithm.
    """

    _target_: str = (
        "refactoring.models.decoding.algorithm.behavior_cloning.BehavioralCloning"
    )


@dataclass
class DiffusionConfig(DecodingAlgorithmConfig):
    """Diffusion algorithm configuration."""

    _target_: str = "refactoring.models.decoding.algorithm.diffusion.Diffusion"

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
    """Flow Matching algorithm configuration."""

    _target_: str = "refactoring.models.decoding.algorithm.flow_matching.FlowMatching"
    sigma: float = 0.0
    ode_solver: str = ODESolver.EULER.value
    num_inference_steps: int = 10


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
            prior=DiffusionPriorConfig(...)
        )

        # NEW: Diffusion with VAE + learned prior
        VariationalAlgorithmConfig(
            base_algorithm=DiffusionConfig(...),
            posterior_encoder=VAETransformerEncoderConfig(...),
            prior=DiffusionPriorConfig(...)
        )

    Args:
        base_algorithm: The base decoding algorithm (BC, FlowMatching, Diffusion, etc.)
        posterior_encoder: Latent encoder for posterior q(z|a,s) (typically VAE)
        prior: Latent prior for p(z|s). If None, auto-creates GaussianPrior.
    """

    _target_: str = (
        "refactoring.models.decoding.algorithm.variational.VariationalAlgorithm"
    )
    base_algorithm: DecodingAlgorithmConfig = MISSING  # type: ignore[assignment]
    posterior_encoder: PosteriorLatentEncoderConfig = MISSING  # type: ignore[assignment]
    prior: PriorLatentEncoderConfig = MISSING
