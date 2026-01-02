"""Latent action posterior and prior network configurations."""
from dataclasses import dataclass

from omegaconf import MISSING

from refactoring.models.layers.activation import ActivationFunction


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

    _target_: str = "refactoring.models.decoding.latent.posterior.transformer_encoder.VAETransformerEncoder"
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


@dataclass
class GaussianPriorConfig(PriorLatentEncoderConfig):
    """Standard Gaussian N(0, I) prior configuration.

    Simple non-learned prior that samples from a standard normal distribution.
    This is the default prior for variational algorithms when no learned prior is specified.

    Args:
        latent_dim: Dimension of latent variable z
        device: Device to place prior on
    """

    _target_: str = (
        "refactoring.models.decoding.latent.prior.gaussian_prior.GaussianPrior"
    )
    latent_dimension: int = 32


@dataclass
class PriorTransformerEncoderConfig(PriorLatentEncoderConfig):
    _target_: str = "refactoring.models.decoding.latent.prior.transformer_encoder.PriorTransformerEncoder"
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


@dataclass
class DiffusionPriorConfig(PriorLatentEncoderConfig):
    """Diffusion-based learned prior configuration.

    Uses a diffusion MLP to learn p(z|s) instead of using N(0,I) prior.
    Trained via denoising to match posterior q(z|a,s) samples from VAE.

    Args:
        latent_dim: Dimension of latent variable z
        conditioning_dim: Dimension of conditioning features (state)
        hidden_dims: Hidden layer dimensions for denoising network
        num_train_timesteps: Number of diffusion timesteps during training
        num_inference_steps: Number of denoising steps during sampling
        beta_start: Starting beta for noise schedule
        beta_end: Ending beta for noise schedule
        beta_schedule: Type of noise schedule
        activation: Activation function for MLP
        dropout: Dropout rate
        device: Device to place prior on
    """

    _target_: str = (
        "refactoring.models.decoding.latent.prior.diffusion_mlp.DiffusionPrior"
    )

    latent_dimension: int = 32
    conditioning_dim: int = (
        128  # Should match the sum of the dimension of the flat state features
    )

    # Denoising network architecture
    hidden_dims: list[int] | None = None  # Defaults to [latent_dim*2, latent_dim*2]

    # Diffusion parameters
    num_train_timesteps: int = 100
    num_inference_steps: int = 10
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "squaredcos_cap_v2"

    # Network parameters
    activation: str = ActivationFunction.RELU.value
    dropout: float = 0.1
