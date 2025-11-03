"""Latent action posterior and prior network configurations."""
from dataclasses import dataclass

from omegaconf import MISSING

from refactoring.models.layers.activation import ActivationFunction


@dataclass
class LatentActionEncoderConfig:
    """Base latent action encoder configuration (for posteriors)."""
    _target_: str = MISSING
    latent_dim: int = MISSING
    output_dim: int = MISSING
    device: str = "${policy.device}"


@dataclass
class LatentPriorConfig:
    """Base latent prior configuration."""
    _target_: str = MISSING
    latent_dim: int = MISSING
    output_dim: int = MISSING
    device: str = "${policy.device}"


@dataclass
class VAETransformerEncoderConfig(LatentActionEncoderConfig):
    """Transformer-based VAE latent action encoder configuration.

    This encoder uses a transformer architecture to encode action sequences into
    a latent space via variational inference.
    """
    _target_: str = "refactoring.models.decoding.latent.vae_posterior.VAETransformerEncoder"

    latent_dim: int = 32
    output_dim: int = 512
    prediction_horizon: int = "${policy.prediction_horizon}"  # type: ignore[assignment]

    number_of_heads: int = 8
    feedforward_dimension: int = 512
    number_of_encoder_layers: int = 4
    activation: str = ActivationFunction.RELU.value
    dropout_rate: float = 0.1
    normalize_before: bool = False

    use_proprioceptive: bool = False


@dataclass
class GaussianPriorConfig(LatentPriorConfig):
    """Standard Gaussian N(0, I) prior configuration.

    Simple non-learned prior that samples from a standard normal distribution.
    This is the default prior for variational algorithms when no learned prior is specified.

    Args:
        latent_dim: Dimension of latent variable z
        output_dim: Dimension to project latent to (for decoder input)
        device: Device to place prior on
    """
    _target_: str = "refactoring.models.decoding.latent.gaussian_prior.GaussianPrior"

    latent_dim: int = 32
    output_dim: int = 512


@dataclass
class DiffusionPriorConfig(LatentPriorConfig):
    """Diffusion-based learned prior configuration.

    Uses a diffusion MLP to learn p(z|s) instead of using N(0,I) prior.
    Trained via denoising to match posterior q(z|a,s) samples from VAE.

    Args:
        latent_dim: Dimension of latent variable z
        conditioning_dim: Dimension of conditioning features (state)
        output_dim: Dimension to project latent to (for decoder input)
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
    _target_: str = "refactoring.models.decoding.latent.diffusion_prior.DiffusionPrior"

    latent_dim: int = 32
    conditioning_dim: int = 128  # Should match the sum of the dimension of the flat state features
    output_dim: int = 512

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


