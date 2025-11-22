"""Diffusion-based learned prior for variational models.

The prior learns to sample from p(z|s) using a diffusion model, where z is the
latent variable and s is the conditioning. During training, it learns to denoise
latent samples from the posterior. During inference, it generates latents via
reverse diffusion.
"""

import torch
import torch.nn as nn

from refactoring.models.layers.diffusion_process import (
    DiffusionSchedulerConfig,
    add_noise_to_tensor,
    create_noise_scheduler,
    sample_random_timesteps,
    setup_inference_timesteps, SchedulerType,
)
from refactoring.models.decoding.constants import PRIOR_PREDICTION_KEY, PRIOR_TARGET_KEY
from refactoring.models.decoding.latent import LatentPrior
from refactoring.models.layers.mlp import MLP
from refactoring.models.layers.activation import ActivationFunction


class DiffusionPrior(LatentPrior):
    """Diffusion-based learned prior for latent variable models.

    Learns to sample from p(z|s) using a simple diffusion MLP model, where z is the latent
    variable and s is the conditioning (state features). This is used instead of
    a standard Gaussian prior N(0,I) to better match the posterior distribution.
    An example use case can be found in https://arxiv.org/html/2508.01622v2.

    During training, the prior is trained to denoise latent samples from the posterior.
    During inference, it generates latent samples via the reverse diffusion process.

    Args:
        latent_dimension: Dimension of latent variable z
        conditioning_dim: Dimension of conditioning features (state)
        hidden_dims: Hidden layer dimensions for denoising network
        num_train_timesteps: Number of diffusion timesteps during training
        num_inference_steps: Number of denoising steps during sampling
        beta_start: Starting beta for noise schedule
        beta_end: Ending beta for noise schedule
        beta_schedule: Type of noise schedule ("linear", "scaled_linear", "squaredcos_cap_v2")
        activation: Activation function for MLP
        dropout: Dropout rate
        device: Device to place prior on
    """

    def __init__(
        self,
        latent_dimension: int,
        conditioning_dim: int,
        device: str,
        hidden_dims: list[int] | None = None,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 10,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "squaredcos_cap_v2",
        activation: str = ActivationFunction.RELU.value,
        dropout: float = 0.1,
    ):
        """Initialize diffusion prior."""
        super().__init__(latent_dimension=latent_dimension, device=device)
        self.conditioning_dim = conditioning_dim
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        scheduler_config = DiffusionSchedulerConfig(
            scheduler_type=SchedulerType.DDPM.value,
            num_train_timesteps=num_train_timesteps,
            num_inference_steps=num_inference_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            prediction_type="epsilon",
            clip_sample=False,  # Don't clip latents
            variance_type=None,  # Use default
        )
        self.noise_scheduler = create_noise_scheduler(scheduler_config)
        # Denoising network: takes (noisy_z, conditioning, timestep) -> predicted_noise
        if hidden_dims is None:
            hidden_dims = [latent_dimension * 2, latent_dimension * 2]

        self.timestep_embed_dim = latent_dimension
        if activation == ActivationFunction.SWIGLU.value:
            activation_fn = ActivationFunction(activation).to_torch_activation()(
                input_dim=self.timestep_embed_dim, hidden_dim=self.timestep_embed_dim)
        else:
            activation_fn = ActivationFunction(activation).to_torch_activation()()
        self.timestep_mlp = nn.Sequential(
            nn.Linear(1, self.timestep_embed_dim),
            activation_fn,
            nn.Linear(self.timestep_embed_dim, self.timestep_embed_dim),
        )
        input_dim = latent_dimension + conditioning_dim + self.timestep_embed_dim
        self.denoising_network = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=latent_dimension,
            activation_function=ActivationFunction(activation).to_torch_activation(),
            dropout=dropout,
        )
        self.to(torch.device(device))

    def sample_prior(self, batch_size: int, conditioning: torch.Tensor | None = None) -> torch.Tensor:
        """Sample latent variable from learned prior p(z|s) via reverse diffusion.

        Args:
            batch_size: Number of samples to generate
            conditioning: Conditioning features (state) with shape (batch_size, conditioning_dim)

        Returns:
            Sampled latent embeddings (batch_size, latent_dim)
        """
        device = next(self.parameters()).device
        if conditioning is None:
            # Unconditional sampling
            conditioning = torch.zeros(batch_size, self.conditioning_dim, device=device)
        # Start from pure noise
        z_t = torch.randn(batch_size, self.latent_dimension, device=device)
        setup_inference_timesteps(self.noise_scheduler, self.num_inference_steps)

        # Reverse diffusion process
        for t in self.noise_scheduler.timesteps:
            timestep = torch.tensor([t], device=device).float().view(1, 1).expand(batch_size, 1)
            timestep_embed = self.timestep_mlp(timestep / self.num_train_timesteps)
            model_input = torch.cat([z_t, conditioning, timestep_embed], dim=-1)
            predicted_noise = self.denoising_network(model_input)
            # Compute previous sample using scheduler
            z_t = self.noise_scheduler.step(predicted_noise, t, z_t).prev_sample
        return z_t

    def forward(
        self,
        target_latents: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute denoising loss for training the prior to match posterior samples.

        Args:
            target_latents: Latent samples from posterior q(z|a,s) with shape (B, latent_dim)
                These should be detached to prevent gradients flowing back to posterior
            conditioning: Conditioning features (state) with shape (B, conditioning_dim)

        Returns:
            Prior network output dictionary containing:
                - Predicted noise or actions , shape (B, latent_dim).
                - 'target': The training target (noise), shape (B, latent_dim).
        """
        batch_size = target_latents.shape[0]
        device = target_latents.device
        timesteps = sample_random_timesteps(
            batch_size=batch_size,
            num_train_timesteps=self.num_train_timesteps,
            device=device,
        )
        noisy_latents, noise = add_noise_to_tensor(
            clean=target_latents,
            noise_scheduler=self.noise_scheduler,
            timesteps=timesteps,
        )

        timestep_input = timesteps.float().unsqueeze(1) / self.num_train_timesteps
        timestep_embed = self.timestep_mlp(timestep_input)
        model_input = torch.cat([noisy_latents, conditioning, timestep_embed], dim=-1)
        predicted_noise = self.denoising_network(model_input)
        return {
            PRIOR_PREDICTION_KEY: predicted_noise,
            PRIOR_TARGET_KEY: noise,
        }
