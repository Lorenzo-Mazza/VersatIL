"""Diffusion-based learned prior for variational models.

This module implements a learned diffusion prior for latent variable models
using shared diffusion process components from algorithm.diffusion_process.

The prior learns to sample from p(z|s) using a diffusion model, where z is the
latent variable and s is the conditioning. During training, it learns to denoise
latent samples from the posterior. During inference, it generates latents via
reverse diffusion.

Shared Components Used:
    - DiffusionSchedulerConfig: Unified configuration for DDPM scheduler
    - create_noise_scheduler(): Factory for creating noise schedulers
    - add_noise_to_tensor(): Forward diffusion (adding noise to posterior latents)
    - sample_random_timesteps(): Uniform timestep sampling for training
    - setup_inference_timesteps(): Configure scheduler for sampling

See algorithm.diffusion_process for detailed documentation of these components.
"""

import torch
import torch.nn as nn

from refactoring.models.decoding.algorithm.diffusion_process import (
    DiffusionSchedulerConfig,
    add_noise_to_tensor,
    create_noise_scheduler,
    sample_random_timesteps,
    setup_inference_timesteps,
)
from refactoring.models.decoding.constants import SchedulerType, PRIOR_PREDICTION_KEY, PRIOR_TARGET_KEY
from refactoring.models.decoding.latent import LatentPrior
from refactoring.models.layers.mlp import MLP
from refactoring.models.layers.activation import ActivationFunction
from tests.models.encoding.proprio.test_base import output_dim


class DiffusionPrior(LatentPrior):
    """Diffusion-based learned prior for latent variable models.

    Learns to sample from p(z|s) using a simple diffusion MLP model, where z is the latent
    variable and s is the conditioning (state features). This is used instead of
    a standard Gaussian prior N(0,I) to better match the posterior distribution.
    An example use case can be found in https://arxiv.org/html/2508.01622v2.

    During training, the prior is trained to denoise latent samples from the posterior.
    During inference, it generates latent samples via the reverse diffusion process.

    Args:
        latent_dim: Dimension of latent variable z
        conditioning_dim: Dimension of conditioning features (state)
        output_dim: Dimension to project latent output to (for decoder input)
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
        latent_dim: int,
        conditioning_dim: int,
        output_dim: int,
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
        """Initialize diffusion prior.

        Args:
            latent_dim: Dimension of latent variable z
            conditioning_dim: Dimension of conditioning features
            output_dim: Output embedding dimension
            device: Device to place prior on
            hidden_dims: Hidden layer dimensions
            num_train_timesteps: Training timesteps
            num_inference_steps: Inference steps
            beta_start: Starting beta
            beta_end: Ending beta
            beta_schedule: Noise schedule type
            activation: Activation function
            dropout: Dropout rate
        """
        super().__init__(latent_dim=latent_dim, device=device, output_dim=output_dim)
        self.conditioning_dim = conditioning_dim
        self.embedding_dimension = output_dim
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
            hidden_dims = [latent_dim * 2, latent_dim * 2]

        self.timestep_embed_dim = latent_dim
        self.timestep_mlp = nn.Sequential(
            nn.Linear(1, self.timestep_embed_dim),
            ActivationFunction(activation).to_torch_activation()(),
            nn.Linear(self.timestep_embed_dim, self.timestep_embed_dim),
        )

        input_dim = latent_dim + conditioning_dim + self.timestep_embed_dim

        self.denoising_network = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=latent_dim,
            activation_function=ActivationFunction(activation).to_torch_activation(),
            dropout=dropout,
        )
        # Projection from latent_dim to embedding_dimension for decoder input
        self.latent_output_projection = nn.Linear(latent_dim, output_dim)
        self.to(torch.device(device))

    def sample_prior(self, batch_size: int, conditioning: torch.Tensor | None = None) -> torch.Tensor:
        """Sample latent variable from learned prior p(z|s) via reverse diffusion.

        Args:
            batch_size: Number of samples to generate
            conditioning: Conditioning features (state) with shape (batch_size, conditioning_dim)

        Returns:
            Sampled latent embeddings (batch_size, embedding_dimension)
        """
        device = next(self.parameters()).device

        if conditioning is None:
            # Unconditional sampling
            conditioning = torch.zeros(batch_size, self.conditioning_dim, device=device)

        # Start from pure noise
        z_t = torch.randn(batch_size, self.latent_dim, device=device)
        setup_inference_timesteps(self.noise_scheduler, self.num_inference_steps)

        # Reverse diffusion process
        for t in self.noise_scheduler.timesteps:
            # Prepare timestep embedding
            timestep = torch.tensor([t], device=device).float().view(1, 1).expand(batch_size, 1)
            timestep_embed = self.timestep_mlp(timestep / self.num_train_timesteps)
            model_input = torch.cat([z_t, conditioning, timestep_embed], dim=-1)
            predicted_noise = self.denoising_network(model_input)

            # Compute previous sample using scheduler
            z_t = self.noise_scheduler.step(predicted_noise, t, z_t).prev_sample

        latent_embedding = self.latent_output_projection(z_t)
        return latent_embedding

    def forward(
        self,
        target_latents: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute denoising loss for training the prior to match posterior samples.

        Args:
            target_latents: Latent samples from posterior q(z|a,s) with shape (B, latent_dim)
                These should be DETACHED to prevent gradients flowing back to posterior
            conditioning: Conditioning features (state) with shape (B, conditioning_dim)

        Returns:
            Prior network output dictionary containing:
                - Predicted noise or actions.
                - 'target': The training target (noise).
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
