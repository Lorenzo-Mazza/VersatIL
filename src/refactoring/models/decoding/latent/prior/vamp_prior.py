"""VampPrior: Variational Mixture of Posteriors Prior.

Implements the VampPrior from "VAE with a VampPrior" (Tomczak & Welling, 2018).
The prior is a mixture of Gaussians where each component is defined by passing
learnable pseudo-inputs through the posterior encoder.

This allows the prior to be more expressive than a standard Gaussian N(0,I)
while maintaining the VAE framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from refactoring.models.decoding.constants import (
    PRIOR_LATENT_KEY,
    PRIOR_LOG_PROB_KEY,
)
from refactoring.models.decoding.latent.prior.base_prior import PriorLatentEncoder
from refactoring.models.decoding.latent.posterior.base_posterior import (
    PosteriorLatentEncoder,
)
from refactoring.models.decoding.constants import MU_KEY, LOGVAR_KEY


def log_normal_diag(z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Compute log probability of z under diagonal Gaussian N(mu, exp(logvar)).

    Args:
        z: Samples, shape (..., latent_dim)
        mu: Mean, shape (..., latent_dim)
        logvar: Log variance, shape (..., latent_dim)

    Returns:
        Log probability, shape (..., latent_dim) - per-dimension log probs
    """
    log_p = -0.5 * (torch.log(2 * torch.pi * torch.ones(1, device=z.device))
                    + logvar
                    + (z - mu) ** 2 / torch.exp(logvar))
    return log_p


class VampPrior(PriorLatentEncoder):
    """Variational Mixture of Posteriors Prior.

    Args:
        latent_dimension: Dimension of latent variable z
        num_components: Number of mixture components K
        pseudo_input_dim: Dimension of pseudo-inputs (should match encoder input)
        device: Device to place prior on
        min_logvar: Optional minimum logvar clamp
    """

    def __init__(
        self,
        latent_dimension: int,
        num_components: int,
        pseudo_input_dim: int,
        device: str,
        min_logvar: float | None = None,
    ):
        super().__init__(latent_dimension=latent_dimension, device=device)
        self.num_components = num_components
        self.pseudo_input_dim = pseudo_input_dim
        self.min_logvar = min_logvar
        self.pseudo_inputs = nn.Parameter(torch.randn(num_components, pseudo_input_dim))
        self.log_weights = nn.Parameter(torch.zeros(num_components, 1, 1))
        self._encoder: PosteriorLatentEncoder | None = None
        self.to(torch.device(device))

    def set_encoder(self, encoder: PosteriorLatentEncoder) -> None:
        """Set the posterior encoder used to compute mixture components.

        Args:
            encoder: Posterior encoder that maps inputs to (mu, logvar)
        """
        self._encoder = encoder

    @property
    def encoder(self) -> PosteriorLatentEncoder:
        """Get the posterior encoder."""
        if self._encoder is None:
            raise RuntimeError(
                "VampPrior encoder not set. Call set_encoder() first or ensure "
                "VariationalAlgorithm properly initializes the prior."
            )
        return self._encoder

    def get_mixture_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute mixture component parameters by passing pseudo-inputs through encoder.

        Returns:
            Tuple of (means, logvars) each of shape (K, latent_dim)
        """
        pseudo_actions = {"pseudo": self.pseudo_inputs.unsqueeze(1)}
        encoder_output = self.encoder.encode(actions=pseudo_actions, observations=None)
        mu = encoder_output[MU_KEY]  # (K, latent_dim)
        logvar = encoder_output[LOGVAR_KEY]  # (K, latent_dim)
        if self.min_logvar is not None:
            logvar = torch.clamp(logvar, min=self.min_logvar)
        return mu, logvar

    def sample_prior(
        self,
        batch_size: int,
        observations: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Sample from the VampPrior mixture.

        Args:
            batch_size: Number of samples to generate
            observations: Optional conditioning (ignored, VampPrior is unconditional)

        Returns:
            Sampled latents of shape (batch_size, latent_dim)
        """
        mu, logvar = self.get_mixture_params()  # (K, latent_dim)
        weights = F.softmax(self.log_weights.squeeze(), dim=0)  # (K,)
        component_indices = torch.multinomial(
            weights, batch_size, replacement=True
        )  # (batch_size,)
        selected_mu = mu[component_indices]  # (batch_size, latent_dim)
        selected_logvar = logvar[component_indices]  # (batch_size, latent_dim)
        std = (selected_logvar / 2).exp()
        eps = torch.randn_like(std)
        z = selected_mu + std * eps
        return z

    def forward(
        self,
        target_latents: torch.Tensor,
        observations: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute prior log probability for training.

        VampPrior is a mixture of Gaussians. Unlike single-Gaussian priors,
        we return log_prob directly since mu/logvar don't represent a mixture.

        Args:
            target_latents: Target latents from posterior (B, latent_dim)
            observations: Conditioning features (ignored for VampPrior)

        Returns:
            Dictionary with PRIOR_LATENT_KEY and PRIOR_LOG_PROB_KEY
        """
        batch_size = target_latents.size(0)
        z = self.sample_prior(batch_size, observations)
        log_prob = self.log_prob(target_latents)
        return {
            PRIOR_LATENT_KEY: z,
            PRIOR_LOG_PROB_KEY: log_prob,
        }

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """Compute log probability of z under the VampPrior mixture.

        Args:
            z: Latent samples of shape (B, latent_dim)

        Returns:
            Log probability of shape (B,) - summed over latent dimensions
        """
        mu, logvar = self.get_mixture_params()  # (K, latent_dim)
        weights = F.softmax(self.log_weights.squeeze(), dim=0)  # (K,)
        z_expanded = z.unsqueeze(0)  # (1, B, latent_dim)
        mu_expanded = mu.unsqueeze(1)  # (K, 1, latent_dim)
        logvar_expanded = logvar.unsqueeze(1)  # (K, 1, latent_dim)
        log_p_components = log_normal_diag(
            z_expanded, mu_expanded, logvar_expanded
        )  # (K, B, latent_dim)
        log_p_components = log_p_components.sum(dim=-1)  # (K, B)
        log_weights = torch.log(weights).unsqueeze(-1)  # (K, 1)
        log_p_weighted = log_p_components + log_weights  # (K, B)
        log_prob = torch.logsumexp(log_p_weighted, dim=0)  # (B,)
        return log_prob