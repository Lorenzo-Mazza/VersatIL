r"""Fixed Gaussian prior for variational inference.

This module implements a standard Gaussian N(0, I) prior for latent variable models.
Unlike learned priors (e.g., DiffusionPrior), this prior requires no training and
simply samples from a standard normal distribution.

This is the default prior used when no learned prior is specified, providing
the traditional approach for imposing a gaussian distribution for the inference model (the approximated posterior q_\phi(z|x)).
"""

import torch

from versatil.models.decoding.constants import LatentKey
from versatil.models.decoding.latent import PriorLatentEncoder


class GaussianPrior(PriorLatentEncoder):
    """Standard Gaussian N(0, I) prior for latent variable models.

    Args:
        latent_dimension: Dimension of latent variable z
        device: Device to place prior on
    """

    def __init__(
        self,
        latent_dimension: int,
        device: str,
        infer_constant_prior: bool = False,
    ):
        """Initialize Gaussian prior."""
        super().__init__(latent_dimension=latent_dimension, device=device)
        self.infer_constant_prior = infer_constant_prior
        self.register_buffer("_device_tracker", torch.zeros(1))
        self.to(torch.device(device))

    def sample_prior(
        self,
        batch_size: int,
        observations: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Sample latent variable from standard multivariate (z dimensional) Gaussian N(0, I).

        Args:
            batch_size: Number of samples to generate
            observations: Optional conditioning features (ignored for Gaussian prior)

        Returns:
            Sampled latent z vector of dimension (batch_size, latent_dim)
        """
        if self.infer_constant_prior:
            # Use constant zero latent for prior (like in ACT)
            return torch.zeros(
                batch_size, self.latent_dimension, device=self._device_tracker.device
            )
        else:
            # Sample from standard normal N(0, I)
            return torch.randn(
                batch_size, self.latent_dimension, device=self._device_tracker.device
            )

    def forward(
        self,
        target_latents: torch.Tensor | None,
        observations: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Forward pass for a fixed Gaussian prior, returning zero mu and unit logvar."""
        if target_latents is None:
            raise ValueError(
                "GaussianPrior.forward() requires target_latents to infer "
                "shape. Use sample_prior() for unconditional sampling."
            )
        mu = torch.zeros_like(target_latents, device=self._device_tracker.device)
        logvar = torch.zeros_like(target_latents, device=self._device_tracker.device)
        z = torch.randn(
            mu.size(0), self.latent_dimension, device=self._device_tracker.device
        )
        return {
            LatentKey.PRIOR_MU.value: mu,
            LatentKey.PRIOR_LOGVAR.value: logvar,
            LatentKey.PRIOR_LATENT.value: z,
        }
