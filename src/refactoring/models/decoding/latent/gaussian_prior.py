"""Simple Gaussian prior for variational models.

This module implements a standard Gaussian N(0, I) prior for latent variable models.
Unlike learned priors (e.g., DiffusionPrior), this prior requires no training and
simply samples from a standard normal distribution.

This is the default prior used when no learned prior is specified, providing
a simple baseline for variational models.
"""

import torch
import torch.nn as nn

from refactoring.models.decoding.latent import LatentPrior


class GaussianPrior(LatentPrior):
    """Standard Gaussian N(0, I) prior for latent variable models.

    Provides a simple, non-learned prior that samples from a standard normal
    distribution. This is the default prior for variational algorithms when
    no learned prior is specified.

    Unlike learned priors (e.g., DiffusionPrior), this prior:
    - Requires no training (forward() returns empty dict)
    - Samples from N(0, I) independently of conditioning
    - Projects to embedding dimension for decoder compatibility
    Optionally, it can use a constant zero latent sample as prior (like in ACT).

    Args:
        latent_dim: Dimension of latent variable z
        output_dim: Dimension to project latent output to (for decoder input)
        device: Device to place prior on
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        device: str,
        infer_constant_prior: bool = False,
    ):
        """Initialize Gaussian prior.

        Args:
            latent_dim: Dimension of latent variable z
            output_dim: Output embedding dimension
            device: Device to place prior on
            infer_constant_prior: Whether to use a constant prior 0 sample (like in ACT) or not
        """
        super().__init__(latent_dim=latent_dim, device=device, output_dim=output_dim)
        self.infer_constant_prior = infer_constant_prior
        self.latent_output_projection = nn.Linear(
            latent_dim,
            output_dim
        )
        self.device = device
        self.to(torch.device(device))

    def sample_prior(
        self,
        batch_size: int,
        conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample latent variable from standard Gaussian N(0, I).

        Args:
            batch_size: Number of samples to generate
            conditioning: Optional conditioning features (ignored for Gaussian prior)

        Returns:
            Sampled latent z variable projected to output dimension (batch_size, output_dim)
        """
        if self.infer_constant_prior:
            # Use constant zero latent for prior (like in ACT)
            z = torch.zeros(batch_size, self.latent_dim, device=self.device)
        else:
            # Sample from standard normal N(0, I)
            z = torch.randn(batch_size, self.latent_dim, device=self.device)

        # Project latent to output dimension for decoder compatibility
        return self.latent_output_projection(z)


    def forward(
        self,
        target_latents: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """No-op forward pass for Gaussian prior (no training needed).

        For a standard Gaussian prior, there is no training objective since
        the prior distribution is fixed. This method returns an empty dictionary
        to indicate no prior loss should be computed.

        Args:
            target_latents: Latent samples from posterior (unused)
            conditioning: Conditioning features (unused)

        Returns:
            Empty dictionary (no prior loss for Gaussian prior)
        """
        return {}
