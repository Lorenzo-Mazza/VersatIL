"""Latent space modules for variational action decoders.

This package provides two types of latent modules:

1. **Latent Action Encoders** (Posteriors):
   Transform action sequences into latent embeddings, modeling q(z|a,s).
   Used during training to encode ground-truth actions.
   - VAETransformerEncoder: Transformer-based VAE for multi-modal action distributions

2. **Latent Priors**:
   Model the prior distribution p(z|s) over latent variables.
   Used during inference when actions are unavailable.
   - GaussianPrior: Simple N(0, I) prior (default, no training required)
   - DiffusionPrior: Learned diffusion-based prior that matches posterior distribution
"""

from refactoring.models.decoding.latent.base_posterior import LatentActionEncoder
from refactoring.models.decoding.latent.base_prior import LatentPrior
from refactoring.models.decoding.latent.diffusion_prior import DiffusionPrior
from refactoring.models.decoding.latent.gaussian_prior import GaussianPrior
from refactoring.models.decoding.latent.vae_posterior import VAETransformerEncoder
__all__ = [
    "LatentActionEncoder",
    "VAETransformerEncoder",
    "LatentPrior",
    "GaussianPrior",
    "DiffusionPrior",
]