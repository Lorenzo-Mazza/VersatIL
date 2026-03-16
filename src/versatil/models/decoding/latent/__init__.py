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

from versatil.models.decoding.latent.posterior.base_posterior import (
    PosteriorLatentEncoder,
)
from versatil.models.decoding.latent.posterior.transformer_encoder import (
    VAETransformerEncoder,
)
from versatil.models.decoding.latent.prior.base_prior import PriorLatentEncoder
from versatil.models.decoding.latent.prior.dit_prior import (
    DiTPrior,
)
from versatil.models.decoding.latent.prior.gaussian_prior import GaussianPrior
from versatil.models.decoding.latent.prior.transformer_encoder import (
    PriorTransformerEncoder,
)
from versatil.models.decoding.latent.prior.vamp_prior import VampPrior

__all__ = [
    "PosteriorLatentEncoder",
    "VAETransformerEncoder",
    "PriorLatentEncoder",
    "PriorTransformerEncoder",
    "GaussianPrior",
    "VampPrior",
    "DiTPrior",
]
