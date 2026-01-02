"""Variational inference wrapper for decoding algorithms.

This module provides a compositional wrapper that adds variational inference
capabilities to any base decoding algorithm.
Variational Inference enables modeling of multi-modal action distributions by introducing
a latent variable z that is conditioned on both observations and actions:
    p(a|s) = p(a|z,s) p(z|s)
Where:
- p(a|z,s) is the base algorithm's decoder (BC, FlowMatching, etc.)
- p(z|s) is the learned conditional prior (Gaussian or learned through a NN)

"""

import logging

import torch

from refactoring.models.decoding.algorithm.base import DecodingAlgorithm
from refactoring.models.decoding.constants import (
    LATENT_KEY,
    LOGVAR_KEY,
    MU_KEY, PRIOR_LATENT_KEY,
)
from refactoring.models.decoding.decoders.base import ActionDecoder
from refactoring.models.decoding.latent import PosteriorLatentEncoder
from refactoring.models.decoding.latent import PriorLatentEncoder
from refactoring.models.decoding.latent.prior.gaussian_prior import GaussianPrior


class VariationalAlgorithm(DecodingAlgorithm):
    """Compositional wrapper adding variational inference to any decoding algorithm.

    Wraps a base algorithm with variational inference, enabling multi-modal
    action prediction through a learned posterior q_phi(z|a,s) and prior p(z|s).

    Training:
        1. Encode actions via approximated posterior: z ~ q_phi(z|a,s)
        2. Train prior to match posterior (if learned prior)
        3. Decode actions via posterior + decoding algorithm: p(a|z,s)q_phi(z|a,s)

    Inference:
        1. Sample latent from prior: z ~ p(z|s)
        2. Decode actions via base algorithm: p(a|z,s)

    Args:
        base_algorithm: The underlying decoding algorithm (BC, FlowMatching, Diffusion, etc.)
        posterior_encoder: Latent action encoder for posterior q_phi(z|a,s) (e.g., a Transformer Encoder)
        prior: Latent prior for p(z|s). If None, auto-creates GaussianPrior.
    """

    def __init__(
        self,
        base_algorithm: DecodingAlgorithm,
        posterior_encoder: PosteriorLatentEncoder,
        prior: PriorLatentEncoder | None = None,
    ):
        """Initialize variational algorithm wrapper."""
        super().__init__()
        self.base_algorithm = base_algorithm
        self.posterior_encoder = posterior_encoder
        self.p_prior = 0.2  # Probability of sampling from prior during training
        if prior is None:
            device = str(posterior_encoder.device)
            self.prior = GaussianPrior(
                latent_dimension=self.posterior_encoder.latent_dim,
                device=device,
            )
            logging.info(
                f"Auto-created GaussianPrior with latent_dim={self.posterior_encoder.latent_dimension}. "
            )
        else:
            self.prior = prior

        if self.prior.latent_dimension != self.posterior_encoder.latent_dimension:
            raise ValueError(
                f"Latent dimension mismatch: prior.latent_dim={self.prior.latent_dimension} "
                f"!= posterior_encoder.latent_dim={self.posterior_encoder.latent_dimension}"
            )


    def _variational_step(
        self,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Encode actions via approximated conditional posterior q_phi(z|a,s) and optionally learn the conditional prior p(z|s).

        Args:
            features: Observation features
            actions: Ground-truth actions

        Returns:
            Tuple of (posterior_output, prior_output)
            where each contain sampled z, mu and logvar from the prior and posterior networks.
        """
        posterior_output = self.posterior_encoder.encode(actions=actions, observations=features)
        z = posterior_output[LATENT_KEY] # (B, posterior.latent_dim)
        prior_output = self.prior.forward(
            target_latents=z.detach(), # Detach z to prevent gradients flowing to posterior encoder
            observations=features,
        )
        return posterior_output, prior_output


    def _sample_prior(
        self,
        features: dict[str, torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        """Sample latent from prior distribution.

        Args:
            features: Observation features
            batch_size: Batch size for sampling

        Returns:
            Sampled latent embedding
        """
        latent_embedding = self.prior.sample_prior(
            batch_size=batch_size,
            observations=features,
        )
        return latent_embedding


    def forward(
        self,
        network: ActionDecoder,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with variational latent encoding.

        Training mode (actions provided):
            - Encodes actions via approximated posterior q_phi(z|a,s)
            - Trains learned prior to match approximated posterior (if learned)
            - Augments features with latent z, sampled from the approximated posterior
            - Delegates to base algorithm

        Args:
            network: Action decoder network
            features: Dictionary of input features from encoding pipeline
            actions: Optional ground-truth actions (for training)

        Returns:
            Dictionary containing:
                - Action predictions from base algorithm
                - Latent variables (mu, logvar) if training
                - Prior outputs (prior_prediction, prior_target) if learned prior
        """
        if actions is None:
            raise ValueError("Actions must be provided during training for variational algorithm.")
        posterior_output, prior_output = self._variational_step(
            features=features,
            actions=actions,
        )
        use_prior = torch.rand(1).item() < self.p_prior
        if use_prior:
            latent = prior_output[PRIOR_LATENT_KEY]
        else:
            latent = posterior_output[LATENT_KEY]
        features_with_latent = {**features, LATENT_KEY: latent} # (B, latent_dimension)
        predictions = self.base_algorithm.forward(
            network=network,
            features=features_with_latent,
            actions=actions,
        )
        predictions.update(posterior_output)
        predictions.update(prior_output)
        return predictions


    def predict(
        self,
        network: ActionDecoder,
        features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Inference mode: sample latent z from prior and decode.

        Args:
            network: Action decoder network
            features: Dictionary of input features

        Returns:
            Dictionary containing action predictions
        """
        batch_size = next(iter(features.values())).shape[0]
        latent_embedding = self._sample_prior(features=features, batch_size=batch_size)
        features_with_latent = {**features, LATENT_KEY: latent_embedding} # (B, latent_dimension)
        return self.base_algorithm.forward(
            network=network,
            features=features_with_latent,
            actions=None,
        )
