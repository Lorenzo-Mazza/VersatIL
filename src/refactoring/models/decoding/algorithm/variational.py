"""Variational inference wrapper for decoding algorithms.

This module provides a compositional wrapper that adds variational inference
capabilities to any base decoding algorithm.
Variational Inference enables modeling of multi-modal action distributions by introducing
a latent variable z that is conditioned on both observations and actions:
    p(a|s) = ∫ p(a|z,s) p(z|s) dz
Where:
- p(a|z,s) is the base algorithm's decoder (BC, FlowMatching, etc.)
- p(z|s) is the prior over latents (Gaussian or learned)
- q(z|a,s) is the posterior encoder (e.g., VAE)

Examples:
    # Behavioral Cloning with VAE
    VariationalAlgorithm(
        base_algorithm=BehavioralCloning(),
        posterior_encoder=VAETransformerEncoder(...),
        # prior=None auto-creates GaussianPrior
    )

    # Variational Flow Matching
    VariationalAlgorithm(
        base_algorithm=FlowMatching(...),
        posterior_encoder=VAETransformerEncoder(...),
        prior=DiffusionPrior(...),
    )

"""

import logging

import torch

from refactoring.models.decoding.algorithm.base import DecodingAlgorithm
from refactoring.models.decoding.constants import (
    LATENT_KEY,
    LOGVAR_KEY,
    MU_KEY,
    PRIOR_PREDICTION_KEY,
    PRIOR_TARGET_KEY, STATE_FEATURE_KEYS,
)
from refactoring.models.decoding.decoders.base import ActionDecoder
from refactoring.models.decoding.latent import LatentActionEncoder
from refactoring.models.decoding.latent import LatentPrior
from refactoring.models.decoding.latent.gaussian_prior import GaussianPrior
from refactoring.models.layers.detr_transformer.vae_transformer import reparametrize


class VariationalAlgorithm(DecodingAlgorithm):
    """Compositional wrapper adding variational inference to any decoding algorithm.

    Wraps a base algorithm with variational latent encoding, enabling multi-modal
    action prediction through a learned posterior q(z|a,s) and prior p(z|s).

    Training:
        1. Encode actions via posterior: z ~ q(z|a,s)
        2. Train prior to match posterior (if learned prior)
        3. Decode actions via posterior + decoding algorithm: p(a|z,s)q(z|a,s)

    Inference:
        1. Sample latent from prior: z ~ p(z|s)
        2. Decode actions via base algorithm: p(a|z,s)

    Args:
        base_algorithm: The underlying decoding algorithm (BC, FlowMatching, Diffusion, etc.)
        posterior_encoder: Latent action encoder for posterior q(z|a,s) (e.g., VAE)
        prior: Latent prior for p(z|s). If None, auto-creates GaussianPrior.
    """

    def __init__(
        self,
        base_algorithm: DecodingAlgorithm,
        posterior_encoder: LatentActionEncoder,
        prior: LatentPrior | None = None,
    ):
        """Initialize variational algorithm wrapper."""
        super().__init__()
        self.base_algorithm = base_algorithm
        self.posterior_encoder = posterior_encoder
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


    @staticmethod
    def _extract_conditioning(features: dict[str, torch.Tensor]) -> torch.Tensor | None:
        """Extract conditioning features from features dictionary.

        Handles temporal features by flattening time dimension (B, T, D) → (B, T*D).
        Ignores spatial features (4D tensors with C, H, W dimensions).

        Args:
            features: Dictionary of input features

        Returns:
            Concatenated conditioning tensor or None if no valid features
        """
        cond = []
        for value in features.values():
            if not isinstance(value, torch.Tensor):
                continue
            if value.ndim == 3:  # (B, T, D) temporal features
                value = value.view(value.shape[0], -1)  # Flatten to (B, T*D)
            if value.ndim == 2:  # (B, D) flat features
                cond.append(value)
            # Skip 4D spatial features (B, C, H, W)

        return torch.cat(cond, dim=-1) if cond else None

    def _encode_posterior(
        self,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor | None]:
        """Encode actions via posterior  q(z|a,s) and optionally train prior.

        Args:
            features: Observation features
            actions: Ground-truth actions

        Returns:
            Tuple of (latent_embedding, mu, logvar, prior_outputs, state features)
            where prior_outputs contains predictions and targets for prior loss
        """
        latent_output = self.posterior_encoder.encode(actions=actions, observations=features)
        z = latent_output[LATENT_KEY] # (B, posterior.latent_dim)
        state_features = latent_output[STATE_FEATURE_KEYS]
        mu = latent_output[MU_KEY]# (B, posterior.latent_dim)
        logvar = latent_output[LOGVAR_KEY] # (B, posterior.latent_dim)
        conditioning = self._extract_conditioning(features)
        prior_outputs = self.prior.forward(
            target_latents=z.detach(), # Detach z to prevent gradients flowing to posterior encoder
            conditioning=conditioning,
        )
        return z, mu, logvar, prior_outputs, state_features

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
        conditioning = self._extract_conditioning(features)
        latent_embedding = self.prior.sample_prior(
            batch_size=batch_size,
            conditioning=conditioning,
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
            - Encodes actions via posterior q(z|a,s)
            - Trains prior to match posterior (if learned)
            - Augments features with latent
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
                - State features used by the posterior encoder (optionally used to compute Optimal Transport Loss)
        """
        if actions is None:
            raise ValueError("Actions must be provided during training for variational algorithm.")
        latent_embedding, mu, logvar, prior_outputs, obs_features = self._encode_posterior(
            features=features,
            actions=actions,
        )
        features_with_latent = {**features, LATENT_KEY: latent_embedding} # (B, latent_dimension)
        predictions = self.base_algorithm.forward(
            network=network,
            features=features_with_latent,
            actions=actions,
        )
        if mu is not None:
            predictions[MU_KEY] = mu
        if logvar is not None:
            predictions[LOGVAR_KEY] = logvar
        if prior_outputs is not None and len(prior_outputs) > 0:
            if PRIOR_PREDICTION_KEY in prior_outputs:
                predictions[PRIOR_PREDICTION_KEY] = prior_outputs[PRIOR_PREDICTION_KEY]
            if PRIOR_TARGET_KEY in prior_outputs:
                predictions[PRIOR_TARGET_KEY] = prior_outputs[PRIOR_TARGET_KEY]
        if obs_features is not None:
            predictions[STATE_FEATURE_KEYS] = obs_features
        return predictions


    def predict(
        self,
        network: ActionDecoder,
        features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Inference mode: sample from prior and decode.

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
