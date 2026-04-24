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

from versatil.configs.experiment import ExperimentConfig
from versatil.models.decoding.algorithm.base import DecodingAlgorithm
from versatil.models.decoding.constants import LatentKey
from versatil.models.decoding.decoders.base import ActionDecoder
from versatil.models.decoding.latent.posterior.base_posterior import (
    PosteriorLatentEncoder,
)
from versatil.models.decoding.latent.prior.base_prior import PriorLatentEncoder
from versatil.models.decoding.latent.prior.gaussian_prior import GaussianPrior
from versatil.models.decoding.latent.protocols import RequiresPosteriorWiring
from versatil.training.callbacks.latent_visualization import LatentVisualizationCallback
from versatil.training.callbacks.provider import CallbackProvider


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
        sampling_from_prior_probability: Probability of sampling from prior during training.
        posterior_decoder_noise_std: Fixed Gaussian jitter added to posterior
            latents before decoder training. This is applied only when training
            decodes from the posterior, not when decoding from prior samples.
    """

    def __init__(
        self,
        base_algorithm: DecodingAlgorithm,
        posterior_encoder: PosteriorLatentEncoder,
        prior: PriorLatentEncoder | None = None,
        sampling_from_prior_probability: float = 0.0,
        posterior_decoder_noise_std: float = 0.0,
    ):
        """Initialize variational algorithm wrapper."""
        super().__init__()
        if posterior_decoder_noise_std < 0.0:
            raise ValueError(
                "posterior_decoder_noise_std must be non-negative, "
                f"got {posterior_decoder_noise_std}."
            )
        self.base_algorithm = base_algorithm
        self.posterior_encoder = posterior_encoder
        self.p_prior = sampling_from_prior_probability
        self.posterior_decoder_noise_std = posterior_decoder_noise_std
        if prior is None:
            device = str(posterior_encoder.device)
            self.prior = GaussianPrior(
                latent_dimension=self.posterior_encoder.latent_dimension,
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
        if isinstance(self.prior, RequiresPosteriorWiring):
            self.prior.wire_posterior(self.posterior_encoder)

    def get_callbacks(self, experiment_config: ExperimentConfig) -> list:
        """Provide callbacks for variational latent monitoring and prior fitting.

        Args:
            experiment_config: Experiment-level callback configuration.

        Returns:
            Callbacks required by the variational algorithm.
        """
        callbacks = [
            LatentVisualizationCallback(
                log_every_n_epochs=experiment_config.val_every,
            )
        ]
        if isinstance(self.prior, CallbackProvider):
            callbacks.extend(
                self.prior.get_callbacks(experiment_config=experiment_config)
            )
        return callbacks

    def get_auxiliary_output_keys(self) -> set[str]:
        """Variational algorithm adds latent variable keys to the output."""
        keys = self.base_algorithm.get_auxiliary_output_keys()
        keys.update(self.posterior_encoder.get_auxiliary_output_keys())
        keys.update(self.prior.get_auxiliary_output_keys())
        return keys

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
        posterior_output = self.posterior_encoder.encode(
            actions=actions, observations=features
        )
        target_latents = self.prior.build_training_target(posterior_output)
        prior_output = self.prior.forward(
            target_latents=target_latents,
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

    def _posterior_latent_for_decoder(
        self,
        posterior_output: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return the posterior latent consumed by the action decoder.

        Optional fixed jitter thickens the posterior support seen by the
        decoder while leaving prior training targets untouched. Those targets
        are built before this method is called.
        """
        latent = posterior_output[LatentKey.POSTERIOR_LATENT.value]
        if self.training and self.posterior_decoder_noise_std > 0.0:
            latent = latent + self.posterior_decoder_noise_std * torch.randn_like(
                latent
            )
        return latent

    def forward(
        self,
        network: ActionDecoder,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with variational latent encoding.

        Training mode (self.training=True):
            - Encodes actions via approximated posterior q_phi(z|a,s)
            - Trains learned prior to match approximated posterior (if learned)
            - Stochastic mixing: samples from prior with probability p_prior
            - Delegates to base algorithm

        Validation/eval mode (self.training=False):
            - Still encodes posterior for loss computation (KL term)
            - But uses ONLY prior samples for action decoding (like inference)
            - Better evaluates actual inference performance

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
            raise ValueError(
                "Actions must be provided during training for variational algorithm."
            )
        posterior_output, prior_output = self._variational_step(
            features=features,
            actions=actions,
        )
        if self.training:
            sample_from_prior = torch.rand(1).item() < self.p_prior
        else:
            sample_from_prior = True
        if sample_from_prior:
            if LatentKey.PRIOR_LATENT.value not in prior_output:
                # Sample from prior only when needed (avoids costly denoising inference during training)
                batch_size = next(iter(features.values())).shape[0]
                prior_output[LatentKey.PRIOR_LATENT.value] = self.prior.sample_prior(
                    batch_size=batch_size, observations=features
                )
            latent = prior_output[LatentKey.PRIOR_LATENT.value]
            posterior_decoder_latent = None
        else:
            latent = self._posterior_latent_for_decoder(posterior_output)
            posterior_decoder_latent = latent
        features_with_latent = {
            **features,
            LatentKey.POSTERIOR_LATENT.value: latent,
        }  # (B, latent_dimension)
        predictions = self.base_algorithm.forward(
            network=network,
            features=features_with_latent,
            actions=actions,
        )
        if posterior_decoder_latent is None:
            predictions.update(posterior_output)
        else:
            predictions.update(
                {
                    **posterior_output,
                    LatentKey.POSTERIOR_LATENT.value: posterior_decoder_latent,
                }
            )
        predictions.update(prior_output)
        return predictions

    @property
    def predicts_in_action_space(self) -> bool:
        """Delegates to the wrapped base algorithm."""
        return self.base_algorithm.predicts_in_action_space

    def get_targets(
        self,
        algorithm_output: dict[str, torch.Tensor],
        ground_truth_actions: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Delegate to the base algorithm's target selection."""
        return self.base_algorithm.get_targets(algorithm_output, ground_truth_actions)

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
        features_with_latent = {
            **features,
            LatentKey.POSTERIOR_LATENT.value: latent_embedding,
        }  # (B, latent_dimension)
        return self.base_algorithm.predict(
            network=network,
            features=features_with_latent,
        )
