"""VampPrior: Variational Mixture of Posteriors Prior.

Implements the VampPrior from "VAE with a VampPrior" (Tomczak & Welling, 2018).
The prior is a mixture of Gaussians where each component is defined by passing
learnable pseudo-inputs through the posterior encoder.

This allows the prior to be more expressive than a standard Gaussian N(0,I)
while maintaining the VAE framework.
"""

import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F

from versatil.data.constants import SampleKey
from versatil.data.task import ActionSpace
from versatil.models.decoding.constants import LatentKey
from versatil.models.decoding.latent.posterior.base_posterior import (
    PosteriorLatentEncoder,
)
from versatil.models.decoding.latent.prior.base_prior import PriorLatentEncoder


def log_normal_diag(
    z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    """Compute log probability of z under diagonal Gaussian N(mu, exp(logvar)).

    Args:
        z: Samples, shape (..., latent_dim)
        mu: Mean, shape (..., latent_dim)
        logvar: Log variance, shape (..., latent_dim)

    Returns:
        Log probability, shape (..., latent_dim) - per-dimension log probs
    """
    log_p = -0.5 * (
        torch.log(2 * torch.pi * torch.ones(1, device=z.device))
        + logvar
        + (z - mu) ** 2 / torch.exp(logvar)
    )
    return log_p


class VampPrior(PriorLatentEncoder):
    """Variational Mixture of Posteriors Prior.

    The posterior encoder in this policy family is conditional,
    ``q_phi(z | a, s)``. Therefore the VampPrior components are conditional
    pseudo-action posteriors, ``q_phi(z | u_k, s)``.

    Args:
        latent_dimension: Dimension of latent variable z
        num_components: Number of mixture components K
        action_space: ActionSpace defining the action dimensions
        prediction_horizon: Number of timesteps in action chunks
        device: Device to place prior on
        min_logvar: Optional minimum logvar clamp
    """

    def __init__(
        self,
        latent_dimension: int,
        num_components: int,
        action_space: ActionSpace,
        prediction_horizon: int,
        device: str,
        min_logvar: float | None = None,
    ):
        super().__init__(latent_dimension=latent_dimension, device=device)
        self.num_components = num_components
        self.prediction_horizon = prediction_horizon
        self.action_dim = action_space.get_total_action_dim()
        self.min_logvar = min_logvar
        self.action_keys, self.action_dimensions = self._get_action_layout(
            action_space=action_space,
        )
        self.pseudo_inputs = nn.Parameter(
            torch.randn(num_components, prediction_horizon, self.action_dim)
        )
        self.log_weights = nn.Parameter(torch.zeros(num_components, 1, 1))
        self._encoder_ref: weakref.ReferenceType[PosteriorLatentEncoder] | None = None
        self.to(torch.device(device))

    def get_auxiliary_output_keys(self) -> set[str]:
        """Return prediction keys produced by the mixture prior."""
        return {
            LatentKey.PRIOR_LATENT.value,
            LatentKey.PRIOR_LOG_PROB.value,
        }

    @staticmethod
    def _get_action_layout(action_space: ActionSpace) -> tuple[list[str], list[int]]:
        action_keys = []
        action_dimensions = []
        for action_key in sorted(action_space.actions_metadata.keys()):
            metadata = action_space.actions_metadata[action_key]
            if metadata.requires_prediction_head:
                action_keys.append(action_key)
                action_dimensions.append(metadata.prediction_dimension)

        total_dimension = sum(action_dimensions)
        expected_dimension = action_space.get_total_action_dim()
        if not action_keys:
            raise ValueError("VampPrior requires at least one predicted action key.")
        if total_dimension != expected_dimension:
            raise ValueError(
                "VampPrior action layout dimension mismatch: "
                f"metadata sums to {total_dimension}, "
                f"but action_space.get_total_action_dim() returned {expected_dimension}."
            )
        return action_keys, action_dimensions

    def _build_pseudo_actions(
        self,
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        """Split flat pseudo-actions into the same keys used by real actions."""
        pseudo_components = torch.split(
            self.pseudo_inputs,
            self.action_dimensions,
            dim=-1,
        )
        pseudo_actions = {}
        for action_key, component in zip(
            self.action_keys,
            pseudo_components,
            strict=True,
        ):
            component = (
                component.unsqueeze(0)
                .expand(batch_size, -1, -1, -1)
                .reshape(
                    batch_size * self.num_components,
                    self.prediction_horizon,
                    component.size(-1),
                )
            )
            pseudo_actions[action_key] = component

        pseudo_actions[SampleKey.IS_PAD_ACTION.value] = torch.zeros(
            batch_size * self.num_components,
            self.prediction_horizon,
            dtype=torch.bool,
            device=self.pseudo_inputs.device,
        )
        return pseudo_actions

    @staticmethod
    def _require_observations(
        observations: dict[str, torch.Tensor] | None,
    ) -> dict[str, torch.Tensor]:
        if not observations:
            raise ValueError(
                "VampPrior requires observations to compute "
                "q_phi(z | pseudo_actions, observations)."
            )
        return observations

    def _get_observation_batch_size(
        self,
        observations: dict[str, torch.Tensor],
    ) -> int:
        return next(iter(observations.values())).size(0)

    def _repeat_observations_for_components(
        self,
        observations: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {
            key: value.repeat_interleave(self.num_components, dim=0)
            for key, value in observations.items()
        }

    def wire_posterior(self, posterior: PosteriorLatentEncoder) -> None:
        """Wire shared state from the posterior encoder.

        Extracts the encoder reference needed to compute mixture
        components from learnable pseudo-inputs.

        Args:
            posterior: Posterior encoder that maps inputs to (mu, logvar).
        """
        self._encoder_ref = weakref.ref(posterior)

    @property
    def encoder(self) -> PosteriorLatentEncoder:
        """Get the posterior encoder."""
        if self._encoder_ref is None:
            raise RuntimeError(
                "VampPrior encoder not set. Call wire_posterior() first or ensure "
                "VariationalAlgorithm properly initializes the prior."
            )
        encoder = self._encoder_ref()
        if encoder is None:
            raise RuntimeError(
                "VampPrior encoder reference is no longer valid. Keep the posterior "
                "encoder alive while the prior is in use."
            )
        return encoder

    def get_mixture_params(
        self,
        observations: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute mixture component parameters by passing pseudo-inputs through encoder.

        Returns:
            Tuple of conditional (means, logvars), each shaped (B, K, latent_dim).
        """
        observations = self._require_observations(observations=observations)
        batch_size = self._get_observation_batch_size(observations=observations)
        encoder_observations = self._repeat_observations_for_components(
            observations=observations,
        )

        pseudo_actions = self._build_pseudo_actions(batch_size=batch_size)
        encoder_output = self.encoder.encode(
            actions=pseudo_actions,
            observations=encoder_observations,
        )
        mu = encoder_output[LatentKey.POSTERIOR_MU.value]  # (K, latent_dim)
        logvar = encoder_output[LatentKey.POSTERIOR_LOGVAR.value]  # (K, latent_dim)
        mu = mu.reshape(batch_size, self.num_components, self.latent_dimension)
        logvar = logvar.reshape(
            batch_size,
            self.num_components,
            self.latent_dimension,
        )
        if self.min_logvar is not None:
            logvar = torch.clamp(logvar, min=self.min_logvar)
        return mu, logvar

    def build_training_target(
        self,
        posterior_output: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Use a differentiable posterior sample for the sample-based KL."""
        return posterior_output[LatentKey.POSTERIOR_LATENT.value]

    def sample_prior(
        self,
        batch_size: int,
        observations: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Sample from the VampPrior mixture.

        Args:
            batch_size: Number of samples to generate
            observations: Required conditioning features

        Returns:
            Sampled latents of shape (batch_size, latent_dim)
        """
        mu, logvar = self.get_mixture_params(observations=observations)
        weights = F.softmax(self.log_weights.view(-1), dim=0)  # (K,)
        component_indices = torch.multinomial(
            weights, batch_size, replacement=True
        )  # (batch_size,)
        if mu.size(0) != batch_size:
            raise ValueError(
                "VampPrior conditional mixture batch size mismatch: "
                f"got {mu.size(0)} component batches for requested "
                f"batch_size {batch_size}."
            )
        batch_indices = torch.arange(batch_size, device=mu.device)
        selected_mu = mu[batch_indices, component_indices]
        selected_logvar = logvar[batch_indices, component_indices]
        std = (selected_logvar / 2).exp()
        eps = torch.randn_like(std)
        z = selected_mu + std * eps
        return z

    def forward(
        self,
        target_latents: torch.Tensor | None,
        observations: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute prior log probability for training.

        VampPrior is a mixture of Gaussians. Unlike single-Gaussian priors,
        we return log_prob directly since mu/logvar don't represent a mixture.

        Args:
            target_latents: Target latents from posterior (B, latent_dim)
            observations: Conditioning features

        Returns:
            Dictionary with LatentKey.PRIOR_LOG_PROB
        """
        if target_latents is None:
            raise ValueError(
                "VampPrior.forward() requires target_latents for log-prob "
                "computation. Use sample_prior() for inference."
            )
        log_prob = self.log_prob(target_latents, observations=observations)
        return {
            LatentKey.PRIOR_LOG_PROB.value: log_prob,
        }

    def log_prob(
        self,
        z: torch.Tensor,
        observations: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Compute log probability of z under the VampPrior mixture.

        Args:
            z: Latent samples of shape (B, latent_dim)
            observations: Required conditioning features

        Returns:
            Log probability of shape (B,) - summed over latent dimensions
        """
        mu, logvar = self.get_mixture_params(observations=observations)
        weights = F.softmax(self.log_weights.view(-1), dim=0)  # (K,)
        log_p_components = log_normal_diag(
            z.unsqueeze(1),
            mu,
            logvar,
        ).sum(dim=-1)  # (B, K)
        log_p_weighted = log_p_components + torch.log(weights).unsqueeze(0)
        log_prob = torch.logsumexp(log_p_weighted, dim=1)  # (B,)
        return log_prob
