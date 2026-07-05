"""Maximum Mean Discrepancy losses for latent distribution matching."""

import math

import torch
import torch.nn.functional as F

from versatil.metrics.base import (
    BaseLoss,
    LossOutput,
    ScalarWeightedLoss,
    WeightsDictionary,
)
from versatil.metrics.constants import MetadataKey, MetricKey
from versatil.metrics.kernels import KernelType
from versatil.models.decoding.constants import DecoderOutputKey, LatentKey


class MaximumMeanDiscrepancyLoss(BaseLoss):
    """MMD loss for regularizing latent distributions toward a prior.

    Ref: [Info-VAE / MMD-VAE](https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/)
    """

    @property
    def weights(self) -> WeightsDictionary:
        """Getter that returns dictionary with weight keys and scalar coefficients."""
        return {
            "weight": self.weight,
            "prior_regularization_weight": self.prior_regularization_weight,
        }

    def set_weights(self, new_weights: WeightsDictionary) -> None:
        """Setter that updates the weight scalar coefficients."""
        self._validate_weights(new_weights)
        self.weight = new_weights["weight"]
        self.prior_regularization_weight = new_weights["prior_regularization_weight"]

    def __init__(
        self,
        weight: float = 1.0,
        prior_regularization_weight: float = 0.0,
        kernel_type: str = KernelType.RBF.value,
        bandwidth_multipliers: list[float] | None = None,
        use_median_heuristic: bool = True,
        use_fixed_gaussian_as_prior: bool = False,
        prior_target_key: str = LatentKey.POSTERIOR_LATENT.value,
    ):
        """Initialize MMD loss.

        Args:
            weight: Loss weight for MMD(posterior, prior).
            prior_regularization_weight: Weight for MMD(prior, N(0,I)) regularization.
                Only meaningful for learned priors.
            kernel_type: Kernel type for MMD computation (see KernelType enum).
            bandwidth_multipliers: Scale factors for bandwidth. When
                use_median_heuristic=True these scale the adaptive median.
                When False these are absolute bandwidth values. WAE
                recommends [2 * latent_dim] with use_median_heuristic=False.
            use_median_heuristic: Adaptive bandwidth via median heuristic
                (True) or fixed absolute bandwidths (False).
            use_fixed_gaussian_as_prior: If True, always use standard Gaussian as prior.
            prior_target_key: Posterior output key used as aggregate prior-matching samples.
                Use ``LatentKey.POSTERIOR_MU`` for deterministic WAE-style matching.
        """
        super().__init__()
        self.weight = weight
        self.prior_regularization_weight = prior_regularization_weight
        self.prior_target_key = prior_target_key
        self.kernel = KernelType(kernel_type).to_kernel(
            bandwidth_multipliers=bandwidth_multipliers,
            use_median_heuristic=use_median_heuristic,
        )
        self.use_fixed_gaussian_as_prior = use_fixed_gaussian_as_prior

    def get_required_keys(self) -> set[str]:
        """Get required keys for MMD loss."""
        keys = {self.prior_target_key}
        if not self.use_fixed_gaussian_as_prior:
            keys.add(LatentKey.PRIOR_LATENT.value)
        return keys

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute MMD between posterior samples and the configured prior.

        Args:
            predictions: Must contain ``prior_target_key`` and, unless
                ``use_fixed_gaussian_as_prior`` is True,
                ``LatentKey.PRIOR_LATENT.value``, each with shape
                (B, latent_dim).
            targets: Unused; prior samples come from ``predictions``.
            is_pad: Unused.

        Returns:
            LossOutput with MMD loss.
        """
        required_keys = self.get_required_keys()
        if not all(k in predictions for k in required_keys):
            raise ValueError(
                f"Predictions must contain '{required_keys}' for MaximumMeanDiscrepancyLoss."
            )

        z_posterior = predictions[self.prior_target_key]  # (B, latent_dim)
        original_z_prior = predictions.get(
            LatentKey.PRIOR_LATENT.value
        )  # (B, latent_dim) or None
        if self.use_fixed_gaussian_as_prior:
            z_prior = torch.randn_like(z_posterior)  # (B, latent_dim)
        else:
            if original_z_prior is None:
                raise ValueError(
                    "Prior latent is required when use_fixed_gaussian_as_prior=False."
                )
            z_prior = original_z_prior  # (B, latent_dim)

        # Resolve one bandwidth from the combined (posterior, prior) samples
        # and reuse it for all three MMD terms. Letting the kernel resolve a
        # fresh bandwidth per call would compute the three terms under
        # different kernels, breaking the MMD^2 definition.
        posterior_prior_samples = torch.cat([z_posterior, z_prior], dim=0)
        shared_bandwidth = self.kernel.resolve_base_bandwidth(posterior_prior_samples)

        k_posterior_posterior = self.kernel(
            z_posterior, z_posterior, bandwidth=shared_bandwidth
        )
        k_prior_prior = self.kernel(z_prior, z_prior, bandwidth=shared_bandwidth)
        k_posterior_prior = self.kernel(
            z_posterior, z_prior, bandwidth=shared_bandwidth
        )

        mmd_sq = (
            k_posterior_posterior.mean()
            + k_prior_prior.mean()
            - 2 * k_posterior_prior.mean()
        )
        mmd_sq = torch.clamp(mmd_sq, min=0.0)

        component_losses = {MetricKey.MMD_LOSS.value: mmd_sq}
        total_loss = self.weight * mmd_sq

        if self.prior_regularization_weight > 0.0:
            z_standard = torch.randn_like(z_prior)  # (B, latent_dim)

            prior_standard_samples = torch.cat([z_prior, z_standard], dim=0)
            regularization_bandwidth = self.kernel.resolve_base_bandwidth(
                prior_standard_samples
            )

            k_prior_prior_regularization = self.kernel(
                z_prior, z_prior, bandwidth=regularization_bandwidth
            )
            k_standard_standard = self.kernel(
                z_standard, z_standard, bandwidth=regularization_bandwidth
            )
            k_prior_standard = self.kernel(
                z_prior, z_standard, bandwidth=regularization_bandwidth
            )

            prior_mmd_sq = (
                k_prior_prior_regularization.mean()
                + k_standard_standard.mean()
                - 2 * k_prior_standard.mean()
            )
            prior_mmd_sq = torch.clamp(prior_mmd_sq, min=0.0)

            component_losses[MetricKey.HYPERPRIOR_MMD_REGULARIZATION.value] = (
                prior_mmd_sq
            )
            total_loss = total_loss + self.prior_regularization_weight * prior_mmd_sq

        metadata = {}
        posterior_latent = predictions.get(LatentKey.POSTERIOR_LATENT.value)
        if posterior_latent is not None:
            metadata[MetadataKey.POSTERIOR_Z.value] = posterior_latent
        posterior_mu = predictions.get(LatentKey.POSTERIOR_MU.value)
        if posterior_mu is not None:
            metadata[MetadataKey.POSTERIOR_MU.value] = posterior_mu
        posterior_logvar = predictions.get(LatentKey.POSTERIOR_LOGVAR.value)
        if posterior_logvar is not None:
            metadata[MetadataKey.POSTERIOR_LOGVAR.value] = posterior_logvar
        if self.use_fixed_gaussian_as_prior:
            metadata[MetadataKey.HYPERPRIOR_Z.value] = z_prior
        if original_z_prior is not None:
            metadata[MetadataKey.PRIOR_Z.value] = original_z_prior
        prior_mu = predictions.get(LatentKey.PRIOR_MU.value)
        if prior_mu is not None:
            metadata[MetadataKey.PRIOR_MU.value] = prior_mu
        prior_logvar = predictions.get(LatentKey.PRIOR_LOGVAR.value)
        if prior_logvar is not None:
            metadata[MetadataKey.PRIOR_LOGVAR.value] = prior_logvar

        return LossOutput(
            total_loss=total_loss,
            component_losses=component_losses,
            metadata=metadata,
        )


class ConditionalMaximumMeanDiscrepancyLoss(BaseLoss):
    """Product-kernel joint MMD for conditional aggregate matching.

    This regularizes ``q(z|s)`` toward ``p(z|s)`` by matching the empirical
    joint samples ``(s, z_posterior)`` and ``(s, z_prior)``. The state vector
    is emitted by the prior and should be action-free. Separate kernels are
    used for state and latent samples so their bandwidths can be controlled
    independently.
    """

    def __init__(
        self,
        weight: float = 1.0,
        state_weight: float = 1.0,
        kernel_type: str = KernelType.RBF.value,
        bandwidth_multipliers: list[float] | None = None,
        use_median_heuristic: bool = False,
        condition_kernel_type: str = KernelType.RBF.value,
        condition_bandwidth_multipliers: list[float] | None = None,
        condition_use_median_heuristic: bool = True,
        prior_target_key: str = LatentKey.POSTERIOR_LATENT.value,
        condition_key: str = LatentKey.PRIOR_CONDITION.value,
        normalize_condition: bool = True,
    ):
        """Initialize conditional MMD loss."""
        super().__init__()
        if state_weight < 0.0:
            raise ValueError(f"state_weight must be non-negative, got {state_weight}.")
        self.weight = weight
        self.state_weight = state_weight
        self.prior_target_key = prior_target_key
        self.condition_key = condition_key
        self.normalize_condition = normalize_condition
        self.latent_kernel = KernelType(kernel_type).to_kernel(
            bandwidth_multipliers=bandwidth_multipliers,
            use_median_heuristic=use_median_heuristic,
        )
        self.condition_kernel = KernelType(condition_kernel_type).to_kernel(
            bandwidth_multipliers=condition_bandwidth_multipliers,
            use_median_heuristic=condition_use_median_heuristic,
        )

    @property
    def weights(self) -> WeightsDictionary:
        """Getter that returns dictionary with weight keys and scalar coefficients."""
        return {"weight": self.weight}

    def set_weights(self, new_weights: WeightsDictionary) -> None:
        """Setter that updates the weight scalar coefficients."""
        self._validate_weights(new_weights)
        self.weight = new_weights["weight"]

    def get_required_keys(self) -> set[str]:
        """Get required keys for conditional MMD loss."""
        return {
            self.prior_target_key,
            LatentKey.PRIOR_LATENT.value,
            self.condition_key,
        }

    def _condition_samples(
        self,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if condition.ndim != 2:
            raise ValueError(
                f"Condition samples must have shape (batch, dimension), got {condition.shape}."
            )
        if self.normalize_condition:
            condition = F.normalize(condition, p=2, dim=-1)
        return condition.detach() * math.sqrt(self.state_weight)

    def _validate_sample_shapes(
        self,
        posterior_latents: torch.Tensor,
        prior_latents: torch.Tensor,
        condition: torch.Tensor,
    ) -> None:
        if posterior_latents.ndim != 2:
            raise ValueError(
                "Posterior latent samples must have shape "
                f"(batch, dimension), got {posterior_latents.shape}."
            )
        if prior_latents.ndim != 2:
            raise ValueError(
                "Prior latent samples must have shape "
                f"(batch, dimension), got {prior_latents.shape}."
            )
        if posterior_latents.shape != prior_latents.shape:
            raise ValueError(
                "Posterior and prior latent samples must have the same shape, "
                f"got {posterior_latents.shape} and {prior_latents.shape}."
            )
        if condition.ndim != 2:
            raise ValueError(
                f"Condition samples must have shape (batch, dimension), got {condition.shape}."
            )
        if posterior_latents.shape[0] != condition.shape[0]:
            raise ValueError(
                "Latent and condition samples must have the same batch size, "
                f"got {posterior_latents.shape[0]} and {condition.shape[0]}."
            )

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute joint MMD between posterior and prior conditioned samples."""
        required_keys = self.get_required_keys()
        if not all(k in predictions for k in required_keys):
            raise ValueError(
                f"Predictions must contain '{required_keys}' for ConditionalMaximumMeanDiscrepancyLoss."
            )

        posterior_latents = predictions[self.prior_target_key].float()
        prior_latents = predictions[LatentKey.PRIOR_LATENT.value].float()
        condition = predictions[self.condition_key].float()
        self._validate_sample_shapes(
            posterior_latents=posterior_latents,
            prior_latents=prior_latents,
            condition=condition,
        )
        condition_samples = self._condition_samples(condition=condition)

        condition_bandwidth = self.condition_kernel.resolve_base_bandwidth(
            condition_samples
        )
        condition_kernel = self.condition_kernel(
            condition_samples,
            condition_samples,
            bandwidth=condition_bandwidth,
        )
        latent_samples = torch.cat([posterior_latents, prior_latents], dim=0)
        latent_bandwidth = self.latent_kernel.resolve_base_bandwidth(latent_samples)
        latent_kernel_posterior = self.latent_kernel(
            posterior_latents,
            posterior_latents,
            bandwidth=latent_bandwidth,
        )
        latent_kernel_prior = self.latent_kernel(
            prior_latents,
            prior_latents,
            bandwidth=latent_bandwidth,
        )
        latent_kernel_cross = self.latent_kernel(
            posterior_latents,
            prior_latents,
            bandwidth=latent_bandwidth,
        )
        conditional_mmd_sq = (
            (condition_kernel * latent_kernel_posterior).mean()
            + (condition_kernel * latent_kernel_prior).mean()
            - 2 * (condition_kernel * latent_kernel_cross).mean()
        )
        conditional_mmd_sq = torch.clamp(conditional_mmd_sq, min=0.0)

        metadata = {}
        posterior_latent = predictions.get(LatentKey.POSTERIOR_LATENT.value)
        if posterior_latent is not None:
            metadata[MetadataKey.POSTERIOR_Z.value] = posterior_latent
        posterior_mu = predictions.get(LatentKey.POSTERIOR_MU.value)
        if posterior_mu is not None:
            metadata[MetadataKey.POSTERIOR_MU.value] = posterior_mu
        posterior_logvar = predictions.get(LatentKey.POSTERIOR_LOGVAR.value)
        if posterior_logvar is not None:
            metadata[MetadataKey.POSTERIOR_LOGVAR.value] = posterior_logvar
        metadata[MetadataKey.PRIOR_Z.value] = predictions[LatentKey.PRIOR_LATENT.value]
        metadata[MetadataKey.PRIOR_CONDITION.value] = condition
        prior_mu = predictions.get(LatentKey.PRIOR_MU.value)
        if prior_mu is not None:
            metadata[MetadataKey.PRIOR_MU.value] = prior_mu
        prior_logvar = predictions.get(LatentKey.PRIOR_LOGVAR.value)
        if prior_logvar is not None:
            metadata[MetadataKey.PRIOR_LOGVAR.value] = prior_logvar

        return LossOutput(
            total_loss=self.weight * conditional_mmd_sq,
            component_losses={MetricKey.CONDITIONAL_MMD_LOSS.value: conditional_mmd_sq},
            metadata=metadata,
        )


class BinaryMaximumMeanDiscrepancyLoss(ScalarWeightedLoss):
    """MMD loss for regularizing binary latent distributions toward a uniform prior.

    Encourages q(b|x) ≈ p(b) where p(b) = Bernoulli(0.5) independent for each bit.
    """

    def __init__(
        self,
        weight: float = 1.0,
        kernel_type: str = KernelType.RBF.value,
        bandwidth_multipliers: list[float] | None = None,
    ):
        """Initialize binary MMD loss.

        Args:
            weight: Loss weight.
            kernel_type: Kernel type for MMD computation (see KernelType enum).
            bandwidth_multipliers: Scale factors for the median heuristic bandwidth.
        """
        super().__init__()
        self.weight = weight
        self.kernel = KernelType(kernel_type).to_kernel(
            bandwidth_multipliers=bandwidth_multipliers
        )

    def get_required_keys(self) -> set[str]:
        """Returns required prediction keys: {DecoderOutputKey.BINARY_LOGITS.value}."""
        return {DecoderOutputKey.BINARY_LOGITS.value}

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute MMD between binary latent samples and uniform Bernoulli prior.

        Args:
            predictions: Must contain DecoderOutputKey.BINARY_LOGITS.value with shape (B, H).
            targets: Unused (prior is implicit).
            is_pad: Unused.

        Returns:
            LossOutput with MMD loss.
        """
        if DecoderOutputKey.BINARY_LOGITS.value not in predictions:
            raise ValueError(
                f"Predictions must contain '{DecoderOutputKey.BINARY_LOGITS.value}' for BinaryMaximumMeanDiscrepancyLoss."
            )

        logits = predictions[DecoderOutputKey.BINARY_LOGITS.value]  # (B, T, H)
        probs = torch.sigmoid(logits.float())  # Cast to fp32 for stability
        z_hard = torch.bernoulli(probs)
        z = (
            z_hard - probs.detach() + probs
        )  # Straight-through: forward=hard, backward=soft
        z_prior = torch.bernoulli(
            0.5 * torch.ones_like(z)
        )  # samples from Bernoulli(0.5)

        # Share bandwidth across the three MMD terms (see MaximumMeanDiscrepancyLoss).
        posterior_prior_samples = torch.cat([z, z_prior], dim=0)
        shared_bandwidth = self.kernel.resolve_base_bandwidth(posterior_prior_samples)

        k_posterior_posterior = self.kernel(z, z, bandwidth=shared_bandwidth)
        k_prior_prior = self.kernel(z_prior, z_prior, bandwidth=shared_bandwidth)
        k_posterior_prior = self.kernel(z, z_prior, bandwidth=shared_bandwidth)
        # MMD^2 = E[k(z, z')] + E[k(p, p')] - 2 E[k(z, p)]
        mmd = (
            k_posterior_posterior.mean()
            + k_prior_prior.mean()
            - 2 * k_posterior_prior.mean()
        )
        metadata = {
            MetadataKey.POSTERIOR_Z.value: z,
        }
        return LossOutput(
            total_loss=self.weight * mmd,
            component_losses={MetricKey.BINARY_MMD_LOSS.value: mmd},
            metadata=metadata,
        )
