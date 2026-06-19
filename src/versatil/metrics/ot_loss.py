"""Loss functions using Sinkhorn Optimal Transport (geomloss + pykeops).

IMPORTANT: geomloss is imported lazily to avoid PyKeOps JIT compilation overhead.
"""

import math

import torch
import torch.nn.functional as F

from versatil.metrics.base import LossOutput, ScalarWeightedLoss
from versatil.metrics.constants import MetadataKey, MetricKey
from versatil.models.decoding.constants import LatentKey


def _reference_scale(dim: int, expected_std: float) -> float:
    """Expected pairwise L2 distance under the assumed latent distribution.

    For two samples x, y ~ N(0, (expected_std)^2 * I) in R^dim, we have
    E[||x - y||_2] ~= sqrt(2 * dim) * expected_std. We use this as a
    reference length scale: blur and reach are specified as dimensionless
    fractions of it, so the same configuration works across different
    dimensions without retuning.

    Args:
        dim: Dimensionality of the sample space.
        expected_std: Expected per-dimension standard deviation. Use 1.0
            for latents from Gaussian-reparameterized or LayerNorm-output
            transformer posteriors/priors. Use ~1/sqrt(3) ~ 0.577 for
            actions normalized to [-1, 1] via LinearNormalizer.
    """
    return (2.0 * dim) ** 0.5 * expected_std


class OptimalTransportLoss(ScalarWeightedLoss):
    """Debiased Sinkhorn divergence between predicted and target action chunks.

    Note:
        Uses GeomLoss ``SamplesLoss(loss="sinkhorn", ...)`` with
        ``debias=True``, implementing the Sinkhorn divergence
        ``S_eps(a, b) = OT_eps(a, b) - 0.5 * OT_eps(a, a) - 0.5 * OT_eps(b, b)``
        (Feydy et al., AISTATS 2019, arXiv:1810.08278). Debiasing is
        essential: raw entropic OT does not vanish on identical
        distributions and its minimizer is a shrunk version of the target.

        Interpolation property (Feydy et al. 2019):
            small blur: S -> Wasserstein distance W_p
            large blur: S -> (1/2) MMD with negative-cost kernel k = -c

        A linear time embedding in [0, 1] * time_scale is concatenated to
        each action vector before the Sinkhorn computation, giving OT a
        soft temporal-alignment preference: coupling actions at similar
        trajectory positions is cheaper than coupling across time.
        time_scale=0 gives permutation-invariant OT over the horizon.

        Unbalanced variant: passing reach_multiplier enables JUMBOT-style
        unbalanced OT (Fatras et al., ICML 2021, arXiv:2103.03606), letting
        outlier mass be dropped at a KL cost rather than force-matched.

        Reference: Genevay, Peyre, Cuturi (AISTATS 2018, arXiv:1706.00292)
        "Learning Generative Models with Sinkhorn Divergences".

        Requires the optional dependencies geomloss and pykeops
        (``pip install geomloss pykeops``). Imported lazily to avoid
        PyKeOps JIT compilation unless this loss is instantiated.
    """

    def __init__(
        self,
        action_keys: list[str],
        weight: float = 1.0,
        p: int = 2,
        blur_fraction: float = 0.1,
        reach_multiplier: float | None = None,
        expected_std: float = 1.0,
        time_scale: float = 1.0,
    ):
        """Initializes the OptimalTransportLoss.

        Args:
            action_keys: List of keys for action tensors in predictions and targets.
            weight: Scaling factor for the total loss.
            p: Exponent for the ground cost. p=1 gives ``||a - a'||_2``,
                p=2 gives ``(1/2) * ||a - a'||_2^2``.
            blur_fraction: Dimensionless Sinkhorn regularization, expressed
                as a fraction of the reference pairwise scale
                sqrt(2 * dim) * expected_std. GeomLoss recommends ~0.1.
            reach_multiplier: Unbalanced OT scale, as a multiple of the
                reference pairwise scale. ``None`` keeps balanced OT.
                Typical values for mild outlier tolerance are 3.0-10.0.
            expected_std: Expected per-dimension standard deviation of
                the action samples. For actions normalized to [-1, 1],
                use ~1/sqrt(3) ~ 0.577.
            time_scale: Scaling factor for the linear time embedding
                concatenated to actions. time_scale=0 gives
                permutation-invariant OT over the horizon.

        Raises:
            ImportError: If geomloss is not installed.
        """
        super().__init__()
        self.weight = weight
        self.action_keys = action_keys
        self.p = p
        self.blur_fraction = blur_fraction
        self.reach_multiplier = reach_multiplier
        self.expected_std = expected_std
        self.time_scale = time_scale
        # Lazy import to avoid PyKeOps compilation overhead unless this loss is used.
        try:
            from geomloss import SamplesLoss  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "OptimalTransportLoss requires geomloss and pykeops. "
                "Install with: pip install geomloss pykeops"
            ) from e

        self._samples_loss_class = SamplesLoss
        # Constructed lazily in forward() since blur and reach are derived
        # from the trailing tensor dim, which is only known at call time.
        self.ot: SamplesLoss | None = None

    def _build_sinkhorn(self, dim: int):
        reference_scale = _reference_scale(dim=dim, expected_std=self.expected_std)
        blur = self.blur_fraction * reference_scale
        reach = (
            self.reach_multiplier * reference_scale
            if self.reach_multiplier is not None
            else None
        )
        return self._samples_loss_class(
            loss="sinkhorn",
            p=self.p,
            blur=blur,
            reach=reach,
            debias=True,
        )

    def get_required_keys(self) -> set[str]:
        """Return the action keys this loss consumes."""
        return set(self.action_keys)

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Computes the forward pass for the OT loss.

        Flattens and masks actions for composite cost ||a - a'||^p then
        applies debiased Sinkhorn OT.

        Args:
            predictions: Dict of predicted action tensors.
            targets: Dict of target action tensors.
            is_pad: Optional padding mask (B, horizon); True where padded.

        Returns:
            LossOutput with total weighted loss and the Sinkhorn component.

        Raises:
            ValueError: If required action keys are missing in predictions
                or targets.
        """
        for action_key in self.action_keys:
            if action_key not in predictions or action_key not in targets:
                raise ValueError(
                    f"Predictions and targets must contain key '{action_key}' "
                    f"for Optimal Transport Loss."
                )
        total_predictions = torch.cat(
            [predictions[k] for k in self.action_keys], dim=-1
        )  # (B, horizon, action_total_dim)
        total_targets = torch.cat(
            [targets[k] for k in self.action_keys], dim=-1
        )  # (B, horizon, action_total_dim)
        batch_size, horizon, _ = total_predictions.shape
        time_embeddings = torch.linspace(
            0, 1, steps=horizon, device=total_predictions.device
        )
        time_embeddings = time_embeddings.view(1, horizon, 1).expand(batch_size, -1, -1)
        time_embeddings = time_embeddings * self.time_scale
        predictions_with_time = torch.cat(
            [total_predictions, time_embeddings], dim=-1
        )  # (B, horizon, action_total_dim + 1)
        targets_with_time = torch.cat(
            [total_targets, time_embeddings], dim=-1
        )  # (B, horizon, action_total_dim + 1)
        if is_pad is None:
            is_pad = torch.zeros(
                (batch_size, horizon), dtype=torch.bool, device=total_predictions.device
            )
        weights = (~is_pad).float()  # 1.0 for valid points, 0.0 for padded points
        weight_sums = weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        normalized_weights = weights / weight_sums
        if self.ot is None:
            self.ot = self._build_sinkhorn(dim=predictions_with_time.shape[-1])
        # GeomLoss API: (weights_x, samples_x, weights_y, samples_y).
        ot_loss = self.ot(
            normalized_weights,
            predictions_with_time,
            normalized_weights,
            targets_with_time,
        ).mean()
        return LossOutput(
            total_loss=self.weight * ot_loss,
            component_losses={MetricKey.OPTIMAL_TRANSPORT_LOSS.value: ot_loss},
        )


class LatentOptimalTransportLoss(ScalarWeightedLoss):
    """Debiased Sinkhorn divergence between posterior and prior latent samples.

    Regularizes the aggregate posterior ``q(z) = E_x[q(z|x)]`` toward the
    prior ``p(z)`` via sample-based OT (WAE strategy with Sinkhorn instead
    of MMD). Only the aggregate posterior is constrained to match the
    prior, so per-sample posteriors remain free to separate.

    Note:
        Assumes the latent samples have approximately unit per-dimension
        standard deviation, which holds for the Gaussian-reparameterized
        and LayerNorm-output transformer posteriors and priors.
    """

    def __init__(
        self,
        weight: float = 1.0,
        p: int = 2,
        blur_fraction: float = 0.1,
        reach_multiplier: float | None = None,
        prior_target_key: str = LatentKey.POSTERIOR_LATENT.value,
    ):
        """Initialize latent OT loss.

        Args:
            weight: Scaling factor for the total loss.
            p: Exponent for the ground cost. p=2 is standard for
                W_2-style regularization of latent distributions.
            blur_fraction: Dimensionless Sinkhorn regularization, as a
                fraction of the reference pairwise scale sqrt(2 * dim).
            reach_multiplier: Unbalanced OT scale, as a multiple of the
                reference pairwise scale. ``None`` keeps balanced OT.
            prior_target_key: Posterior output key used as aggregate prior-matching samples.
                Use ``LatentKey.POSTERIOR_MU`` for deterministic WAE-style matching.

        Raises:
            ImportError: If geomloss is not installed.
        """
        super().__init__()
        self.weight = weight
        self.p = p
        self.blur_fraction = blur_fraction
        self.reach_multiplier = reach_multiplier
        self.prior_target_key = prior_target_key
        try:
            from geomloss import SamplesLoss  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "LatentOptimalTransportLoss requires geomloss and pykeops. "
                "Install with: pip install geomloss pykeops"
            ) from e
        self._samples_loss_class = SamplesLoss
        self.ot: SamplesLoss | None = None

    def _build_sinkhorn(self, dim: int):
        reference_scale = _reference_scale(dim=dim, expected_std=1.0)
        blur = self.blur_fraction * reference_scale
        reach = (
            self.reach_multiplier * reference_scale
            if self.reach_multiplier is not None
            else None
        )
        return self._samples_loss_class(
            loss="sinkhorn",
            p=self.p,
            blur=blur,
            reach=reach,
            debias=True,
        )

    def get_required_keys(self) -> set[str]:
        """Get required prediction keys."""
        return {self.prior_target_key, LatentKey.PRIOR_LATENT.value}

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute Sinkhorn divergence between posterior and prior latents.

        Args:
            predictions: Must contain posterior and prior latent tensors
                of shape (B, latent_dim).
            targets: Unused.
            is_pad: Unused.

        Returns:
            LossOutput with Sinkhorn divergence loss.
        """
        required = self.get_required_keys()
        if not all(k in predictions for k in required):
            raise ValueError(
                f"Predictions must contain {required} for LatentOptimalTransportLoss."
            )
        z_posterior = predictions[self.prior_target_key]  # (B, latent_dim)
        z_prior = predictions[LatentKey.PRIOR_LATENT.value]  # (B, latent_dim)

        if self.ot is None:
            self.ot = self._build_sinkhorn(dim=z_posterior.shape[-1])

        ot_loss = self.ot(z_posterior, z_prior).mean()

        metadata: dict[str, torch.Tensor] = {MetadataKey.PRIOR_Z.value: z_prior}
        posterior_latent = predictions.get(LatentKey.POSTERIOR_LATENT.value)
        if posterior_latent is not None:
            metadata[MetadataKey.POSTERIOR_Z.value] = posterior_latent
        posterior_mu = predictions.get(LatentKey.POSTERIOR_MU.value)
        if posterior_mu is not None:
            metadata[MetadataKey.POSTERIOR_MU.value] = posterior_mu
        prior_mu = predictions.get(LatentKey.PRIOR_MU.value)
        if prior_mu is not None:
            metadata[MetadataKey.PRIOR_MU.value] = prior_mu
        posterior_logvar = predictions.get(LatentKey.POSTERIOR_LOGVAR.value)
        if posterior_logvar is not None:
            metadata[MetadataKey.POSTERIOR_LOGVAR.value] = posterior_logvar
        prior_logvar = predictions.get(LatentKey.PRIOR_LOGVAR.value)
        if prior_logvar is not None:
            metadata[MetadataKey.PRIOR_LOGVAR.value] = prior_logvar
        return LossOutput(
            total_loss=self.weight * ot_loss,
            component_losses={MetricKey.SINKHORN_LOSS.value: ot_loss},
            metadata=metadata,
        )


class RelaxedConditionalLatentOptimalTransportLoss(LatentOptimalTransportLoss):
    """Relaxed conditional Sinkhorn loss over joint state-latent samples.

    This matches empirical ``(s, z_posterior)`` samples to ``(s, z_prior)``
    samples. A finite ``state_weight`` lets mass move across nearby state
    vectors.
    """

    def __init__(
        self,
        weight: float = 1.0,
        p: int = 2,
        blur_fraction: float = 0.1,
        reach_multiplier: float | None = None,
        prior_target_key: str = LatentKey.POSTERIOR_LATENT.value,
        condition_key: str = LatentKey.PRIOR_CONDITION.value,
        state_weight: float = 1.0,
        normalize_condition: bool = True,
    ):
        """Initialize relaxed conditional latent OT loss."""
        if state_weight < 0.0:
            raise ValueError(f"state_weight must be non-negative, got {state_weight}.")
        super().__init__(
            weight=weight,
            p=p,
            blur_fraction=blur_fraction,
            reach_multiplier=reach_multiplier,
            prior_target_key=prior_target_key,
        )
        self.condition_key = condition_key
        self.state_weight = state_weight
        self.normalize_condition = normalize_condition

    def get_required_keys(self) -> set[str]:
        """Get required prediction keys."""
        return {
            self.prior_target_key,
            LatentKey.PRIOR_LATENT.value,
            self.condition_key,
        }

    def _joint_samples(
        self,
        latents: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if latents.ndim != 2:
            raise ValueError(
                f"Latent samples must have shape (batch, dimension), got {latents.shape}."
            )
        if condition.ndim != 2:
            raise ValueError(
                f"Condition samples must have shape (batch, dimension), got {condition.shape}."
            )
        if latents.shape[0] != condition.shape[0]:
            raise ValueError(
                "Latent and condition samples must have the same batch size, "
                f"got {latents.shape[0]} and {condition.shape[0]}."
            )
        if self.normalize_condition:
            condition = F.normalize(condition, p=2, dim=-1)
        condition = condition.detach() * math.sqrt(self.state_weight)
        return torch.cat([condition, latents], dim=-1)

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute relaxed conditional Sinkhorn divergence."""
        required = self.get_required_keys()
        if not all(k in predictions for k in required):
            raise ValueError(
                "Predictions must contain "
                f"{required} for RelaxedConditionalLatentOptimalTransportLoss."
            )
        posterior_latents = predictions[self.prior_target_key].float()
        prior_latents = predictions[LatentKey.PRIOR_LATENT.value].float()
        condition = predictions[self.condition_key].float()
        posterior_joint = self._joint_samples(
            latents=posterior_latents,
            condition=condition,
        )
        prior_joint = self._joint_samples(
            latents=prior_latents,
            condition=condition,
        )

        if self.ot is None:
            self.ot = self._build_sinkhorn(dim=posterior_joint.shape[-1])

        ot_loss = self.ot(posterior_joint, prior_joint).mean()

        metadata: dict[str, torch.Tensor] = {
            MetadataKey.PRIOR_Z.value: prior_latents,
            MetadataKey.PRIOR_CONDITION.value: condition,
        }
        posterior_latent = predictions.get(LatentKey.POSTERIOR_LATENT.value)
        if posterior_latent is not None:
            metadata[MetadataKey.POSTERIOR_Z.value] = posterior_latent
        posterior_mu = predictions.get(LatentKey.POSTERIOR_MU.value)
        if posterior_mu is not None:
            metadata[MetadataKey.POSTERIOR_MU.value] = posterior_mu
        prior_mu = predictions.get(LatentKey.PRIOR_MU.value)
        if prior_mu is not None:
            metadata[MetadataKey.PRIOR_MU.value] = prior_mu
        posterior_logvar = predictions.get(LatentKey.POSTERIOR_LOGVAR.value)
        if posterior_logvar is not None:
            metadata[MetadataKey.POSTERIOR_LOGVAR.value] = posterior_logvar
        prior_logvar = predictions.get(LatentKey.PRIOR_LOGVAR.value)
        if prior_logvar is not None:
            metadata[MetadataKey.PRIOR_LOGVAR.value] = prior_logvar
        return LossOutput(
            total_loss=self.weight * ot_loss,
            component_losses={
                MetricKey.RELAXED_CONDITIONAL_SINKHORN_LOSS.value: ot_loss
            },
            metadata=metadata,
        )
