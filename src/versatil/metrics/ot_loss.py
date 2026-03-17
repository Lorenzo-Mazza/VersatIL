"""Loss functions using Sinkhorn Optimal Transport (geomloss + pykeops).

IMPORTANT: geomloss is imported lazily to avoid PyKeOps JIT compilation overhead.
"""

import torch

from versatil.metrics import BaseLoss, LossOutput, MetadataKey, MetricKey
from versatil.models.decoding.constants import LatentKey


class OptimalTransportLoss(BaseLoss):
    """Computes an entropic-smoothed version of Kantorovich Optimal Transport (K-OT) loss
    using Sinkhorn divergence.

    Note:
        This loss computes a differentiable OT cost between a predicted and a target probability
        distribution, using the Sinkhorn divergence algorithm.
        Entropic smoothing generates a family of losses interpolating between Wasserstein (OT)
        and Maximum Mean Discrepancy (MMD), thus allowing to find a sweet spot leveraging the
        geometry of OT and the favorable high-dimensional sample complexity of MMD which comes
        with unbiased gradient estimates.
        When the regularization parameter epsilon goes to zero, the loss converges to the
        Wasserstein distance, while for epsilon going to infinity, it converges to MMD with a Gaussian kernel.
        Ref. "Learning Generative Models with Sinkhorn Divergences" (Cuturi et al., 2019)
        https://arxiv.org/abs/1706.00292

        NB: It requires the optional dependencies geomloss and pykeops.
        Install with: pip install geomloss pykeops
        The geomloss library is imported lazily during __init__ to avoid
        triggering PyKeOps JIT compilation unless this loss is actually used.
    """

    def __init__(
        self,
        action_keys: list[str],
        weight: float = 0.1,
        p: int = 2,
        epsilon: float = 0.01,
        time_scale: float = 1.0,
    ):
        """Initializes the OptimalTransportLoss.

        Args:
            action_keys: List of keys for action tensors in predictions and targets.
            weight: Scaling factor for the total loss.
            p: Exponent for the ground cost, 1 for ||a - a'||_2, 2 for 1/2(||a - a'||)^2_2.
            epsilon: Regularization parameter for Sinkhorn (blur = epsilon^p).
            time_scale: Scaling factor for time embedding to be concatenated to actions.

        Raises:
            ImportError: If geomloss is not installed.
        """
        super().__init__()
        self.weight = weight
        self.action_keys = action_keys
        self.time_scale = time_scale
        # Lazy import to avoid PyKeOps compilation overhead unless this loss is used
        try:
            from geomloss import SamplesLoss  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "OptimalTransportLoss requires geomloss and pykeops. "
                "Install with: pip install geomloss pykeops"
            ) from e

        self.ot = SamplesLoss(loss="sinkhorn", p=p, blur=epsilon ** (1 / p))

    def get_required_keys(self) -> set[str]:
        """Gets the required keys for predictions and targets.

        Returns:
            Set of required action keys.
        """
        return set(self.action_keys)

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Computes the forward pass for the OT loss.

        Flattens and masks actions  for composite cost ||a - a'||^2 then applies Sinkhorn OT.

        Args:
            predictions: Dict of predicted action tensors.
            targets: Dict of target action tensors.
            is_pad: Optional padding mask (B, horizon); True where padded.

        Returns:
            LossOutput with total weighted loss and component 'k_ot'.

        Raises:
            ValueError: If required action keys are missing in predictions or targets.
        """
        for action_key in self.action_keys:
            if action_key not in predictions or action_key not in targets:
                raise ValueError(
                    f"Predictions and targets must contain key '{action_key}' for Optimal Transport Loss."
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
        # We need to pass (Weights_X, Samples_X, Weights_Y, Samples_Y) as args here because of GeomLoss API
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


class LatentOptimalTransportLoss(BaseLoss):
    """Sinkhorn divergence for latent regularization between posterior and prior."""

    def __init__(
        self,
        weight: float = 1.0,
        p: int = 2,
        epsilon: float = 0.01,
    ):
        """Initialize latent OT loss.

        Args:
            weight: Scaling factor for the total loss.
            p: Exponent for the ground cost.
            epsilon: Sinkhorn regularization parameter.

        Raises:
            ImportError: If geomloss is not installed.
        """
        super().__init__()
        self.weight = weight
        try:
            from geomloss import SamplesLoss  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "LatentOptimalTransportLoss requires geomloss and pykeops. "
                "Install with: pip install geomloss pykeops"
            ) from e
        self.ot = SamplesLoss(loss="sinkhorn", p=p, blur=epsilon ** (1 / p))

    def get_required_keys(self) -> set[str]:
        """Get required prediction keys."""
        return {LatentKey.POSTERIOR_LATENT.value, LatentKey.PRIOR_LATENT.value}

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute Sinkhorn divergence between posterior and prior latents.

        Args:
            predictions: Must contain posterior and prior latent tensors (B, latent_dim).
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
        z_posterior = predictions[LatentKey.POSTERIOR_LATENT.value]  # (B, latent_dim)
        z_prior = predictions[LatentKey.PRIOR_LATENT.value]  # (B, latent_dim)

        ot_loss = self.ot(z_posterior, z_prior).mean()

        metadata: dict[str, torch.Tensor] = {
            MetadataKey.POSTERIOR_Z.value: z_posterior,
            MetadataKey.PRIOR_Z.value: z_prior,
        }
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
