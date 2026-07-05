"""Denoising loss for learned diffusion priors."""

import torch
import torch.nn.functional as F

from versatil.metrics.base import LossOutput, ScalarWeightedLoss
from versatil.metrics.constants import MetadataKey, MetricKey
from versatil.models.decoding.constants import LatentKey


class PriorDenoisingLoss(ScalarWeightedLoss):
    """Denoising loss for learned diffusion prior.

    Computes MSE loss between predicted noise and target noise from the
    diffusion prior. Used in variational models to train the prior p(z|s)
    to match the posterior q(z|a,s).
    """

    def __init__(self, weight: float = 1.0):
        """Initialize prior denoising loss.

        Args:
            weight: Weight for this loss component
        """
        super().__init__()
        self.weight = weight

    def get_required_keys(self) -> set[str]:
        """Return required prediction keys."""
        return {LatentKey.PRIOR_PREDICTION.value, LatentKey.PRIOR_TARGET.value}

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute prior denoising loss.

        Args:
            predictions: Dictionary containing LatentKey.PRIOR_PREDICTION.value and LatentKey.PRIOR_TARGET.value
            targets: Not used (targets are in predictions dict)
            is_pad: Not used (prior loss doesn't need padding)

        Returns:
            LossOutput with weighted MSE loss

        Raises:
            ValueError: If required keys are missing from predictions
        """
        if LatentKey.PRIOR_PREDICTION.value not in predictions:
            raise ValueError(
                f"Predictions must contain '{LatentKey.PRIOR_PREDICTION.value}' for PriorDenoisingLoss."
            )
        if LatentKey.PRIOR_TARGET.value not in predictions:
            raise ValueError(
                f"Predictions must contain '{LatentKey.PRIOR_TARGET.value}' for PriorDenoisingLoss."
            )
        prior_loss = F.mse_loss(
            predictions[LatentKey.PRIOR_PREDICTION.value],
            predictions[LatentKey.PRIOR_TARGET.value],
        )
        target = predictions[LatentKey.PRIOR_TARGET.value].float()
        target_var = target.var(unbiased=False)
        target_std = torch.sqrt(target_var + 1e-8)
        normalized_mse = prior_loss / (target_var + 1e-8)
        normalized_rmse = torch.sqrt(prior_loss) / target_std
        metadata: dict[str, torch.Tensor] = {}
        if LatentKey.POSTERIOR_LATENT.value in predictions:
            metadata[MetadataKey.POSTERIOR_Z.value] = predictions[
                LatentKey.POSTERIOR_LATENT.value
            ]
        if LatentKey.POSTERIOR_MU.value in predictions:
            metadata[MetadataKey.POSTERIOR_MU.value] = predictions[
                LatentKey.POSTERIOR_MU.value
            ]
        if LatentKey.POSTERIOR_LOGVAR.value in predictions:
            metadata[MetadataKey.POSTERIOR_LOGVAR.value] = predictions[
                LatentKey.POSTERIOR_LOGVAR.value
            ]
        if LatentKey.PRIOR_LATENT.value in predictions:
            metadata[MetadataKey.PRIOR_Z.value] = predictions[
                LatentKey.PRIOR_LATENT.value
            ]
        if LatentKey.PRIOR_MU.value in predictions:
            metadata[MetadataKey.PRIOR_MU.value] = predictions[LatentKey.PRIOR_MU.value]
        if LatentKey.PRIOR_LOGVAR.value in predictions:
            metadata[MetadataKey.PRIOR_LOGVAR.value] = predictions[
                LatentKey.PRIOR_LOGVAR.value
            ]

        return LossOutput(
            total_loss=self.weight * prior_loss,
            component_losses={
                MetricKey.PRIOR_DENOISING_LOSS.value: prior_loss,
                MetricKey.PRIOR_DENOISING_TARGET_STD.value: target_std,
                MetricKey.PRIOR_DENOISING_NORMALIZED_MSE.value: normalized_mse,
                MetricKey.PRIOR_DENOISING_NORMALIZED_RMSE.value: normalized_rmse,
            },
            metadata=metadata,
        )
