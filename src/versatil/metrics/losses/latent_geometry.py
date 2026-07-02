"""Moment and covariance regularizers for latent geometry."""

import torch
import torch.nn.functional as F

from versatil.metrics.base import BaseLoss, LossOutput, WeightsDictionary
from versatil.metrics.constants import MetricKey
from versatil.models.decoding.constants import LatentKey


class VICLatentLoss(BaseLoss):
    """VICReg-style covariance + variance loss for latent decorrelation and anti-collapse.

    Note:
        Combines two regularization terms:
        - Covariance: Penalizes off-diagonal covariance to encourage independent dimensions
        - Variance: Hinge loss forcing std >= gamma per dimension to prevent collapse
        Ref. https://arxiv.org/pdf/2105.04906
    """

    def __init__(
        self,
        key: str = LatentKey.POSTERIOR_MU.value,
        covariance_weight: float = 3.0,
        variance_weight: float = 10.0,
        gamma: float = 0.3,
    ):
        """Initialize VICReg latent loss.

        Args:
            key: Prediction key for latent mu tensor.
            covariance_weight: Weight for off-diagonal covariance penalty.
            variance_weight: Weight for variance hinge loss.
            gamma: Hinge threshold for per-dimension standard deviation.
        """
        super().__init__()
        self.key = key
        self.covariance_weight = covariance_weight
        self.variance_weight = variance_weight
        self.gamma = gamma

    @property
    def weights(self) -> WeightsDictionary:
        """Getter that returns dictionary with weight keys and scalar coefficients."""
        return {
            "covariance_weight": self.covariance_weight,
            "variance_weight": self.variance_weight,
        }

    def set_weights(self, new_weights: WeightsDictionary) -> None:
        """Setter that updates the weight scalar coefficients."""
        self._validate_weights(new_weights)
        self.covariance_weight = new_weights["covariance_weight"]
        self.variance_weight = new_weights["variance_weight"]

    def get_required_keys(self) -> set[str]:
        """Get required prediction keys."""
        return {self.key}

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute VICReg loss combining covariance and variance terms.

        Args:
            predictions: Must contain self.key with shape (B, latent_dim).
            targets: Unused.
            is_pad: Unused.

        Returns:
            LossOutput with weighted covariance and variance penalties.
        """
        if self.key not in predictions:
            raise ValueError(
                f"Predictions must contain '{self.key}' for VICLatentLoss."
            )
        latent_vectors = predictions[self.key].float()
        batch_size, latent_dimension = latent_vectors.shape
        centered = latent_vectors - latent_vectors.mean(dim=0)
        standard_deviation = torch.sqrt(centered.var(dim=0) + 1e-6)
        variance_loss = torch.mean(F.relu(self.gamma - standard_deviation))
        covariance = (centered.T @ centered) / (batch_size - 1)
        diagonal_mask = torch.eye(latent_dimension, device=latent_vectors.device)
        off_diagonal_covariance = covariance * (1 - diagonal_mask)
        covariance_loss = off_diagonal_covariance.pow(2).sum() / latent_dimension
        total_loss = (
            self.covariance_weight * covariance_loss
            + self.variance_weight * variance_loss
        )
        return LossOutput(
            total_loss=total_loss,
            component_losses={
                MetricKey.COVARIANCE_LOSS.value: self.covariance_weight
                * covariance_loss,
                MetricKey.VARIANCE_LOSS.value: self.variance_weight * variance_loss,
            },
        )


class PosteriorGeometryLoss(BaseLoss):
    """Moment regularizer for posterior latent geometry.

    The loss keeps posterior means centered, controls per-dimension latent
    scale, optionally caps large standard deviations, and decorrelates latent
    dimensions. Unlike ``VICLatentLoss``, this regularizer penalizes excessive
    posterior spread.
    """

    def __init__(
        self,
        key: str = LatentKey.POSTERIOR_MU.value,
        mean_weight: float = 0.0,
        std_weight: float = 0.0,
        target_std: float = 1.0,
        max_std_weight: float = 0.0,
        max_std: float = 2.0,
        covariance_weight: float = 0.0,
        eps: float = 1e-6,
    ):
        """Initialize posterior geometry loss.

        Args:
            key: Prediction key for latent vectors.
            mean_weight: Weight for squared batch-mean penalty.
            std_weight: Weight for squared deviation from ``target_std``.
            target_std: Desired per-dimension posterior standard deviation.
            max_std_weight: Weight for hinge penalty above ``max_std``.
            max_std: Maximum tolerated per-dimension standard deviation.
            covariance_weight: Weight for off-diagonal covariance penalty.
            eps: Numerical epsilon for standard deviation.
        """
        super().__init__()
        if target_std <= 0.0:
            raise ValueError(f"target_std must be positive, got {target_std}.")
        if max_std <= 0.0:
            raise ValueError(f"max_std must be positive, got {max_std}.")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}.")
        self.key = key
        self.mean_weight = mean_weight
        self.std_weight = std_weight
        self.target_std = target_std
        self.max_std_weight = max_std_weight
        self.max_std = max_std
        self.covariance_weight = covariance_weight
        self.eps = eps

    @property
    def weights(self) -> WeightsDictionary:
        """Getter that returns dictionary with weight keys and scalar coefficients."""
        return {
            "mean_weight": self.mean_weight,
            "std_weight": self.std_weight,
            "max_std_weight": self.max_std_weight,
            "covariance_weight": self.covariance_weight,
        }

    def set_weights(self, new_weights: WeightsDictionary) -> None:
        """Setter that updates the weight scalar coefficients."""
        self._validate_weights(new_weights)
        self.mean_weight = new_weights["mean_weight"]
        self.std_weight = new_weights["std_weight"]
        self.max_std_weight = new_weights["max_std_weight"]
        self.covariance_weight = new_weights["covariance_weight"]

    def get_required_keys(self) -> set[str]:
        """Get required prediction keys."""
        return {self.key}

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute posterior moment and covariance penalties."""
        del targets, is_pad
        if self.key not in predictions:
            raise ValueError(
                f"Predictions must contain '{self.key}' for PosteriorGeometryLoss."
            )
        latent_vectors = predictions[self.key].float()
        if latent_vectors.ndim != 2:
            raise ValueError(
                f"PosteriorGeometryLoss expects '{self.key}' with shape "
                f"(batch_size, latent_dimension), got {tuple(latent_vectors.shape)}."
            )

        batch_size, latent_dimension = latent_vectors.shape
        mean = latent_vectors.mean(dim=0)
        centered = latent_vectors - mean
        standard_deviation = torch.sqrt(centered.square().mean(dim=0) + self.eps)

        mean_loss = mean.square().mean()
        std_loss = (standard_deviation - self.target_std).square().mean()
        max_std_loss = F.relu(standard_deviation - self.max_std).square().mean()
        covariance_denominator = max(batch_size - 1, 1)
        covariance = (centered.T @ centered) / covariance_denominator
        diagonal_mask = torch.eye(latent_dimension, device=latent_vectors.device)
        off_diagonal_covariance = covariance * (1 - diagonal_mask)
        covariance_loss = off_diagonal_covariance.square().sum() / latent_dimension

        weighted_mean_loss = self.mean_weight * mean_loss
        weighted_std_loss = self.std_weight * std_loss
        weighted_max_std_loss = self.max_std_weight * max_std_loss
        weighted_covariance_loss = self.covariance_weight * covariance_loss
        total_loss = (
            weighted_mean_loss
            + weighted_std_loss
            + weighted_max_std_loss
            + weighted_covariance_loss
        )
        return LossOutput(
            total_loss=total_loss,
            component_losses={
                MetricKey.POSTERIOR_GEOMETRY_MEAN_LOSS.value: weighted_mean_loss,
                MetricKey.POSTERIOR_GEOMETRY_STD_LOSS.value: weighted_std_loss,
                MetricKey.POSTERIOR_GEOMETRY_MAX_STD_LOSS.value: weighted_max_std_loss,
                MetricKey.POSTERIOR_GEOMETRY_COVARIANCE_LOSS.value: weighted_covariance_loss,
            },
        )
