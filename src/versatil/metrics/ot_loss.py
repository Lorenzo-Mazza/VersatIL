"""Module for add-on loss functions that use computationally intensive libraries.

IMPORTANT: This module uses geomloss+pykeops which trigger slow JIT compilation.
To avoid this overhead, geomloss is imported lazily (only when OptimalTransportLoss
is instantiated).

Usage via Hydra config (recommended):
    loss:
        _target_: versatil.metrics.add-ons.OptimalTransportLoss
        action_keys: [position_action]
        weight: 0.1

This ensures the import only happens when the loss is actually configured.
"""
import torch

from versatil.metrics import BaseLoss, LossOutput, MetricKey


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
            from geomloss import SamplesLoss
        except ImportError as e:
            raise ImportError(
                "OptimalTransportLoss requires geomloss and pykeops. "
                "Install with: pip install geomloss pykeops"
            ) from e

        self.ot = SamplesLoss(
            loss="sinkhorn", p=p, blur=epsilon ** 1/p
        )

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
        total_predictions = torch.cat([predictions[k] for k in self.action_keys], dim=-1) # (B, horizon, action_total_dim)
        total_targets = torch.cat([targets[k] for k in self.action_keys], dim=-1) # (B, horizon, action_total_dim)
        batch_size, horizon, _ = total_predictions.shape
        time_embeddings = torch.linspace(0, 1, steps=horizon, device=total_predictions.device)
        time_embeddings = time_embeddings.view(1, horizon, 1).expand(batch_size, -1, -1)
        time_embeddings = time_embeddings * self.time_scale
        predictions_with_time = torch.cat([total_predictions, time_embeddings], dim=-1) # (B, horizon, action_total_dim + 1)
        targets_with_time = torch.cat([total_targets, time_embeddings], dim=-1) # (B, horizon, action_total_dim + 1)
        if is_pad is None:
            is_pad = torch.zeros((batch_size, horizon), dtype=torch.bool, device=total_predictions.device)
        weights = (~is_pad).float() # 1.0 for valid points, 0.0 for padded points
        weight_sums = weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        normalized_weights = weights / weight_sums
        # We need to pass (Weights_X, Samples_X, Weights_Y, Samples_Y) as args here because of GeomLoss API
        ot_loss = self.ot(normalized_weights, predictions_with_time, normalized_weights, targets_with_time).mean()
        return LossOutput(
            total_loss=self.weight * ot_loss,
            component_losses={MetricKey.OPTIMAL_TRANSPORT_LOSS.value: ot_loss},
        )
