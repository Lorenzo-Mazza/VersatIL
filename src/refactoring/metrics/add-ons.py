"""Module for add-on loss functions that use computationally intensive libraries.

IMPORTANT: This module uses geomloss+pykeops which trigger slow JIT compilation.
To avoid this overhead, geomloss is imported lazily (only when OptimalTransportLoss
is instantiated).

Usage via Hydra config (recommended):
    loss:
        _target_: refactoring.metrics.add-ons.OptimalTransportLoss
        action_keys: [position_action]
        weight: 0.1

This ensures the import only happens when the loss is actually configured.
"""
import torch

from refactoring.metrics import BaseLoss, LossOutput, MetricKey


class OptimalTransportLoss(BaseLoss):
    """Computes Kantorovich Optimal Transport (K-OT) loss using Sinkhorn divergence.

    This loss regularizes action distributions by computing a differentiable OT cost
    between predicted and target actions.

    Note:
        This loss requires the optional dependencies geomloss and pykeops.
        Install with: pip install geomloss pykeops

        The geomloss library is imported lazily during __init__ to avoid
        triggering PyKeOps JIT compilation unless this loss is actually used.

    Limitations:
        - For action spaces with mixed data types (e.g., continuous like position/orientation deltas alongside binary like gripper states),
           the Euclidean metric may yield suboptimal results. Binary dimensions can be overshadowed by continuous ones, leading to under-penalized mismatches.
           Consider removing binary action dimensions when using this loss.
    """

    def __init__(
        self,
        action_keys: list[str],
        weight: float = 0.1,
        epsilon: float = 0.01,
    ):
        """Initializes the OptimalTransportLoss.

        Args:
            action_keys: List of keys for action tensors in predictions and targets.
            weight: Scaling factor for the total loss.
            epsilon: Regularization parameter for Sinkhorn (blur = sqrt(epsilon)).

        Raises:
            ImportError: If geomloss is not installed.
        """
        super().__init__()
        self.weight = weight
        self.action_keys = action_keys
        # Lazy import to avoid PyKeOps compilation overhead unless this loss is used
        try:
            from geomloss import SamplesLoss
        except ImportError as e:
            raise ImportError(
                "OptimalTransportLoss requires geomloss and pykeops. "
                "Install with: pip install geomloss pykeops"
            ) from e

        self.ot = SamplesLoss(loss="sinkhorn", p=2, blur=epsilon**0.5)  # Sinkhorn with p=2 (Euclid)

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
                raise ValueError(f"Predictions and targets must contain key '{action_key}' for Optimal Transport Loss.")

        # Flat actions (B*horizon, sum_dims)
        pred_a = torch.cat([predictions[k] for k in self.action_keys], dim=-1)
        target_a = torch.cat([targets[k] for k in self.action_keys], dim=-1)
        B, horizon, adim = pred_a.shape
        pred_a = pred_a.view(-1, adim)
        target_a = target_a.view(-1, adim)
        if is_pad is not None:
            flat_mask = ~is_pad.view(-1)
            pred_a = pred_a[flat_mask]
            target_a = target_a[flat_mask]


        # Geomloss Sinkhorn (i.e. Optimal Transport cost in Euclidean space).
        ot_loss = self.ot(pred_a, target_a)
        return LossOutput(
            total_loss=self.weight * ot_loss,
            component_losses={MetricKey.OPTIMAL_TRANSPORT_LOSS.value: ot_loss},
        )