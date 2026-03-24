"""Per-layer structured weight pruning using Lp-norm channel ranking."""

from torch import nn
from torch.nn.utils import prune

from versatil.post_training_compression.constants import (
    PrunableLayerType,
    PruningTargetAttribute,
)
from versatil.post_training_compression.pruning.base import BasePruner


class StructuredPruner(BasePruner):
    """Per-layer structured pruning along a specified dimension.

    Ranks channels by their Lp-norm magnitude and zeros the lowest-ranked
    fraction. The ``norm_order`` parameter specifies p in the Lp-norm
    (e.g., 1 for L1-norm, 2 for L2-norm).
    """

    def __init__(
        self,
        amount: float,
        norm_order: int = 2,
        dimension: int = 0,
        layer_types: list[str] | None = None,
    ) -> None:
        """Initialize with pruning parameters.

        Args:
            amount: Fraction of channels to zero per layer, must be in (0, 1).
            norm_order: The p in Lp-norm used to rank channels
                (e.g., 1 for L1, 2 for L2).
            dimension: Weight tensor dimension along which to prune.
            layer_types: PrunableLayerType values to target. Defaults
                to Conv1d and Conv2d.

        Raises:
            ValueError: If amount is not in the open interval (0, 1).
        """
        if amount <= 0.0 or amount >= 1.0:
            raise ValueError(f"Pruning amount must be in (0, 1), got {amount}")
        if layer_types is None:
            layer_types = [
                PrunableLayerType.CONV1D.value,
                PrunableLayerType.CONV2D.value,
                PrunableLayerType.LINEAR.value,
            ]
        self._amount = amount
        self._norm_order = norm_order
        self._dimension = dimension
        self._layer_types = tuple(
            PrunableLayerType(name).to_module_type() for name in layer_types
        )

    @property
    def amount(self) -> float:
        """Fraction of channels to prune per layer."""
        return self._amount

    @property
    def norm_order(self) -> int:
        """Ln norm order for channel ranking."""
        return self._norm_order

    @property
    def dimension(self) -> int:
        """Weight tensor dimension along which pruning is applied."""
        return self._dimension

    @property
    def layer_types(self) -> tuple[type[nn.Module], ...]:
        """Module types targeted for pruning."""
        return self._layer_types

    def prune(self, module: nn.Module) -> tuple[int, int]:
        """Apply per-layer Ln structured pruning.

        Iterates over all target layers and applies Ln structured pruning
        individually, then removes the pruning reparametrization.

        Args:
            module: Neural network module to prune.

        Returns:
            Tuple of (total_parameters, zero_parameters).
        """
        for child_module in module.modules():
            if isinstance(child_module, self._layer_types):
                prune.ln_structured(
                    child_module,
                    name=PruningTargetAttribute.WEIGHT.value,
                    amount=self._amount,
                    n=self._norm_order,
                    dim=self._dimension,
                )
                prune.remove(child_module, PruningTargetAttribute.WEIGHT.value)
        return self.compute_sparsity(module)
