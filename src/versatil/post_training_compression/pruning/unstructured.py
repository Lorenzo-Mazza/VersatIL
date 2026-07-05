"""Global L1 unstructured weight pruning."""

from torch import nn
from torch.nn.utils import prune

from versatil.post_training_compression.constants import (
    PrunableLayerType,
    PruningTargetAttribute,
)
from versatil.post_training_compression.pruning.base import BasePruner


class UnstructuredPruner(BasePruner):
    """Global L1 unstructured pruning across all target layers."""

    def __init__(
        self,
        amount: float,
        layer_types: list[str] | None = None,
    ) -> None:
        """Initialize with pruning amount and target layer types.

        Args:
            amount: Fraction of weights to zero, must be in (0, 1).
            layer_types: PrunableLayerType values to target. Defaults to
                convolution and linear layers (normalization scales and
                embedding tables are usually not good pruning targets).

        Raises:
            ValueError: If amount is not in the open interval (0, 1).
        """
        if amount <= 0.0 or amount >= 1.0:
            raise ValueError(f"Pruning amount must be in (0, 1), got {amount}")
        self._amount = amount
        if layer_types is None:
            layer_types = [
                PrunableLayerType.CONV1D.value,
                PrunableLayerType.CONV2D.value,
                PrunableLayerType.LINEAR.value,
            ]
        self._layer_types = tuple(
            PrunableLayerType(name).to_module_type() for name in layer_types
        )

    @property
    def amount(self) -> float:
        """Fraction of weights to prune."""
        return self._amount

    @property
    def layer_types(self) -> tuple[type[nn.Module], ...] | None:
        """Module types targeted for pruning. None means all."""
        return self._layer_types

    def prune(self, module: nn.Module) -> tuple[int, int]:
        """Apply global L1 unstructured pruning.

        Identifies all target layers, applies global L1 unstructured pruning,
        then removes the pruning reparametrization to make weights permanent.

        Args:
            module: Neural network module to prune.

        Returns:
            Tuple of (total_parameters, zero_parameters).
        """
        parameters_to_prune = [
            (child_module, PruningTargetAttribute.WEIGHT.value)
            for child_module in module.modules()
            if isinstance(
                getattr(child_module, PruningTargetAttribute.WEIGHT.value, None),
                nn.Parameter,
            )
            and isinstance(child_module, self._layer_types)
        ]
        if not parameters_to_prune:
            raise ValueError(
                "Unstructured pruning selected no modules; the target module "
                f"contains no {[t.__name__ for t in self._layer_types]} layers."
            )
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,  # type: ignore[arg-type]
            amount=self._amount,
        )
        for child_module, name in parameters_to_prune:
            prune.remove(child_module, name)
        return self.compute_sparsity(module)
