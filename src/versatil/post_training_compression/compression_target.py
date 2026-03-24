"""Compression target defining what to apply to a policy submodule."""

from versatil.configs.post_training_compression import PreparationConfig
from versatil.post_training_compression.pruning.base import BasePruner
from versatil.quantization.strategies import (
    PT2EStrategy,
    QuantizeApiStrategy,
)


class CompressionTarget:
    """Defines preparation, pruning, and quantization for a PyTorch submodule.

    Validates quantization config invariants on construction.
    """

    def __init__(
        self,
        module_path: str,
        preparation: PreparationConfig | None = None,
        pruning: list[BasePruner] | None = None,
        quantization: PT2EStrategy | QuantizeApiStrategy | None = None,
    ) -> None:
        """Initialize and validate compression components.

        Args:
            module_path: Dotted path to the target submodule,
                or empty string for the full policy.
            preparation: BN replacement and fusion settings.
            pruning: Pruning strategies to apply sequentially.
            quantization: Quantization strategy or None to skip.

        Raises:
            ValueError: If quantize_() config uses static activation
                (only supported via PT2E).
        """
        self.module_path = module_path
        self.preparation = preparation
        self.pruning: list[BasePruner] = pruning or []
        self.quantization = quantization
        self._validate_quantization()

    def _validate_quantization(self) -> None:
        """Validate that quantize_() configs don't use static activation."""
        if not isinstance(self.quantization, QuantizeApiStrategy):
            return
        if hasattr(self.quantization.quantize_config, "act_quant_scale"):
            label = self.module_path or "(root)"
            raise ValueError(
                f"Module '{label}' uses a static activation quantize_() "
                f"config. Static quantization is only supported via PT2E. "
                f"Use PT2EStrategy or a dynamic/weight-only config."
            )
