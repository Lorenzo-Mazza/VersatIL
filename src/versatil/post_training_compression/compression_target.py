"""Compression target defining what workflows (layer fusion, pruning, quantization) to apply to a policy submodule."""

from versatil.configs.post_training_compression import PreparationConfig
from versatil.post_training_compression.pruning.base import BasePruner
from versatil.quantization.workflows.base import BaseQuantizationWorkflow


class CompressionTarget:
    """Stores compression components for one PyTorch submodule."""

    def __init__(
        self,
        module_path: str,
        preparation: PreparationConfig | None = None,
        pruning: list[BasePruner] | None = None,
        quantization: BaseQuantizationWorkflow | None = None,
    ) -> None:
        """Initialize compression components.

        Args:
            module_path: Dotted path to the target submodule,
                or empty string for the full policy.
            preparation: BN replacement and fusion settings.
            pruning: Pruning strategies to apply sequentially.
            quantization: Quantization workflow. ``None`` means no
                quantization for this target.
        """
        self.module_path = module_path
        self.preparation = preparation
        self.pruning: list[BasePruner] = pruning or []
        self.quantization = quantization
