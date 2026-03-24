"""Post-training compressor for a trained policy."""

import torch.nn as nn

from versatil.configs.post_training_compression import PreparationConfig
from versatil.post_training_compression.pruning.base import BasePruner
from versatil.quantization.strategies import (
    PT2EStrategy,
    QuantizeApiStrategy,
)
from versatil.training.constants import CheckpointFilename


class ModuleCompressor:
    """Compression scheme for a single policy submodule.

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


class PostTrainingCompressor:
    """Post-training global compressor for a trained policy.

    Holds per-module compressors and validates module
    paths against the loaded policy.
    """

    def __init__(
        self,
        checkpoint_path: str,
        modules: list[ModuleCompressor],
        preparation: PreparationConfig,
        device: str = "cpu",
        calibration_steps: int = 128,
        checkpoint_name: str = CheckpointFilename.DEFAULT_CHECKPOINT.value,
        output_directory: str | None = None,
        pruning: list[BasePruner] | None = None,
        quantization: PT2EStrategy | QuantizeApiStrategy | None = None,
    ) -> None:
        """Initialize the compression pipeline.

        Args:
            checkpoint_path: Path to the training checkpoint directory.
            modules: Per-module compression schemes (empty = global).
            preparation: Global preparation settings.
            device: Device for loading and compression.
            calibration_steps: Number of calibration batches for
                static quantization.
            checkpoint_name: Checkpoint filename inside the directory.
            output_directory: Where to save compressed output.
                Defaults to checkpoint_path/compressed.
            pruning: Global pruning strategies (inherited by modules).
            quantization: Global quantization strategy (inherited by
                modules).
        """
        self.checkpoint_path = checkpoint_path
        self.checkpoint_name = checkpoint_name
        self.output_directory = output_directory
        self.device = device
        self.calibration_steps = calibration_steps
        self.modules = modules
        self.preparation = preparation
        self.pruning: list[BasePruner] = pruning or []
        self.quantization = quantization

    def resolve_modules(self) -> list[ModuleCompressor]:
        """Return the compression targets for this run.

        Supports two configuration modes: per-module (explicit
        ``modules`` list targeting specific submodules) and global
        (``modules`` is empty, applying the top-level preparation,
        pruning, and quantization to the entire policy).

        Returns:
            Non-empty list of ModuleCompressor instances.
        """
        if self.modules:
            return self.modules
        return [
            ModuleCompressor(
                module_path="",
                preparation=self.preparation,
                pruning=self.pruning,
                quantization=self.quantization,
            )
        ]

    def validate(self, policy: nn.Module) -> None:
        """Validate module paths and strategy compatibility.

        Args:
            policy: The loaded policy model.

        Raises:
            ValueError: If a module_path doesn't match a submodule,
                or if PT2E and quantize_() strategies are both present.
        """
        has_pt2e = any(isinstance(m.quantization, PT2EStrategy) for m in self.modules)
        has_quantize_api = any(
            isinstance(m.quantization, QuantizeApiStrategy) for m in self.modules
        )
        if has_pt2e and has_quantize_api:
            raise ValueError(
                "PT2E and quantize_() strategies cannot be combined. "
                "PT2E operates on the exported FX graph while "
                "quantize_() requires eager nn.Module submodules. "
                "Use one strategy per compression run."
            )
        for module in self.modules:
            if module.module_path == "":
                continue
            try:
                policy.get_submodule(module.module_path)
            except AttributeError as error:
                available = list(dict(policy.named_children()).keys())
                raise ValueError(
                    f"Module path '{module.module_path}' not found in "
                    f"policy. Available top-level modules: {available}"
                ) from error
