"""Quantization config strategies for torchao integration.

`PT2EStrategy`: graph-level quantization with operator fusion via torch.export.
`QuantizeApiStrategy`: eager mode quantization via `torchao` `quantize_()`.
`QATStrategy`: training-time fake quantization via torchao `QATConfig`.
"""

import logging

import torch.nn as nn
from torchao.quantization import quantize_
from torchao.quantization.granularity import PerGroup
from torchao.quantization.qat import QATConfig
from torchao.quantization.quant_api import AOBaseConfig

from versatil.quantization.backends.base import BasePT2EBackend

logger = logging.getLogger(__name__)


class PT2EStrategy:
    """PT2E quantization strategy. Graph-level with operator fusion."""

    def __init__(self, pt2e_backend: BasePT2EBackend) -> None:
        """Initialize with a PT2E backend.

        Args:
            pt2e_backend: Backend providing quantization config and
                environment context.
        """
        self.pt2e_backend = pt2e_backend

    @property
    def needs_calibration(self) -> bool:
        """Static PT2E requires calibration, dynamic does not."""
        return not self.pt2e_backend.is_dynamic


class QuantizeApiStrategy:
    """quantize_() API strategy. Eager mode, no operator fusion."""

    def __init__(self, quantize_config: AOBaseConfig) -> None:
        """Initialize with a torchao AOBaseConfig instance.

        Args:
            quantize_config: torchao quantization config.
        """
        self.quantize_config = quantize_config


class QATStrategy:
    """Quantization-aware training strategy using torchao QATConfig.

    Note: cf. https://docs.pytorch.org/ao/stable/api_reference/api_ref_quantization.html
     for available base quantization configs.
    """

    def __init__(
        self,
        base_config: AOBaseConfig,
        module_paths: list[str] | None = None,
        auto_filter_incompatible_linears: bool = True,
    ) -> None:
        """Initialize QAT strategy.

        Args:
            base_config: torchao PTQ base config passed to ``QATConfig``.
            module_paths: Dotted module paths to scope QAT. Empty means all
                eligible linear modules in the model.
            auto_filter_incompatible_linears: Whether to skip linears whose
                ``in_features`` are incompatible with the config group size.
        """
        self.base_config = base_config
        self.module_paths = module_paths or []
        self.auto_filter_incompatible_linears = auto_filter_incompatible_linears
        self._prepared_module_names: set[str] = set()

    def prepare_model(self, model: nn.Module) -> None:
        """Apply fake quantization modules in-place before QAT training.

        Args:
            model: Model to prepare.

        Raises:
            ValueError: If configured module paths are invalid or no eligible
                linear modules are selected.
        """
        selected, skipped = self._select_linear_modules(model=model)
        if not selected:
            skipped_text = "; ".join(f"{name}: {reason}" for name, reason in skipped)
            raise ValueError(
                "QAT selected zero eligible nn.Linear modules. "
                f"Skipped modules: {skipped_text or 'none'}."
            )
        selected_names = set(selected)
        self._prepared_module_names = selected_names
        for name, reason in skipped:
            logger.info("Skipping QAT module %s: %s", name, reason)
        logger.info(
            "Preparing %d nn.Linear modules for QAT with %s.",
            len(selected_names),
            type(self.base_config).__name__,
        )
        quantize_(
            model=model,
            config=QATConfig(base_config=self.base_config, step="prepare"),
            filter_fn=lambda module, fqn: fqn in selected_names,
        )

    def convert_model(self, model: nn.Module) -> None:
        """Convert prepared fake-quant modules to quantized modules in-place.

        Args:
            model: Prepared model to convert.

        Raises:
            ValueError: If ``prepare_model`` has not selected any modules.
        """
        if not self._prepared_module_names:
            raise ValueError("QAT convert_model() requires prepare_model() first.")
        selected_names = set(self._prepared_module_names)
        quantize_(
            model=model,
            config=QATConfig(base_config=self.base_config, step="convert"),
            filter_fn=lambda module, fqn: fqn in selected_names,
        )

    def _select_linear_modules(
        self,
        model: nn.Module,
    ) -> tuple[list[str], list[tuple[str, str]]]:
        """Return selected and skipped linear module names."""
        self._validate_module_paths(model=model)
        group_size = self._weight_group_size()
        selected: list[str] = []
        skipped: list[tuple[str, str]] = []
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not self._is_in_scope(module_name=name):
                continue
            if (
                self.auto_filter_incompatible_linears
                and group_size is not None
                and module.in_features % group_size != 0
            ):
                skipped.append(
                    (
                        name,
                        f"in_features {module.in_features} is not divisible by "
                        f"group_size {group_size}",
                    )
                )
                continue
            selected.append(name)
        return selected, skipped

    def _validate_module_paths(self, model: nn.Module) -> None:
        """Validate configured module paths exist on the model."""
        for module_path in self.module_paths:
            if module_path == "":
                continue
            try:
                model.get_submodule(module_path)
            except AttributeError as error:
                available = list(dict(model.named_children()).keys())
                raise ValueError(
                    f"QAT module path '{module_path}' not found in model. "
                    f"Available top-level modules: {available}."
                ) from error

    def _is_in_scope(self, module_name: str) -> bool:
        """Return whether a module name is inside configured QAT scopes."""
        if not self.module_paths:
            return True
        return any(
            module_name == module_path or module_name.startswith(module_path + ".")
            for module_path in self.module_paths
        )

    def _weight_group_size(self) -> int | None:
        """Return the base config's weight group size when one is declared."""
        group_size = getattr(self.base_config, "group_size", None)
        if isinstance(group_size, int):
            return group_size
        for attribute_name in ("weight_granularity", "granularity"):
            granularity = getattr(self.base_config, attribute_name, None)
            if isinstance(granularity, PerGroup):
                return granularity.group_size
        return None
