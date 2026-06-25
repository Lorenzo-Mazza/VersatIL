"""Eager torchao quantization workflow."""

import logging

import torch.nn as nn
from torchao.quantization import quantize_
from torchao.quantization.granularity import PerGroup
from torchao.quantization.qat import QATConfig
from torchao.quantization.quant_api import AOBaseConfig

from versatil.models.exportable_policy import ExportablePolicy
from versatil.post_training_compression.compression_target import CompressionTarget
from versatil.post_training_compression.constants import QuantizationWorkflow
from versatil.post_training_compression.export import (
    build_example_inputs,
    export_policy,
)
from versatil.post_training_compression.policy_loading import (
    load_float_policy_context,
    load_qat_policy_context,
)
from versatil.quantization.constants import QuantizationMode
from versatil.quantization.workflows.base import (
    BaseQuantizationWorkflow,
    PolicyContext,
    QuantizedContext,
)

logger = logging.getLogger(__name__)


class EagerQuantizationWorkflow(BaseQuantizationWorkflow):
    """Eager torchao quantization workflow for PTQ and QAT."""

    def __init__(
        self,
        quantize_config: AOBaseConfig,
        is_qat: bool = False,
        module_paths: list[str] | None = None,
        auto_filter_incompatible_linears: bool = True,
    ) -> None:
        """Initialize eager torchao quantization.

        Args:
            quantize_config: torchao quantization config. For QAT this is
                wrapped in ``QATConfig`` during prepare and convert.
            is_qat: Whether this workflow is used for QAT checkpoint training
                and conversion.
            module_paths: Dotted module paths to scope QAT. Empty means all
                eligible linear modules in the model.
            auto_filter_incompatible_linears: Whether to skip linears whose
                ``in_features`` are incompatible with the config group size.
        """
        self.quantize_config = quantize_config
        self._is_qat = is_qat
        self.module_paths = module_paths or []
        self.auto_filter_incompatible_linears = auto_filter_incompatible_linears
        self._prepared_module_names: set[str] = set()

    @property
    def quantization_mode(self) -> str:
        """Return ``eager`` because this workflow mutates modules before export."""
        return QuantizationMode.EAGER.value

    @property
    def quantization_workflow(self) -> str:
        """Return the serialized quantization workflow value."""
        return QuantizationWorkflow.EAGER.value

    @property
    def is_qat(self) -> bool:
        """Return whether this eager workflow handles QAT checkpoints."""
        return self._is_qat

    @property
    def base_config(self) -> AOBaseConfig:
        """Compatibility alias for QAT configs."""
        return self.quantize_config

    def load_policy_context(
        self,
        checkpoint_path: str,
        checkpoint_name: str,
    ) -> PolicyContext:
        """Load a float or QAT-prepared checkpoint."""
        if self.is_qat:
            return load_qat_policy_context(
                checkpoint_path=checkpoint_path,
                checkpoint_name=checkpoint_name,
                quantization=self,
            )
        return load_float_policy_context(
            checkpoint_path=checkpoint_path,
            checkpoint_name=checkpoint_name,
        )

    def quantize(
        self,
        context: PolicyContext,
        exportable: ExportablePolicy,
        modules: list[CompressionTarget],
        calibration_steps: int,
    ) -> QuantizedContext:
        """Apply eager quantization and export the policy."""
        if self.is_qat:
            self.convert_model(model=context.policy)
        else:
            self._apply_ptq(model=context.policy, modules=modules)

        example_inputs = build_example_inputs(
            exportable=exportable,
            observation_space=context.observation_space,
            observation_horizon=context.observation_horizon,
            tokenizer=context.tokenizer,
        )
        exported = export_policy(exportable=exportable, example_inputs=example_inputs)
        return QuantizedContext(
            float_model=exported,
            quantized_model=exported,
            example_inputs=example_inputs,
            quantization_workflow=self.quantization_workflow,
        )

    def prepare_model(self, model: nn.Module) -> None:
        """Apply fake quantization modules in-place before QAT training."""
        if not self.is_qat:
            raise ValueError("prepare_model() requires is_qat=True.")
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
            type(self.quantize_config).__name__,
        )
        quantize_(
            model=model,
            config=QATConfig(base_config=self.quantize_config, step="prepare"),
            filter_fn=lambda module, fqn: fqn in selected_names,
        )

    def convert_model(self, model: nn.Module) -> None:
        """Convert prepared fake-quant modules to quantized modules in-place."""
        if not self.is_qat:
            raise ValueError("convert_model() requires is_qat=True.")
        if not self._prepared_module_names:
            raise ValueError("QAT convert_model() requires prepare_model() first.")
        selected_names = set(self._prepared_module_names)
        quantize_(
            model=model,
            config=QATConfig(base_config=self.quantize_config, step="convert"),
            filter_fn=lambda module, fqn: fqn in selected_names,
        )

    @staticmethod
    def _apply_ptq(model: nn.Module, modules: list[CompressionTarget]) -> None:
        """Apply eager PTQ to targeted modules."""
        eager_modules = [
            module
            for module in modules
            if isinstance(module.quantization, EagerQuantizationWorkflow)
            and not module.quantization.is_qat
        ]
        for module in eager_modules:
            module_path = module.module_path
            label = module_path or "(root)"
            logger.info("quantize_() target: %s", label)

            if module_path == "":
                quantize_(model, module.quantization.quantize_config)
            else:
                quantize_(
                    model,
                    module.quantization.quantize_config,
                    filter_fn=lambda mod, fqn, mp=module_path: (
                        (fqn == mp or fqn.startswith(mp + "."))
                        and isinstance(mod, nn.Linear)
                    ),
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
        """Return the config's weight group size when one is declared."""
        group_size = getattr(self.quantize_config, "group_size", None)
        if isinstance(group_size, int):
            return group_size
        for attribute_name in ("weight_granularity", "granularity"):
            granularity = getattr(self.quantize_config, attribute_name, None)
            if isinstance(granularity, PerGroup):
                return granularity.group_size
        return None
