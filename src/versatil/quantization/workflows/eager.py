"""Eager torchao quantization workflow."""

import logging
from dataclasses import dataclass

import torch.nn as nn
from torchao.quantization import quantize_
from torchao.quantization.granularity import PerGroup
from torchao.quantization.qat import QATConfig

from versatil.models.exportable_policy import ExportablePolicy
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
from versatil.quantization.module_target import EagerQuantizationModuleTarget
from versatil.quantization.workflows.base import (
    BaseQuantizationWorkflow,
    PolicyContext,
    QuantizedContext,
)

logger = logging.getLogger(__name__)


@dataclass
class _PreparedEagerTarget:
    """QAT target torch module state, captured after fake-quant preparation.

    Attributes:
        target: Eager quantization target whose config prepared the modules.
        module_names: Fully qualified module names selected during preparation.
            Conversion reuses this set so only prepared modules are converted.
    """

    target: EagerQuantizationModuleTarget
    module_names: set[str]


class EagerQuantizationWorkflow(BaseQuantizationWorkflow):
    """Eager torchao quantization workflow for PTQ and QAT."""

    def __init__(
        self,
        targets: list[EagerQuantizationModuleTarget],
        is_qat: bool = False,
        auto_filter_incompatible_linears: bool = True,
    ) -> None:
        """Initialize eager torchao quantization.

        Args:
            targets: Module-level eager quantization targets.
            is_qat: Whether this workflow is used for QAT checkpoint training
                and conversion.
            auto_filter_incompatible_linears: Whether to skip linears whose
                ``in_features`` are incompatible with the config group size.
        """
        if not targets:
            raise ValueError("EagerQuantizationWorkflow requires at least one target.")
        self._targets = targets
        self._is_qat = is_qat
        self.auto_filter_incompatible_linears = auto_filter_incompatible_linears
        self._prepared_targets: list[_PreparedEagerTarget] = []

    @property
    def targets(self) -> list[EagerQuantizationModuleTarget]:
        """Return eager quantization targets."""
        return self._targets

    @property
    def quantization_mode(self) -> str:
        """Return quantization mode name."""
        return QuantizationMode.EAGER.value

    @property
    def is_qat(self) -> bool:
        """Return whether this eager workflow handles QAT checkpoints."""
        return self._is_qat

    def load_policy_context(
        self,
        checkpoint_path: str,
        checkpoint_name: str,
    ) -> PolicyContext:
        """Load a float or QAT-prepared checkpoint.

        Args:
            checkpoint_path: Directory containing the training checkpoint.
            checkpoint_name: Checkpoint filename to load from the directory.

        Returns:
            Float policy context for PTQ, or QAT-prepared policy context for
            QAT conversion.
        """
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
        calibration_steps: int,
    ) -> QuantizedContext:
        """Apply eager quantization and export the policy.

        Args:
            context: Loaded policy context containing the eager policy to
                mutate.
            exportable: Export wrapper around the same policy.
            calibration_steps: Unused for eager quantization.

        Returns:
            Exported eager-quantized model and example inputs for deployment.
        """
        example_inputs = build_example_inputs(
            exportable=exportable,
            observation_space=context.observation_space,
            observation_horizon=context.observation_horizon,
            tokenizer=context.tokenizer,
        )
        # Export the float baseline before quantize_() mutates the policy.
        float_exported = export_policy(
            exportable=exportable, example_inputs=example_inputs
        )
        if self.is_qat:
            self.convert_model(model=context.policy)
        else:
            self._apply_ptq(model=context.policy)
        exported = export_policy(exportable=exportable, example_inputs=example_inputs)
        return QuantizedContext(
            float_model=float_exported,
            quantized_model=exported,
            example_inputs=example_inputs,
            quantization_workflow=QuantizationWorkflow.EAGER.value,
        )

    def prepare_model(self, model: nn.Module) -> None:
        """Apply fake quantization modules in-place before QAT training.

        Args:
            model: Policy model to prepare for QAT.

        Raises:
            ValueError: If the workflow is not a QAT workflow, if a target path
                is invalid, or if a target selects no eligible ``nn.Linear``
                modules.
        """
        if not self.is_qat:
            raise ValueError("prepare_model() requires is_qat=True.")
        self.validate_targets(model=model)
        self._prepared_targets = []
        for target in self.targets:
            selected, skipped = self._select_linear_modules(
                model=model,
                target=target,
            )
            if not selected:
                skipped_text = "; ".join(
                    f"{name}: {reason}" for name, reason in skipped
                )
                raise ValueError(
                    f"QAT target '{target.label}' selected zero eligible "
                    "nn.Linear modules. "
                    f"Skipped modules: {skipped_text or 'none'}."
                )
            selected_names = set(selected)
            self._prepared_targets.append(
                _PreparedEagerTarget(
                    target=target,
                    module_names=selected_names,
                )
            )
            for name, reason in skipped:
                logger.info(f"Skipping QAT module {name}: {reason}")
            logger.info(
                f"Preparing {len(selected_names)} nn.Linear modules in "
                f"{target.label} for QAT with "
                f"{type(target.quantize_config).__name__}."
            )
            quantize_(
                model=model,
                config=QATConfig(base_config=target.quantize_config, step="prepare"),
                filter_fn=lambda module, fqn, names=selected_names: fqn in names,
            )

    def convert_model(self, model: nn.Module) -> None:
        """Convert prepared fake-quant modules to quantized modules in-place.

        Args:
            model: Policy model previously prepared by ``prepare_model()``.

        Raises:
            ValueError: If the workflow is not a QAT workflow, or if
                ``prepare_model()`` has not captured prepared targets.
        """
        if not self.is_qat:
            raise ValueError("convert_model() requires is_qat=True.")
        if not self._prepared_targets:
            raise ValueError("QAT convert_model() requires prepare_model() first.")
        for prepared in self._prepared_targets:
            selected_names = set(prepared.module_names)
            quantize_(
                model=model,
                config=QATConfig(
                    base_config=prepared.target.quantize_config,
                    step="convert",
                ),
                filter_fn=lambda module, fqn, names=selected_names: fqn in names,
            )

    def _apply_ptq(self, model: nn.Module) -> None:
        """Apply eager PTQ to targeted modules.

        Linears whose ``in_features`` are incompatible with the config group
        size are skipped when ``auto_filter_incompatible_linears`` is enabled.

        Args:
            model: Policy model to quantize in-place.

        Raises:
            ValueError: If a target path is invalid or targets overlap.
        """
        self.validate_targets(model=model)
        for target in self.targets:
            selected, skipped = self._select_linear_modules(
                model=model,
                target=target,
            )
            for name, reason in skipped:
                logger.info(f"Skipping PTQ module {name}: {reason}")
            selected_names = set(selected)
            logger.info(f"quantize_() target: {target.label}")
            quantize_(
                model=model,
                config=target.quantize_config,
                filter_fn=lambda module, fqn, names=selected_names: fqn in names,
            )

    def _select_linear_modules(
        self,
        model: nn.Module,
        target: EagerQuantizationModuleTarget,
    ) -> tuple[list[str], list[tuple[str, str]]]:
        """Return selected and skipped linear module names.

        Args:
            model: Policy model whose modules should be inspected.
            target: Eager target that scopes the module selection.

        Returns:
            Two lists: selected fully qualified ``nn.Linear`` module names, and
            skipped module names paired with skip reasons.
        """
        group_size = self._weight_group_size(target=target)
        selected: list[str] = []
        skipped: list[tuple[str, str]] = []
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not target.contains_module(module_name=name):
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

    @staticmethod
    def _weight_group_size(target: EagerQuantizationModuleTarget) -> int | None:
        """Return the target config's weight group size when one is declared.

        Args:
            target: Eager target whose torchao config should be inspected.

        Returns:
            Integer group size for grouped weight quantization, otherwise
            ``None``.
        """
        group_size = getattr(target.quantize_config, "group_size", None)
        if isinstance(group_size, int):
            return group_size
        for attribute_name in ("weight_granularity", "granularity"):
            granularity = getattr(target.quantize_config, attribute_name, None)
            if isinstance(granularity, PerGroup):
                return granularity.group_size
        return None
