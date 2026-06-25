"""PT2E quantization workflow."""

import logging

import torch
import torch.nn as nn
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchao.quantization.pt2e.quantizer.composable_quantizer import (
    ComposableQuantizer,
)

from versatil.data.dataloader import get_dataloaders
from versatil.models.exportable_policy import ExportablePolicy
from versatil.post_training_compression.constants import QuantizationWorkflow
from versatil.post_training_compression.export import (
    build_example_inputs,
    export_policy,
)
from versatil.post_training_compression.policy_loading import load_float_policy_context
from versatil.quantization.calibration import CalibrationDataProvider
from versatil.quantization.constants import FXNodePattern, QuantizationMode
from versatil.quantization.module_target import PT2EQuantizationModuleTarget
from versatil.quantization.pt2e.backends.base import BasePT2EBackend
from versatil.quantization.workflows.base import (
    BaseQuantizationWorkflow,
    PolicyContext,
    QuantizedContext,
)

logger = logging.getLogger(__name__)


class PT2EQuantizationWorkflow(BaseQuantizationWorkflow):
    """PT2E graph quantization workflow."""

    def __init__(self, targets: list[PT2EQuantizationModuleTarget]) -> None:
        """Initialize with PT2E module targets.

        Args:
            targets: module-level PT2E quantization targets.
        """
        if not targets:
            raise ValueError("PT2EQuantizationWorkflow requires at least one target.")
        self._targets = targets
        if self.is_qat:
            raise NotImplementedError("PT2E QAT configuration is not supported yet.")

    @property
    def targets(self) -> list[PT2EQuantizationModuleTarget]:
        """Return PT2E quantization targets."""
        return self._targets

    @property
    def pt2e_backend(self) -> BasePT2EBackend:
        """Return the first target backend for runtime environment setup."""
        return self.targets[0].pt2e_backend

    @property
    def pt2e_backend_names(self) -> tuple[str, ...]:
        """Return serialized PT2E backend names used by all targets."""
        return tuple(target.pt2e_backend.name for target in self.targets)

    @property
    def quantization_mode(self) -> str:
        """Return ``pt2e`` because this workflow quantizes an exported graph."""
        return QuantizationMode.PT2E.value

    @property
    def needs_calibration(self) -> bool:
        """Static PT2E requires calibration, dynamic does not."""
        return any(target.needs_calibration for target in self.targets)

    @property
    def is_qat(self) -> bool:
        """Return whether the PT2E backend is configured for QAT."""
        return any(target.pt2e_backend.is_qat for target in self.targets)

    def load_policy_context(
        self,
        checkpoint_path: str,
        checkpoint_name: str,
    ) -> PolicyContext:
        """Load the policy checkpoint required by PT2E quantization.

        Args:
            checkpoint_path: Directory containing the training checkpoint.
            checkpoint_name: Checkpoint filename to load from the directory.

        Returns:
            Float policy context used for PT2E export, preparation,
            calibration, and conversion.
        """
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
        """Export, prepare, optionally calibrate, and convert with PT2E.

        Args:
            context: Loaded float policy context.
            exportable: Policy wrapper exposing positional tensor inputs for
                ``torch.export``.
            calibration_steps: Maximum number of training batches used for
                static PT2E calibration.

        Returns:
            Float exported model, PT2E-converted model, and example inputs.

        Raises:
            ValueError: If a target path is invalid or targets overlap.
        """
        self.validate_targets(model=context.policy)
        calibration = self._build_calibration(
            context=context,
            exportable=exportable,
            targets=self.targets,
            calibration_steps=calibration_steps,
        )
        example_inputs = (
            calibration.get_single_batch()
            if calibration is not None
            else build_example_inputs(
                exportable=exportable,
                observation_space=context.observation_space,
                observation_horizon=context.observation_horizon,
                tokenizer=context.tokenizer,
            )
        )
        exported = export_policy(exportable=exportable, example_inputs=example_inputs)
        converted = self._convert_exported_model(
            exported=exported,
            targets=self.targets,
            calibration=calibration,
        )
        return QuantizedContext(
            float_model=exported,
            quantized_model=converted,
            example_inputs=example_inputs,
            quantization_workflow=QuantizationWorkflow.PT2E.value,
        )

    @staticmethod
    def _build_calibration(
        context: PolicyContext,
        exportable: ExportablePolicy,
        targets: list[PT2EQuantizationModuleTarget],
        calibration_steps: int,
    ) -> CalibrationDataProvider | None:
        """Build calibration data for static PT2E quantization.

        Args:
            context: Loaded policy context containing the training dataloader
                config.
            exportable: Policy wrapper whose observation key order determines
                calibration batch layout.
            targets: PT2E targets that determine whether calibration is needed.
            calibration_steps: Maximum number of calibration batches.

        Returns:
            Calibration provider for static targets, or ``None`` when all
            targets are dynamic.
        """
        needs_calibration = any(target.needs_calibration for target in targets)
        if not needs_calibration:
            return None
        train_loader, _, _, _, _ = get_dataloaders(config=context.config)
        return CalibrationDataProvider(
            dataloader=train_loader,
            observation_keys=exportable.observation_keys,
            num_calibration_steps=calibration_steps,
        )

    @staticmethod
    def _convert_exported_model(
        exported: nn.Module,
        targets: list[PT2EQuantizationModuleTarget],
        calibration: CalibrationDataProvider | None,
    ) -> nn.Module:
        """Apply PT2E prepare/calibrate/convert to an exported model.

        Args:
            exported: Exported float graph module.
            targets: PT2E targets used to create backend quantizers.
            calibration: Calibration batches for static PT2E targets.

        Returns:
            Converted PT2E graph module.

        Raises:
            ValueError: If any target needs calibration but no calibration
                provider was supplied.
        """
        if not targets:
            return exported
        needs_calibration = any(target.needs_calibration for target in targets)
        if needs_calibration and calibration is None:
            raise ValueError(
                "PT2E static quantization requires calibration data "
                "but no CalibrationDataProvider was supplied."
            )
        quantizers = []
        for target in targets:
            backend = target.pt2e_backend
            quantizers.append(backend.create_quantizer(module_path=target.module_path))
            logger.info("PT2E target: %s", target.label)

        composed = ComposableQuantizer(quantizers)
        first_backend = targets[0].pt2e_backend
        with first_backend.environment_context():
            prepared = prepare_pt2e(exported, composed)
            if calibration is not None:
                logger.info("Calibrating PT2E...")
                with torch.no_grad():
                    for batch in calibration:
                        prepared(*batch)
            converted = convert_pt2e(prepared)
        logger.info(
            "PT2E done, static ops: %d",
            str(converted.graph).count(FXNodePattern.QUANTIZE_PER_TENSOR.value),
        )
        return converted
