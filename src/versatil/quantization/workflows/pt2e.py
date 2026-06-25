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
from versatil.post_training_compression.compression_target import CompressionTarget
from versatil.post_training_compression.constants import QuantizationWorkflow
from versatil.post_training_compression.export import (
    build_example_inputs,
    export_policy,
)
from versatil.post_training_compression.policy_loading import load_float_policy_context
from versatil.quantization.calibration import CalibrationDataProvider
from versatil.quantization.constants import FXNodePattern, QuantizationMode
from versatil.quantization.pt2e.backends.base import BasePT2EBackend
from versatil.quantization.workflows.base import (
    BaseQuantizationWorkflow,
    PolicyContext,
    QuantizedContext,
)

logger = logging.getLogger(__name__)


class PT2EQuantizationWorkflow(BaseQuantizationWorkflow):
    """PT2E graph quantization workflow."""

    def __init__(self, pt2e_backend: BasePT2EBackend) -> None:
        """Initialize with a PT2E backend.

        Args:
            pt2e_backend: Backend providing quantization config and
                environment context.
        """
        self.pt2e_backend = pt2e_backend
        if self.is_qat:
            raise NotImplementedError("PT2E QAT configuration is not supported yet.")

    @property
    def quantization_mode(self) -> str:
        """Return ``pt2e`` because this workflow quantizes an exported graph."""
        return QuantizationMode.PT2E.value

    @property
    def needs_calibration(self) -> bool:
        """Static PT2E requires calibration, dynamic does not."""
        return not self.pt2e_backend.is_dynamic

    @property
    def is_qat(self) -> bool:
        """Return whether the PT2E backend is configured for QAT."""
        return self.pt2e_backend.is_qat

    def load_policy_context(
        self,
        checkpoint_path: str,
        checkpoint_name: str,
    ) -> PolicyContext:
        """Load the policy checkpoint required by PT2E quantization."""
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
        """Export, prepare, optionally calibrate, and convert with PT2E."""
        pt2e_modules = [
            module
            for module in modules
            if isinstance(module.quantization, PT2EQuantizationWorkflow)
        ]
        calibration = self._build_calibration(
            context=context,
            exportable=exportable,
            pt2e_modules=pt2e_modules,
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
            pt2e_modules=pt2e_modules,
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
        pt2e_modules: list[CompressionTarget],
        calibration_steps: int,
    ) -> CalibrationDataProvider | None:
        """Build calibration data for static PT2E quantization."""
        needs_calibration = any(
            module.quantization.needs_calibration for module in pt2e_modules
        )
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
        pt2e_modules: list[CompressionTarget],
        calibration: CalibrationDataProvider | None,
    ) -> nn.Module:
        """Apply PT2E prepare/calibrate/convert to an exported model."""
        if not pt2e_modules:
            return exported
        needs_calibration = any(
            module.quantization.needs_calibration for module in pt2e_modules
        )
        if needs_calibration and calibration is None:
            raise ValueError(
                "PT2E static quantization requires calibration data "
                "but no CalibrationDataProvider was supplied."
            )
        quantizers = []
        for module in pt2e_modules:
            backend = module.quantization.pt2e_backend
            quantizers.append(backend.create_quantizer(module_path=module.module_path))
            logger.info("PT2E target: %s", module.module_path or "(root)")

        composed = ComposableQuantizer(quantizers)
        first_backend = pt2e_modules[0].quantization.pt2e_backend
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
