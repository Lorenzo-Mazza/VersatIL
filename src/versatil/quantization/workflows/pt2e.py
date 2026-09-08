"""PT2E quantization workflow."""

import copy
import logging

import torch
import torch.nn as nn
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchao.quantization.pt2e.quantizer.composable_quantizer import (
    ComposableQuantizer,
)

from versatil.models.exportable.base import ExportablePolicy
from versatil.models.exportable.metadata import PolicyExportMetadata
from versatil.post_training_compression.constants import (
    QuantizationWorkflow,
)
from versatil.post_training_compression.deployment_backends.base import (
    DeploymentBackend,
)
from versatil.post_training_compression.export import (
    build_example_inputs,
    export_policy,
)
from versatil.post_training_compression.policy_loading import load_float_policy_context
from versatil.quantization.calibration import (
    CalibrationDataProvider,
    build_calibration_data,
)
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
        deployment_backend: DeploymentBackend | None = None,
    ) -> QuantizedContext:
        """Export, prepare, optionally calibrate, and convert with PT2E.

        Args:
            context: Loaded float policy context.
            exportable: Policy wrapper exposing positional tensor inputs for
                ``torch.export``.
            calibration_steps: Maximum number of training batches used for
                static PT2E calibration.
            deployment_backend: Destination selected by compression, which checks
                compatibility with the configured PT2E quantizers.

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
        # Export always uses synthetic batch>=2 example inputs: a raw
        # training batch with batch_size=1 would trip torch.export's 0/1
        # specialization on the dynamic batch dimension. Calibration still
        # runs on real dataloader batches.
        example_inputs = build_example_inputs(
            exportable=exportable,
            observation_space=context.observation_space,
            observation_horizon=context.observation_horizon,
            tokenizer=context.tokenizer,
        )  # (batch, ...)
        exported = export_policy(exportable=exportable, example_inputs=example_inputs)
        # prepare_pt2e/convert_pt2e mutate the exported graph in place; keep a
        # genuinely float copy so the report compares against the pre-
        # quantization model instead of the mutated graph itself.
        float_exported = copy.deepcopy(exported)
        converted = self._convert_exported_model(
            exported=exported,
            targets=self.targets,
            calibration=calibration,
            example_inputs=example_inputs,
            observation_keys=exportable.observation_keys,
            export_metadata=exportable.export_metadata,
        )
        return QuantizedContext(
            float_model=float_exported,
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
            exportable: Policy wrapper identifying the required observation keys.
            targets: PT2E targets that determine whether calibration is needed.
            calibration_steps: Maximum number of calibration batches.

        Returns:
            Calibration provider for static targets, or ``None`` when all
            targets are dynamic.

        Raises:
            ValueError: If a static target requests fewer than one calibration batch.
        """
        needs_calibration = any(target.needs_calibration for target in targets)
        if not needs_calibration:
            return None
        if calibration_steps < 1:
            raise ValueError(
                "Static PT2E quantization requires calibration_steps >= 1, "
                f"got {calibration_steps}."
            )
        return build_calibration_data(
            context=context,
            observation_keys=exportable.observation_keys,
            num_calibration_steps=calibration_steps,
            device=torch.device("cpu"),
        )

    @staticmethod
    def _convert_exported_model(
        exported: nn.Module,
        targets: list[PT2EQuantizationModuleTarget],
        calibration: CalibrationDataProvider | None,
        example_inputs: tuple[torch.Tensor, ...],
        observation_keys: list[str],
        export_metadata: PolicyExportMetadata | None = None,
    ) -> nn.Module:
        """Apply PT2E prepare/calibrate/convert to an exported model.

        Note:
            A forward pass with the export inputs initializes weight observers for
            dynamic activation quantization, including weights shared across
            denoising steps. Static calibration appends the noise inputs described
            by the policy export metadata to each observation batch.
            ``B`` denotes batch size. Output dimensions depend on whether the
            exported policy produces action tensors or action-token IDs.

        Args:
            exported: Exported float graph module.
            targets: PT2E targets used to create backend quantizers.
            calibration: Calibration batches for static PT2E targets.
            example_inputs: Export inputs used to initialize weight observers
                when no calibration provider is needed.
            observation_keys: Observation keys in the exported graph's input order.
            export_metadata: Additional graph inputs appended to observations
                during static calibration. The default uses observation inputs only.

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
            logger.info(f"PT2E target: {target.label}")

        composed = ComposableQuantizer(quantizers)
        first_backend = targets[0].pt2e_backend
        with first_backend.environment_context():
            prepared = prepare_pt2e(exported, composed)
            if calibration is not None:
                logger.info("Calibrating PT2E...")
                export_metadata = export_metadata or PolicyExportMetadata()
                with torch.no_grad():
                    for observation in calibration:
                        inputs = export_metadata.prepare_inputs(
                            observations=tuple(
                                observation[key] for key in observation_keys
                            )
                        )  # (B, ...)
                        prepared(*inputs)  # (B, ...)
            else:
                logger.info("Initializing PT2E weight observers with export inputs.")
                with torch.no_grad():
                    prepared(*example_inputs)  # (B, ...)
            converted = convert_pt2e(prepared)
        static_op_count = str(converted.graph).count(
            FXNodePattern.QUANTIZE_PER_TENSOR.value
        )
        logger.info(f"PT2E done, static ops: {static_op_count}")
        return converted
