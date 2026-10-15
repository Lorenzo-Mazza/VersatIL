"""Eager torchao quantization workflow."""

import logging
from contextlib import nullcontext

import torch
import torch.nn as nn
from torchao.quantization import quantize_

from versatil.models.exportable.base import ExportablePolicy
from versatil.models.policy import Policy
from versatil.post_training_compression.constants import (
    CompressionDeviceType,
    QuantizationWorkflow,
)
from versatil.post_training_compression.deployment_backends.base import (
    DeploymentBackend,
)
from versatil.post_training_compression.export import (
    build_example_inputs,
    export_policy,
    move_exported_model,
)
from versatil.post_training_compression.policy_loading import (
    load_float_policy_context,
    load_qat_policy_context,
)
from versatil.quantization.calibration import (
    CalibrationDataProvider,
    build_calibration_data,
    calibrate_policy,
)
from versatil.quantization.constants import QuantizationMode
from versatil.quantization.metadata import (
    QuantizationTargetMetadata,
)
from versatil.quantization.module_target import EagerQuantizationModuleTarget
from versatil.quantization.resolution import (
    ResolvedQuantizationTarget,
    resolve_quantization_targets,
    validate_target_configuration,
)
from versatil.quantization.workflows.base import (
    BaseQuantizationWorkflow,
    PolicyContext,
    QuantizedContext,
)
from versatil.training.constants import PrecisionType

logger = logging.getLogger(__name__)


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
            auto_filter_incompatible_linears: Skip linear and embedding layers
                whose weight row widths conflict with the configured group size.

        Raises:
            ValueError: If the target list is empty.
        """
        if not targets:
            raise ValueError("EagerQuantizationWorkflow requires at least one target.")
        self._targets = targets
        self._is_qat = is_qat
        self.auto_filter_incompatible_linears = auto_filter_incompatible_linears
        self._prepared_targets: list[ResolvedQuantizationTarget] = []

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
        device: torch.device = torch.device(CompressionDeviceType.CPU.value),
    ) -> PolicyContext:
        """Load a float or QAT-prepared checkpoint.

        Args:
            checkpoint_path: Directory containing the training checkpoint.
            checkpoint_name: Checkpoint filename to load from the directory.
            device: Device for the restored policy and preprocessing state.

        Returns:
            Float policy context for PTQ, or QAT-prepared policy context for
            QAT conversion.
        """
        if self.is_qat:
            return load_qat_policy_context(
                checkpoint_path=checkpoint_path,
                checkpoint_name=checkpoint_name,
                quantization=self,
                device=device,
            )
        return load_float_policy_context(
            checkpoint_path=checkpoint_path,
            checkpoint_name=checkpoint_name,
            device=device,
        )

    def quantize(
        self,
        context: PolicyContext,
        exportable: ExportablePolicy,
        calibration_steps: int,
        deployment_backend: DeploymentBackend | None = None,
        deployment_device: torch.device | None = None,
    ) -> QuantizedContext:
        """Apply eager quantization and export the policy.

        Args:
            context: Loaded policy context containing the eager policy to
                mutate.
            exportable: Export wrapper around the same policy.
            calibration_steps: Maximum observation batches for schemas that
                require calibration. Direct PTQ and existing QAT conversion skip it.
            deployment_backend: Backend selected for the exported artifact. When
                provided, its representation and conversion-device requirements
                are checked before quantization preparation and conversion.
            deployment_device: Destination for final weight conversion and export.
                Calibration uses the loaded policy's device. None uses that device
                for conversion as well.

        Returns:
            Exported eager-quantized model and example inputs for deployment.

        Note:
            Floating export initializes dynamically created layers. PTQ then resolves
            the complete layer selection and validates every target before quantization.
        """
        source_device = self._resolve_deployment_device(
            device=context.policy.device, source_device=context.policy.device
        )
        device = self._resolve_deployment_device(
            device=deployment_device, source_device=source_device
        )
        execution_precision = context.precision or context.autocast_precision
        activation_dtype = (
            execution_precision.get_model_dtype()
            if execution_precision is not None
            else None
        )
        if self.is_qat:
            prepared_targets = self._prepared_targets
            self._validate_prepared_targets(
                model=context.policy,
                deployment_backend=deployment_backend,
                activation_dtype=activation_dtype,
                conversion_device=device,
            )
        calibration = self._build_calibration(
            context=context, calibration_steps=calibration_steps
        )
        example_inputs = build_example_inputs(
            exportable=exportable,
            observation_space=context.observation_space,
            observation_horizon=context.observation_horizon,
            tokenizer=context.tokenizer,
            device=context.policy.device,
        )  # (batch, ...)
        # Export the float baseline before quantize_() mutates the policy.
        float_exported = export_policy(
            exportable=exportable, example_inputs=example_inputs
        )
        if device != source_device:
            float_exported = move_exported_model(
                model=float_exported, example_inputs=example_inputs, device=device
            )
        if self.is_qat:
            self._move_for_conversion(
                model=context.policy, prepared_targets=prepared_targets, device=device
            )
            quantization_targets = self._convert_targets(
                model=context.policy, prepared_targets=prepared_targets
            )
            calibration_batches = 0
        else:
            prepared_targets = resolve_quantization_targets(
                model=context.policy,
                targets=self.targets,
                is_qat=self.is_qat,
                auto_filter_incompatible_linears=self.auto_filter_incompatible_linears,
                deployment_backend=deployment_backend,
                activation_dtype=activation_dtype,
                conversion_device=device,
            )
            calibration_batches, quantization_targets = self._execute_ptq(
                model=context.policy,
                prepared_targets=prepared_targets,
                calibration=calibration,
                conversion_device=device,
                autocast_precision=context.autocast_precision,
            )
        if device != source_device:
            example_inputs = tuple(
                tensor.to(device=device) for tensor in example_inputs
            )  # each (batch, ...)
        exported = export_policy(exportable=exportable, example_inputs=example_inputs)
        return QuantizedContext(
            float_model=float_exported,
            quantized_model=exported,
            example_inputs=example_inputs,
            quantization_workflow=QuantizationWorkflow.EAGER.value,
            calibration_batches=calibration_batches,
            quantization_targets=quantization_targets,
        )

    def _build_calibration(
        self, context: PolicyContext, calibration_steps: int
    ) -> CalibrationDataProvider | None:
        """Build representative observations when a quantization schema needs them.

        Args:
            context: Loaded policy and its training dataset configuration.
            calibration_steps: Maximum number of observation batches.

        Returns:
            Named observations on the policy device, or None when unused.

        Raises:
            ValueError: If a calibrated configuration is requested with fewer than one
                batch.
        """
        if self.is_qat or not any(
            target.schema.needs_calibration for target in self.targets
        ):
            return None
        if calibration_steps < 1:
            raise ValueError(
                "Calibrated module quantization requires calibration_steps >= 1, "
                f"got {calibration_steps}."
            )
        return build_calibration_data(
            context=context,
            observation_keys=context.policy.input_keys,
            num_calibration_steps=calibration_steps,
            device=context.policy.device,
        )

    def prepare_model(self, model: nn.Module) -> None:
        """Apply fake quantization modules in-place before QAT training.

        Args:
            model: Policy model to prepare for QAT.

        Raises:
            ValueError: If is_qat is false, a target path is invalid, or a target
                selects zero eligible layers.
        """
        if not self.is_qat:
            raise ValueError("prepare_model() requires is_qat=True.")
        prepared_targets = resolve_quantization_targets(
            model=model,
            targets=self.targets,
            is_qat=self.is_qat,
            auto_filter_incompatible_linears=self.auto_filter_incompatible_linears,
            for_conversion=False,
        )
        for prepared in prepared_targets:
            self._prepare_target(model=model, prepared=prepared)
        self._prepared_targets = prepared_targets

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
        self._validate_prepared_targets(model=model)
        self._convert_targets(model=model, prepared_targets=self._prepared_targets)

    def _validate_prepared_targets(
        self,
        model: nn.Module,
        deployment_backend: DeploymentBackend | None = None,
        activation_dtype: torch.dtype | None = None,
        conversion_device: torch.device | None = None,
    ) -> None:
        """Check conversion requirements for layers already prepared for QAT.

        Args:
            model: Model with its restored fake-quantization layers.
            deployment_backend: Backend selected for the exported artifact.
            activation_dtype: Configured floating-point input dtype for conversion.
            conversion_device: Destination for final weight conversion and packing.

        Raises:
            ValueError: If preparation is required or a conversion requirement fails.
        """
        if not self._prepared_targets:
            raise ValueError("QAT convert_model() requires prepare_model() first.")
        for prepared in self._prepared_targets:
            validate_target_configuration(
                model=model,
                target=prepared.target,
                module_names=prepared.module_names,
                deployment_backend=deployment_backend,
                activation_dtype=activation_dtype,
                conversion_device=conversion_device,
            )

    def _apply_ptq(
        self,
        model: nn.Module,
        activation_dtype: torch.dtype | None = None,
        calibration: CalibrationDataProvider | None = None,
        deployment_backend: DeploymentBackend | None = None,
    ) -> tuple[int, list[QuantizationTargetMetadata]]:
        """Convert selected policy weights before graph export.

        Args:
            model: Policy or module with floating-point weights. The weights must
                already be on the device and have the dtype required by the
                selected TorchAO configurations.
            activation_dtype: Floating-point dtype of inputs to quantized linear
                layers during inference. Providing it enables activation-dtype
                validation.
            calibration: Representative model observations for schemas that require
                them. Calibrated conversion requires a Policy in evaluation mode.
            deployment_backend: Backend selected for the exported artifact. Its
                representation requirements are checked when supplied.

        Returns:
            Consumed calibration batches and metadata for the converted targets.

        Raises:
            ValueError: If this is a QAT workflow, a module path is invalid, a
                target selects no eligible layers, or a compatibility check fails.

        Note:
            Conversion modifies the selected weights in place. Every target is
            checked before any weights are converted.
            Targets share one calibration pass through the prepared floating
            policy. Conversion starts after every schema's calibration checks pass.
            A failed calibration leaves observer modules installed; reload the
            floating checkpoint before retrying the compression run.
        """
        if self.is_qat:
            raise ValueError(
                "_apply_ptq() requires is_qat=False; use prepare_model() and convert_model() for QAT."
            )
        prepared_targets = resolve_quantization_targets(
            model=model,
            targets=self.targets,
            is_qat=self.is_qat,
            auto_filter_incompatible_linears=self.auto_filter_incompatible_linears,
            activation_dtype=activation_dtype,
            deployment_backend=deployment_backend,
        )
        return self._execute_ptq(
            model=model, prepared_targets=prepared_targets, calibration=calibration
        )

    def _execute_ptq(
        self,
        model: nn.Module,
        prepared_targets: list[ResolvedQuantizationTarget],
        calibration: CalibrationDataProvider | None,
        conversion_device: torch.device | None = None,
        autocast_precision: PrecisionType | None = None,
    ) -> tuple[int, list[QuantizationTargetMetadata]]:
        """Prepare, calibrate and convert the previously validated layer selection.

        Args:
            model: Model whose selected floating-point weights will be converted.
            prepared_targets: Resolved layer names and TorchAO configurations.
            calibration: Representative observations for calibrated schemas.
            conversion_device: Destination for final conversion and packing after
                calibration. None retains the model's current device.
            autocast_precision: Checkpoint precision used to execute mixed-dtype
                layers during calibration. None uses ordinary tensor dispatch.

        Returns:
            Consumed observation batches and successful target conversion metadata.

        Raises:
            ValueError: If calibration inputs, evaluation mode or collected layer
                statistics are required before conversion.
        """
        requires_calibration = any(
            prepared.target.schema.needs_calibration for prepared in prepared_targets
        )
        if requires_calibration:
            if calibration is None:
                raise ValueError(
                    "The selected quantization schemas require calibration observations."
                )
            if not isinstance(model, Policy) or model.training:
                raise ValueError(
                    "Calibrated module quantization requires a Policy in evaluation mode."
                )
        for prepared in prepared_targets:
            self._prepare_target(model=model, prepared=prepared)
        calibration_batches = 0
        if requires_calibration:
            with (
                autocast_precision.autocast(device_type=model.device.type)
                if autocast_precision is not None
                else nullcontext()
            ):
                calibration_batches = calibrate_policy(
                    policy=model, calibration=calibration
                )
        for prepared in prepared_targets:
            prepared.target.schema.validate_calibration(
                model=model, module_names=prepared.module_names
            )
        if conversion_device is not None:
            self._move_for_conversion(
                model=model,
                prepared_targets=prepared_targets,
                device=conversion_device,
            )
        return calibration_batches, self._convert_targets(
            model=model, prepared_targets=prepared_targets
        )

    @staticmethod
    def _move_for_conversion(
        model: nn.Module,
        prepared_targets: list[ResolvedQuantizationTarget],
        device: torch.device,
    ) -> None:
        """Move prepared weights and calibration state before runtime packing.

        Args:
            model: Policy with calibrated observers or restored fake-quant layers.
            prepared_targets: Resolved layer selection and its schemas.
            device: Destination for conversion and execution.
        """
        model.to(device=device)
        if isinstance(model, Policy):
            model.device = device
        for prepared in prepared_targets:
            prepared.target.schema.move_calibration_state(
                model=model, module_names=prepared.module_names, device=device
            )

    def _convert_targets(
        self, model: nn.Module, prepared_targets: list[ResolvedQuantizationTarget]
    ) -> list[QuantizationTargetMetadata]:
        """Convert the resolved layers and describe their weight representations.

        Args:
            model: Model whose selected layers are ready for TorchAO conversion.
            prepared_targets: Validated layer names and conversion configurations.

        Returns:
            Per-target settings, layer dimensions, skipped names and weight classes.
        """
        metadata = [
            prepared.target.build_metadata(
                model=model,
                module_names=prepared.module_names,
                skipped=prepared.skipped,
            )
            for prepared in prepared_targets
        ]
        for prepared in prepared_targets:
            logger.info(f"Converting quantization target: {prepared.target.label}")
            quantize_(
                model=model,
                config=prepared.conversion_config,
                filter_fn=lambda module, module_name, names=prepared.module_names: (
                    module_name in names
                ),
            )
        for prepared, target_metadata in zip(prepared_targets, metadata, strict=True):
            target_metadata.weight_representations = {
                name: type(model.get_submodule(name).weight).__qualname__
                for name in sorted(prepared.module_names)
            }
        return metadata

    @staticmethod
    def _prepare_target(model: nn.Module, prepared: ResolvedQuantizationTarget) -> None:
        """Install observer or fake-quant modules using the schema's configuration.

        Args:
            model: Model whose selected layers will be replaced in place.
            prepared: Resolved layer names and their preparation configuration.
        """
        if prepared.preparation_config is not None:
            quantize_(
                model=model,
                config=prepared.preparation_config,
                filter_fn=lambda module, module_name, names=prepared.module_names: (
                    module_name in names
                ),
            )
