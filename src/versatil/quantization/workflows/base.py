"""Shared quantization workflow types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn

from versatil.models.exportable.base import ExportablePolicy
from versatil.post_training_compression.constants import CompressionDeviceType
from versatil.post_training_compression.deployment_backends.base import (
    DeploymentBackend,
)
from versatil.post_training_compression.policy_context import PolicyContext
from versatil.quantization.metadata import QuantizationTargetMetadata
from versatil.quantization.module_target import QuantizationModuleTarget
from versatil.quantization.resolution import validate_quantization_targets


class BaseQuantizationWorkflow(ABC):
    """Base class for ordered quantization workflows."""

    @property
    @abstractmethod
    def quantization_mode(self) -> str:
        """Return the quantization path name: ``none``, ``pt2e`` or ``eager``."""

    @abstractmethod
    def load_policy_context(
        self,
        checkpoint_path: str,
        checkpoint_name: str,
        device: torch.device = torch.device(CompressionDeviceType.CPU.value),
    ) -> PolicyContext:
        """Load the policy context required by this workflow.

        Args:
            checkpoint_path: Directory containing the training checkpoint.
            checkpoint_name: Checkpoint filename to load from the directory.
            device: Device for the restored policy and preprocessing state.

        Returns:
            Policy context with the loaded policy, metadata, normalizer, and
            optional tokenizer.
        """

    @abstractmethod
    def quantize(
        self,
        context: PolicyContext,
        exportable: ExportablePolicy,
        calibration_steps: int,
        deployment_backend: DeploymentBackend | None = None,
        deployment_device: torch.device | None = None,
    ) -> "QuantizedContext":
        """Run the workflow and return exported deployment inputs.

        Args:
            context: Loaded policy context used by the workflow.
            exportable: Policy wrapper exposing positional tensor inputs for
                ``torch.export``.
            calibration_steps: Maximum calibration batches for workflows that
                require calibration.
            deployment_backend: Destination used to validate backend-specific
                quantization requirements. None leaves those checks to the
                deployment caller.
            deployment_device: Device for final conversion, export and execution.
                None uses the loaded policy's device.

        Returns:
            Quantized context containing the float export, selected deployment
            model, example inputs, and serialized workflow name.
        """

    def validate_deployment_device(self, device: torch.device) -> None:
        """Check workflow-specific requirements for the artifact's execution device.

        Args:
            device: Device on which the deployed graph will execute.

        Note:
            Deployment backends and eager schemas apply their own execution and
            packing requirements. PT2E workflows also check their quantizer target.
        """
        return None

    @staticmethod
    def _resolve_deployment_device(
        device: torch.device | None, source_device: torch.device
    ) -> torch.device:
        """Resolve the destination to a concrete device before comparing placements.

        Args:
            device: Requested deployment device, or None to retain the source.
            source_device: Device of the loaded policy's tensors.

        Returns:
            Destination with an explicit CUDA index when CUDA is requested.

        Note:
            Resolving the current CUDA device avoids transferring graphs between
            equivalent ``cuda`` and ``cuda:0`` device names.
        """
        if device is None:
            return source_device
        if device.type == CompressionDeviceType.CUDA.value and device.index is None:
            return torch.device(
                CompressionDeviceType.CUDA.value, torch.cuda.current_device()
            )
        return device

    @property
    def is_qat(self) -> bool:
        """Return whether the workflow expects QAT-trained weights."""
        return False

    def prepare_model(self, model: nn.Module) -> None:
        """Prepare fake-quant modules before loading or training QAT weights.

        Args:
            model: Eager policy model to mutate in-place.

        Raises:
            NotImplementedError: If the workflow does not implement QAT
                preparation.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support QAT preparation."
        )

    @property
    def targets(self) -> list[QuantizationModuleTarget]:
        """Return module-level quantization targets."""
        return []

    @property
    def pt2e_backend_names(self) -> tuple[str, ...]:
        """Return serialized PT2E backend names used by this workflow."""
        return ()

    def validate_targets(self, model: nn.Module) -> None:
        """Validate this workflow's target module paths and overlaps.

        Args:
            model: Policy model whose submodule tree should contain every
                target.

        Raises:
            ValueError: If a target path does not exist or two targets overlap.
        """
        validate_quantization_targets(model=model, targets=self.targets)


@dataclass
class QuantizedContext:
    """Quantized export result plus inputs and metadata needed for deployment.

    Note:
        ``float_model`` captures the policy before conversion. PTQ uses floating
        weights; QAT retains the restored fake-quantization operations in this
        reference graph.
        ``calibration_batches`` counts representative batches when the workflow
        reports that count. None means the workflow does not report it.
        ``quantization_targets`` records the actual module selection and converted
        weight types as typed metadata supplied by the workflow.
    """

    float_model: nn.Module
    quantized_model: nn.Module
    example_inputs: tuple[torch.Tensor, ...]
    quantization_workflow: str
    calibration_batches: int | None = None
    quantization_targets: list[QuantizationTargetMetadata] | None = None
