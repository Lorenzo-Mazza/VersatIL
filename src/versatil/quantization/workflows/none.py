"""No-quantization workflow for float model export."""

import torch
import torch.nn as nn

from versatil.models.exportable.base import ExportablePolicy
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
)
from versatil.post_training_compression.policy_loading import load_float_policy_context
from versatil.quantization.constants import QuantizationMode
from versatil.quantization.workflows.base import (
    BaseQuantizationWorkflow,
    PolicyContext,
    QuantizedContext,
)


class NoQuantizationWorkflow(BaseQuantizationWorkflow):
    """Export the float policy without applying quantization."""

    @property
    def quantization_mode(self) -> str:
        """Return ``none`` because this workflow skips quantization."""
        return QuantizationMode.NONE.value

    @property
    def quantization_workflow(self) -> str:
        """Return the serialized no-quantization workflow value."""
        return QuantizationWorkflow.NONE.value

    def prepare_model(self, model: nn.Module) -> None:
        """Leave the model unchanged before training.

        Args:
            model: Policy model passed by the training workspace.
        """
        return None

    def load_policy_context(
        self,
        checkpoint_path: str,
        checkpoint_name: str,
        device: torch.device = torch.device(CompressionDeviceType.CPU.value),
    ) -> PolicyContext:
        """Load the float policy checkpoint.

        Args:
            checkpoint_path: Directory containing the training checkpoint.
            checkpoint_name: Checkpoint filename to load from the directory.
            device: Device for the restored policy and preprocessing state.

        Returns:
            Float policy context for unquantized export.
        """
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
        """Export the policy without modifying weights or graph quantization.

        Args:
            context: Loaded float policy context.
            exportable: Policy wrapper exposing positional tensor inputs for
                ``torch.export``.
            calibration_steps: Unused for unquantized export.
            deployment_backend: Destination selected for the floating-point artifact.
            deployment_device: Execution device for the exported artifact.
                None uses the loaded policy's device.

        Returns:
            Quantized context whose float and deployment models are the same
            exported graph.
        """
        source_device = self._resolve_deployment_device(
            device=context.policy.device, source_device=context.policy.device
        )
        device = self._resolve_deployment_device(
            device=deployment_device, source_device=source_device
        )
        if device != source_device:
            context.policy.to(device=device)
            context.policy.device = device
        example_inputs = build_example_inputs(
            exportable=exportable,
            observation_space=context.observation_space,
            observation_horizon=context.observation_horizon,
            tokenizer=context.tokenizer,
            device=device,
        )
        exported = export_policy(exportable=exportable, example_inputs=example_inputs)
        return QuantizedContext(
            float_model=exported,
            quantized_model=exported,
            example_inputs=example_inputs,
            quantization_workflow=self.quantization_workflow,
        )
