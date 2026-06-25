"""No-quantization workflow for float model export."""

import torch.nn as nn

from versatil.models.exportable_policy import ExportablePolicy
from versatil.post_training_compression.constants import QuantizationWorkflow
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
    ) -> PolicyContext:
        """Load the float policy checkpoint.

        Args:
            checkpoint_path: Directory containing the training checkpoint.
            checkpoint_name: Checkpoint filename to load from the directory.

        Returns:
            Float policy context for unquantized export.
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
        """Export the policy without modifying weights or graph quantization.

        Args:
            context: Loaded float policy context.
            exportable: Policy wrapper exposing positional tensor inputs for
                ``torch.export``.
            calibration_steps: Unused for unquantized export.

        Returns:
            Quantized context whose float and deployment models are the same
            exported graph.
        """
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
