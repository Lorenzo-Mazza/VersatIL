"""Shared quantization workflow types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn

from versatil.models.exportable.base import ExportablePolicy
from versatil.post_training_compression.deployment_backends.base import (
    DeploymentBackend,
)
from versatil.post_training_compression.policy_context import PolicyContext
from versatil.quantization.metadata import QuantizationTargetMetadata
from versatil.quantization.module_target import QuantizationModuleTarget


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
    ) -> PolicyContext:
        """Load the policy context required by this workflow.

        Args:
            checkpoint_path: Directory containing the training checkpoint.
            checkpoint_name: Checkpoint filename to load from the directory.

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

        Returns:
            Quantized context containing the float export, selected deployment
            model, example inputs, and serialized workflow name.
        """

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


def validate_quantization_targets(
    model: nn.Module,
    targets: list[QuantizationModuleTarget],
) -> None:
    """Validate quantization target paths and overlap.

    Args:
        model: Policy model whose submodule tree should contain every target.
        targets: Module targets configured for one quantization workflow.

    Raises:
        ValueError: If a target path does not exist or two targets overlap.
    """
    for target in targets:
        if target.module_path == "":
            continue
        try:
            model.get_submodule(target.module_path)
        except AttributeError as error:
            available = list(dict(model.named_children()).keys())
            raise ValueError(
                f"Quantization target '{target.module_path}' not found in model. "
                f"Available top-level modules: {available}."
            ) from error
    for index, target in enumerate(targets):
        for other in targets[index + 1 :]:
            if target.overlaps(other=other):
                raise ValueError(
                    "Quantization targets overlap: "
                    f"'{target.module_path}' and '{other.module_path}'."
                )


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
