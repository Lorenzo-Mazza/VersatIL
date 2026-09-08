"""Policy deployment backend contracts."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn

from versatil.post_training_compression.constants import ArtifactFormat
from versatil.quantization.constants import QuantizationMode
from versatil.quantization.module_target import EagerQuantizationModuleTarget


@dataclass
class DeploymentArtifact:
    """Artifact emitted by a deployment backend."""

    converted_model: nn.Module | None
    example_inputs: tuple[torch.Tensor, ...]
    model_filename: str
    artifact_format: ArtifactFormat
    backend_name: str
    model_bytes: bytes | None = None


class DeploymentBackend(ABC):
    """Base class for deployment artifact generation."""

    name: str
    artifact_format: ArtifactFormat
    model_filename: str
    supported_quantization_modes: tuple[str, ...] = ()
    supported_pt2e_backends: tuple[str, ...] = ()

    def validate_quantization(
        self, mode: str, pt2e_backend_names: tuple[str, ...] = ()
    ) -> None:
        """Check the workflow and graph quantizers selected for this backend.

        Args:
            mode: Quantization workflow identifier.
            pt2e_backend_names: Graph quantizer identifiers used by a PT2E workflow.

        Raises:
            ValueError: If the workflow or a PT2E quantizer conflicts with the
                backend's declared supported values.
        """
        if mode not in self.supported_quantization_modes:
            raise ValueError(
                f"Deployment backend {self.name} supports quantization modes "
                f"{list(self.supported_quantization_modes)}, got '{mode}'."
            )
        if mode == QuantizationMode.PT2E.value and any(
            name not in self.supported_pt2e_backends for name in pt2e_backend_names
        ):
            raise ValueError(
                f"Deployment backend {self.name} supports PT2E backends "
                f"{list(self.supported_pt2e_backends)}, got {list(pt2e_backend_names)}."
            )

    def validate_eager_target(
        self,
        model: nn.Module,
        target: EagerQuantizationModuleTarget,
        module_names: set[str],
        for_conversion: bool = True,
    ) -> None:
        """Apply deployment-specific requirements to selected eager layers.

        Args:
            model: Model containing the selected layers.
            target: Quantization schema and module scope selected by the workflow.
            module_names: Fully qualified names of the selected linear layers.
            for_conversion: Whether to check the device requirements for conversion.

        Note:
            Numerical requirements belong to the target's quantization schema.
            Backends override this method for their additional representation and
            device requirements.
        """
        return None

    @abstractmethod
    def export(
        self,
        model: nn.Module,
        example_inputs: tuple[torch.Tensor, ...],
    ) -> DeploymentArtifact:
        """Create a deployment artifact from an exportable PyTorch module."""
