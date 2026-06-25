"""ExecuTorch XNNPACK deployment backend for .pte artifacts, to deploy policies on mobile Arm and x86 CPUs, ref. https://docs.pytorch.org/executorch/main/backends/xnnpack/xnnpack-overview.html."""

import importlib
from types import ModuleType

import torch
import torch.nn as nn

from versatil.post_training_compression.constants import (
    ArtifactFormat,
    CompressionBackendName,
    CompressionFilename,
)
from versatil.post_training_compression.deployment_backends.base import (
    DeploymentArtifact,
    DeploymentBackend,
)
from versatil.post_training_compression.export import _export_with_dynamic_batch


class ExecutorchXNNPACKBackend(DeploymentBackend):
    """Backend that lowers exported programs to ExecuTorch XNNPACK."""

    name = CompressionBackendName.EXECUTORCH_XNNPACK.value
    artifact_format = ArtifactFormat.EXECUTORCH_PTE
    model_filename = CompressionFilename.EXECUTORCH_MODEL.value

    def export(
        self,
        model: nn.Module,
        example_inputs: tuple[torch.Tensor, ...],
    ) -> DeploymentArtifact:
        """Lower a PyTorch module into an ExecuTorch .pte buffer."""
        exported_program = _export_with_dynamic_batch(
            model=model,
            example_inputs=example_inputs,
        )
        model_bytes = self._lower_to_pte_buffer(exported_program=exported_program)
        return DeploymentArtifact(
            converted_model=None,
            example_inputs=example_inputs,
            model_filename=self.model_filename,
            artifact_format=self.artifact_format,
            backend_name=self.name,
            model_bytes=model_bytes,
        )

    @staticmethod
    def _lower_to_pte_buffer(
        exported_program: torch.export.ExportedProgram,
    ) -> bytes:
        """Lower an exported program to an ExecuTorch PTE buffer."""
        executorch_exir = importlib.import_module("executorch.exir")
        xnnpack_partitioner = importlib.import_module(
            "executorch.backends.xnnpack.partition.xnnpack_partitioner"
        )  # This avoids a hard dependency on executorch for the entire versatil package, only requiring it when this adapter is used.
        return _lower_exported_program(
            exported_program=exported_program,
            executorch_exir=executorch_exir,
            xnnpack_partitioner=xnnpack_partitioner,
        )


def _lower_exported_program(
    exported_program: torch.export.ExportedProgram,
    executorch_exir: ModuleType,
    xnnpack_partitioner: ModuleType,
) -> bytes:
    """Lower an exported program using imported ExecuTorch modules."""
    edge_program = executorch_exir.to_edge_transform_and_lower(
        exported_program,
        partitioner=[xnnpack_partitioner.XnnpackPartitioner()],
    )
    executorch_program = edge_program.to_executorch()
    return bytes(executorch_program.buffer)
