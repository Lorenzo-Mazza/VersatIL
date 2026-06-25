"""Torch Inductor deployment backend for .pt2 artifacts, to deploy policies on x86 CPUs with optimized C++/Triton kernels, ref. https://docs.pytorch.org/ao/stable/pt2e_quantization/pt2e_quant_x86_inductor.html."""

import torch
import torch.nn as nn

from versatil.post_training_compression.constants import (
    ArtifactFormat,
    CompressionFilename,
    DeploymentBackendName,
)
from versatil.post_training_compression.deployment_backends.base import (
    DeploymentArtifact,
    DeploymentBackend,
)


class TorchInductorBackend(DeploymentBackend):
    """Backend that saves .pt2 artifacts for torch.compile lowering."""

    name = DeploymentBackendName.TORCH_INDUCTOR.value
    artifact_format = ArtifactFormat.TORCH_EXPORT_PT2
    model_filename = CompressionFilename.COMPRESSED_MODEL.value

    def export(
        self,
        model: nn.Module,
        example_inputs: tuple[torch.Tensor, ...],
    ) -> DeploymentArtifact:
        """Return a .pt2 deployment artifact descriptor."""
        return DeploymentArtifact(
            converted_model=model,
            example_inputs=example_inputs,
            model_filename=self.model_filename,
            artifact_format=self.artifact_format,
            backend_name=self.name,
        )
