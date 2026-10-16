"""Torch Export artifact backend for PyTorch CPU and CUDA runtimes."""

import torch
import torch.nn as nn

from versatil.post_training_compression.constants import (
    ArtifactFormat,
    CompressionDeviceType,
    CompressionFilename,
    DeploymentBackendName,
)
from versatil.post_training_compression.deployment_backends.base import (
    DeploymentArtifact,
    DeploymentBackend,
)
from versatil.quantization.constants import PT2EBackendName, QuantizationMode


class TorchInductorBackend(DeploymentBackend):
    """Backend that saves .pt2 artifacts for torch.compile lowering."""

    name = DeploymentBackendName.TORCH_INDUCTOR.value
    artifact_format = ArtifactFormat.TORCH_EXPORT_PT2
    model_filename = CompressionFilename.COMPRESSED_MODEL.value
    supported_quantization_modes = (
        QuantizationMode.NONE.value,
        QuantizationMode.PT2E.value,
        QuantizationMode.EAGER.value,
    )
    supported_pt2e_backends = (PT2EBackendName.X86_INDUCTOR.value,)
    supported_device_types = (
        CompressionDeviceType.CPU.value,
        CompressionDeviceType.CUDA.value,
    )

    def export(
        self,
        model: nn.Module,
        example_inputs: tuple[torch.Tensor, ...],
    ) -> DeploymentArtifact:
        """Validate graph execution and return an artifact descriptor.

        Args:
            model: Exported model selected for deployment.
            example_inputs: Representative tensors on the export device.

        Returns:
            Artifact descriptor ready for serialization.

        Note:
            Some quantized operations use a fallback for shapes their kernel
            cannot handle. Export can capture the direct kernel call, so we run
            the exported graph with the example inputs on the deployment device
            before saving to catch those failures during compression.

            B denotes batch size.
        """
        with torch.no_grad():
            model(*example_inputs)  # (B, ...)
        return DeploymentArtifact(
            converted_model=model,
            example_inputs=example_inputs,
            model_filename=self.model_filename,
            artifact_format=self.artifact_format,
            backend_name=self.name,
        )
