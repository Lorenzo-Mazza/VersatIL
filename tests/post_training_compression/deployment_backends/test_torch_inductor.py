"""Tests for versatil.post_training_compression.deployment_backends.torch_inductor module."""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from versatil.post_training_compression.constants import (
    ArtifactFormat,
    CompressionBackendName,
    CompressionFilename,
)
from versatil.post_training_compression.deployment_backends.torch_inductor import (
    TorchInductorBackend,
)


@pytest.mark.unit
def test_torch_inductor_backend_returns_pt2_artifact_descriptor():
    model = MagicMock(spec=nn.Module)
    example_inputs = (torch.zeros(2, 4),)

    artifact = TorchInductorBackend().export(
        model=model,
        example_inputs=example_inputs,
    )

    assert artifact.converted_model is model
    assert artifact.example_inputs is example_inputs
    assert artifact.model_bytes is None
    assert artifact.model_filename == CompressionFilename.COMPRESSED_MODEL.value
    assert artifact.artifact_format == ArtifactFormat.TORCH_EXPORT_PT2
    assert artifact.backend_name == CompressionBackendName.TORCH_INDUCTOR.value
