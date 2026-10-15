"""Tests for versatil.post_training_compression.deployment_backends.base module."""

import re

import pytest
import torch

from versatil.post_training_compression.deployment_backends.base import (
    DeploymentBackend,
)
from versatil.post_training_compression.deployment_backends.executorch_xnnpack import (
    ExecutorchXNNPACKBackend,
)
from versatil.post_training_compression.deployment_backends.torch_inductor import (
    TorchInductorBackend,
)


@pytest.mark.unit
@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:1"])
def test_inductor_accepts_cpu_and_cuda_devices(device: str) -> None:
    TorchInductorBackend().validate_device(device=torch.device(device))


@pytest.mark.unit
def test_xnnpack_rejects_cuda_device() -> None:
    backend = ExecutorchXNNPACKBackend(max_batch_size=8)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Deployment backend executorch_xnnpack supports devices ('cpu',), got 'cuda:1'."
        ),
    ):
        backend.validate_device(device=torch.device("cuda:1"))


@pytest.mark.unit
def test_deployment_backend_requires_export_implementation():
    with pytest.raises(
        TypeError,
        match=re.escape(
            "Can't instantiate abstract class DeploymentBackend without an "
            "implementation for abstract method 'export'"
        ),
    ):
        DeploymentBackend()
