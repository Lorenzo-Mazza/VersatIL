"""Tests for versatil.post_training_compression.deployment_backends.torch_inductor module."""

import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from torchao.quantization import Int8WeightOnlyConfig

from versatil.post_training_compression.constants import (
    ArtifactFormat,
    CompressionFilename,
    DeploymentBackendName,
)
from versatil.post_training_compression.deployment_backends.torch_inductor import (
    TorchInductorBackend,
)
from versatil.quantization.constants import PT2EBackendName, QuantizationMode


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
    assert artifact.backend_name == DeploymentBackendName.TORCH_INDUCTOR.value


@pytest.mark.unit
@pytest.mark.parametrize("mode", [*list(QuantizationMode), "unsupported"])
def test_declares_supported_quantization_workflows(mode: str) -> None:
    backend = TorchInductorBackend()
    expectation = (
        pytest.raises(
            ValueError,
            match=re.escape(
                "Deployment backend torch_inductor supports quantization modes "
                "['none', 'pt2e', 'eager'], got 'unsupported'."
            ),
        )
        if mode == "unsupported"
        else does_not_raise()
    )
    with expectation:
        backend.validate_quantization(mode=mode)


@pytest.mark.unit
@pytest.mark.parametrize("pt2e_backend", list(PT2EBackendName))
def test_pt2e_export_requires_the_x86_quantizer(pt2e_backend: PT2EBackendName) -> None:
    backend = TorchInductorBackend()
    expectation = (
        does_not_raise()
        if pt2e_backend == PT2EBackendName.X86_INDUCTOR
        else pytest.raises(
            ValueError,
            match=re.escape(
                "Deployment backend torch_inductor supports PT2E backends "
                "['x86_inductor'], got ['xnnpack']."
            ),
        )
    )
    with expectation:
        backend.validate_quantization(
            mode=QuantizationMode.PT2E.value,
            pt2e_backend_names=(pt2e_backend.value,),
        )


@pytest.mark.unit
def test_eager_export_leaves_intrinsic_checks_with_the_schema(
    deployment_target_factory: Callable[..., MagicMock],
    deployment_model_factory: Callable[..., MagicMock],
) -> None:
    model = deployment_model_factory(device="cpu")
    target = deployment_target_factory(config=MagicMock(spec=Int8WeightOnlyConfig))

    TorchInductorBackend().validate_eager_target(
        model=model, target=target, module_names={"projection"}, for_conversion=True
    )

    model.get_submodule.assert_not_called()
    assert target.schema.mock_calls == []
