"""Tests for versatil.post_training_compression.deployment_backends.executorch_xnnpack module."""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from versatil.post_training_compression.constants import (
    ArtifactFormat,
    CompressionFilename,
    DeploymentBackendName,
)
from versatil.post_training_compression.deployment_backends.executorch_xnnpack import (
    ExecutorchXNNPACKBackend,
    _lower_exported_program,
)

XNNPACK_MODULE = (
    "versatil.post_training_compression.deployment_backends.executorch_xnnpack"
)


@pytest.mark.unit
class TestExecutorchXNNPACKBackend:
    def test_export_lowers_model_to_pte_bytes(self):
        backend = ExecutorchXNNPACKBackend()
        model = MagicMock(spec=nn.Module)
        example_inputs = (torch.zeros(2, 4),)
        exported_program = MagicMock()

        with (
            patch(
                f"{XNNPACK_MODULE}._export_with_dynamic_batch",
                return_value=exported_program,
            ) as mock_export,
            patch.object(
                backend,
                "_lower_to_pte_buffer",
                return_value=b"pte-bytes",
            ) as mock_lower,
        ):
            artifact = backend.export(model=model, example_inputs=example_inputs)

        mock_export.assert_called_once_with(
            model=model,
            example_inputs=example_inputs,
        )
        mock_lower.assert_called_once_with(exported_program=exported_program)
        assert artifact.converted_model is None
        assert artifact.example_inputs is example_inputs
        assert artifact.model_bytes == b"pte-bytes"
        assert artifact.model_filename == CompressionFilename.EXECUTORCH_MODEL.value
        assert artifact.artifact_format == ArtifactFormat.EXECUTORCH_PTE
        assert artifact.backend_name == DeploymentBackendName.EXECUTORCH_XNNPACK.value


@pytest.mark.unit
class TestLowerExportedProgram:
    def test_delegates_to_executorch_xnnpack_partitioner(self):
        exported_program = MagicMock()
        edge_program = MagicMock()
        executorch_program = MagicMock()
        executorch_program.buffer = b"pte"
        edge_program.to_executorch.return_value = executorch_program
        executorch_exir = MagicMock()
        executorch_exir.to_edge_transform_and_lower.return_value = edge_program
        partitioner = MagicMock()
        xnnpack_partitioner = MagicMock()
        xnnpack_partitioner.XnnpackPartitioner.return_value = partitioner

        result = _lower_exported_program(
            exported_program=exported_program,
            executorch_exir=executorch_exir,
            xnnpack_partitioner=xnnpack_partitioner,
        )

        executorch_exir.to_edge_transform_and_lower.assert_called_once_with(
            exported_program,
            partitioner=[partitioner],
        )
        edge_program.to_executorch.assert_called_once_with()
        assert result == b"pte"
