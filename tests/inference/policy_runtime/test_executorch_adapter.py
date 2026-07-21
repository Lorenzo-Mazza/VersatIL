"""Tests for versatil.inference.policy_runtime.executorch_adapter module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from versatil.inference.policy_runtime.executorch_adapter import (
    ExecuTorchModuleAdapter,
)
from versatil.post_training_compression.constants import CompressionFilename
from versatil.post_training_compression.deployment_backends.executorch_xnnpack import (
    ExecutorchXNNPACKBackend,
)

EXECUTORCH_ADAPTER_MODULE = "versatil.inference.policy_runtime.executorch_adapter"


class TestExecutorchModuleAdapter:
    @pytest.mark.unit
    def test_forward_makes_every_input_contiguous_before_executorch(self) -> None:
        portable_library = MagicMock()
        executorch_module = MagicMock()
        output_tensor = torch.zeros(1)
        executorch_module.forward.return_value = (output_tensor,)
        portable_library._load_for_executorch.return_value = executorch_module
        contiguous_tensor = torch.zeros(1, 4)
        channel_last_frames = torch.zeros(1, 4, 4, 3).permute(0, 3, 1, 2)
        non_contiguous_tensor = torch.stack([channel_last_frames])
        if non_contiguous_tensor.is_contiguous():
            raise RuntimeError("Test input must be non-contiguous.")

        with patch(
            f"{EXECUTORCH_ADAPTER_MODULE}.importlib.import_module",
            return_value=portable_library,
        ):
            adapter = ExecuTorchModuleAdapter(model_path="policy.pte")
            outputs = adapter(
                observation_tensors=(non_contiguous_tensor, contiguous_tensor)
            )

        forwarded_tensors = executorch_module.forward.call_args.args[0]
        assert all(tensor.is_contiguous() for tensor in forwarded_tensors)
        torch.testing.assert_close(forwarded_tensors[0], non_contiguous_tensor)
        assert forwarded_tensors[1] is contiguous_tensor
        executorch_module.forward.assert_called_once()
        assert len(outputs) == 1
        torch.testing.assert_close(outputs[0], output_tensor)

    @pytest.mark.integration
    @pytest.mark.requires_executorch
    def test_runs_non_contiguous_camera_batch_through_pte(
        self,
        tmp_path: Path,
    ) -> None:
        model = nn.ReLU().eval()
        example_tensor = torch.zeros(2, 1, 3, 256, 256)
        backend = ExecutorchXNNPACKBackend(max_batch_size=2)
        artifact = backend.export(model=model, example_inputs=(example_tensor,))
        model_path = tmp_path / CompressionFilename.EXECUTORCH_MODEL.value
        model_path.write_bytes(artifact.model_bytes)
        adapter = ExecuTorchModuleAdapter(model_path=str(model_path))
        channel_last_frames = torch.ones(1, 256, 256, 3).permute(0, 3, 1, 2)
        non_contiguous_tensor = torch.stack([channel_last_frames])
        if non_contiguous_tensor.is_contiguous():
            raise RuntimeError("Test input must be non-contiguous.")

        outputs = adapter(observation_tensors=(non_contiguous_tensor,))

        assert len(outputs) == 1
        torch.testing.assert_close(outputs[0], non_contiguous_tensor)
