"""ExecuTorch runtime adapter for compressed policy inference running a binary program through python."""

import importlib
from typing import Any

import torch
import torch.nn as nn


class ExecuTorchModuleAdapter(nn.Module):
    """Adapter exposing an ExecuTorch PTE program through nn.Module."""

    def __init__(self, model_path: str) -> None:
        """Register quantized kernels and load a PTE program from disk.

        Args:
            model_path: Path to the serialized ExecuTorch program.

        Note:
            ExecuTorch is an optional dependency, loaded when this adapter is
            constructed.
        """
        super().__init__()
        importlib.import_module("executorch.kernels.quantized")
        portable_lib = importlib.import_module(
            "executorch.extension.pybindings.portable_lib"
        )
        self._module: Any = portable_lib._load_for_executorch(model_path)

    def forward(
        self,
        observation_tensors: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Run the ExecuTorch forward method."""
        contiguous_tensors = tuple(
            tensor.contiguous() for tensor in observation_tensors
        )
        outputs = self._module.forward(contiguous_tensors)
        if isinstance(outputs, torch.Tensor):
            return (outputs,)
        return tuple(outputs)
