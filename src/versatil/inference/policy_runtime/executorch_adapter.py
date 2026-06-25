"""ExecuTorch runtime adapter for compressed policy inference running a binary program through python."""

import importlib
from typing import Any

import torch
import torch.nn as nn


class ExecuTorchModuleAdapter(nn.Module):
    """Adapter exposing an ExecuTorch PTE program through nn.Module."""

    def __init__(self, model_path: str) -> None:
        """Load a PTE program from disk."""
        super().__init__()
        portable_lib = importlib.import_module(
            "executorch.extension.pybindings.portable_lib"
        )  # This avoids a hard dependency on executorch for the entire versatil package, only requiring it when this adapter is used.
        self._module: Any = portable_lib._load_for_executorch(model_path)

    def forward(
        self,
        observation_tensors: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Run the ExecuTorch forward method."""
        outputs = self._module.forward(observation_tensors)
        if isinstance(outputs, torch.Tensor):
            return (outputs,)
        return tuple(outputs)
