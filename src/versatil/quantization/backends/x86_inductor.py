"""X86 Inductor backend for PT2E quantized operator lowering."""

import os
from collections.abc import Generator
from contextlib import contextmanager

import torch
import torch._inductor.config as inductor_config
import torch.nn as nn
from torchao.quantization.pt2e.lowering import lower_pt2e_quantized_to_x86
from torchao.quantization.pt2e.quantizer import Quantizer
from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
    X86InductorQuantizer,
    get_default_x86_inductor_quantization_config,
)

from versatil.quantization.backends.base import BasePT2EBackend

_CUDA_VISIBLE_DEVICES_KEY = "CUDA_VISIBLE_DEVICES"
_TORCHINDUCTOR_FREEZING_KEY = "TORCHINDUCTOR_FREEZING"


class X86InductorBackend(BasePT2EBackend):
    """X86 Inductor backend for PT2E quantization and lowering."""

    def __init__(
        self,
        is_dynamic: bool = False,
        is_qat: bool = False,
        reduce_range: bool = False,
    ) -> None:
        """Initialize X86 Inductor backend configuration.

        Args:
            is_dynamic: Use dynamic activation quantization.
            is_qat: Use quantization-aware training observers.
            reduce_range: Reduce quantization range for older CPUs
                without VNNI.
        """
        self._is_dynamic = is_dynamic
        self._is_qat = is_qat
        self._reduce_range = reduce_range

    @property
    def is_dynamic(self) -> bool:
        """Whether this backend uses dynamic activation quantization."""
        return self._is_dynamic

    def create_quantizer(self, module_path: str) -> Quantizer:
        """Create an X86InductorQuantizer targeting a specific module.

        Args:
            module_path: Dotted path to the target submodule.
                Empty string means global (whole model).

        Returns:
            Configured X86InductorQuantizer.
        """
        quantizer = X86InductorQuantizer()
        config = get_default_x86_inductor_quantization_config(
            is_dynamic=self._is_dynamic,
            is_qat=self._is_qat,
            reduce_range=self._reduce_range,
        )
        if module_path == "":
            quantizer.set_global(config)
        else:
            quantizer.set_module_name_qconfig(module_path, config)
        return quantizer

    @contextmanager
    def environment_context(self) -> Generator[None]:
        """Context manager that sets and restores X86 Inductor env vars.

        Sets CUDA_VISIBLE_DEVICES to empty (CPU-only), enables
        TORCHINDUCTOR_FREEZING, and enables cpp_wrapper. Restores
        original values on exit.

        Yields:
            None.
        """
        saved = {
            _CUDA_VISIBLE_DEVICES_KEY: os.environ.get(_CUDA_VISIBLE_DEVICES_KEY),
            _TORCHINDUCTOR_FREEZING_KEY: os.environ.get(_TORCHINDUCTOR_FREEZING_KEY),
        }
        saved_cpp_wrapper = inductor_config.cpp_wrapper
        os.environ[_CUDA_VISIBLE_DEVICES_KEY] = ""
        os.environ[_TORCHINDUCTOR_FREEZING_KEY] = "1"
        inductor_config.cpp_wrapper = True
        try:
            yield
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            inductor_config.cpp_wrapper = saved_cpp_wrapper

    def lower(
        self,
        converted_model: nn.Module,
        example_inputs: tuple[torch.Tensor, ...],
    ) -> nn.Module:
        """Apply X86 Inductor lowering to the converted model.

        Args:
            converted_model: The PT2E-converted model.
            example_inputs: Example inputs for lowering optimization.

        Returns:
            The lowered model with X86 Inductor operator fusion.
        """
        return lower_pt2e_quantized_to_x86(converted_model, example_inputs)
