"""Abstract base class for PT2E quantization backends."""

from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager

import torch
import torch.nn as nn
from torchao.quantization.pt2e.quantizer import Quantizer


class BasePT2EBackend(ABC):
    """Base class for PT2E quantization backends.

    Each backend provides quantizer creation, environment setup,
    and operator lowering for a specific hardware target.
    """

    @property
    @abstractmethod
    def is_dynamic(self) -> bool:
        """Whether this backend uses dynamic activation quantization."""

    @abstractmethod
    def create_quantizer(self, module_path: str) -> Quantizer:
        """Create a configured quantizer targeting a specific module.

        Args:
            module_path: Dotted path to the target submodule.
                Empty string means global (whole model).

        Returns:
            A quantizer instance ready for ComposableQuantizer.
        """

    @abstractmethod
    @contextmanager
    def environment_context(self) -> Generator[None]:
        """Context manager that sets and restores backend-specific env vars."""

    @abstractmethod
    def lower(
        self,
        converted_model: nn.Module,
        example_inputs: tuple[torch.Tensor, ...],
    ) -> nn.Module:
        """Apply backend-specific lowering to the converted model."""
