"""Shared quantization workflow types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn

from versatil.configs.main import MainConfig
from versatil.data.task import ObservationSpace
from versatil.data.tokenization.tokenizer import Tokenizer
from versatil.models.exportable_policy import ExportablePolicy
from versatil.models.policy import Policy


@dataclass
class PolicyContext:
    """Loaded policy context used by quantization workflows."""

    policy: Policy
    config: MainConfig
    tokenizer: Tokenizer | None
    observation_space: ObservationSpace
    observation_horizon: int
    checkpoint_path: str
    checkpoint_name: str


class CompressionTargetProtocol(Protocol):
    """Compression target fields consumed by quantization workflows."""

    module_path: str


class BaseQuantizationWorkflow(ABC):
    """Base class for ordered quantization workflows."""

    @property
    @abstractmethod
    def quantization_mode(self) -> str:
        """Return the quantization path name: ``none``, ``pt2e`` or ``eager``."""

    @abstractmethod
    def load_policy_context(
        self,
        checkpoint_path: str,
        checkpoint_name: str,
    ) -> PolicyContext:
        """Load the policy context required by this workflow."""

    @abstractmethod
    def quantize(
        self,
        context: PolicyContext,
        exportable: ExportablePolicy,
        modules: list[CompressionTargetProtocol],
        calibration_steps: int,
    ) -> "QuantizedContext":
        """Return the exported context produced by this workflow."""

    @property
    def is_qat(self) -> bool:
        """Return whether the workflow expects QAT-trained weights."""
        return False

    def prepare_model(self, model: nn.Module) -> None:
        """Prepare fake-quant modules before loading or training QAT weights."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support QAT preparation."
        )


@dataclass
class QuantizedContext:
    """Quantized export result plus inputs and metadata needed for deployment."""

    float_model: nn.Module
    quantized_model: nn.Module
    example_inputs: tuple[torch.Tensor, ...]
    quantization_workflow: str
