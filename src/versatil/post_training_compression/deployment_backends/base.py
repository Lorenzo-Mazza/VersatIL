"""Policy deployment backend contracts."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn

from versatil.post_training_compression.constants import ArtifactFormat


@dataclass
class DeploymentArtifact:
    """Artifact emitted by a deployment backend."""

    converted_model: nn.Module | None
    example_inputs: tuple[torch.Tensor, ...]
    model_filename: str
    artifact_format: ArtifactFormat
    backend_name: str
    model_bytes: bytes | None = None


class DeploymentBackend(ABC):
    """Base class for deployment artifact generation."""

    name: str
    artifact_format: ArtifactFormat
    model_filename: str

    @abstractmethod
    def export(
        self,
        model: nn.Module,
        example_inputs: tuple[torch.Tensor, ...],
    ) -> DeploymentArtifact:
        """Create a deployment artifact from an exportable PyTorch module."""
