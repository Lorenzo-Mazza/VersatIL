"""Base contracts for generative language models used for action generation."""

from __future__ import annotations

import abc
from dataclasses import dataclass

import torch
from transformers.cache_utils import Cache

from versatil.common.module_attr_mixin import ModuleAttrMixin
from versatil.data.metadata import BaseMetadata
from versatil.models.input_specification import InputSpecification
from versatil.training.constants import PrecisionType


@dataclass
class CausalLanguageModelOutput:
    """Causal language-model output used by generative language models."""

    hidden_states: tuple[torch.Tensor, ...]
    logits: torch.Tensor
    past_key_values: Cache | tuple[tuple[torch.Tensor, ...], ...] | None


class GenerativeLanguageModel(ModuleAttrMixin, abc.ABC):
    """Base class for generative language models."""

    hidden_dimension: int

    def __init__(
        self,
        input_specification: InputSpecification,
        pretrained: bool = False,
        frozen: bool = False,
        device: str | None = None,
        model_dtype: str | None = None,
    ) -> None:
        """Initialize common generative language-model metadata."""
        super().__init__()
        input_specification.validate()
        self.input_specification = input_specification
        self.pretrained = pretrained
        self.frozen = frozen
        if model_dtype is not None:
            valid_values = [p.value for p in PrecisionType]
            if model_dtype not in valid_values:
                raise ValueError(
                    f"Invalid model_dtype '{model_dtype}'. "
                    f"Must be one of: {valid_values}"
                )
            self.precision_type: PrecisionType | None = PrecisionType(model_dtype)
            self.model_dtype: torch.dtype | None = self.precision_type.get_model_dtype()
        else:
            self.precision_type = None
            self.model_dtype = None
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    def _freeze_weights(self) -> None:
        """Freeze model weights."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def train(self, mode: bool = True) -> GenerativeLanguageModel:
        """Set train/eval mode while keeping fully frozen models eval-locked."""
        super().train(mode)
        parameters = list(self.parameters())
        if (
            mode
            and self.frozen
            and parameters
            and all(not parameter.requires_grad for parameter in parameters)
        ):
            super().train(False)
        return self

    def _apply_model_dtype(self) -> None:
        """Cast the module tree to the configured model dtype.

        Note:
            Under mixed precision, trainable parameters are
            kept in float32 storage: autocast already runs the compute in the low
            precision, while float32 masters prevent optimizer updates smaller
            than the parameter's low-precision rounding step from vanishing.

            The method must be called after ``requires_grad`` flags are final
            (freezing, LoRA wrapping).
        """
        target_dtype = (
            self.model_dtype if self.model_dtype is not None else torch.float32
        )
        self.to(target_dtype)
        if self.precision_type is not None and self.precision_type.is_mixed():
            for parameter in self.parameters():
                if parameter.requires_grad:
                    parameter.data = parameter.data.float()

    def get_vocab_size(self) -> int | None:
        """Get vocabulary size if applicable, else None."""
        return None

    def resize_token_embeddings(self, vocabulary_size: int) -> None:
        """Resize token embeddings when the model exposes a token vocabulary."""
        raise ValueError(
            f"{type(self).__name__} does not support token embedding resizing."
        )

    def validate_input_metadata(self, key: str, metadata: BaseMetadata) -> str | None:
        """Check that observation metadata is compatible with this model."""
        return None
