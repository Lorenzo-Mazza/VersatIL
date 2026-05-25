"""Base classes for encoder input/output specifications."""

import abc
from abc import abstractmethod

import torch

from versatil.common.module_attr_mixin import ModuleAttrMixin
from versatil.data.metadata import BaseMetadata
from versatil.models.feature_meta import FeatureMetadata
from versatil.models.input_specification import InputSpecification
from versatil.training.constants import PrecisionType

EncoderInput = InputSpecification


class EncodingMixin(ModuleAttrMixin, abc.ABC):
    """Base interface for all encoders, conditional and non-conditional."""

    def __init__(
        self,
        input_specification: EncoderInput,
        pretrained: bool = False,
        frozen: bool = False,
        device: str | None = None,
        model_dtype: str | None = None,
    ) -> None:
        """Initialize base encoder.

        Args:
            input_specification: Structured input specification for this encoder
            pretrained: Whether to use pretrained weights
            frozen: Whether to freeze encoder weights
            device: Device to place the encoder on (will use "cuda" if available, else "cpu" if None)
            model_dtype: Precision string from experiment config (e.g. ``"bf16-mixed"``).
                Resolved to ``torch.dtype`` via ``PrecisionType``.
        """
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
            self.model_dtype: torch.dtype | None = PrecisionType(
                model_dtype
            ).get_model_dtype()
        else:
            self.model_dtype = None
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    def _freeze_weights(self):
        """Freeze model weights."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def train(self, mode: bool = True) -> "EncodingMixin":
        """Set train/eval mode while keeping fully frozen encoders eval-locked."""
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
        """Cast the entire encoder module tree to a consistent dtype.

        When ``model_dtype`` is set explicitly, casts to that dtype.
        When ``model_dtype`` is None, casts to ``torch.float32``.

        Called by child encoders at the end of ``__init__`` and after any
        deferred-build method (``set_image_size``, ``_build_network``, …) so
        a rebuild cannot leak mismatched-dtype submodules back into the tree.
        """
        target_dtype = (
            self.model_dtype if self.model_dtype is not None else torch.float32
        )
        self.to(target_dtype)

    @abstractmethod
    def get_output_specification(self) -> list[FeatureMetadata]:
        """Get encoder structured output specification."""
        raise NotImplementedError

    def get_vocab_size(self) -> int | None:
        """Get vocabulary size if applicable, else None."""
        return None

    def validate_input_metadata(self, key: str, metadata: BaseMetadata) -> str | None:
        """Check that observation metadata is compatible with this encoder.

        Note:
            Called by the encoding pipeline per input key during setup.
            Subclasses override to check type, channels, dimensions, etc.

        Args:
            key: Observation key being validated.
            metadata: Metadata from the observation space for this key.

        Returns:
            Error message if incompatible, None if valid.
        """
        return None

    def set_image_size(self, image_height: int, image_width: int) -> None:
        """Set the target image size for this encoder.

        Called by the encoding pipeline after Hydra instantiation, with per-camera
        image dimensions from the observation space. Image encoders override this
        to configure their backbone (ViT img_size) or pooling head (CNN spatial dims).

        Args:
            image_height: Target image height.
            image_width: Target image width.
        """
        pass
