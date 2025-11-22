import abc
from abc import abstractmethod
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class Metadata:
    features: list[str]  # list of feature names as `EncoderOutputKeys.value`.
    dimensions: dict[str, int | tuple]  # feature_name -> dimension

@dataclass
class SingleOutput:
    output_name: str
    output_dim: int | tuple[int, ...]
    tensor: torch.Tensor
    @property
    def is_spatial(self) -> bool:
        """Check if output has spatial dimensions."""
        return isinstance(self.output_dim, tuple) and len(self.output_dim) == 3

    @property
    def is_flat(self) -> bool:
        """Check if output is flat (no spatial or sequence dimensions)."""
        if isinstance(self.output_dim, int):
            return True
        return len(self.output_dim) == 1

@dataclass
class StructuredOutput:
    outputs: list[SingleOutput]
    @property
    def is_multi_output(self) -> bool:
        return len(self.outputs) > 1


@dataclass
class EncoderOutput:
    """Structured encoder output specification."""
    features: list[str]  # list of feature names as `EncoderOutputKeys.value`.
    dimensions: dict[str, int | tuple]  # feature_name -> dimension (skips batch and time dimension)

    @property
    def is_multi_output(self) -> bool:
        return len(self.features) > 1


@dataclass
class EncoderInput:
    """Structured input specification for encoders."""
    keys: str | list[str]
    #: The encoder needs these input observation keys
    required: list[str] = field(default_factory=list)
    #: The encoder needs exactly one input observation key from each of these groups
    one_of_groups: list[list[str]] = field(default_factory=list)
    #: The encoder needs at least one input observation key from these groups
    at_least_one_of_groups: list[list[str]] = field(default_factory=list)
    # For conditional encoders
    conditioning_key: str | None = None
    conditioning_required: list[str] = field(default_factory=list)
    conditioning_one_of_groups: list[list[str]] = field(default_factory=list)
    # For validating the data tokenizer vocabulary to be consistent with the encoder language models
    requires_tokenized: bool = False

    def __post_init__(self):
        if isinstance(self.keys, str):
            self.keys = [self.keys]

    def validate(self):
        key_set = set(self.keys)
        missing = set(self.required) - key_set
        if missing:
            raise ValueError(f"Missing required inputs: {missing}")
        for group in self.one_of_groups:
            matches = key_set.intersection(group)
            if len(matches) != 1:
                raise ValueError(f"Exactly one from {group} required, got {matches}")
        for group in self.at_least_one_of_groups:
            matches = key_set.intersection(group)
            if len(matches) < 1:
                raise ValueError(f"At least one from {group} required, got {matches}")
        if self.conditioning_key:
            conditioning_set = {self.conditioning_key}
            missing_conditioning = set(self.conditioning_required) - conditioning_set
            if missing_conditioning:
                raise ValueError(f"Missing required conditioning: {missing_conditioning}")
            for group in self.conditioning_one_of_groups:
                matches = conditioning_set.intersection(group)
                if len(matches) != 1:
                    raise ValueError(f"Exactly one from {group} required for conditioning")


class EncodingMixin(nn.Module, abc.ABC):
    """Base interface for all encoders, conditional and non-conditional."""
    def __init__(
            self,
            input_specification: EncoderInput,
            pretrained: bool = False,
            frozen: bool = False,
            device: str | None = None,
    ):
        """Initialize base encoder.

        Args:
            input_specification: Structured input specification for this encoder
            pretrained: Whether to use pretrained weights
            frozen: Whether to freeze encoder weights
            device: Device to place the encoder on (will use "cuda" if available, else "cpu" if None)
        """
        super().__init__()
        input_specification.validate()
        self.input_specification = input_specification
        self.pretrained = pretrained
        self.frozen = frozen
        # Store device as torch.device for consistency with Policy and Decoder
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)


    def _freeze_weights(self):
        """Freeze model weights."""
        for param in self.parameters():
            param.requires_grad = False


    @abstractmethod
    def get_output_specification(self) -> EncoderOutput:
        """Get encoder structured output specification."""
        raise NotImplementedError

    def get_vocab_size(self) -> int | None:
        """Get vocabulary size if applicable, else None."""
        return None

