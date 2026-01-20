from abc import abstractmethod

import torch

from versatil.models.encoding.encoders.base import EncoderInput, EncodingMixin


class Encoder(EncodingMixin):
    """Base class for all unconditional encoders."""

    def __init__(
        self,
        input_specification: EncoderInput,
        pretrained: bool = False,
        frozen: bool = False,
        device: str | None = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize base encoder.

        Args:
            input_specification: Structured input specification for this encoder
            pretrained: Whether to use pretrained weights
            frozen: Whether to freeze encoder weights
            device: Device to place the encoder on
        """
        super().__init__(
            input_specification=input_specification,
            pretrained=pretrained,
            frozen=frozen,
            device=device,
        )

    @abstractmethod
    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Args:
            inputs: Dict mapping input_keys to tensors
        Returns:
            A dictionary with as keys the feature names and as values the corresponding feature torch tensors.
        """
        raise NotImplementedError
