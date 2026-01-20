from abc import abstractmethod

import torch

from versatil.models.encoding.encoders.base import EncoderInput, EncodingMixin


class ConditionalEncoder(EncodingMixin):
    """Encoder that conditions its outputs based on an external feature."""

    def __init__(
        self,
        input_specification: EncoderInput,
        pretrained: bool = False,
        frozen: bool = False,
        device: str | None = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        if not input_specification.conditioning_key:
            raise ValueError("Conditional encoder requires conditioning_key")
        super().__init__(
            input_specification=input_specification,
            pretrained=pretrained,
            frozen=frozen,
            device=device,
        )

    @abstractmethod
    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        conditioning: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass to extract features from images.
        Args:
            inputs: Dict mapping input_keys to tensors
            conditioning: Conditioning tensor from another encoder

        Returns:
            A dictionary with as keys the feature names and as values the corresponding feature torch tensors.
        """
        raise NotImplementedError
