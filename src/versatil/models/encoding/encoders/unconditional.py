from abc import abstractmethod

import torch

from versatil.models.encoding.encoders.base import EncoderInput, EncodingMixin


class Encoder(EncodingMixin):
    """Base class for all unconditional encoders.

    Subclasses implement `encode()`. The base class
    `forward()` handles temporal flatten/unflatten — `encode()` receives
    tensors without a time dimension (images as (B, C, H, W), not (B, T, C, H, W)).
    """

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

    def _validate_inputs(self, inputs: dict[str, torch.Tensor]) -> None:
        """Validate that all input tensors have a temporal dimension (B, T, ...).

        Args:
            inputs: Dict mapping keys to tensors.

        Raises:
            ValueError: If any value is not a tensor or has fewer than 3 dimensions.
        """
        for key, value in inputs.items():
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"Encoder input '{key}' must be a torch.Tensor, "
                    f"got {type(value).__name__}."
                )
            if value.dim() < 3:
                raise ValueError(
                    f"Encoder input '{key}' has shape {tuple(value.shape)} "
                    f"but all inputs must have a temporal dimension (B, T, ...)."
                )

    def _flatten_temporal(
        self, inputs: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], int, int]:
        """Merge temporal dimension into batch for all input tensors.

        Args:
            inputs: Dict mapping keys to tensors with temporal dimension.
                All tensors must have shape (B, T, ...).

        Returns:
            Flattened dict (B*T, ...), batch_size, temporal_length.
        """
        first_tensor = next(iter(inputs.values()))
        batch_size = first_tensor.shape[0]
        temporal_length = first_tensor.shape[1]
        flattened = {}
        for key, tensor in inputs.items():
            flattened[key] = tensor.reshape(
                batch_size * temporal_length, *tensor.shape[2:]
            )
        return flattened, batch_size, temporal_length

    def _unflatten_temporal(
        self,
        outputs: dict[str, torch.Tensor],
        batch_size: int,
        temporal_length: int,
    ) -> dict[str, torch.Tensor]:
        """Restore temporal dimension in output tensors.

        Args:
            outputs: Dict mapping feature names to tensors (B*T, ...).
            batch_size: Original batch size.
            temporal_length: Temporal length to restore.

        Returns:
            Dict with shape (B, T, ...) for each tensor.
        """
        unflattened = {}
        for key, tensor in outputs.items():
            unflattened[key] = tensor.reshape(
                batch_size, temporal_length, *tensor.shape[1:]
            )
        return unflattened

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass with temporal flatten/unflatten.

        All input tensors must have a temporal dimension: images as (B, T, C, H, W),
        sequences as (B, T, S), vectors as (B, T, D). The temporal dimension is
        flattened into batch before calling encode(), then restored after.

        Args:
            inputs: Dict mapping input_keys to tensors with temporal dimension.

        Returns:
            Dict mapping feature names to feature tensors with temporal dimension.
        """
        self._validate_inputs(inputs)
        inputs, batch_size, temporal_length = self._flatten_temporal(inputs)
        outputs = self.encode(inputs)
        return self._unflatten_temporal(outputs, batch_size, temporal_length)

    @abstractmethod
    def encode(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Encode inputs without temporal dimension.

        Subclasses implement this. Images are (B, C, H, W),
        tokenized text is (B, S), proprioceptive state is (B, D).

        Args:
            inputs: Dict mapping input_keys to tensors without temporal dimension.

        Returns:
            Dict mapping feature names to feature tensors.
        """
        raise NotImplementedError
