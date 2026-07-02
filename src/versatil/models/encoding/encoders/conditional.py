from abc import abstractmethod

import torch

from versatil.models.encoding.encoders.base import EncoderInput, EncodingMixin


class ConditionalEncoder(EncodingMixin):
    """Encoder that conditions its outputs based on an external feature.

    Subclasses implement `encode()`. The base class `forward()` handles
    temporal flatten/unflatten — `encode()` receives tensors without a
    time dimension.
    """

    def __init__(
        self,
        input_specification: EncoderInput,
        pretrained: bool = False,
        frozen: bool = False,
        device: str | None = "cuda" if torch.cuda.is_available() else "cpu",
        model_dtype: str | None = None,
    ):
        """Initialize conditional encoder.

        Args:
            input_specification: Structured input specification with conditioning key.
            pretrained: Whether to use pretrained weights.
            frozen: Whether to freeze encoder weights.
            device: Device to place the encoder on.
            model_dtype: Precision string from experiment config (e.g. ``"bf16-mixed"``).
        """
        if not input_specification.conditioning_key:
            raise ValueError("Conditional encoder requires conditioning_key")
        super().__init__(
            input_specification=input_specification,
            pretrained=pretrained,
            frozen=frozen,
            device=device,
            model_dtype=model_dtype,
        )
        self.condition_key = input_specification.conditioning_key

    def _flatten_temporal(
        self, inputs: dict[str, torch.Tensor], conditioning: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, int, int]:
        """Merge temporal dimension into batch for inputs and conditioning.

        Args:
            inputs: Dict mapping keys to tensors (B, T, ...).
            conditioning: Conditioning tensor (B, T, D) or (B, D).

        Returns:
            Flattened inputs, flattened conditioning, batch_size, temporal_length.
        """
        first_tensor = next(iter(inputs.values()))
        batch_size = first_tensor.shape[0]
        temporal_length = first_tensor.shape[1]
        flattened = {}
        for key, tensor in inputs.items():
            flattened[key] = tensor.reshape(
                batch_size * temporal_length, *tensor.shape[2:]
            )
        if conditioning.dim() >= 3 and conditioning.shape[1] == temporal_length:
            conditioning = conditioning.reshape(
                batch_size * temporal_length, *conditioning.shape[2:]
            )
        elif conditioning.dim() >= 3 and conditioning.shape[1] == 1:
            # Single-timestep conditioning (e.g. language) replicates across
            # the image temporal length.
            conditioning = conditioning.expand(
                batch_size, temporal_length, *conditioning.shape[2:]
            ).reshape(batch_size * temporal_length, *conditioning.shape[2:])
        elif conditioning.dim() == 2:
            # Conditioning has no temporal dim — replicate across time
            conditioning = (
                conditioning.unsqueeze(1)
                .expand(batch_size, temporal_length, *conditioning.shape[1:])
                .reshape(batch_size * temporal_length, *conditioning.shape[1:])
            )
        else:
            raise ValueError(
                f"Conditioning shape {tuple(conditioning.shape)} does not match "
                f"the image temporal length {temporal_length}: expected "
                f"(B, {temporal_length}, ...) or a time-less (B, ...) tensor. "
                "Passing it through unflattened would broadcast against a "
                f"(B*T={batch_size * temporal_length}, ...) batch downstream."
            )
        return flattened, conditioning, batch_size, temporal_length

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

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        conditioning: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with temporal flatten/unflatten.

        Args:
            inputs: Dict mapping input_keys to tensors with temporal dimension.
            conditioning: Conditioning tensor from another encoder.

        Returns:
            Dict mapping feature names to feature tensors with temporal dimension.
        """
        self._validate_inputs(inputs)
        inputs, conditioning, batch_size, temporal_length = self._flatten_temporal(
            inputs, conditioning
        )
        outputs = self.encode(inputs, conditioning)
        return self._unflatten_temporal(outputs, batch_size, temporal_length)

    @abstractmethod
    def encode(
        self,
        inputs: dict[str, torch.Tensor],
        conditioning: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Encode inputs with conditioning, without temporal dimension.

        Args:
            inputs: Dict mapping input_keys to tensors without temporal dimension.
            conditioning: Conditioning tensor (B, D).

        Returns:
            Dict mapping feature names to feature tensors.
        """
        raise NotImplementedError
