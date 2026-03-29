"""Mixin for encoders that process tokenized language inputs."""

import logging

import torch

from versatil.data.constants import SampleKey
from versatil.models.encoding.encoders.constants import EncoderOutputKeys, PoolingMethod


class LanguageEncoderMixin:
    """Shared logic for encoders that process tokenized text sequences.

    Provides language key setup, text padding/truncation, and input
    extraction with validation. Mixed into LanguageEncoder, TwoTowerVLM,
    PaliGemma, and SmolVLM encoders.
    """

    def _setup_language_keys(self, output_modality: str) -> None:
        """Set language key and padding mask name.

        Args:
            output_modality: The output feature key used to construct the
                padding mask name (e.g. ``language`` or ``fused_rgb_language``).
        """
        self.language_key: str = SampleKey.TOKENIZED_OBSERVATIONS.value
        self.padding_mask_name: str = (
            f"{output_modality}_{EncoderOutputKeys.PADDING_MASK.value}"
        )

    def _extract_text_inputs(
        self, inputs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Extract and validate tokenized text and optional padding mask from inputs.

        Args:
            inputs: Dict of input tensors.

        Returns:
            Tuple of (text_input_ids, language_mask).

        Raises:
            ValueError: If the language key is missing from inputs.
        """
        if self.language_key not in inputs:
            raise ValueError(
                f"{self.__class__.__name__} expects pre-tokenized input. "
                f"Expected key '{self.language_key}' not found in inputs. "
                f"Ensure tokenization is enabled in DataloaderConfig."
            )
        text_input_ids = inputs[self.language_key]
        language_mask = inputs.get(SampleKey.IS_PAD_OBSERVATION.value)
        return text_input_ids, language_mask

    def _pad_text_inputs(
        self,
        text_input_ids: torch.Tensor,
        language_mask: torch.Tensor | None,
        max_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Pad or truncate text inputs to a fixed length.

        Args:
            text_input_ids: Tokenized text tensor of shape (B, S).
            language_mask: Optional boolean padding mask of shape (B, S).
            max_length: Target sequence length to pad or truncate to.

        Returns:
            Tuple of (padded_text_input_ids, padded_language_mask).
        """
        current_length = text_input_ids.shape[1]
        if current_length > max_length:
            text_input_ids = text_input_ids[:, :max_length]
            if language_mask is not None:
                language_mask = language_mask[:, :max_length]
            logging.warning(
                f"Input text length {current_length} exceeds max_length "
                f"{max_length}. Truncating input."
            )
        elif current_length < max_length:
            pad_length = max_length - current_length
            batch_size = text_input_ids.shape[0]
            pad_tensor = torch.zeros(
                (batch_size, pad_length),
                dtype=text_input_ids.dtype,
                device=text_input_ids.device,
            )
            text_input_ids = torch.cat([text_input_ids, pad_tensor], dim=1)
            if language_mask is not None:
                pad_mask = torch.ones(
                    (batch_size, pad_length),
                    dtype=language_mask.dtype,
                    device=language_mask.device,
                )
                language_mask = torch.cat([language_mask, pad_mask], dim=1)
        return text_input_ids, language_mask

    def _build_attention_mask(
        self,
        language_mask: torch.Tensor | None,
        text_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Build integer attention mask from optional boolean padding mask.

        Args:
            language_mask: Optional boolean padding mask (True = padded).
            text_input_ids: Token IDs tensor, used for shape/device if mask is None.

        Returns:
            Long attention mask (1 = attend, 0 = ignore).
        """
        if language_mask is not None:
            attention_mask = ~language_mask
        else:
            attention_mask = torch.ones_like(text_input_ids, dtype=torch.bool)
        return attention_mask.to(torch.long)

    def _build_output_padding_mask(
        self,
        attention_mask: torch.Tensor,
        pooling_method: str,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build output padding mask based on pooling method.

        For sequence outputs (NONE pooling), returns the per-token padding
        mask. For pooled outputs, returns a scalar False per batch item.

        Args:
            attention_mask: Long attention mask (1 = attend, 0 = ignore).
            pooling_method: Pooling strategy from PoolingMethod enum.
            batch_size: Batch size for scalar mask.
            device: Device for the output tensor.

        Returns:
            Boolean padding mask — (B, S) for NONE, (B,) for pooled.
        """
        if pooling_method == PoolingMethod.NONE.value:
            return ~attention_mask.bool()
        else:
            return torch.zeros(batch_size, dtype=torch.bool, device=device)
