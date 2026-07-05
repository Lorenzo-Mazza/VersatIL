"""Shared base class for action decoders that contain a vision-language generative model backbone (e.g. OpenVLA, Pi0, SmolVLA, ...)."""

import torch

from versatil.models.decoding.generative_language_models.vision_language.base import (
    GenerativeVLM,
)


class VLMBackboneDecoderMixin:
    """Mixin for decoders that own a vision-language model backbone."""

    vlm_backbone: GenerativeVLM

    @staticmethod
    def _vlm_decoder_input_keys(
        input_keys: list[str],
        vlm_backbone: GenerativeVLM,
    ) -> list[str]:
        """Return decoder input keys with VLM raw inputs first-class.

        Args:
            input_keys: Additional decoder feature keys.
            vlm_backbone: VLM backbone with an input specification.

        Returns:
            De-duplicated decoder input keys.
        """
        return list(
            dict.fromkeys([*input_keys, *vlm_backbone.input_specification.keys])
        )

    @staticmethod
    def _validate_no_extra_input_keys(
        decoder_name: str,
        input_keys: list[str],
    ) -> None:
        """Validate decoders whose prefix is fully owned by the VLM backbone.

        Args:
            decoder_name: Name used in validation errors.
            input_keys: Additional decoder feature keys from configuration.
        """
        if input_keys:
            raise ValueError(
                f"{decoder_name} builds its prefix from vlm_backbone inputs. "
                f"Set input_keys to an empty list, got {input_keys}."
            )

    def _build_vlm_prefix(
        self,
        features: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Build prefix embeddings with the attached VLM backbone.

        Args:
            features: Raw camera tensors and tokenized text inputs requested by
                ``self.vlm_backbone.input_specification``.

        Returns:
            Prefix embeddings with shape ``(B, P, hidden_dimension)`` and an
            optional padding mask with shape ``(B, P)``.
        """
        return self.vlm_backbone.build_prefix(inputs=features)
