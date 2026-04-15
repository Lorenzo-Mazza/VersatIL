"""PaliGemma encoder — decomposes the VLM into vision tower + LM for multi-camera support."""

import math

import torch
import torch.nn as nn
from transformers import AutoConfig

from versatil.models.encoding.encoders.constants import (
    AttentionImplementation,
    PaliGemmaModelType,
)
from versatil.models.encoding.encoders.cross_modal.vision_language.generative_vlm import (
    GenerativeVLMEncoder,
)
from versatil.models.encoding.encoders.image_mixin import resize_to_target_size


class PaliGemmaEncoder(GenerativeVLMEncoder):
    """PaliGemma encoder with per-camera sequential image encoding.

    Each camera image is encoded through SigLIP + multi-modal projector
    separately, scaled by ``sqrt(hidden_dim)`` (Gemma convention), then
    concatenated with language embeddings before the Gemma LM pass.
    """

    def __init__(
        self,
        input_keys: str | list[str],
        pretrained: bool,
        frozen: bool,
        model_name: str = PaliGemmaModelType.PALIGEMMA2_3B_224.value,
        attention_type: str = AttentionImplementation.SDPA.value,
        use_embeddings_only: bool = False,
        model_dtype: str | None = None,
        max_text_length: int | None = None,
    ):
        """Initialize the PaliGemma encoder.

        Args:
            input_keys: Input keys for cameras and tokenized text.
            pretrained: Whether to load pretrained HuggingFace weights.
            frozen: Whether to freeze all encoder weights.
            model_name: HuggingFace model identifier for PaliGemma.
            attention_type: Attention implementation (e.g. SDPA, eager).
            use_embeddings_only: If True, return raw image + language embeddings
                without running the LM. The LM layers remain available for
                external use (e.g. interleaved expert decoders).
            model_dtype: Precision string from experiment config (e.g. ``"bf16-mixed"``).
            max_text_length: Maximum text sequence length. Defaults to model's
                max_position_embeddings if None.
        """
        super().__init__(
            input_keys=input_keys,
            pretrained=pretrained,
            frozen=frozen,
            model_name=model_name,
            attention_type=attention_type,
            use_embeddings_only=use_embeddings_only,
            model_dtype=model_dtype,
            max_text_length=max_text_length,
        )

    def _compute_num_image_tokens(self, config: AutoConfig) -> int:
        """PaliGemma stores num_image_tokens directly in the vision config."""
        return config.vision_config.num_image_tokens

    def _get_language_model(self) -> nn.Module:
        """PaliGemma wraps Gemma2Model as ``vlm.language_model``."""
        return self.vlm.language_model

    def _embed_images(
        self, inputs: dict[str, torch.Tensor], batch_size: int
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Encode each camera through SigLIP + projector sequentially.

        Args:
            inputs: Dict with camera images as (B, C, H, W) per camera key.
            batch_size: Batch size.

        Returns:
            (image_embeddings, image_pad_masks) — one tensor per camera.
        """
        image_embeddings = []
        image_pad_masks = []
        for camera_key in self.camera_keys:
            images = resize_to_target_size(
                images=inputs[camera_key],
                target_height=self.image_size,
                target_width=self.image_size,
            )
            features = self.vlm.get_image_features(images)
            camera_embeddings = features.pooler_output * math.sqrt(self.hidden_dim)
            image_embeddings.append(camera_embeddings)
            image_pad_masks.append(
                torch.zeros(
                    batch_size,
                    self.num_image_tokens_per_camera,
                    dtype=torch.bool,
                    device=camera_embeddings.device,
                )
            )
        return image_embeddings, image_pad_masks

    def _scale_language_embeddings(
        self, language_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Gemma applies sqrt(hidden_size) scaling after embedding lookup."""
        return language_embeddings * math.sqrt(self.hidden_dim)
