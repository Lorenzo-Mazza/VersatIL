"""SmolVLM/Idefics3 encoder — decomposes the VLM into vision tower + LM for multi-camera support."""

import torch
import torch.nn as nn
from transformers import AutoConfig

from versatil.models.encoding.encoders.constants import (
    AttentionImplementation,
    SmolVLMModelType,
)
from versatil.models.encoding.encoders.cross_modal.vision_language.generative_vlm import (
    GenerativeVLMEncoder,
)
from versatil.models.encoding.encoders.image_mixin import resize_to_target_size


class SmolVLMEncoder(GenerativeVLMEncoder):
    """SmolVLM/Idefics3 encoder with native multi-image support.

    Camera images are stacked along the ``num_images`` dimension and processed
    through SigLIP + connector in a single call, then concatenated with
    language embeddings before the SmolLM pass.
    """

    def __init__(
        self,
        input_keys: str | list[str],
        pretrained: bool,
        frozen: bool,
        model_name: str = SmolVLMModelType.SMOLVLM_256M.value,
        attention_type: str = AttentionImplementation.SDPA.value,
        use_embeddings_only: bool = False,
        model_dtype: str | None = None,
    ):
        """Initialize the SmolVLM encoder.

        Args:
            input_keys: Input keys for cameras and tokenized text.
            pretrained: Whether to load pretrained HuggingFace weights.
            frozen: Whether to freeze all encoder weights.
            model_name: HuggingFace model identifier for SmolVLM.
            attention_type: Attention implementation (e.g. SDPA, eager).
            use_embeddings_only: If True, return raw image + language embeddings
                without running the LM. The LM layers remain available for
                external use (e.g. interleaved expert decoders).
            model_dtype: Precision string from experiment config (e.g. ``"bf16-mixed"``).
        """
        super().__init__(
            input_keys=input_keys,
            pretrained=pretrained,
            frozen=frozen,
            model_name=model_name,
            attention_type=attention_type,
            use_embeddings_only=use_embeddings_only,
            model_dtype=model_dtype,
        )

    def _compute_num_image_tokens(self, config: AutoConfig) -> int:
        """Idefics3 computes image tokens from patch grid and scale factor."""
        num_patches_per_side = (
            config.vision_config.image_size // config.vision_config.patch_size
        )
        num_patches = num_patches_per_side * num_patches_per_side
        scale_factor = config.scale_factor
        return num_patches // (scale_factor * scale_factor)

    def _get_language_model(self) -> nn.Module:
        """SmolVLM wraps a Llama-style model as ``vlm.text_model``."""
        return self.vlm.text_model

    def _embed_images(
        self, inputs: dict[str, torch.Tensor], batch_size: int
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Stack all cameras along num_images dim and encode in a single call.

        Args:
            inputs: Dict with camera images as (B, C, H, W) per camera key.
            batch_size: Batch size.

        Returns:
            ([combined_embeddings], [combined_pad_mask]) — single-element lists
            since all cameras are processed together.
        """
        camera_images = []
        for camera_key in self.camera_keys:
            camera_images.append(
                resize_to_target_size(
                    images=inputs[camera_key],
                    target_height=self.image_size,
                    target_width=self.image_size,
                )
            )
        pixel_values = torch.stack(camera_images, dim=1)
        image_features = self.vlm.get_image_features(pixel_values)
        image_embeddings = image_features.pooler_output
        image_pad_mask = torch.zeros(
            batch_size,
            self.total_image_tokens,
            dtype=torch.bool,
            device=image_embeddings.device,
        )
        return [image_embeddings], [image_pad_mask]
