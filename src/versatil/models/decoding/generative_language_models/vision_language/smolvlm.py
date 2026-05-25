"""SmolVLM/Idefics3 component for VLA decoders."""

import math

import torch
import torch.nn as nn
from transformers import AutoConfig

from versatil.models.adaptation.lora import LoRAAdaptation
from versatil.models.decoding.generative_language_models.constants import (
    SmolVLMModelType,
)
from versatil.models.decoding.generative_language_models.vision_language.huggingface import (
    HuggingFaceGenerativeVLM,
)
from versatil.models.encoding.encoders.constants import AttentionImplementation
from versatil.models.encoding.encoders.image_mixin import resize_to_target_size


class SmolVLM(HuggingFaceGenerativeVLM):
    """SmolVLM/Idefics3 component with native multi-image support.

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
        model_dtype: str | None = None,
        max_text_length: int | None = None,
        lora_config: LoRAAdaptation | None = None,
    ):
        """Initialize the SmolVLM component.

        Args:
            input_keys: Input keys for cameras and tokenized text.
            pretrained: Whether to load pretrained HuggingFace weights.
            frozen: Whether to freeze all model weights.
            model_name: HuggingFace model identifier for SmolVLM.
            attention_type: Attention implementation (e.g. SDPA, eager).
            model_dtype: Precision string from experiment config (e.g. ``"bf16-mixed"``).
            max_text_length: Maximum text sequence length. Defaults to model's
                max_position_embeddings if None.
            lora_config: Optional LoRA adapter configuration.
        """
        super().__init__(
            input_keys=input_keys,
            pretrained=pretrained,
            frozen=frozen,
            model_name=model_name,
            attention_type=attention_type,
            model_dtype=model_dtype,
            max_text_length=max_text_length,
            lora_config=lora_config,
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

    def _scale_language_embeddings(
        self, language_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Scale language embeddings by sqrt(hidden_dim) to match SmolVLM convention."""
        return language_embeddings * math.sqrt(self.hidden_dim)

    def _embed_images(
        self, inputs: dict[str, torch.Tensor], batch_size: int
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Stack all cameras along num_images dim and encode in a single call.

        Stacks N camera images into (B, N, C, H, W), passes through the vision
        encoder which returns (B*N, tokens_per_camera, hidden_dim), then reshapes
        back to (B, N*tokens_per_camera, hidden_dim).

        Args:
            inputs: Dict with camera images as (B, C, H, W) per camera key.
            batch_size: Batch size.

        Returns:
            ([embeddings], [pad_mask]) where embeddings is
            (B, N*tokens_per_camera, hidden_dim) and pad_mask is
            (B, N*tokens_per_camera).
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
        num_cameras = len(self.camera_keys)
        pixel_values = torch.stack(camera_images, dim=1)  # (B, num_cameras, C, H, W)
        image_features = self.vlm.get_image_features(pixel_values)
        # pooler_output: (B * num_cameras, tokens_per_camera, hidden_dim)
        image_embeddings = image_features.pooler_output
        # SmolVLM convention: scale embeddings by sqrt(hidden_dim) to match the
        # magnitude of the LM's token embeddings (which use the same scaling).
        image_embeddings = image_embeddings * math.sqrt(image_embeddings.shape[-1])
        # Merge camera and token dims back into batch: (B, num_cameras * tokens_per_camera, hidden_dim)
        tokens_per_camera = image_embeddings.shape[1]
        image_embeddings = image_embeddings.reshape(
            batch_size, num_cameras * tokens_per_camera, image_embeddings.shape[2]
        )
        image_pad_mask = torch.zeros(
            batch_size,
            self.total_image_tokens,
            dtype=torch.bool,
            device=image_embeddings.device,
        )
        return [image_embeddings], [image_pad_mask]
