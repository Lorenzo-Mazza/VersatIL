"""PaliGemma VLM component for VLA decoders."""

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.cache_utils import Cache

from versatil.models.adaptation.lora import LoRAAdaptation
from versatil.models.decoding.generative_language_models.base import (
    CausalLanguageModelOutput,
)
from versatil.models.decoding.generative_language_models.constants import (
    PaliGemmaModelType,
)
from versatil.models.decoding.generative_language_models.vision_language.huggingface import (
    HuggingFaceGenerativeVLM,
)
from versatil.models.encoding.encoders.constants import AttentionImplementation
from versatil.models.encoding.encoders.image_mixin import resize_to_target_size
from versatil.models.encoding.explainability import (
    ActivationLayout,
    ExplanationTargetKind,
    VisionExplanationTarget,
)


class PaliGemmaVLM(HuggingFaceGenerativeVLM):
    """PaliGemma VLM with per-camera sequential image encoding.

    Each camera image is encoded through SigLIP + multi-modal projector
    separately, then concatenated with language embeddings before the Gemma
    language-model pass. Scaling follows the HF reference: text embeddings are
    scaled by ``sqrt(hidden_dimension)`` inside Gemma's embedding module, image
    tokens enter unscaled.
    """

    def __init__(
        self,
        input_keys: str | list[str],
        pretrained: bool,
        frozen: bool,
        model_name: str = PaliGemmaModelType.PALIGEMMA2_3B_224.value,
        attention_type: str = AttentionImplementation.SDPA.value,
        model_dtype: str | None = None,
        max_text_length: int | None = None,
        lora_config: LoRAAdaptation | None = None,
    ):
        """Initialize the PaliGemma VLM component.

        Args:
            input_keys: Input keys for cameras and tokenized text.
            pretrained: Whether to load pretrained HuggingFace weights.
            frozen: Whether to freeze all model weights.
            model_name: HuggingFace model identifier for PaliGemma.
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
        """PaliGemma stores num_image_tokens directly in the vision config."""
        return config.vision_config.num_image_tokens

    def _get_language_model(self) -> nn.Module:
        """Return the nested Gemma2 language model."""
        return self._get_paligemma_model().language_model

    def _get_paligemma_model(self) -> nn.Module:
        """Return the inner PaliGemma model under optional PEFT wrapping."""
        if self.lora_config is not None and self.lora_config.enabled:
            return self.vlm.model.model
        return self.vlm.model

    def forward_language_model(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | tuple[tuple[torch.Tensor, ...], ...] | None = None,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        output_hidden_states: bool = True,
    ) -> CausalLanguageModelOutput:
        """Run the Gemma language tower with PaliGemma 1-indexed positions."""
        paligemma_position_ids = None
        if position_ids is not None:
            paligemma_position_ids = position_ids + 1
        return super().forward_language_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_ids=paligemma_position_ids,
            output_hidden_states=output_hidden_states,
        )

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
            # The HF PaliGemma reference inserts projector outputs into the
            # prefix unscaled; only text embeddings carry the Gemma
            # sqrt(hidden) scale, applied inside GemmaTextScaledWordEmbedding.
            camera_embeddings = features.pooler_output
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

    def get_explainability_targets(self) -> list[VisionExplanationTarget]:
        """Return the projector output used as PaliGemma visual context.

        PaliGemma encodes each camera independently with SigLIP and projects
        patch tokens into the Gemma hidden size through ``multi_modal_projector``.
        For multi-camera inputs this layer is invoked once per camera, and the
        base VLM ``is_multi_camera`` flag lets the explainability runner select
        the requested camera invocation.

        Returns:
            One token-sequence target over PaliGemma image tokens.
        """
        return [
            VisionExplanationTarget(
                layer=self._get_paligemma_model().multi_modal_projector,
                target_kind=ExplanationTargetKind.TOKEN_SEQUENCE.value,
                activation_layout=ActivationLayout.NLC.value,
                patch_grid=self._get_image_token_grid(),
            )
        ]
