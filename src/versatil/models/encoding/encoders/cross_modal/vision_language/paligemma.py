"""PaliGemma encoder — decomposes the VLM into vision tower + LM for multi-camera support."""

import math

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PretrainedConfig

from versatil.data.constants import (
    RGB_CAMERAS,
    SampleKey,
)
from versatil.data.metadata import BaseMetadata, CameraMetadata
from versatil.models.encoding.encoders.base import EncoderInput
from versatil.models.encoding.encoders.constants import (
    AttentionImplementation,
    EncoderOutputKeys,
    PaliGemmaModelType,
)
from versatil.models.encoding.encoders.image_mixin import resize_to_target_size
from versatil.models.encoding.encoders.language_mixin import LanguageEncoderMixin
from versatil.models.encoding.encoders.unconditional import Encoder
from versatil.models.feature_meta import FeatureMetadata, FeatureType


class PaliGemmaEncoder(LanguageEncoderMixin, Encoder):
    """PaliGemma encoder that decomposes the VLM into separate tower calls.

    Encodes each camera image through SigLIP + multi-modal projector separately,
    embeds language tokens via the Gemma embedding layer, concatenates all
    embeddings, and runs the Gemma LM to produce contextualized features.

    Supports multiple cameras: each image is encoded through the shared vision
    tower sequentially, producing a single fused output sequence.
    """

    def __init__(
        self,
        input_keys: str | list[str],
        pretrained: bool,
        frozen: bool,
        model_name: str = PaliGemmaModelType.PALIGEMMA2_3B_224.value,
        attention_type: str = AttentionImplementation.SDPA.value,
        use_embeddings_only: bool = False,
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
        """
        specification = EncoderInput(
            keys=input_keys,
            at_least_one_of_groups=[RGB_CAMERAS],
            required=[SampleKey.TOKENIZED_OBSERVATIONS.value],
            requires_tokenized=True,
        )
        super().__init__(
            input_specification=specification, pretrained=pretrained, frozen=frozen
        )
        self.camera_keys = [
            key for key in self.input_specification.keys if key in RGB_CAMERAS
        ]
        self._setup_language_keys(
            output_modality=EncoderOutputKeys.FUSED_RGB_LANGUAGE.value
        )
        self.model_name = model_name
        config = AutoConfig.from_pretrained(model_name)
        if pretrained:
            self.vlm = AutoModel.from_pretrained(
                model_name, attn_implementation=attention_type
            )
        else:
            self.vlm = AutoModel.from_config(config, attn_implementation=attention_type)
        self.image_size: int = config.vision_config.image_size
        self.hidden_dim: int = config.text_config.hidden_size
        self.num_image_tokens_per_camera: int = config.vision_config.num_image_tokens
        self.max_text_length: int = config.text_config.max_position_embeddings
        self.use_embeddings_only = use_embeddings_only
        if frozen:
            super()._freeze_weights()

    @property
    def total_image_tokens(self) -> int:
        """Total image tokens across all cameras."""
        return int(self.num_image_tokens_per_camera * len(self.camera_keys))

    def _embed_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images through SigLIP vision tower + multi-modal projector.

        Args:
            images: Image tensor of shape (B, C, H, W).

        Returns:
            Projected image embeddings of shape (B, num_image_tokens, hidden_dim).
        """
        image_features = self.vlm.get_image_features(images)
        # Gemma applies hidden_size**0.5 scaling after embedding lookup in its forward().
        return image_features.pooler_output * math.sqrt(self.hidden_dim)

    def _embed_language(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Convert pre-tokenized observation IDs to embeddings via the Gemma embedding table.

        The data pipeline's ObservationTokenizer produces integer token IDs.
        This method maps them to dense vectors through the LM's embedding layer.

        Args:
            token_ids: Pre-tokenized observation IDs of shape (B, S), as integers.

        Returns:
            Token embeddings of shape (B, S, hidden_dim).
        """
        return self.vlm.language_model.embed_tokens(token_ids)

    def encode(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Encode multi-camera images + text through decomposed PaliGemma.

        Each camera image is encoded through SigLIP separately. Language tokens
        are embedded via the Gemma embedding layer. All embeddings are concatenated
        and run through the Gemma LM for cross-modal contextualization.

        Args:
            inputs: Dict with camera images as (B, C, H, W) per camera key,
                tokenized text as (B, S), and optional padding mask.

        Returns:
            Dict with fused features and padding mask.
        """
        text_input_ids, language_mask = self._extract_text_inputs(inputs=inputs)
        batch_size = text_input_ids.shape[0]
        text_input_ids, language_mask = self._pad_text_inputs(
            text_input_ids=text_input_ids,
            language_mask=language_mask,
            max_length=self.max_text_length,
        )
        image_embeddings = []
        image_pad_masks = []
        for camera_key in self.camera_keys:
            images = resize_to_target_size(
                images=inputs[camera_key],
                target_height=self.image_size,
                target_width=self.image_size,
            )
            camera_embeddings = self._embed_image(images)
            image_embeddings.append(camera_embeddings)
            image_pad_masks.append(
                torch.zeros(
                    batch_size,
                    self.num_image_tokens_per_camera,
                    dtype=torch.bool,
                    device=camera_embeddings.device,
                )
            )
        language_embeddings = self._embed_language(text_input_ids)
        # Gemma applies hidden_size**0.5 scaling after embedding lookup in its forward().
        # See GemmaModel.forward() in transformers.
        language_embeddings = language_embeddings * math.sqrt(self.hidden_dim)
        all_embeddings = image_embeddings + [language_embeddings]
        inputs_embeds = torch.cat(all_embeddings, dim=1)
        attention_mask = self._build_attention_mask(
            language_mask=language_mask, text_input_ids=text_input_ids
        )
        text_pad_mask = ~attention_mask.bool()
        full_padding_mask = torch.cat(image_pad_masks + [text_pad_mask], dim=1)
        full_attention_mask = (~full_padding_mask).to(torch.long)
        if self.use_embeddings_only:
            fused_features = inputs_embeds
        else:
            lm_outputs = self.vlm.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
            )
            fused_features = lm_outputs.last_hidden_state
        return {
            EncoderOutputKeys.FUSED_RGB_LANGUAGE.value: fused_features,
            self.padding_mask_name: full_padding_mask,
        }

    def validate_input_metadata(self, key: str, metadata: BaseMetadata) -> str | None:
        """Validate that camera keys have RGB metadata and non-camera keys are not images.

        Args:
            key: Observation key being validated.
            metadata: Metadata from the observation space for this key.

        Returns:
            Error message if incompatible, None if valid.
        """
        if key in self.camera_keys:
            if not isinstance(metadata, CameraMetadata):
                return f"Expected CameraMetadata for '{key}', got {type(metadata).__name__}"
            if not metadata.is_rgb:
                return (
                    f"Expected 3-channel RGB for '{key}', "
                    f"got {metadata.channels} channels"
                )
        else:
            if isinstance(metadata, CameraMetadata):
                return (
                    f"PaliGemmaEncoder cannot process image data for '{key}'. "
                    f"Got CameraMetadata, expected tokenized text input."
                )
        return None

    def get_output_specification(self) -> list[FeatureMetadata]:
        """Return the output feature names and dimensions for this encoder.

        Returns:
            List of FeatureMetadata with fused RGB-language features and padding mask.
        """
        total_sequence_length = self.total_image_tokens + self.max_text_length
        return [
            FeatureMetadata(
                key=EncoderOutputKeys.FUSED_RGB_LANGUAGE.value,
                feature_type=FeatureType.SEQUENTIAL.value,
                dimension=(total_sequence_length, self.hidden_dim),
            ),
            FeatureMetadata(
                key=self.padding_mask_name,
                feature_type=FeatureType.FLAT.value,
                dimension=(total_sequence_length,),
            ),
        ]

    def get_vocab_size(self) -> int:
        """Get the vocabulary size of the Gemma language model."""
        return self.vlm.language_model.config.vocab_size

    def get_backbone_layers(self) -> nn.ModuleList:
        """Return the Gemma LM transformer layers for interleaved decoding."""
        return self.vlm.language_model.model.layers

    def get_rotary_embedding(self) -> nn.Module:
        """Return the Gemma RoPE module."""
        return self.vlm.language_model.model.rotary_emb

    def get_backbone_hidden_dim(self) -> int:
        """Return the Gemma LM hidden dimension."""
        return self.hidden_dim

    def get_text_config(self) -> PretrainedConfig:
        """Return the Gemma LM text config for expert creation."""
        return self.vlm.language_model.config
