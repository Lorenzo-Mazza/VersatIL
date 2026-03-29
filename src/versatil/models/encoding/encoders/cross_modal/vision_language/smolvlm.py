"""SmolVLM/Idefics3 encoder — decomposes the VLM into vision tower + LM for multi-camera support."""

import torch
from transformers import AutoConfig, AutoModel

from versatil.data.constants import (
    RGB_CAMERAS,
    SampleKey,
)
from versatil.data.metadata import BaseMetadata, CameraMetadata
from versatil.models.encoding.encoders.base import EncoderInput
from versatil.models.encoding.encoders.constants import (
    AttentionImplementation,
    EncoderOutputKeys,
    SmolVLMModelType,
)
from versatil.models.encoding.encoders.image_mixin import resize_to_target_size
from versatil.models.encoding.encoders.language_mixin import LanguageEncoderMixin
from versatil.models.encoding.encoders.unconditional import Encoder
from versatil.models.feature_meta import FeatureMetadata, FeatureType


class SmolVLMEncoder(LanguageEncoderMixin, Encoder):
    """SmolVLM/Idefics3 encoder that decomposes the VLM into separate tower calls.

    Encodes camera images through SigLIP + connector, embeds language tokens
    via the SmolLM embedding layer, concatenates all embeddings, and runs the
    SmolLM to produce contextualized features. Natively supports multiple
    cameras via the num_images dimension in pixel_values.
    """

    def __init__(
        self,
        input_keys: str | list[str],
        pretrained: bool,
        frozen: bool,
        model_name: str = SmolVLMModelType.SMOLVLM_256M.value,
        attention_type: str = AttentionImplementation.SDPA.value,
        use_embeddings_only: bool = False,
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
        self.num_image_tokens_per_camera: int = self._compute_num_image_tokens(
            config=config
        )
        self.max_text_length: int = config.text_config.max_position_embeddings
        self.use_embeddings_only = use_embeddings_only
        if frozen:
            super()._freeze_weights()

    @staticmethod
    def _compute_num_image_tokens(config: AutoConfig) -> int:
        """Compute the number of image tokens per camera from patch grid and scale factor.

        Args:
            config: HuggingFace model config with vision_config and scale_factor.

        Returns:
            Number of image tokens per camera image.
        """
        num_patches_per_side = (
            config.vision_config.image_size // config.vision_config.patch_size
        )
        num_patches = num_patches_per_side * num_patches_per_side
        scale_factor = config.scale_factor
        return num_patches // (scale_factor * scale_factor)

    @property
    def total_image_tokens(self) -> int:
        """Total image tokens across all cameras."""
        return self.num_image_tokens_per_camera * len(self.camera_keys)

    def _embed_images(self, camera_images: list[torch.Tensor]) -> torch.Tensor:
        """Encode multiple camera images through SigLIP + connector.

        Stacks camera images along the num_images dimension and processes them
        through the vision tower in a single call (native Idefics3 multi-image).

        Args:
            camera_images: List of image tensors, each (B, C, H, W).

        Returns:
            Image embeddings of shape (B, total_image_tokens, hidden_dim).
        """
        # Stack cameras along num_images dim: (B, num_cameras, C, H, W)
        num_images_dim = 1
        pixel_values = torch.stack(camera_images, dim=num_images_dim)
        image_features = self.vlm.get_image_features(pixel_values)
        return image_features.pooler_output

    def _embed_language(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Convert pre-tokenized observation IDs to embeddings via the SmolLM embedding table.

        The data pipeline's ObservationTokenizer produces integer token IDs.
        This method maps them to dense vectors through the LM's embedding layer.

        Args:
            token_ids: Pre-tokenized observation IDs of shape (B, S), as integers.

        Returns:
            Token embeddings of shape (B, S, hidden_dim).
        """
        return self.vlm.text_model.get_input_embeddings()(token_ids)

    def encode(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Encode multi-camera images + text through decomposed SmolVLM.

        Camera images are stacked along the num_images dimension and encoded
        through SigLIP + connector in one call. Language tokens are embedded
        via the SmolLM embedding layer. All embeddings are concatenated and
        run through the SmolLM for cross-modal contextualization.

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
        camera_images = []
        for camera_key in self.camera_keys:
            camera_images.append(
                resize_to_target_size(
                    images=inputs[camera_key],
                    target_height=self.image_size,
                    target_width=self.image_size,
                )
            )
        image_embeddings = self._embed_images(camera_images)
        language_embeddings = self._embed_language(text_input_ids)
        inputs_embeds = torch.cat([image_embeddings, language_embeddings], dim=1)
        image_pad_mask = torch.zeros(
            batch_size,
            self.total_image_tokens,
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
        text_attention_mask = self._build_attention_mask(
            language_mask=language_mask, text_input_ids=text_input_ids
        )
        text_pad_mask = ~text_attention_mask.bool()
        full_padding_mask = torch.cat([image_pad_mask, text_pad_mask], dim=1)
        full_attention_mask = (~full_padding_mask).to(torch.long)
        if self.use_embeddings_only:
            fused_features = inputs_embeds
        else:
            lm_outputs = self.vlm.text_model(
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
                    f"SmolVLMEncoder cannot process image data for '{key}'. "
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
        """Get the vocabulary size of the SmolLM text model."""
        return self.vlm.text_model.config.vocab_size
