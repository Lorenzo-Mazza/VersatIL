"""Two-tower VLM encoder with separate vision and language blocks (CLIP-style)."""

import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

from versatil.data.constants import (
    RGB_CAMERAS,
    SampleKey,
)
from versatil.data.metadata import BaseMetadata, CameraMetadata
from versatil.models.encoding.encoders.base import EncoderInput
from versatil.models.encoding.encoders.constants import (
    AttentionImplementation,
    EncoderOutputKeys,
    ImageTextModelType,
    PoolingMethod,
)
from versatil.models.encoding.encoders.image_mixin import (
    RGBEncoderMixin,
    resize_to_target_size,
)
from versatil.models.encoding.encoders.language_mixin import LanguageEncoderMixin
from versatil.models.encoding.encoders.unconditional import Encoder
from versatil.models.feature_meta import (
    FeatureMetadata,
    FeatureType,
    infer_feature_type,
)
from versatil.models.layers.pooling.pooling_head import create_token_pooling_head


class TwoTowerVLMEncoder(LanguageEncoderMixin, RGBEncoderMixin, Encoder):
    """Two-tower VLM encoder with separate vision and language outputs."""

    def __init__(
        self,
        input_keys: str | list[str],
        pretrained: bool,
        frozen: bool,
        pooling_method: str,
        model_name: str = ImageTextModelType.CLIP_VITB32.value,
        attention_type: str = AttentionImplementation.SDPA.value,
        model_dtype: str | None = None,
    ):
        """Initialize the two-tower VLM encoder.

        Args:
            input_keys: Input keys for cameras and tokenized text.
            pretrained: Whether to load pretrained HuggingFace weights.
            frozen: Whether to freeze all encoder weights.
            pooling_method: Feature pooling strategy for vision and language outputs.
            model_name: HuggingFace model identifier for the VLM.
            attention_type: Attention implementation (e.g. SDPA, eager).
            model_dtype: Precision string from experiment config (e.g. ``"bf16-mixed"``).
        """
        if isinstance(input_keys, str):
            input_keys = [input_keys]
        all_keys = list(input_keys) + [
            SampleKey.TOKENIZED_OBSERVATIONS.value,
            SampleKey.IS_PAD_OBSERVATION.value,
        ]
        specification = EncoderInput(
            keys=all_keys,
            at_least_one_of_groups=[RGB_CAMERAS],
            required=[SampleKey.TOKENIZED_OBSERVATIONS.value],
            requires_tokenized=True,
        )
        super().__init__(
            input_specification=specification,
            pretrained=pretrained,
            frozen=frozen,
            model_dtype=model_dtype,
        )
        self._setup_camera_keys(input_keys=self.input_specification.keys)
        self._setup_language_keys(output_modality=EncoderOutputKeys.LANGUAGE.value)
        self.pooling_method = pooling_method
        config = AutoConfig.from_pretrained(model_name)
        if pretrained:
            self.encoder = AutoModel.from_pretrained(
                model_name, attn_implementation=attention_type
            )
        else:
            self.encoder = AutoModel.from_config(
                config, attn_implementation=attention_type
            )
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_name,
            do_rescale=False,
            do_normalize=False,
            do_convert_rgb=False,
            do_resize=False,
        )
        vision_config = self.encoder.vision_model.config
        self.image_size: int | None = getattr(vision_config, "image_size", None)
        self.max_text_length: int = (
            self.encoder.text_model.config.max_position_embeddings
        )
        self.hidden_vision_dim: int = vision_config.hidden_size
        self.hidden_language_dim: int = self.encoder.text_model.config.hidden_size
        vision_has_cls = (
            hasattr(self.encoder.vision_model.embeddings, "class_embedding")
            and self.encoder.vision_model.embeddings.class_embedding is not None
        )
        self.vision_pooling_head = create_token_pooling_head(
            pooling_method=pooling_method,
            input_dimension=self.hidden_vision_dim,
            num_prefix_tokens=1 if vision_has_cls else 0,
        )
        self.language_pooling_head = create_token_pooling_head(
            pooling_method=pooling_method,
            input_dimension=self.hidden_language_dim,
            sequence_length=self.max_text_length,
            num_prefix_tokens=0,
        )
        self.output_vision_dim = self.vision_pooling_head.output_dim
        self.output_language_dim = self.language_pooling_head.output_dim
        self.output_padding_mask_dim = (
            (self.max_text_length,) if pooling_method == PoolingMethod.NONE.value else 1
        )
        if frozen:
            super()._freeze_weights()
        self._apply_model_dtype()

    def _pool_features(
        self,
        outputs: BaseModelOutputWithPooling,
        modality: str,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool extracted features from encoder outputs.

        Args:
            outputs: HuggingFace model output containing hidden states and pooler output.
            modality: Modality key determining which pooling head to use.
            padding_mask: Optional token padding mask where ``True`` means padded.

        Returns:
            Pooled feature tensor.

        Raises:
            RuntimeError: If encoder outputs are missing required fields.
        """
        if outputs.pooler_output is None or outputs.last_hidden_state is None:
            raise RuntimeError("Encoder outputs are missing required fields.")
        if self.pooling_method == PoolingMethod.DEFAULT.value:
            return outputs.pooler_output  # HF model's built-in pooler
        if modality == EncoderOutputKeys.RGB.value:
            return self.vision_pooling_head(outputs.last_hidden_state)
        else:
            return self.language_pooling_head(
                outputs.last_hidden_state,
                padding_mask=padding_mask,
            )

    def _encode_single_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a single camera's images through the vision tower.

        Args:
            images: Image tensor of shape (B, C, H, W).

        Returns:
            Pooled vision features.
        """
        if self.image_size is not None:
            images = resize_to_target_size(
                images=images,
                target_height=self.image_size,
                target_width=self.image_size,
            )
        vision_inputs = self.image_processor(images=images, return_tensors="pt")
        # Processor uses 'pixel_attention_mask', vision_model expects 'attention_mask'. Huggingface inconstitency.
        if "pixel_attention_mask" in vision_inputs:
            vision_inputs["attention_mask"] = vision_inputs.pop("pixel_attention_mask")
        vision_output = self.encoder.vision_model(**vision_inputs)
        return self._pool_features(vision_output, modality=EncoderOutputKeys.RGB.value)

    def encode(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Encode images and text through the two-tower VLM.

        Args:
            inputs: Dict with camera images as (B, C, H, W) per camera key,
                tokenized text as (B, S), and optional padding mask.

        Returns:
            Dict with per-camera RGB features, language features, and padding mask.
        """
        text_input_ids, language_mask = self._extract_text_inputs(inputs=inputs)
        batch_size = text_input_ids.shape[0]
        text_input_ids, language_mask = self._pad_text_inputs(
            text_input_ids=text_input_ids,
            language_mask=language_mask,
            max_length=self.max_text_length,
        )
        attention_mask = self._build_attention_mask(
            language_mask=language_mask, text_input_ids=text_input_ids
        )
        text_output = self.encoder.text_model(
            input_ids=text_input_ids,
            attention_mask=attention_mask,
        )
        language_features = self._pool_features(
            text_output,
            modality=EncoderOutputKeys.LANGUAGE.value,
            padding_mask=~attention_mask.bool(),
        )
        token_padding_mask = self._build_output_padding_mask(
            attention_mask=attention_mask,
            pooling_method=self.pooling_method,
            batch_size=batch_size,
            device=language_features.device,
            num_prefix_tokens=self.language_pooling_head.num_prefix_tokens,
        )
        result = {
            EncoderOutputKeys.LANGUAGE.value: language_features,
            self.padding_mask_name: token_padding_mask,
        }
        result.update(self._encode_vision(inputs))
        return result

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
                    f"TwoTowerVLMEncoder cannot process image data for '{key}'. "
                    f"Got CameraMetadata, expected tokenized text input."
                )
        return None

    def get_output_specification(self) -> list[FeatureMetadata]:
        """Return the output feature names and dimensions for this encoder.

        Returns:
            List of FeatureMetadata with per-camera vision features, language features,
            and the language padding mask.
        """
        vision_names = self._get_vision_feature_names()
        vision_dim = (
            (self.output_vision_dim,)
            if isinstance(self.output_vision_dim, int)
            else self.output_vision_dim
        )
        language_dim = (
            (self.output_language_dim,)
            if isinstance(self.output_language_dim, int)
            else self.output_language_dim
        )
        padding_dim = (
            (self.output_padding_mask_dim,)
            if isinstance(self.output_padding_mask_dim, int)
            else self.output_padding_mask_dim
        )
        result = [
            FeatureMetadata(
                key=name,
                feature_type=infer_feature_type(vision_dim),
                dimension=vision_dim,
            )
            for name in vision_names
        ]
        result.append(
            FeatureMetadata(
                key=EncoderOutputKeys.LANGUAGE.value,
                feature_type=infer_feature_type(language_dim),
                dimension=language_dim,
            )
        )
        result.append(
            FeatureMetadata(
                key=self.padding_mask_name,
                feature_type=FeatureType.FLAT.value,
                dimension=padding_dim,
            )
        )
        return result

    def get_vocab_size(self) -> int:
        """Get the vocabulary size of the text encoder.

        Returns:
            Vocabulary size of the language model component
        """
        return self.encoder.text_model.config.vocab_size
