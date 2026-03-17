import logging

import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

from versatil.data.constants import (
    RGB_CAMERAS,
    SampleKey,
)
from versatil.models.encoding.encoders.base import EncoderInput, EncoderOutput
from versatil.models.encoding.encoders.constants import (
    AttentionImplementation,
    EncoderOutputKeys,
    ImageTextModelType,
    PoolingMethod,
)
from versatil.models.encoding.encoders.unconditional import Encoder
from versatil.models.layers import LearnedAggregation


class VLMEncoder(Encoder):
    """Vision-Language Model (VLM) encoder using Transformers API."""

    def __init__(
        self,
        input_keys: str | list[str],
        pretrained: bool,
        frozen: bool,
        pooling_method: str,
        model_name: str = ImageTextModelType.CLIP_VITB32.value,
        attention_type: str = AttentionImplementation.SDPA.value,
    ):
        specification = EncoderInput(
            keys=input_keys,
            one_of_groups=[RGB_CAMERAS],
            required=[SampleKey.TOKENIZED_OBSERVATIONS.value],
            requires_tokenized=True,
        )
        super().__init__(
            input_specification=specification, pretrained=pretrained, frozen=frozen
        )
        self.camera_key = next(
            key
            for key in self.input_specification.keys
            if key in self.input_specification.one_of_groups[0]
        )
        self.language_key = specification.required[0]
        self.pooling_method = pooling_method
        self.num_register_tokens = 0
        self.pooling_head: LearnedAggregation | None = None
        config = AutoConfig.from_pretrained(model_name)
        if pretrained:
            self.encoder = AutoModel.from_pretrained(
                model_name, attn_implementation=attention_type
            )
        else:
            self.encoder = AutoModel.from_config(
                config, attn_implementation=attention_type
            )
        vision_config = self.encoder.vision_model.config
        self.requires_fixed_size = (
            hasattr(vision_config, "image_size")
            and vision_config.image_size is not None
        )
        if self.requires_fixed_size:
            self.image_size = vision_config.image_size
        else:
            self.image_size = None  # Flexible size (e.g., SigLIP naflex)
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_name,
            do_rescale=False,
            do_normalize=False,
            do_convert_rgb=False,
            do_center_crop=False,
            do_resize=False,
        )
        self.max_text_length = self.encoder.text_model.config.max_position_embeddings
        self.hidden_vision_dim = vision_config.hidden_size
        self.hidden_language_dim = self.encoder.text_model.config.hidden_size
        self.padding_mask_name = (
            f"{EncoderOutputKeys.LANGUAGE.value}_{EncoderOutputKeys.PADDING_MASK.value}"
        )
        self._setup_pooling()
        if frozen:
            super()._freeze_weights()

    def _setup_pooling(self):
        """Set-up pooling heads and output dimensionality accordingly."""
        if self.pooling_method == PoolingMethod.LEARNED_AGGREGATION.value:
            self.pooling_heads = torch.nn.ModuleDict(
                {
                    EncoderOutputKeys.RGB.value: LearnedAggregation(
                        self.hidden_vision_dim
                    ),
                    EncoderOutputKeys.LANGUAGE.value: LearnedAggregation(
                        self.hidden_language_dim
                    ),
                }
            )

        if self.pooling_method == PoolingMethod.NONE.value:
            self.output_vision_dim = (
                -1,
                self.hidden_vision_dim,
            )  # -1 indicates the variable sequence dimension
            self.output_language_dim = (self.max_text_length, self.hidden_language_dim)
            self.output_padding_mask_dim = (self.max_text_length,)
        else:
            self.output_vision_dim = self.hidden_vision_dim
            self.output_language_dim = self.hidden_language_dim
            self.output_padding_mask_dim = 1

    def _pool_features(
        self, outputs: BaseModelOutputWithPooling, modality: str
    ) -> torch.Tensor:
        """Pool extracted features using the encoder pooling heads."""
        if outputs.pooler_output is None or outputs.last_hidden_state is None:
            raise RuntimeError("Encoder outputs are missing required fields.")
        if self.pooling_method == PoolingMethod.DEFAULT.value:
            return outputs.pooler_output
        elif self.pooling_method == PoolingMethod.AVERAGE.value:
            if modality == EncoderOutputKeys.RGB.value:
                return outputs.last_hidden_state[:, 1:].mean(
                    dim=1
                )  # GAP on patches (exclude CLS)
            else:
                return outputs.last_hidden_state.mean(dim=1)
        elif self.pooling_method == PoolingMethod.NONE.value:
            return outputs.last_hidden_state
        elif self.pooling_method == PoolingMethod.LEARNED_AGGREGATION.value:
            if self.pooling_heads is None:
                raise RuntimeError(
                    "pooling_head must be initialized for LEARNED_AGGREGATION"
                )
            result: torch.Tensor = self.pooling_heads[modality](
                outputs.last_hidden_state
            )
            return result
        else:
            raise ValueError(
                f"Unsupported feature extraction method: {self.pooling_method}"
            )

    def _resize_images(self, images: torch.Tensor) -> torch.Tensor:
        """Resize images manually to the expected input size of the encoder.

        Note:
          This enables the user to pass in images of arbitrary size, rather than being
          constrained to the size expected by the pretrained model. The internal resizing
          uses bicubic interpolation.
        """
        if images.ndim != 4:
            raise ValueError(f"(b,c,h,w) expected, but {images.shape}")
        cur_height, cur_width = images.shape[2:]
        target_size = self.image_size
        ratio = max(cur_width / target_size, cur_height / target_size)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)
        resized_img = torch.nn.functional.interpolate(
            images,
            size=(resized_height, resized_width),
            mode="bicubic",
            align_corners=False,
        )
        pad_height = max(0, target_size - resized_height)
        pad_width = max(0, target_size - resized_width)
        padded_img = torch.nn.functional.pad(
            resized_img, (pad_width, 0, pad_height, 0), value=0.0
        )
        return padded_img

    def _pad_text_inputs(
        self, text_input_ids: torch.Tensor, language_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Pad or truncate text inputs to max_text_length."""
        if text_input_ids.shape[1] > self.max_text_length:
            text_input_ids = text_input_ids[:, : self.max_text_length]
            if language_mask is not None:
                language_mask = language_mask[:, : self.max_text_length]
            logging.warning(
                f"Input text length {text_input_ids.shape[1]} exceeds max_text_length "
                f"{self.max_text_length}. Truncating input."
            )
        elif text_input_ids.shape[1] < self.max_text_length:
            pad_length = self.max_text_length - text_input_ids.shape[1]
            pad_tensor = torch.zeros(
                (text_input_ids.shape[0], pad_length),
                dtype=text_input_ids.dtype,
                device=text_input_ids.device,
            )
            text_input_ids = torch.cat([text_input_ids, pad_tensor], dim=1)
            if language_mask is not None:
                pad_mask = torch.ones(
                    (language_mask.shape[0], pad_length),
                    dtype=language_mask.dtype,
                    device=language_mask.device,
                )
                language_mask = torch.cat([language_mask, pad_mask], dim=1)
            logging.warning(
                f"Input text length {text_input_ids.shape[1]} less than max_text_length "
                f"{self.max_text_length}. Padding input."
            )
        return text_input_ids, language_mask

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        images = inputs[self.camera_key]
        if not isinstance(images, torch.Tensor):
            raise ValueError("images must be a tensor")

        if self.language_key not in inputs:
            raise ValueError(
                f"VLM encoder expects pre-tokenized input. "
                f"Expected key '{self.language_key}' not found in inputs. "
                f"Ensure tokenization is enabled in DataloaderConfig."
            )

        text_input_ids = inputs[self.language_key]
        if not isinstance(text_input_ids, torch.Tensor):
            raise ValueError("tokenized_observations must be a tensor")

        language_mask = inputs.get(SampleKey.IS_PAD_OBSERVATION.value)

        T = None
        if images.dim() == 5:
            B, T, C, H, W = images.shape
            images = images.reshape(B * T, C, H, W)
            text_input_ids = text_input_ids.reshape(B * T, -1)
            language_mask = (
                language_mask.reshape(B * T, -1) if language_mask is not None else None
            )
            has_time = True
        else:
            B = images.shape[0]
            has_time = False
        images = self._resize_images(images) if self.requires_fixed_size else images
        text_input_ids, language_mask = self._pad_text_inputs(
            text_input_ids, language_mask
        )
        if not self.requires_fixed_size:
            # SigLIP naflex requires padding
            images = self.image_processor(
                images=images, return_tensors="pt", padding=True
            ).to(images.device)
        else:
            images = self.image_processor(
                images=images,
                return_tensors="pt",
            ).to(images.device)

        # Create attention mask from padding mask
        if language_mask is not None:
            attention_mask = ~language_mask
        else:
            attention_mask = torch.ones_like(text_input_ids, dtype=torch.bool)

        attention_mask = attention_mask.to(torch.long)
        encoder_inputs = {
            "input_ids": text_input_ids,
            "attention_mask": attention_mask,
            **images,
        }

        outputs = self.encoder(**encoder_inputs)
        image_features = self._pool_features(
            outputs.vision_model_output, modality=EncoderOutputKeys.RGB.value
        )
        language_features = self._pool_features(
            outputs.text_model_output, modality=EncoderOutputKeys.LANGUAGE.value
        )
        token_padding_mask = ~attention_mask  # bool, True where padded
        if has_time:
            if self.pooling_method == PoolingMethod.NONE.value:
                if image_features.ndim < 3 or language_features.ndim != 3:
                    raise RuntimeError(
                        f"Expected image_features.ndim >= 3 and language_features.ndim == 3, "
                        f"got {image_features.ndim} and {language_features.ndim}"
                    )
                vision_seq_len = image_features.shape[1]
                image_features = image_features.view(
                    B, T, vision_seq_len, self.hidden_vision_dim
                )
                language_features = language_features.view(
                    B, T, self.max_text_length, self.hidden_language_dim
                )
                token_padding_mask = token_padding_mask.view(B, T, self.max_text_length)
            else:
                if image_features.ndim != 2 or language_features.ndim != 2:
                    raise RuntimeError(
                        f"Expected image_features.ndim == 2 and language_features.ndim == 2, "
                        f"got {image_features.ndim} and {language_features.ndim}"
                    )
                image_features = image_features.view(B, T, self.hidden_vision_dim)
                language_features = language_features.view(
                    B, T, self.hidden_language_dim
                )
                token_padding_mask = torch.zeros(
                    B, T, dtype=torch.bool, device=image_features.device
                )
        else:
            if self.pooling_method != PoolingMethod.NONE.value:
                token_padding_mask = torch.zeros(
                    B, dtype=torch.bool, device=image_features.device
                )
        return {
            EncoderOutputKeys.RGB.value: image_features,
            EncoderOutputKeys.LANGUAGE.value: language_features,
            self.padding_mask_name: token_padding_mask,
        }

    def get_output_specification(self) -> EncoderOutput:
        return EncoderOutput(
            features=[
                EncoderOutputKeys.RGB.value,
                EncoderOutputKeys.LANGUAGE.value,
                self.padding_mask_name,
            ],
            dimensions={
                EncoderOutputKeys.RGB.value: self.output_vision_dim,
                EncoderOutputKeys.LANGUAGE.value: self.output_language_dim,
                self.padding_mask_name: self.output_padding_mask_dim,
            },
        )

    def get_vocab_size(self) -> int:
        """Get the vocabulary size of the text encoder.

        Returns:
            Vocabulary size of the language model component
        """
        return self.encoder.text_model.config.vocab_size
