
import torch
from transformers import AutoModel, AutoProcessor
from transformers.modeling_outputs import BaseModelOutputWithPooling

from refactoring.data.constants import LANGUAGE_KEY, Cameras
from refactoring.models.encoding.encoders.base import EncoderInput, EncoderOutput
from refactoring.models.encoding.encoders.constants import (
    AttentionImplementation,
    EncoderOutputKeys,
    FeatureExtractionMethod,
    ImageTextModelType,
)
from refactoring.models.encoding.encoders.unconditional import Encoder
from refactoring.models.layers import LearnedAggregation


class VLMEncoder(Encoder):
    """Vision-Language Model (VLM) encoder using Transformers API."""
    def __init__(
            self,
            input_keys: str | list[str],
            pretrained: bool,
            frozen: bool,
            feature_extraction_method: str,
            model_name: str = ImageTextModelType.CLIP_VITB32,
            attention_type: str = AttentionImplementation.SDPA.value,
    ):
        specification = EncoderInput(keys=input_keys, one_of_groups=[[Cameras.LEFT.value, Cameras.RIGHT.value]], required=[LANGUAGE_KEY])
        super().__init__(input_specification=specification, pretrained=pretrained, frozen=frozen)
        self.camera_key = next(key for key in self.input_specification.keys if key in self.input_specification.one_of_groups[0])
        self.language_key = specification.required[0]
        self.feature_extraction_method = feature_extraction_method
        self.num_register_tokens = 0
        self.pooling_head: LearnedAggregation | None = None
        self.encoder = AutoModel.from_pretrained(model_name, device_map="auto",
                                                 attn_implementation=attention_type, use_safetensors=True)
        self.processor = AutoProcessor.from_pretrained(model_name, do_rescale=False, do_normalize=False,
                                                       do_convert_rgb=False, do_center_crop=False, do_resize=False)
        vision_config = self.encoder.vision_model.config
        self.requires_fixed_size = hasattr(vision_config, 'image_size') and vision_config.image_size is not None
        if self.requires_fixed_size:
            self.image_size = vision_config.image_size
        else:
            self.image_size = None  # Flexible size (e.g., SigLIP naflex)
        self.feature_dim = vision_config.projection_dim if hasattr(vision_config, 'projection_dim') else vision_config.hidden_size
        self._setup_pooling()
        if frozen:
            super()._freeze_weights()



    def _setup_pooling(self):
        if self.feature_extraction_method == FeatureExtractionMethod.LEARNED_AGGREGATION.value:
            self.pooling_head = LearnedAggregation(self.feature_dim)


    def _extract_features(self, outputs: BaseModelOutputWithPooling) -> torch.Tensor:
        if self.feature_extraction_method == FeatureExtractionMethod.CLS_TOKEN.value:
            if outputs.pooler_output is None:
                raise RuntimeError("pooler_output must be present in model output")
            return outputs.pooler_output
        elif self.feature_extraction_method == FeatureExtractionMethod.AVERAGE_PATCH_TOKENS.value:
            if outputs.last_hidden_state is None:
                raise RuntimeError("last_hidden_state must be present in model output")
            return outputs.last_hidden_state[:, 1:].mean(dim=1)  # GAP on patches (exclude CLS)
        elif self.feature_extraction_method == FeatureExtractionMethod.LEARNED_AGGREGATION.value:
            if self.pooling_head is None:
                raise RuntimeError("pooling_head must be initialized for LEARNED_AGGREGATION")
            if outputs.last_hidden_state is None:
                raise RuntimeError("last_hidden_state must be present in model output")
            result: torch.Tensor = self.pooling_head(outputs.last_hidden_state[:, 1:])  # Learned agg on tokens (exclude CLS)
            return result
        else:
            raise ValueError(f"Unsupported feature extraction method: {self.feature_extraction_method}")


    def _resize_images(self, images: torch.Tensor) -> torch.Tensor:
        """Resize images manually to the expected input size of the encoder.

        Note:
          This enables the user to pass in images of arbitrary size, rather than being
          constrained to the size expected by the pretrained model. The internal resizing
          uses bicubic interpolation.
        """
        result: torch.Tensor = torch.nn.functional.interpolate(images, size=(self.image_size, self.image_size), mode='bicubic', align_corners=False)
        return result

    def forward(self, inputs: dict[str, torch.Tensor | list[list[str]] | list[str]]) -> dict[str, torch.Tensor]:  # type: ignore[override]
        images = inputs[self.camera_key]
        if not isinstance(images, torch.Tensor):
            raise ValueError("images must be a tensor")
        language_instruction: list[list[str]] | list[str] = inputs[self.language_key]  # type: ignore[assignment]
        T = None
        if images.dim() == 5:
            B, T, C, H, W = images.shape
            images = images.reshape(B * T, C, H, W)
            # Flatten the time dimension of the language instruction
            language_instruction = [time for batch in language_instruction for time in batch]
            has_time = True
        else:
            B = images.shape[0]
            has_time = False
        images = self._resize_images(images) if self.requires_fixed_size else images
        inputs = self.processor(
            text=language_instruction,
            images=images,
            return_tensors="pt",
            padding=True
        ).to(images.device)
        outputs = self.encoder(**inputs)
        image_features = outputs["image_embeds"]
        language_features = outputs["text_embeds"]
        if has_time:
            if T is None:
                raise RuntimeError("T must be set when has_time is True")
            image_features = image_features.reshape(B, T, -1)
            language_features = language_features.reshape(B, T, -1)
        return {EncoderOutputKeys.RGB.value: image_features, EncoderOutputKeys.LANGUAGE.value: language_features}


    def get_output_specification(self) -> EncoderOutput:
        return EncoderOutput(
            features=[EncoderOutputKeys.RGB.value, EncoderOutputKeys.LANGUAGE.value],
            dimensions={EncoderOutputKeys.LANGUAGE.value: self.feature_dim,
                        EncoderOutputKeys.RGB.value: self.feature_dim},
        )
