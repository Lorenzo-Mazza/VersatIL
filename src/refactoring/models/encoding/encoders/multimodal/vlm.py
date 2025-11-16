
import torch
from transformers import AutoModel, AutoProcessor, AutoConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling

from refactoring.data.constants import LANGUAGE_KEY, Cameras
from refactoring.models.encoding.encoders.base import EncoderInput, EncoderOutput
from refactoring.models.encoding.encoders.constants import (
    AttentionImplementation,
    EncoderOutputKeys,
    PoolingMethod,
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
        config = AutoConfig.from_pretrained(model_name)
        if pretrained:
            self.encoder = AutoModel.from_pretrained(model_name,attn_implementation=attention_type, use_safetensors=True)
        else:
            self.encoder = AutoModel.from_config(config, attn_implementation=attention_type)
        self.processor = AutoProcessor.from_pretrained(model_name, do_rescale=False, do_normalize=False,
                                                       do_convert_rgb=False, do_center_crop=False, do_resize=False)
        vision_config = self.encoder.vision_model.config
        self.requires_fixed_size = hasattr(vision_config, 'image_size') and vision_config.image_size is not None
        if self.requires_fixed_size:
            self.image_size = vision_config.image_size
        else:
            self.image_size = None  # Flexible size (e.g., SigLIP naflex)
        self.hidden_vision_dim = vision_config.hidden_size
        self.hidden_language_dim = self.encoder.text_model.config.hidden_size
        self._setup_pooling()
        if frozen:
            super()._freeze_weights()



    def _setup_pooling(self):
        if self.feature_extraction_method == PoolingMethod.LEARNED_AGGREGATION.value:
            self.pooling_heads = torch.nn.ModuleDict({
                EncoderOutputKeys.RGB.value: LearnedAggregation(self.hidden_vision_dim),
                EncoderOutputKeys.LANGUAGE.value: LearnedAggregation(self.hidden_language_dim)
            })


    def _extract_features(self, outputs: BaseModelOutputWithPooling, modality: str) -> torch.Tensor:
        if outputs.pooler_output is None or outputs.last_hidden_state is None:
            raise RuntimeError("Encoder outputs are missing required fields.")
        if self.feature_extraction_method == PoolingMethod.DEFAULT.value:
            return outputs.pooler_output
        elif self.feature_extraction_method == PoolingMethod.AVERAGE.value:
            return outputs.last_hidden_state.mean(dim=1)  # GAP on patches (exclude CLS)
        elif self.feature_extraction_method == PoolingMethod.NONE.value:
            return outputs.last_hidden_state
        elif self.feature_extraction_method == PoolingMethod.LEARNED_AGGREGATION.value:
            if self.pooling_heads is None:
                raise RuntimeError("pooling_head must be initialized for LEARNED_AGGREGATION")
            result: torch.Tensor = self.pooling_heads[modality](outputs.last_hidden_state)
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
        if images.ndim != 4:
            raise ValueError(f"(b,c,h,w) expected, but {images.shape}")
        cur_height, cur_width = images.shape[2:]
        target_size = self.image_size
        ratio = max(cur_width / target_size, cur_height / target_size)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)
        resized_img = torch.nn.functional.interpolate(
            images, size=(resized_height, resized_width), mode='bicubic', align_corners=False
        )
        pad_height = max(0, target_size - resized_height)
        pad_width = max(0, target_size - resized_width)
        # Pad with zeros
        padded_img = torch.nn.functional.pad(resized_img, (pad_width, 0, pad_height, 0), value=-1)
        return padded_img


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
        image_features = self._extract_features(outputs.vision_model_output, modality=EncoderOutputKeys.RGB.value)
        language_features = self._extract_features(outputs.text_model_output, modality=EncoderOutputKeys.LANGUAGE.value)
        if has_time:
            if T is None:
                raise RuntimeError("T must be set when has_time is True")
            image_features = image_features.reshape(B, T, *image_features.shape[1:])
            language_features = language_features.reshape(B, T, *language_features.shape[1:])
        return {EncoderOutputKeys.RGB.value: image_features, EncoderOutputKeys.LANGUAGE.value: language_features}


    def get_output_specification(self) -> EncoderOutput:
        return EncoderOutput(
            features=[EncoderOutputKeys.RGB.value, EncoderOutputKeys.LANGUAGE.value],
            dimensions={EncoderOutputKeys.RGB.value: self.hidden_vision_dim,
                        EncoderOutputKeys.LANGUAGE.value: self.hidden_language_dim},
        )
