
import torch
from transformers import AutoConfig, AutoModel
from transformers.models.timm_wrapper.modeling_timm_wrapper import (
    TimmWrapperModelOutput,
)

from refactoring.data.constants import Cameras
from refactoring.models.encoding.encoders.base import EncoderInput, EncoderOutput
from refactoring.models.encoding.encoders.constants import (
    EncoderOutputKeys,
    PoolingMethod,
    RGBBackboneType,
)
from refactoring.models.encoding.encoders.unconditional import Encoder
from refactoring.models.layers import LearnedAggregation


class ViTEncoder(Encoder):
    ONE_OF_GROUPS = [[Cameras.LEFT.value, Cameras.RIGHT.value]]
    def __init__(
            self,
            input_keys: str | list[str],
            pretrained: bool,
            frozen: bool,
            feature_extraction_method: str,
            backbone: str = RGBBackboneType.DINOV2_VITB14.value,
    ):
        """Vision Transformer (ViT) encoder using Transformers library and TIMM models."""
        specification = EncoderInput(keys=input_keys,one_of_groups=[[Cameras.LEFT.value, Cameras.RIGHT.value]])
        super().__init__(input_specification=specification, pretrained=pretrained, frozen=frozen)
        self.feature_extraction_method = feature_extraction_method
        self.num_register_tokens = 0
        self.pooling_head: LearnedAggregation | None = None
        self.backbone_name = backbone
        self._build_backbone()
        self.feature_dim = self.backbone.config.num_features
        self._setup_feature_extractor()
        if frozen:
            super()._freeze_weights()


    def _build_backbone(self):
        """Build backbone using Transformer library."""
        config = AutoConfig.from_pretrained(self.backbone_name)
        config.model_args = {"dynamic_img_size": True} # Allows the underlying timm model to accept input images of arbitrary sizes
        if self.pretrained:
            self.backbone = AutoModel.from_pretrained(self.backbone_name, config=config, use_safetensors=True)
        else:
            self.backbone = AutoModel.from_config(config)


    def _setup_feature_extractor(self):
        """Setup feature extraction layer based on configuration."""
        if self.feature_extraction_method == PoolingMethod.LEARNED_AGGREGATION.value:
            self.pooling_head = LearnedAggregation(self.feature_dim)


    def _extract_features(self, outputs: TimmWrapperModelOutput) -> torch.Tensor:
        last_hidden_state = outputs.last_hidden_state
        if self.feature_extraction_method == PoolingMethod.DEFAULT.value:
            return last_hidden_state[:, 0] # CLS token
        elif self.feature_extraction_method == PoolingMethod.AVERAGE.value:
            return last_hidden_state[:, 1:].mean(dim=1)  # GAP on patches (exclude CLS)
        elif self.feature_extraction_method == PoolingMethod.LEARNED_AGGREGATION.value:
            if self.pooling_head is None:
                raise RuntimeError("pooling_head must be initialized for LEARNED_AGGREGATION")
            result: torch.Tensor = self.pooling_head(last_hidden_state[:, 1]) # Learned aggregation on patches
            return result
        elif self.feature_extraction_method == PoolingMethod.NONE.value:
            return last_hidden_state[:, 1:]  # Return all patch tokens (exclude CLS)
        else:
            raise ValueError(f"Unknown feature extraction method: {self.feature_extraction_method}")

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass to extract features from images.

        Args:
            inputs: Dict with single key from input_keys

        Returns:
            A dictionary containing key `RGB_FEATURES` and tensor with shape (batch size, *feature dim) or
            (batch size, time steps, *feature dim) if input has temporal dimension.
        """
        img = inputs[self.input_specification.keys[0]]
        T = None
        if img.dim() == 5:
            B, T, C, H, W = img.shape
            img = img.reshape(B * T, C, H, W)
            has_time = True
        else:
            B = img.shape[0]
            has_time = False
        outputs = self.backbone(img)
        features = self._extract_features(outputs)
        if has_time:
            if T is None:
                raise RuntimeError("T must be set when has_time is True")
            features = features.reshape(B, T, *features.shape[1:])  # Batch, Time, Features
        return {EncoderOutputKeys.RGB.value: features}

    def get_output_specification(self) -> EncoderOutput:
        return EncoderOutput(
            features=[EncoderOutputKeys.RGB.value],
            dimensions={EncoderOutputKeys.RGB.value: self.feature_dim},
        )
