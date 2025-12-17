
import torch
from transformers import AutoConfig, AutoModel
from transformers.models.timm_wrapper.modeling_timm_wrapper import (
    TimmWrapperModelOutput,
)

from refactoring.data.constants import RGB_CAMERAS
from refactoring.models.encoding.encoders.base import EncoderInput, EncoderOutput
from refactoring.models.encoding.encoders.constants import (
    EncoderOutputKeys,
    PoolingMethod,
    RGBBackboneType,
)
from refactoring.models.encoding.encoders.unconditional import Encoder
from refactoring.models.layers import LearnedAggregation


class ViTEncoder(Encoder):
    def __init__(
            self,
            input_keys: str | list[str],
            pretrained: bool,
            frozen: bool,
            pooling_method: str,
            backbone: str = RGBBackboneType.DINOV2_VITB14.value,
    ):
        """Vision Transformer (ViT) encoder using Transformers library and TIMM."""
        specification = EncoderInput(keys=input_keys,one_of_groups=[RGB_CAMERAS])
        super().__init__(input_specification=specification, pretrained=pretrained, frozen=frozen)
        if backbone not in [e.value for e in RGBBackboneType]:
            valid_backbones = [e.value for e in RGBBackboneType if not any(
                x in e.value for x in ["efficientnet", "resnet", "edgenext"])]
            raise ValueError(
                f"Invalid backbone '{backbone}'. Must be one Vision Transformer of the following: {valid_backbones}"
            )

        self.pooling_method = pooling_method
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
        """Set-up pooling head and output dimensionality accordingly."""
        if self.pooling_method == PoolingMethod.LEARNED_AGGREGATION.value:
            self.pooling_head = LearnedAggregation(self.feature_dim-1) # feature_dim-1 to exclude CLS token
        if self.pooling_method == PoolingMethod.NONE.value:
            self.output_dim = (-1, self.feature_dim-1) # -1 indicates the variable seq dimension, feature_dim-1 to exclude CLS token
        else:
            self.output_dim = self.feature_dim


    def _extract_features(self, outputs: TimmWrapperModelOutput) -> torch.Tensor:
        """Pool extracted features using the encoder pooling head."""
        last_hidden_state = outputs.last_hidden_state
        if self.pooling_method == PoolingMethod.DEFAULT.value:
            return last_hidden_state[:, 0] # CLS token
        elif self.pooling_method == PoolingMethod.AVERAGE.value:
            return last_hidden_state[:, 1:].mean(dim=1)  # GAP on patches (exclude CLS)
        elif self.pooling_method == PoolingMethod.LEARNED_AGGREGATION.value:
            if self.pooling_head is None:
                raise RuntimeError("pooling_head must be initialized for LEARNED_AGGREGATION")
            result: torch.Tensor = self.pooling_head(last_hidden_state[..., 1:]) # Learned aggregation on patches
            return result
        elif self.pooling_method == PoolingMethod.NONE.value:
            return last_hidden_state[:, 1:]  # Return all patch tokens (exclude CLS)
        else:
            raise ValueError(f"Unknown feature extraction method: {self.pooling_method}")

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
            features = features.reshape(B, T, *features.shape[1:])  # B, T, Emb or B, T, Seq, Emb
        return {EncoderOutputKeys.RGB.value: features}

    def get_output_specification(self) -> EncoderOutput:
        return EncoderOutput(
            features=[EncoderOutputKeys.RGB.value],
            dimensions={EncoderOutputKeys.RGB.value: self.output_dim},
        )
