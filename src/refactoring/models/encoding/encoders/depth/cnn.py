
import torch
from transformers import TimmBackbone, TimmBackboneConfig
from transformers.modeling_outputs import BackboneOutput

from refactoring.data.constants import Cameras
from refactoring.models.encoding.encoders.base import EncoderInput, EncoderOutput
from refactoring.models.encoding.encoders.constants import (
    EncoderOutputKeys,
    PoolingMethod,
    RGBBackboneType,
)
from refactoring.models.encoding.encoders.unconditional import Encoder
from refactoring.models.layers.convert_layers import replace_batchnorm_with_groupnorm
from refactoring.models.layers.pooling.pooling_head import create_pooling_head


class DepthCNNEncoder(Encoder):
    """Convolutional Neural Network encoder supporting multiple backbones via TIMM for depth images."""
    def __init__(
            self,
            input_keys: str | list[str],
            backbone: str = RGBBackboneType.RESNET18.value,
            pooling_method: str = PoolingMethod.AVERAGE.value,
            use_group_norm: bool = True,
            pretrained: bool = False,
            frozen: bool = False,
    ):
        specification = EncoderInput(keys=input_keys,required=[Cameras.DEPTH.value])
        super().__init__(input_specification=specification, pretrained=pretrained, frozen=frozen)
        self.use_group_norm = use_group_norm
        self.pooling_method = pooling_method
        self.backbone_name = backbone
        self._build_backbone()
        self.feature_dim = self.backbone.num_features[-1]
        self._setup_pooling()
        if frozen:
            super()._freeze_weights()


    def _build_backbone(self):
        """Build backbone using TIMM library."""
        backbone_config = TimmBackboneConfig(self.backbone_name, use_pretrained_backbone=self.pretrained, features_only=True, num_channels=1)
        self.backbone = TimmBackbone(config=backbone_config)
        if self.use_group_norm:
            self.backbone = replace_batchnorm_with_groupnorm(self.backbone)  # type: ignore[assignment]


    def _setup_pooling(self):
        """Setup pooling layer based on configuration."""
        with torch.no_grad():
            mock_input = torch.zeros(1, 1, 224, 224)
            mock_output: BackboneOutput = self.backbone(mock_input)
            mock_features = mock_output.feature_maps[-1]  # type: ignore[index]
            _, c, h, w = mock_features.shape
        self.pooling_head = create_pooling_head(
            pooling_method=self.pooling_method,
            feature_channels=self.feature_dim,
            spatial_height=h,
            spatial_width=w,
        )
        self.output_dim = self.pooling_head.get_output_dim(self.feature_dim)


    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass to extract features from images.

        Args:
            inputs: Dict with single key from input_keys

        Returns:
            A dictionary containing key `DEPTH_FEATURES` and tensor with shape (batch size, feature dim) or
            (batch size, time steps, feature dim) if input has temporal dimension.

        Note:
            Feature dimension size depends on the pooling method used. If no pooling is applied, the raw feature maps are returned and  the output shape will
            be (batch size, channels, height, width) or (batch size, time steps, channels, height, width).
        """
        img = inputs[self.input_specification.keys[0]]
        T = None
        if img.dim() == 5:
            B, T, C, H, W = img.shape  # Batch, Time, Channels, Height, Width
            img = img.reshape(B * T, C, H, W)
            has_time = True
        else:
            B = img.shape[0]
            has_time = False
        backbone_output = self.backbone(img)
        features = backbone_output.feature_maps[-1]
        pooled_features = self.pooling_head(features)
        if has_time:
            pooled_features = pooled_features.reshape(B, T, *pooled_features.shape[1:])  # Batch, Time, Features
        return {EncoderOutputKeys.DEPTH.value: pooled_features}


    def get_output_specification(self) -> EncoderOutput:
        return EncoderOutput(
            features=[EncoderOutputKeys.DEPTH.value],
            dimensions={EncoderOutputKeys.DEPTH.value: self.output_dim},
        )
