import torch
from timm.layers import freeze_batch_norm_2d
from transformers import TimmBackbone, TimmBackboneConfig
from transformers.modeling_outputs import BackboneOutput

from refactoring.data.constants import Cameras, RGB_CAMERAS
from refactoring.models.encoding.encoders.base import EncoderInput, EncoderOutput
from refactoring.models.encoding.encoders.constants import (
    EncoderOutputKeys,
    PoolingMethod,
    RGBBackboneType,
    BatchNormHandling
)
from refactoring.models.encoding.encoders.unconditional import Encoder
from refactoring.models.layers.convert_layers import replace_batchnorm_with_groupnorm
from refactoring.models.layers.pooling.pooling_head import create_pooling_head


class CNNEncoder(Encoder):
    """Convolutional Neural Network encoder supporting multiple backbones via TIMM."""

    def __init__(
            self,
            input_keys: str | list[str],
            backbone: str = RGBBackboneType.RESNET18.value,
            pooling_method: str = PoolingMethod.AVERAGE.value,
            batch_norm_handling: str = BatchNormHandling.FROZEN.value,
            pretrained: bool = False,
            frozen: bool = False,
    ):
        specification = EncoderInput(keys=input_keys, one_of_groups=[RGB_CAMERAS])
        super().__init__(
            input_specification=specification, pretrained=pretrained, frozen=frozen
        )
        # Validate backbone type at instantiation
        if backbone not in [e.value for e in RGBBackboneType]:
            valid_backbones = [e.value for e in RGBBackboneType if "vit" not in e.value]
            raise ValueError(
                f"Invalid backbone '{backbone}'. Must be one of: {valid_backbones}"
            )
        self.batch_norm_handling = batch_norm_handling
        self.pooling_method = pooling_method
        self.backbone_name = backbone
        self._build_backbone()
        self.feature_dim = self.backbone.num_features[-1]
        self._setup_pooling()
        if frozen:
            super()._freeze_weights()

    def _build_backbone(self):
        """Build backbone using TIMM library."""
        backbone_config = TimmBackboneConfig(
            self.backbone_name,
            use_pretrained_backbone=self.pretrained,
            features_only=True,
        )
        self.backbone = TimmBackbone(config=backbone_config)
        match self.batch_norm_handling:
            case BatchNormHandling.FROZEN.value:
                self.backbone.apply(freeze_batch_norm_2d)  # type: ignore[arg-type]
            case BatchNormHandling.CONVERT_TO_GROUPNORM.value:
                self.backbone = replace_batchnorm_with_groupnorm(self.backbone)
            case BatchNormHandling.DEFAULT.value:
                pass  # keep as-is
            case _:
                raise ValueError(f"Unknown batch norm handling: {self.batch_norm_handling}")

    def _setup_pooling(self):
        """Setup mock pooling head. The actual pooling head will be created in forward()."""
        with torch.no_grad():
            mock_input = torch.zeros(1, 3, 224, 224)
            mock_output: BackboneOutput = self.backbone(mock_input)
            mock_features = mock_output.feature_maps[0]  # type: ignore[index]
            _, c, h, w = mock_features.shape
        mock_pooling_head = create_pooling_head(
            pooling_method=self.pooling_method,
            feature_channels=self.feature_dim,
            spatial_height=h,
            spatial_width=w,
        )
        self.pooling_head = (
            None  # Will be created in forward() with correct patch dimensions
        )
        self.output_dim = mock_pooling_head.get_output_dim(self.feature_dim)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass to extract features from images.

        Args:
            inputs: Dict with single key from input_keys

        Returns:
            A dictionary containing key `RGB_FEATURES` and tensor with shape (batch size, feature dim) or
            (batch size, time steps, feature dim) if input has temporal dimension.

        Note:
            Feature dimension size depends on the pooling method used. If no pooling is applied, the raw feature maps are returned and the output shape will
            be (batch size, channels, height, width) or (batch size, time steps, channels, height, width).
            If pooling is used, the output shape will be (batch size, channels).
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
        _, _, H_feature_maps, W_feature_maps = features.shape
        if self.pooling_head is None:
            self.pooling_head = create_pooling_head(
                pooling_method=self.pooling_method,
                feature_channels=self.feature_dim,
                spatial_height=H_feature_maps,
                spatial_width=W_feature_maps,
            ).to(self.device)
        pooled_features = self.pooling_head(features)
        if has_time:
            # Reshape back to (B, T, C) or (B, T, C, H, W)
            pooled_features = pooled_features.reshape(B, T, *pooled_features.shape[1:])
        return {EncoderOutputKeys.RGB.value: pooled_features}

    def get_output_specification(self) -> EncoderOutput:
        return EncoderOutput(
            features=[EncoderOutputKeys.RGB.value],
            dimensions={EncoderOutputKeys.RGB.value: self.output_dim},
        )
