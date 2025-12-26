
# mypy: ignore-errors
import timm
import torch
import torch.nn as nn
from timm.layers import freeze_batch_norm_2d

from refactoring.data.constants import Cameras, RGB_CAMERAS
from refactoring.models.encoding.encoders.base import EncoderInput, EncoderOutput
from refactoring.models.encoding.encoders.conditional import ConditionalEncoder
from refactoring.models.encoding.encoders.constants import (
    EncoderOutputKeys,
    PoolingMethod,
    RGBBackboneType, BatchNormHandling,
)
from refactoring.models.layers.convert_layers import replace_batchnorm_with_groupnorm
from refactoring.models.layers.modulation.film_residual_block import FiLMedResBlock
from refactoring.models.layers.pooling.pooling_head import create_pooling_head


class ConditionalCNNEncoder(ConditionalEncoder):
    """CNN encoder with FiLM conditioning for conditioned vision, e.g. from language features."""
    BACKBONE_CONFIGS = {
        RGBBackboneType.RESNET18.value: {
            "layers": [2, 2, 2, 2],
            "feature_dim": 512,
        },
        RGBBackboneType.RESNET34.value: {
            "layers": [3, 4, 6, 3],
            "feature_dim": 512,
        },
    }
    def __init__(
            self,
            input_keys: str | list[str],
            condition_key: str,
            condition_dim: int,
            backbone: str = RGBBackboneType.RESNET18.value,
            pooling_method: str = PoolingMethod.SPATIAL_SOFTMAX.value,
            batch_norm_handling: str = BatchNormHandling.FROZEN.value,
            pretrained: bool = False,
            frozen: bool = False,
    ):
        specification = EncoderInput(keys=input_keys,one_of_groups=[RGB_CAMERAS],
                                     conditioning_key=condition_key)
        super().__init__(input_specification=specification, pretrained=pretrained, frozen=frozen)
        self.condition_key = condition_key
        self.condition_dim = condition_dim
        self.batch_norm_handling = batch_norm_handling
        self.backbone_name = backbone
        self.pooling_method = pooling_method

        if backbone not in self.BACKBONE_CONFIGS:
            raise ValueError(
                f"Backbone {backbone} not supported for FiLM Conditioning. "
                f"Supported: {list(self.BACKBONE_CONFIGS.keys())}"
            )

        self._build_filmed_backbone()
        self.feature_dim = self.BACKBONE_CONFIGS[backbone]["feature_dim"]
        self._setup_pooling()
        if frozen:
            super()._freeze_weights()

    def _build_filmed_backbone(self):
        """Build FiLMed ResNet backbone."""
        config = self.BACKBONE_CONFIGS[self.backbone_name]
        base_model = timm.create_model(self.backbone_name, pretrained=self.pretrained, num_classes=0)
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.act1
        self.maxpool = base_model.maxpool
        self.in_channels = 64
        self.layer1 = self._make_filmed_layer(64, config["layers"][0], stride=1)
        self.layer2 = self._make_filmed_layer(128, config["layers"][1], stride=2)
        self.layer3 = self._make_filmed_layer(256, config["layers"][2], stride=2)
        self.layer4 = self._make_filmed_layer(512, config["layers"][3], stride=2)
        if self.pretrained:
            self._copy_pretrained_weights(base_model)
        self._apply_batch_norm_handling()


    def _apply_batch_norm_handling(self) -> None:
        """Apply BatchNorm handling strategy to all layers."""
        match self.batch_norm_handling:
            case BatchNormHandling.FROZEN.value:
                freeze_batch_norm_2d(self.bn1)
                for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                    for block in layer:
                        block.apply(lambda m: freeze_batch_norm_2d(m) or None)
            case BatchNormHandling.CONVERT_TO_GROUPNORM.value:
                num_channels = self.bn1.num_features
                self.bn1 = nn.GroupNorm(num_channels // 16, num_channels)
                self.layer1 = replace_batchnorm_with_groupnorm(self.layer1)
                self.layer2 = replace_batchnorm_with_groupnorm(self.layer2)
                self.layer3 = replace_batchnorm_with_groupnorm(self.layer3)
                self.layer4 = replace_batchnorm_with_groupnorm(self.layer4)
            case BatchNormHandling.DEFAULT.value:
                pass
            case _:
                raise ValueError(f"Unknown batch norm handling: {self.batch_norm_handling}")


    def _make_filmed_layer(
            self,
            out_channels: int,
            num_blocks: int,
            stride: int
    ) -> nn.ModuleList:
        """Create a layer with FiLMedResBlocks, for FiLM conditioning."""
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            down_layers = [
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
            downsample = nn.Sequential(*down_layers)

        blocks = nn.ModuleList()
        blocks.append(
            FiLMedResBlock(self.in_channels, out_channels, self.condition_dim, stride, downsample)
        )
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            blocks.append(
                FiLMedResBlock(self.in_channels, out_channels, self.condition_dim, stride=1)
            )

        return blocks

    def _copy_pretrained_weights(self, base_model):
        """Copy weights from pretrained ResNet to FiLMedResBlocks where applicable."""
        base_layers = [base_model.layer1, base_model.layer2, base_model.layer3, base_model.layer4]
        self_layers = [self.layer1, self.layer2, self.layer3, self.layer4]

        for base_layer, self_layer in zip(base_layers, self_layers):
            for i, base_block in enumerate(base_layer):
                self_block = self_layer[i]

                self_block.conv1.load_state_dict(base_block.conv1.state_dict())

                if isinstance(self_block.bn1, nn.BatchNorm2d):
                    self_block.bn1.load_state_dict(base_block.bn1.state_dict())

                self_block.conv2.load_state_dict(base_block.conv2.state_dict())

                if isinstance(self_block.bn2, nn.BatchNorm2d):
                    self_block.bn2.load_state_dict(base_block.bn2.state_dict())

                if base_block.downsample is not None and self_block.downsample is not None:
                    self_block.downsample[0].load_state_dict(base_block.downsample[0].state_dict())
                    if isinstance(self_block.downsample[1], nn.BatchNorm2d):
                        self_block.downsample[1].load_state_dict(base_block.downsample[1].state_dict())


    def _setup_pooling(self):
        """Setup pooling layer."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            dummy_condition = torch.zeros(1, self.condition_dim)
            x = self.conv1(dummy_input)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            for block in self.layer1:
                x = block(x, dummy_condition)
            for block in self.layer2:
                x = block(x, dummy_condition)
            for block in self.layer3:
                x = block(x, dummy_condition)
            for block in self.layer4:
                x = block(x, dummy_condition)
            _, c, h, w = x.shape

        mock_pooling_head = create_pooling_head(
            pooling_method=self.pooling_method,
            feature_channels=self.feature_dim,
            spatial_height=h,
            spatial_width=w,
        )
        self.pooling_head = None # Will be created in forward() with correct patch dimensions
        self.output_dim = mock_pooling_head.get_output_dim(self.feature_dim)


    def forward(
            self,
            inputs: dict[str, torch.Tensor],
            conditioning: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass to extract features from images.

        Note:
            Feature dimension size depends on the pooling method used. If no pooling is applied, the raw feature maps are returned and  the output shape will
            be (batch size, channels, height, width) or (batch size, time steps, channels, height, width).
        """
        img = inputs[self.input_specification.keys[0]]
        B, T, C, H, W = None, None, None, None, None
        if img.dim() == 5:
            B, T, C, H, W = img.shape  # Batch, Time, Channels, Height, Width
            img = img.reshape(B * T, C, H, W)
            if conditioning.dim() == 3 and conditioning.shape[1] == T:  # Already (B, T, D)
                conditioning = conditioning.reshape(B * T, -1)
            elif conditioning.dim() == 2:  # (B, D), replicate over T
                conditioning = conditioning.unsqueeze(1).repeat(1, T, 1).reshape(B * T, -1)
            else:
                raise ValueError(f"Unexpected conditioning shape: {conditioning.shape}. Conditioning must be (B, D) or (B, T, D).")
            has_time = True
        else:
            B = img.shape[0]
            has_time = False

        x = self.conv1(img)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        for block in self.layer1:
            x = block(x, conditioning)
        for block in self.layer2:
            x = block(x, conditioning)
        for block in self.layer3:
            x = block(x, conditioning)
        for block in self.layer4:
            x = block(x, conditioning)

        _, _, H_feature_maps, W_feature_maps = x.shape
        if self.pooling_head is None:
            self.pooling_head = create_pooling_head(
                pooling_method=self.pooling_method,
                feature_channels=self.feature_dim,
                spatial_height=H_feature_maps,
                spatial_width=W_feature_maps,
            ).to(self.device)
        pooled_features = self.pooling_head(x)
        if has_time:
            # Reshape back to (B, T, C) or (B, T, C, H, W)
            pooled_features = pooled_features.reshape(B, T, *pooled_features.shape[1:])
        return {EncoderOutputKeys.RGB.value: pooled_features}


    def get_output_specification(self) -> EncoderOutput:
        return EncoderOutput(
            features=[EncoderOutputKeys.RGB.value],
            dimensions={EncoderOutputKeys.RGB.value: self.output_dim},
        )
