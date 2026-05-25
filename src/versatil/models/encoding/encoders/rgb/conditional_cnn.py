"""FiLM-conditioned CNN encoder for conditioned vision encoding."""

import timm
import torch
import torch.nn as nn
from timm.layers import freeze_batch_norm_2d

from versatil.data.constants import CameraModality
from versatil.data.metadata import BaseMetadata, CameraMetadata
from versatil.models.adaptation.lora import LoRAAdaptation, apply_lora_config
from versatil.models.encoding.encoders.base import EncoderInput
from versatil.models.encoding.encoders.conditional import ConditionalEncoder
from versatil.models.encoding.encoders.constants import (
    BatchNormHandling,
    EncoderOutputKeys,
    PoolingMethod,
    SpatialBackboneType,
)
from versatil.models.encoding.encoders.image_mixin import RGBEncoderMixin
from versatil.models.feature_meta import FeatureMetadata, infer_feature_type
from versatil.models.layers.convert_layers import replace_batchnorm_with_groupnorm
from versatil.models.layers.modulation.film_residual_block import FiLMedResBlock
from versatil.models.layers.pooling.pooling_head import (
    PoolingHead,
    create_spatial_pooling_head,
)


class _ConditionalBackbone(nn.Module):
    """Backbone module that applies conditioned residual layers."""

    def __init__(
        self,
        conv1: nn.Module,
        bn1: nn.Module,
        activation: nn.Module,
        maxpool: nn.Module,
        layer1: nn.ModuleList,
        layer2: nn.ModuleList,
        layer3: nn.ModuleList,
        layer4: nn.ModuleList,
    ) -> None:
        """Initialize the retained conditional backbone.

        Args:
            conv1: Initial convolution module.
            bn1: Normalization module after ``conv1``.
            activation: Activation module after ``bn1``.
            maxpool: Initial pooling module.
            layer1: First conditioned residual block stage.
            layer2: Second conditioned residual block stage.
            layer3: Third conditioned residual block stage.
            layer4: Fourth conditioned residual block stage.
        """
        super().__init__()
        self.conv1 = conv1
        self.bn1 = bn1
        self.activation = activation
        self.maxpool = maxpool
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4

    def forward(self, images: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """Run the conditioned backbone.

        Args:
            images: Image tensor with shape ``(B, C, H, W)``.
            conditioning: Conditioning tensor with shape ``(B, D)``.

        Returns:
            Spatial feature tensor with shape ``(B, C_out, H_out, W_out)``.
        """
        features = self.conv1(images)
        features = self.bn1(features)
        features = self.activation(features)
        features = self.maxpool(features)
        for block in self.layer1:
            features = block(features, conditioning)
        for block in self.layer2:
            features = block(features, conditioning)
        for block in self.layer3:
            features = block(features, conditioning)
        for block in self.layer4:
            features = block(features, conditioning)
        return features


class ConditionalCNNEncoder(RGBEncoderMixin, ConditionalEncoder):
    """CNN encoder with FiLM conditioning for conditioned vision, e.g. from language features."""

    BACKBONE_CONFIGS = {
        SpatialBackboneType.RESNET18.value: {
            "layers": [2, 2, 2, 2],
            "feature_dim": 512,
        },
        SpatialBackboneType.RESNET34.value: {
            "layers": [3, 4, 6, 3],
            "feature_dim": 512,
        },
    }

    def __init__(
        self,
        input_keys: str | list[str],
        condition_key: str,
        condition_dim: int,
        backbone: str = SpatialBackboneType.RESNET18.value,
        pooling_method: str = PoolingMethod.SPATIAL_SOFTMAX.value,
        batch_norm_handling: str = BatchNormHandling.FROZEN.value,
        pretrained: bool = False,
        frozen: bool = False,
        model_dtype: str | None = None,
        lora_config: LoRAAdaptation | None = None,
    ) -> None:
        """Initialize FiLM-conditioned CNN encoder.

        Args:
            input_keys: Camera observation keys.
            condition_key: Key for the conditioning feature tensor.
            condition_dim: Dimensionality of the conditioning feature.
            backbone: timm ResNet model name.
            pooling_method: Feature pooling strategy.
            batch_norm_handling: How to handle batch normalization layers.
            pretrained: Whether to load pretrained weights.
            frozen: Whether to freeze all parameters.
            model_dtype: Precision string from experiment config (e.g. ``"bf16-mixed"``).
            lora_config: Optional PEFT LoRA adapter configuration.
        """
        specification = EncoderInput(
            keys=input_keys,
            required_camera_modalities=[CameraModality.RGB],
            conditioning_key=condition_key,
        )
        super().__init__(
            input_specification=specification,
            pretrained=pretrained,
            frozen=frozen,
            model_dtype=model_dtype,
        )
        self._setup_camera_keys(input_keys=self.input_specification.keys)
        self.condition_key = condition_key
        self.condition_dim = condition_dim
        self.batch_norm_handling = batch_norm_handling
        self.backbone_name = backbone
        self.pooling_method = pooling_method
        self.lora_config = lora_config

        if backbone not in self.BACKBONE_CONFIGS:
            raise ValueError(
                f"Backbone {backbone} not supported for FiLM Conditioning. "
                f"Supported: {list(self.BACKBONE_CONFIGS.keys())}"
            )

        self._build_filmed_backbone()
        self.feature_dim = self.BACKBONE_CONFIGS[backbone]["feature_dim"]
        self.pooling_head: PoolingHead | None = None
        self.output_dim: int | tuple[int, ...] = self.feature_dim
        if frozen:
            super()._freeze_weights()
        self._apply_model_dtype()

    def _build_filmed_backbone(self) -> None:
        """Build FiLMed ResNet backbone."""
        config = self.BACKBONE_CONFIGS[self.backbone_name]
        base_model = timm.create_model(
            self.backbone_name, pretrained=self.pretrained, num_classes=0
        )

        self.in_channels = 64
        layer1 = self._make_filmed_layer(64, config["layers"][0], stride=1)
        layer2 = self._make_filmed_layer(128, config["layers"][1], stride=2)
        layer3 = self._make_filmed_layer(256, config["layers"][2], stride=2)
        layer4 = self._make_filmed_layer(512, config["layers"][3], stride=2)
        self.backbone = _ConditionalBackbone(
            conv1=base_model.conv1,
            bn1=base_model.bn1,
            activation=base_model.act1,
            maxpool=base_model.maxpool,
            layer1=layer1,
            layer2=layer2,
            layer3=layer3,
            layer4=layer4,
        )
        if self.pretrained:
            self._copy_pretrained_weights(base_model)
        self._apply_batch_norm_handling()
        self.backbone = apply_lora_config(
            model=self.backbone,
            lora_config=self.lora_config,
            frozen=self.frozen,
        )

    def _apply_batch_norm_handling(self) -> None:
        """Apply BatchNorm handling strategy to all layers."""
        match self.batch_norm_handling:
            case BatchNormHandling.FROZEN.value:
                freeze_batch_norm_2d(self.backbone.bn1)
                for layer in [
                    self.backbone.layer1,
                    self.backbone.layer2,
                    self.backbone.layer3,
                    self.backbone.layer4,
                ]:
                    for block in layer:
                        block.apply(lambda m: freeze_batch_norm_2d(m) or None)
            case BatchNormHandling.CONVERT_TO_GROUPNORM.value:
                num_channels = self.backbone.bn1.num_features
                # 16 groups matches ResNet channel multiples (64, 128, 256, 512)
                self.backbone.bn1 = nn.GroupNorm(num_channels // 16, num_channels)
                self.backbone.layer1 = replace_batchnorm_with_groupnorm(
                    self.backbone.layer1
                )
                self.backbone.layer2 = replace_batchnorm_with_groupnorm(
                    self.backbone.layer2
                )
                self.backbone.layer3 = replace_batchnorm_with_groupnorm(
                    self.backbone.layer3
                )
                self.backbone.layer4 = replace_batchnorm_with_groupnorm(
                    self.backbone.layer4
                )
            case BatchNormHandling.DEFAULT.value:
                pass
            case _:
                raise ValueError(
                    f"Unknown batch norm handling: {self.batch_norm_handling}"
                )

    def _make_filmed_layer(
        self, out_channels: int, num_blocks: int, stride: int
    ) -> nn.ModuleList:
        """Create a ResNet layer composed of FiLMedResBlocks.

        Args:
            out_channels: Number of output channels for this layer.
            num_blocks: Number of residual blocks in the layer.
            stride: Stride for the first block's convolution.

        Returns:
            ModuleList of FiLMedResBlocks.
        """
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            down_layers = [
                nn.Conv2d(
                    self.in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            ]
            downsample = nn.Sequential(*down_layers)

        blocks = nn.ModuleList()
        blocks.append(
            FiLMedResBlock(
                self.in_channels, out_channels, self.condition_dim, stride, downsample
            )
        )
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            blocks.append(
                FiLMedResBlock(
                    self.in_channels, out_channels, self.condition_dim, stride=1
                )
            )

        return blocks

    def _copy_pretrained_weights(self, base_model: nn.Module) -> None:
        """Copy weights from pretrained ResNet to FiLMedResBlocks where applicable."""
        base_layers = [
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
        ]
        self_layers = [
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
        ]

        for base_layer, self_layer in zip(base_layers, self_layers):
            for i, base_block in enumerate(base_layer):
                self_block = self_layer[i]

                self_block.conv1.load_state_dict(base_block.conv1.state_dict())

                if isinstance(self_block.bn1, nn.BatchNorm2d):
                    self_block.bn1.load_state_dict(base_block.bn1.state_dict())

                self_block.conv2.load_state_dict(base_block.conv2.state_dict())

                if isinstance(self_block.bn2, nn.BatchNorm2d):
                    self_block.bn2.load_state_dict(base_block.bn2.state_dict())

                if (
                    base_block.downsample is not None
                    and self_block.downsample is not None
                ):
                    self_block.downsample[0].load_state_dict(
                        base_block.downsample[0].state_dict()
                    )
                    if isinstance(self_block.downsample[1], nn.BatchNorm2d):
                        self_block.downsample[1].load_state_dict(
                            base_block.downsample[1].state_dict()
                        )

    def _setup_pooling(self, spatial_height: int, spatial_width: int) -> None:
        """Create pooling head from feature map spatial dimensions.

        Args:
            spatial_height: Height of the backbone's output feature map.
            spatial_width: Width of the backbone's output feature map.
        """
        self.pooling_head = create_spatial_pooling_head(
            pooling_method=self.pooling_method,
            input_dimension=self.feature_dim,
            spatial_height=spatial_height,
            spatial_width=spatial_width,
        )
        self.output_dim = self.pooling_head.output_dim

    def _encode_single_image(self, images: torch.Tensor) -> torch.Tensor:
        """Not used directly — conditioning requires ``_encode_conditioned_image``.

        Raises:
            RuntimeError: Always. Use ``encode()`` which passes conditioning.
        """
        raise RuntimeError(
            "ConditionalCNNEncoder requires conditioning. Use encode() directly."
        )

    def _encode_conditioned_image(
        self, images: torch.Tensor, conditioning: torch.Tensor
    ) -> torch.Tensor:
        """Encode a single camera's images through the FiLM backbone and pooling.

        Args:
            images: Image tensor of shape (B, C, H, W).
            conditioning: Conditioning tensor of shape (B, D).

        Returns:
            Pooled feature tensor.
        """
        if self.pooling_head is None:
            raise RuntimeError(
                "pooling_head is not initialized. Call set_image_size() before forward."
            )
        features = self.backbone(images, conditioning)
        return self.pooling_head(features)

    def encode(
        self,
        inputs: dict[str, torch.Tensor],
        conditioning: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Encode images with FiLM conditioning.

        Args:
            inputs: Dict with camera images as (B, C, H, W) per camera key.
            conditioning: Conditioning tensor as (B, D).

        Returns:
            Dict with RGB features. Single camera: key is ``rgb``.
            Multiple cameras: keys are ``rgb:{camera_key}`` per camera.
        """
        modality = EncoderOutputKeys.RGB.value
        if self.is_multi_camera:
            result = {}
            for camera_key in self.camera_keys:
                features = self._encode_conditioned_image(
                    images=inputs[camera_key], conditioning=conditioning
                )
                result[f"{modality}:{camera_key}"] = features
            return result
        features = self._encode_conditioned_image(
            images=inputs[self.camera_keys[0]], conditioning=conditioning
        )
        return {modality: features}

    def set_image_size(self, image_height: int, image_width: int) -> None:
        """Compute feature map dimensions and create pooling head.

        Args:
            image_height: Target image height.
            image_width: Target image width.
        """
        probe_dtype = (
            self.model_dtype if self.model_dtype is not None else torch.float32
        )
        with torch.no_grad():
            mock_input = torch.zeros(1, 3, image_height, image_width, dtype=probe_dtype)
            mock_condition = torch.zeros(1, self.condition_dim, dtype=probe_dtype)
            features = self.backbone(mock_input, mock_condition)
            _, _, spatial_height, spatial_width = features.shape
        self._setup_pooling(spatial_height=spatial_height, spatial_width=spatial_width)
        if self.frozen:
            self._freeze_weights()
        self._apply_model_dtype()

    def validate_input_metadata(self, key: str, metadata: BaseMetadata) -> str | None:
        """Validate that input metadata is camera metadata.

        Args:
            key: Observation key being validated.
            metadata: Metadata from the observation space.

        Returns:
            Error message if incompatible, None if valid.
        """
        if not isinstance(metadata, CameraMetadata):
            return f"Expected CameraMetadata for '{key}', got {type(metadata).__name__}"
        return None

    def get_output_specification(self) -> list[FeatureMetadata]:
        """Get structured output specification with feature names and dimensions.

        Returns:
            List of FeatureMetadata with per-camera feature names and pooled dimensions.
        """
        feature_names = self._get_vision_feature_names()
        dimension = (
            (self.output_dim,) if isinstance(self.output_dim, int) else self.output_dim
        )
        return [
            FeatureMetadata(
                key=name,
                feature_type=infer_feature_type(dimension),
                dimension=dimension,
            )
            for name in feature_names
        ]
