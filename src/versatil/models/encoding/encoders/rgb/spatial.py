"""Spatial RGB encoder producing (B, C, H, W) feature maps via timm features_only."""

import timm
import torch
from timm.layers import freeze_batch_norm_2d

from versatil.data.constants import CameraModality
from versatil.data.metadata import BaseMetadata, CameraMetadata
from versatil.models.adaptation.lora import LoRAAdaptation, apply_lora_config
from versatil.models.encoding.encoders.base import EncoderInput
from versatil.models.encoding.encoders.constants import (
    BatchNormHandling,
    PoolingMethod,
    SpatialBackboneType,
)
from versatil.models.encoding.encoders.image_mixin import RGBEncoderMixin
from versatil.models.encoding.encoders.unconditional import Encoder
from versatil.models.feature_meta import FeatureMetadata, infer_feature_type
from versatil.models.layers.convert_layers import replace_batchnorm_with_groupnorm
from versatil.models.layers.pooling.pooling_head import (
    PoolingHead,
    create_spatial_pooling_head,
)


class SpatialRGBEncoder(RGBEncoderMixin, Encoder):
    """RGB encoder for backbones that output spatial feature maps.

    Supports any timm backbone compatible with ``features_only=True``,
    regardless of whether the architecture is convolutional (ResNet,
    EfficientNet, ConvNeXt) or attention-based (Swin, TinyViT).
    Handles both NCHW and NHWC output layouts transparently.
    """

    def __init__(
        self,
        input_keys: str | list[str],
        backbone: str = SpatialBackboneType.RESNET18.value,
        pooling_method: str = PoolingMethod.AVERAGE.value,
        batch_norm_handling: str = BatchNormHandling.FROZEN.value,
        intermediate_layer_index: int | None = None,
        pretrained: bool = False,
        frozen: bool = False,
        model_dtype: str | None = None,
        lora_config: LoRAAdaptation | None = None,
    ) -> None:
        """Initialize spatial RGB encoder with timm backbone.

        Args:
            input_keys: Camera observation keys.
            backbone: timm model name from SpatialBackboneType.
            pooling_method: Feature pooling strategy.
            batch_norm_handling: How to handle batch normalization layers.
            intermediate_layer_index: Optional timm intermediate layer index
                to pool. Negative values index from the end; ``None`` uses
                the last layer.
            pretrained: Whether to load pretrained weights.
            frozen: Whether to freeze all parameters.
            model_dtype: Precision string from experiment config (e.g. ``"bf16-mixed"``).
            lora_config: Optional PEFT LoRA adapter configuration.
        """
        specification = EncoderInput(
            keys=input_keys,
            required_camera_modalities=[CameraModality.RGB],
        )
        super().__init__(
            input_specification=specification,
            pretrained=pretrained,
            frozen=frozen,
            model_dtype=model_dtype,
        )
        valid_backbones = [e.value for e in SpatialBackboneType]
        if backbone not in valid_backbones:
            raise ValueError(
                f"Invalid backbone '{backbone}'. Must be one of: {valid_backbones}"
            )
        pooling = PoolingMethod(pooling_method)
        if not pooling.supports_spatial:
            raise ValueError(
                f"Pooling method '{pooling_method}' is not compatible with "
                f"spatial feature maps. Use one of: "
                f"{[p.value for p in PoolingMethod if p.supports_spatial]}"
            )
        self._setup_camera_keys(input_keys=self.input_specification.keys)
        self.batch_norm_handling = batch_norm_handling
        self.pooling_method = pooling_method
        self.intermediate_layer_index = intermediate_layer_index
        self.backbone_name = backbone
        self.lora_config = lora_config
        self._channels_last = False
        self._build_backbone()
        self.feature_dim = self._get_intermediate_layer_channels()
        self.pooling_head: PoolingHead | None = None
        self.output_dim: int | tuple[int, ...] = self.feature_dim
        if frozen:
            super()._freeze_weights()
        self._apply_model_dtype()

    def _get_intermediate_layer_channels(self) -> int:
        """Return the channel count for the configured intermediate layer."""
        channels = self.backbone.feature_info.channels()
        layer_index = self._resolve_intermediate_layer_index(
            intermediate_layer_index=self.intermediate_layer_index,
            output_count=len(channels),
        )
        return channels[layer_index]

    def _build_backbone(self, img_size: tuple[int, int] | None = None) -> None:
        """Build backbone using timm features_only mode.

        Args:
            img_size: Optional image size override for strict-input-size backbones.
        """
        kwargs: dict[str, bool | tuple[int, int]] = {
            "pretrained": self.pretrained,
            "features_only": True,
        }
        if img_size is not None:
            kwargs["img_size"] = img_size

        self.backbone = timm.create_model(self.backbone_name, **kwargs)
        self._apply_batch_norm_handling()
        self.backbone = apply_lora_config(
            model=self.backbone,
            lora_config=self.lora_config,
            frozen=self.frozen,
        )

    def _apply_batch_norm_handling(self) -> None:
        """Apply configured batch normalization handling to the backbone."""
        match self.batch_norm_handling:
            case BatchNormHandling.FROZEN.value:
                self.backbone.apply(freeze_batch_norm_2d)
            case BatchNormHandling.CONVERT_TO_GROUPNORM.value:
                self.backbone = replace_batchnorm_with_groupnorm(self.backbone)
            case BatchNormHandling.DEFAULT.value:
                pass
            case _:
                raise ValueError(
                    f"Unknown batch norm handling: {self.batch_norm_handling}"
                )

    def _has_strict_image_size(self) -> bool:
        """Check if backbone requires exact input dimensions."""
        patch_embed = getattr(self.backbone, "patch_embed", None)
        if patch_embed is None:
            return False
        return getattr(patch_embed, "strict_img_size", False)

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
        """Encode a single camera's images through the backbone and pooling.

        Args:
            images: Image tensor of shape (B, C, H, W).

        Returns:
            Pooled feature tensor.
        """
        if self.pooling_head is None:
            raise RuntimeError(
                "pooling_head is not initialized. Call set_image_size() before forward."
            )
        intermediate_outputs = self.backbone(images)
        layer_index = self._resolve_intermediate_layer_index(
            intermediate_layer_index=self.intermediate_layer_index,
            output_count=len(intermediate_outputs),
        )
        features = intermediate_outputs[layer_index]
        if self._channels_last:
            features = features.permute(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)
        return self.pooling_head(features)

    def encode(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Encode images into features.

        Args:
            inputs: Dict mapping camera keys to image tensors (B, C, H, W).

        Returns:
            Dict with RGB features. Single camera: key is ``rgb``.
            Multiple cameras: keys are ``rgb:{camera_key}`` per camera.
        """
        return self._encode_vision(inputs)

    def set_image_size(self, image_height: int, image_width: int) -> None:
        """Compute feature map dimensions and create pooling head.

        For backbones with strict input size requirements (e.g. Swin), this
        rebuilds the backbone with the target dimensions.

        Args:
            image_height: Target image height.
            image_width: Target image width.
        """
        if self._has_strict_image_size():
            self._build_backbone(img_size=(image_height, image_width))
            self.feature_dim = self._get_intermediate_layer_channels()
            if self.frozen:
                self._freeze_weights()

        probe_dtype = (
            self.model_dtype if self.model_dtype is not None else torch.float32
        )
        with torch.no_grad():
            mock_input = torch.zeros(1, 3, image_height, image_width, dtype=probe_dtype)
            intermediate_outputs = self.backbone(mock_input)
            layer_index = self._resolve_intermediate_layer_index(
                intermediate_layer_index=self.intermediate_layer_index,
                output_count=len(intermediate_outputs),
            )
            mock_features = intermediate_outputs[layer_index]

        expected_channels = self.feature_dim
        if mock_features.shape[1] == expected_channels:
            self._channels_last = False
            _, _, spatial_height, spatial_width = mock_features.shape
        elif mock_features.shape[-1] == expected_channels:
            self._channels_last = True
            _, spatial_height, spatial_width, _ = mock_features.shape
        else:
            raise RuntimeError(
                f"Backbone '{self.backbone_name}' output shape {mock_features.shape} "
                f"does not match expected channels {expected_channels} in "
                f"either NCHW or NHWC layout."
            )

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
