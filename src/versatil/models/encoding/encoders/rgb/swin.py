"""Swin Transformer encoder with spatial feature map output via timm."""

import timm
import torch

from versatil.data.constants import RGB_CAMERAS
from versatil.data.metadata import BaseMetadata, CameraMetadata
from versatil.models.encoding.encoders.base import EncoderInput
from versatil.models.encoding.encoders.constants import (
    PoolingMethod,
    SwinBackboneType,
)
from versatil.models.encoding.encoders.image_mixin import ImageEncoderMixin
from versatil.models.encoding.encoders.unconditional import Encoder
from versatil.models.feature_meta import FeatureMetadata, infer_feature_type
from versatil.models.layers.pooling.pooling_head import (
    PoolingHead,
    create_spatial_pooling_head,
)


class SwinEncoder(ImageEncoderMixin, Encoder):
    """Swin Transformer encoder producing spatial feature maps.

    Swin outputs channels-last (B, H, W, C) spatial features, which are
    permuted to (B, C, H, W) and processed by spatial pooling heads.

    Args:
        input_keys: Camera observation keys.
        pretrained: Whether to load pretrained weights.
        frozen: Whether to freeze all parameters.
        pooling_method: Spatial pooling strategy for feature maps.
        backbone: timm model name for the Swin backbone.
    """

    def __init__(
        self,
        input_keys: str | list[str],
        pretrained: bool,
        frozen: bool,
        pooling_method: str = PoolingMethod.AVERAGE.value,
        backbone: str = SwinBackboneType.SWIN_TINY.value,
    ):
        specification = EncoderInput(
            keys=input_keys, at_least_one_of_groups=[RGB_CAMERAS]
        )
        super().__init__(
            input_specification=specification, pretrained=pretrained, frozen=frozen
        )
        valid_backbones = [e.value for e in SwinBackboneType]
        if backbone not in valid_backbones:
            raise ValueError(
                f"Invalid backbone '{backbone}'. Must be one of: {valid_backbones}"
            )

        self._setup_camera_keys(input_keys=self.input_specification.keys)
        self.pooling_method = pooling_method
        self.backbone_name = backbone
        self._build_backbone()
        self.feature_dim: int = int(self.backbone.num_features)
        self.pooling_head: PoolingHead | None = None
        self.output_dim: int | tuple[int, ...] = self.feature_dim
        if frozen:
            super()._freeze_weights()

    def _build_backbone(self) -> None:
        """Build Swin backbone using timm."""
        self.backbone = timm.create_model(
            self.backbone_name,
            pretrained=self.pretrained,
        )
        patch_embedding = self.backbone.patch_embed
        self.expected_image_size = patch_embedding.img_size
        self.patch_size = patch_embedding.patch_size

    def _setup_pooling(self, spatial_height: int, spatial_width: int) -> None:
        """Create spatial pooling head from feature map dimensions.

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
        # Swin outputs (B, H, W, C) channels-last spatial features
        features_nhwc = self.backbone.forward_features(images)
        # Permute to (B, C, H, W) for spatial pooling heads
        features_nchw = features_nhwc.permute(0, 3, 1, 2)
        return self.pooling_head(features_nchw)

    def encode(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Encode images into features.

        Args:
            inputs: Dict mapping camera keys to image tensors (B, C, H, W).

        Returns:
            Dict with RGB features. Single camera: key is ``rgb``.
            Multiple cameras: keys are ``rgb.{camera_key}`` per camera.
        """
        return self._encode_vision(inputs)

    def set_image_size(self, image_height: int, image_width: int) -> None:
        """Rebuild backbone and create pooling head for the target image size.

        Args:
            image_height: Target image height.
            image_width: Target image width.
        """
        self.backbone = timm.create_model(
            self.backbone_name,
            pretrained=self.pretrained,
            img_size=(image_height, image_width),
        )
        self.feature_dim = int(self.backbone.num_features)
        self.expected_image_size = self.backbone.patch_embed.img_size
        with torch.no_grad():
            mock_input = torch.zeros(1, 3, image_height, image_width)
            mock_output = self.backbone.forward_features(mock_input)
            # mock_output is (B, H, W, C)
            spatial_height = mock_output.shape[1]
            spatial_width = mock_output.shape[2]
        self._setup_pooling(spatial_height=spatial_height, spatial_width=spatial_width)

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
