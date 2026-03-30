"""CNN encoder with multi-backbone support via timm."""

import timm
import torch
from timm.layers import freeze_batch_norm_2d

from versatil.data.constants import RGB_CAMERAS
from versatil.data.metadata import BaseMetadata, CameraMetadata
from versatil.models.encoding.encoders.base import EncoderInput
from versatil.models.encoding.encoders.constants import (
    BatchNormHandling,
    CNNBackboneType,
    PoolingMethod,
)
from versatil.models.encoding.encoders.image_mixin import ImageEncoderMixin
from versatil.models.encoding.encoders.unconditional import Encoder
from versatil.models.feature_meta import FeatureMetadata, infer_feature_type
from versatil.models.layers.convert_layers import replace_batchnorm_with_groupnorm
from versatil.models.layers.pooling.pooling_head import (
    PoolingHead,
    create_spatial_pooling_head,
)


class CNNEncoder(ImageEncoderMixin, Encoder):
    """Convolutional Neural Network encoder supporting multiple backbones via TIMM."""

    def __init__(
        self,
        input_keys: str | list[str],
        backbone: str = CNNBackboneType.RESNET18.value,
        pooling_method: str = PoolingMethod.AVERAGE.value,
        batch_norm_handling: str = BatchNormHandling.FROZEN.value,
        pretrained: bool = False,
        frozen: bool = False,
        model_dtype: str | None = None,
    ):
        """Initialize CNN encoder with timm backbone.

        Args:
            input_keys: Camera observation keys.
            backbone: timm model name for the CNN backbone.
            pooling_method: Feature pooling strategy.
            batch_norm_handling: How to handle batch normalization layers.
            pretrained: Whether to load pretrained weights.
            frozen: Whether to freeze all parameters.
            model_dtype: Precision string from experiment config (e.g. ``"bf16-mixed"``).
        """
        specification = EncoderInput(
            keys=input_keys, at_least_one_of_groups=[RGB_CAMERAS]
        )
        super().__init__(
            input_specification=specification,
            pretrained=pretrained,
            frozen=frozen,
            model_dtype=model_dtype,
        )
        valid_backbones = [e.value for e in CNNBackboneType]
        if backbone not in valid_backbones:
            raise ValueError(
                f"Invalid backbone '{backbone}'. Must be one of: {valid_backbones}"
            )
        self._setup_camera_keys(input_keys=self.input_specification.keys)
        self.batch_norm_handling = batch_norm_handling
        self.pooling_method = pooling_method
        self.backbone_name = backbone
        self._build_backbone()
        self.feature_dim = self.backbone.feature_info.channels()[-1]
        self.pooling_head: PoolingHead | None = None
        self.output_dim: int | tuple[int, ...] = self.feature_dim
        if frozen:
            super()._freeze_weights()

    def _build_backbone(self):
        """Build backbone using timm library."""
        self.backbone = timm.create_model(
            self.backbone_name,
            pretrained=self.pretrained,
            features_only=True,
        )
        match self.batch_norm_handling:
            case BatchNormHandling.FROZEN.value:
                self.backbone.apply(freeze_batch_norm_2d)
            case BatchNormHandling.CONVERT_TO_GROUPNORM.value:
                self.backbone = replace_batchnorm_with_groupnorm(self.backbone)
            case BatchNormHandling.DEFAULT.value:
                pass  # keep as-is
            case _:
                raise ValueError(
                    f"Unknown batch norm handling: {self.batch_norm_handling}"
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
        feature_maps = self.backbone(images)
        features = feature_maps[-1]
        return self.pooling_head(features)

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
        """Compute feature map dimensions and create pooling head.

        Args:
            image_height: Target image height.
            image_width: Target image width.
        """
        with torch.no_grad():
            mock_input = torch.zeros(1, 3, image_height, image_width)
            mock_features = self.backbone(mock_input)[-1]
            _, _, spatial_height, spatial_width = mock_features.shape
        self._setup_pooling(spatial_height=spatial_height, spatial_width=spatial_width)

    def validate_input_metadata(self, key: str, metadata: BaseMetadata) -> str | None:
        """Validate that input metadata is RGB camera metadata.

        Args:
            key: Observation key being validated.
            metadata: Metadata from the observation space.

        Returns:
            Error message if incompatible, None if valid.
        """
        if not isinstance(metadata, CameraMetadata):
            return f"Expected CameraMetadata for '{key}', got {type(metadata).__name__}"
        if not metadata.is_rgb:
            return (
                f"Expected 3-channel RGB for '{key}', got {metadata.channels} channels"
            )
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
