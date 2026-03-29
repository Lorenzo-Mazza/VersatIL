"""Single-channel depth CNN encoder via timm."""

import timm
import torch
from timm.layers import freeze_batch_norm_2d

from versatil.data.constants import Cameras
from versatil.data.metadata import BaseMetadata, CameraMetadata
from versatil.models.encoding.encoders.base import EncoderInput
from versatil.models.encoding.encoders.constants import (
    BatchNormHandling,
    CNNBackboneType,
    EncoderOutputKeys,
    PoolingMethod,
)
from versatil.models.encoding.encoders.unconditional import Encoder
from versatil.models.feature_meta import FeatureMetadata, infer_feature_type
from versatil.models.layers.convert_layers import replace_batchnorm_with_groupnorm
from versatil.models.layers.pooling.pooling_head import (
    PoolingHead,
    create_spatial_pooling_head,
)


class DepthCNNEncoder(Encoder):
    """Convolutional Neural Network encoder supporting multiple backbones via TIMM for depth images."""

    def __init__(
        self,
        input_keys: str | list[str],
        backbone: str = CNNBackboneType.RESNET18.value,
        pooling_method: str = PoolingMethod.AVERAGE.value,
        batch_norm_handling: str = BatchNormHandling.FROZEN.value,
        pretrained: bool = False,
        frozen: bool = False,
    ):
        """Initialize depth CNN encoder with single-channel timm backbone.

        Args:
            input_keys: Depth camera observation keys.
            backbone: timm model name for the CNN backbone.
            pooling_method: Feature pooling strategy.
            batch_norm_handling: How to handle batch normalization layers.
            pretrained: Whether to load pretrained weights.
            frozen: Whether to freeze all parameters.
        """
        specification = EncoderInput(keys=input_keys, required=[Cameras.DEPTH.value])
        super().__init__(
            input_specification=specification, pretrained=pretrained, frozen=frozen
        )
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
            in_chans=1,
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

    def encode(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Encode depth images into features.

        Args:
            inputs: Dict with single key from input_keys, depth images as (B, 1, H, W).

        Returns:
            Dict with depth features.
        """
        if self.pooling_head is None:
            raise RuntimeError(
                "pooling_head is not initialized. Call set_image_size() before forward."
            )
        images = inputs[self.input_specification.keys[0]]
        features = self.backbone(images)[-1]
        pooled_features = self.pooling_head(features)
        return {EncoderOutputKeys.DEPTH.value: pooled_features}

    def set_image_size(self, image_height: int, image_width: int) -> None:
        """Compute feature map dimensions and create pooling head.

        Args:
            image_height: Target image height.
            image_width: Target image width.
        """
        with torch.no_grad():
            mock_input = torch.zeros(1, 1, image_height, image_width)
            mock_features = self.backbone(mock_input)[-1]
            _, _, spatial_height, spatial_width = mock_features.shape
        self._setup_pooling(spatial_height=spatial_height, spatial_width=spatial_width)

    def validate_input_metadata(self, key: str, metadata: BaseMetadata) -> str | None:
        """Validate that input metadata is single-channel camera metadata.

        Args:
            key: Observation key being validated.
            metadata: Metadata from the observation space.

        Returns:
            Error message if incompatible, None if valid.
        """
        if not isinstance(metadata, CameraMetadata):
            return f"Expected CameraMetadata for '{key}', got {type(metadata).__name__}"
        if not metadata.is_single_channel:
            return (
                f"Expected single-channel depth for '{key}', "
                f"got {metadata.channels} channels"
            )
        return None

    def get_output_specification(self) -> list[FeatureMetadata]:
        """Get structured output specification with feature name and dimension.

        Returns:
            List of FeatureMetadata with depth feature name and pooled dimension.
        """
        dimension = (
            (self.output_dim,) if isinstance(self.output_dim, int) else self.output_dim
        )
        return [
            FeatureMetadata(
                key=EncoderOutputKeys.DEPTH.value,
                feature_type=infer_feature_type(dimension),
                dimension=dimension,
            )
        ]
