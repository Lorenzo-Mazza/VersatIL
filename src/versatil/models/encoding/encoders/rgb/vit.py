"""Vision Transformer encoder with multi-backbone support via timm."""

import timm
import torch

from versatil.data.constants import RGB_CAMERAS
from versatil.data.metadata import BaseMetadata, CameraMetadata
from versatil.models.encoding.encoders.base import EncoderInput
from versatil.models.encoding.encoders.constants import (
    PoolingMethod,
    ViTBackboneType,
)
from versatil.models.encoding.encoders.image_mixin import (
    ImageEncoderMixin,
    resize_to_target_size,
)
from versatil.models.encoding.encoders.unconditional import Encoder
from versatil.models.feature_meta import FeatureMetadata, infer_feature_type
from versatil.models.layers.pooling.pooling_head import create_token_pooling_head


class ViTEncoder(ImageEncoderMixin, Encoder):
    def __init__(
        self,
        input_keys: str | list[str],
        pretrained: bool,
        frozen: bool,
        pooling_method: str = PoolingMethod.DEFAULT.value,
        backbone: str = ViTBackboneType.DINOV2_VITB14.value,
    ):
        """Vision Transformer encoder using timm library.

        Args:
            input_keys: Camera observation keys.
            pretrained: Whether to load pretrained weights.
            frozen: Whether to freeze all parameters.
            pooling_method: Feature pooling strategy for patch tokens.
                Defaults to CLS token selection.
            backbone: timm model name for the ViT backbone.
        """
        specification = EncoderInput(
            keys=input_keys, at_least_one_of_groups=[RGB_CAMERAS]
        )
        super().__init__(
            input_specification=specification, pretrained=pretrained, frozen=frozen
        )
        valid_backbones = [e.value for e in ViTBackboneType]
        if backbone not in valid_backbones:
            raise ValueError(
                f"Invalid backbone '{backbone}'. Must be one of: {valid_backbones}"
            )

        self._setup_camera_keys(input_keys=self.input_specification.keys)
        self.pooling_method = pooling_method
        self.backbone_name = backbone
        self.image_size: int | tuple[int, int] | None = None
        self._build_backbone()
        self.feature_dim: int = int(self.backbone.num_features)
        self.token_pooling_head = create_token_pooling_head(
            pooling_method=pooling_method,
            input_dimension=self.feature_dim,
            exclude_cls=True,
        )
        self.output_dim = self.token_pooling_head.output_dim
        if frozen:
            super()._freeze_weights()

    def _build_backbone(self):
        """Build backbone using timm library."""
        pretrained_config = timm.get_pretrained_cfg(self.backbone_name)
        fixed_input_size = getattr(pretrained_config, "fixed_input_size", False)
        if fixed_input_size and self.image_size is not None:
            self.backbone = timm.create_model(
                self.backbone_name,
                pretrained=self.pretrained,
                img_size=self.image_size,
            )
        elif fixed_input_size:
            self.backbone = timm.create_model(
                self.backbone_name,
                pretrained=self.pretrained,
                img_size=pretrained_config.input_size[-1],
            )
        else:
            self.backbone = timm.create_model(
                self.backbone_name,
                pretrained=self.pretrained,
            )
        patch_embedding = getattr(self.backbone, "patch_embed", None)
        self.requires_strict_image_size = (
            getattr(patch_embedding, "strict_img_size", False)
            if patch_embedding is not None
            else False
        )
        if patch_embedding is not None:
            self.expected_image_size = patch_embedding.img_size
            self.patch_size = patch_embedding.patch_size
        else:
            self.expected_image_size = None
            self.patch_size = None

    def _encode_single_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a single camera's images through the backbone and pooling.

        Args:
            images: Image tensor of shape (B, C, H, W).

        Returns:
            Feature tensor.
        """
        if self.expected_image_size is not None:
            expected_height, expected_width = self.expected_image_size
            images = resize_to_target_size(
                images=images,
                target_height=expected_height,
                target_width=expected_width,
            )
        last_hidden_state = self.backbone.forward_features(images)
        return self.token_pooling_head(last_hidden_state)

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
        """Rebuild the backbone with the target image size.

        Args:
            image_height: Target image height.
            image_width: Target image width.
        """
        self.image_size = (image_height, image_width)
        self._build_backbone()
        self.feature_dim: int = int(self.backbone.num_features)
        self.token_pooling_head = create_token_pooling_head(
            pooling_method=self.pooling_method,
            input_dimension=self.feature_dim,
            exclude_cls=True,
        )
        self.output_dim = self.token_pooling_head.output_dim

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
