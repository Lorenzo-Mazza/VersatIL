"""DINOv2+SigLIP RGB encoder producing fused patch-token sequences."""

from dataclasses import dataclass

import torch

from versatil.data.constants import (
    IMAGENET_RGB_MEAN,
    IMAGENET_RGB_STD,
    SIGLIP_RGB_MEAN,
    SIGLIP_RGB_STD,
    CameraModality,
)
from versatil.data.metadata import BaseMetadata, CameraMetadata
from versatil.models.adaptation.lora import LoRAAdaptation
from versatil.models.encoding.encoders.base import EncoderInput
from versatil.models.encoding.encoders.constants import (
    DinoV2SigLIPBackboneType,
    FlatBackboneType,
    PoolingMethod,
)
from versatil.models.encoding.encoders.image_mixin import (
    RGBEncoderMixin,
    resize_to_target_size,
)
from versatil.models.encoding.encoders.rgb.flat import FlatRGBEncoder
from versatil.models.encoding.encoders.unconditional import Encoder
from versatil.models.feature_meta import FeatureMetadata, infer_feature_type


@dataclass(frozen=True)
class DinoV2SigLIPBackboneConfig:
    """Resolved timm tower configuration for a paired DINOv2+SigLIP backbone."""

    dino_backbone: FlatBackboneType
    siglip_backbone: FlatBackboneType
    image_size: int


DINOV2_SIGLIP_BACKBONE_CONFIGS: dict[
    DinoV2SigLIPBackboneType,
    DinoV2SigLIPBackboneConfig,
] = {
    DinoV2SigLIPBackboneType.DINOV2_SIGLIP_VIT_SO_224PX: DinoV2SigLIPBackboneConfig(
        dino_backbone=FlatBackboneType.DINOV2_VITL14_REG4,
        siglip_backbone=FlatBackboneType.SIGLIP_SO400M_224,
        image_size=224,
    ),
    DinoV2SigLIPBackboneType.DINOV2_SIGLIP_VIT_SO_384PX: DinoV2SigLIPBackboneConfig(
        dino_backbone=FlatBackboneType.DINOV2_VITL14_REG4,
        siglip_backbone=FlatBackboneType.SIGLIP_SO400M_384,
        image_size=384,
    ),
}


class DinoV2SigLIPRGBEncoder(RGBEncoderMixin, Encoder):
    """RGB encoder that concatenates DINOv2 and SigLIP patch features."""

    def __init__(
        self,
        input_keys: str | list[str],
        pretrained: bool,
        frozen: bool,
        backbone: str = DinoV2SigLIPBackboneType.DINOV2_SIGLIP_VIT_SO_224PX.value,
        model_dtype: str | None = None,
        lora_config: LoRAAdaptation | None = None,
    ) -> None:
        """Initialize paired timm vision towers.

        Args:
            input_keys: RGB camera observation keys.
            pretrained: Whether timm should load pretrained tower weights.
            frozen: Whether to freeze both vision towers.
            backbone: DINOv2+SigLIP paired backbone identifier.
            model_dtype: Precision string from experiment config (e.g. ``"bf16-mixed"``).
            lora_config: Optional LoRA adapter configuration for the timm towers.
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
        self._setup_camera_keys(input_keys=self.input_specification.keys)
        backbone_config = self._resolve_backbone_config(backbone=backbone)
        self.backbone_name = backbone
        self.dino_model_name = backbone_config.dino_backbone.value
        self.siglip_model_name = backbone_config.siglip_backbone.value
        self.image_size = backbone_config.image_size
        self.lora_config = lora_config
        self.dino_encoder = self._build_flat_encoder(
            backbone=backbone_config.dino_backbone
        )
        self.siglip_encoder = self._build_flat_encoder(
            backbone=backbone_config.siglip_backbone
        )
        self.register_buffer(
            "dino_standardization_mean",
            torch.tensor(IMAGENET_RGB_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "dino_standardization_std",
            torch.tensor(IMAGENET_RGB_STD).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "siglip_standardization_mean",
            torch.tensor(SIGLIP_RGB_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "siglip_standardization_std",
            torch.tensor(SIGLIP_RGB_STD).view(1, 3, 1, 1),
            persistent=False,
        )
        self.num_patches = int(self.dino_encoder.backbone.patch_embed.num_patches)
        siglip_num_patches = int(self.siglip_encoder.backbone.patch_embed.num_patches)
        if self.num_patches != siglip_num_patches:
            raise ValueError(
                "DINO and SigLIP patch counts must match, got "
                f"{self.num_patches} and {siglip_num_patches}."
            )
        self.feature_dim = int(
            self.dino_encoder.feature_dim + self.siglip_encoder.feature_dim
        )
        self.embedding_dimension = self.feature_dim
        self.output_dim = (-1, self.feature_dim)
        if frozen:
            super()._freeze_weights()
        self._apply_model_dtype()

    @staticmethod
    def _resolve_backbone_config(backbone: str) -> DinoV2SigLIPBackboneConfig:
        """Validate and resolve a paired DINOv2+SigLIP backbone id."""
        valid_backbones = [model_type.value for model_type in DinoV2SigLIPBackboneType]
        if backbone not in valid_backbones:
            raise ValueError(
                f"Invalid DINOv2+SigLIP backbone '{backbone}'. "
                f"Must be one of: {valid_backbones}."
            )
        return DINOV2_SIGLIP_BACKBONE_CONFIGS[DinoV2SigLIPBackboneType(backbone)]

    def _build_flat_encoder(self, backbone: FlatBackboneType) -> FlatRGBEncoder:
        """Build one flat timm tower with Prismatic patch-token settings."""
        return FlatRGBEncoder(
            input_keys=self.input_specification.keys,
            pretrained=self.pretrained,
            frozen=self.frozen,
            pooling_method=PoolingMethod.NONE.value,
            backbone=backbone.value,
            image_size=self.image_size,
            intermediate_layer_index=-2,
            model_dtype=self.model_dtype,
            lora_config=self.lora_config,
        )

    @staticmethod
    def _standardize_images(
        pixel_values: torch.Tensor,
        mean: torch.Tensor,
        standard_deviation: torch.Tensor,
    ) -> torch.Tensor:
        """Standardize zero-to-one RGB images for one tower.

        Args:
            pixel_values: RGB tensor with shape ``(B, 3, H, W)`` in ``[0, 1]``.
            mean: Per-channel mean with shape ``(1, 3, 1, 1)``.
            standard_deviation: Per-channel standard deviation with shape
                ``(1, 3, 1, 1)``.

        Returns:
            Standardized RGB tensor with shape ``(B, 3, H, W)``.
        """
        return (pixel_values - mean.to(pixel_values)) / standard_deviation.to(
            pixel_values
        )

    def encode_image_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into fused DINOv2+SigLIP patch tokens.

        Args:
            images: RGB tensor with shape ``(B, 3, H, W)``.

        Returns:
            Fused patch tokens with shape ``(B, P, D_dino + D_siglip)``.
        """
        images = resize_to_target_size(
            images=images,
            target_height=self.image_size,
            target_width=self.image_size,
        )
        dino_pixel_values = self._standardize_images(
            pixel_values=images,
            mean=self.dino_standardization_mean,
            standard_deviation=self.dino_standardization_std,
        )
        siglip_pixel_values = self._standardize_images(
            pixel_values=images,
            mean=self.siglip_standardization_mean,
            standard_deviation=self.siglip_standardization_std,
        )
        dino_features = self.dino_encoder._encode_single_image(dino_pixel_values)
        siglip_features = self.siglip_encoder._encode_single_image(siglip_pixel_values)
        return torch.cat([dino_features, siglip_features], dim=2)

    def _encode_single_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a single camera image batch into fused patch tokens."""
        return self.encode_image_tokens(images=images)

    def encode(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Encode images into RGB patch-token features."""
        return self._encode_vision(inputs)

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
        """Get output specification for fused patch-token features."""
        feature_names = self._get_vision_feature_names()
        dimension = (-1, self.feature_dim)
        return [
            FeatureMetadata(
                key=name,
                feature_type=infer_feature_type(dimension),
                dimension=dimension,
            )
            for name in feature_names
        ]
