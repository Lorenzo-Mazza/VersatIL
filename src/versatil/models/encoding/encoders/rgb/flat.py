"""Flat RGB encoder producing (B, S, D) token sequences via timm forward_features."""

import timm
import torch

from versatil.data.constants import CameraModality
from versatil.data.metadata import BaseMetadata, CameraMetadata
from versatil.models.adaptation.lora import LoRAAdaptation, apply_lora_config
from versatil.models.encoding.encoders.base import EncoderInput
from versatil.models.encoding.encoders.constants import (
    FlatBackboneType,
    PoolingMethod,
)
from versatil.models.encoding.encoders.image_mixin import (
    RGBEncoderMixin,
    resize_to_target_size,
)
from versatil.models.encoding.encoders.unconditional import Encoder
from versatil.models.feature_meta import FeatureMetadata, infer_feature_type
from versatil.models.layers.pooling.pooling_head import create_token_pooling_head


class FlatRGBEncoder(RGBEncoderMixin, Encoder):
    """RGB encoder for backbones that output flat token sequences."""

    def __init__(
        self,
        input_keys: str | list[str],
        pretrained: bool,
        frozen: bool,
        pooling_method: str = PoolingMethod.DEFAULT.value,
        backbone: str = FlatBackboneType.DINOV2_VITB14.value,
        image_size: int | tuple[int, int] | None = None,
        intermediate_layer_index: int | None = None,
        model_dtype: str | None = None,
        lora_config: LoRAAdaptation | None = None,
    ) -> None:
        """Initialize flat RGB encoder with timm backbone.

        Args:
            input_keys: Camera observation keys.
            pretrained: Whether to load pretrained weights.
            frozen: Whether to freeze all parameters.
            pooling_method: Feature pooling strategy for patch tokens.
                Defaults to CLS token selection.
            backbone: timm model name for the backbone.
            image_size: Optional image size passed to timm during backbone
                construction.
            intermediate_layer_index: Optional intermediate layer index for
                feature extraction. Negative values index from the end.
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
        valid_backbones = [e.value for e in FlatBackboneType]
        if backbone not in valid_backbones:
            raise ValueError(
                f"Invalid backbone '{backbone}'. Must be one of: {valid_backbones}"
            )

        pooling = PoolingMethod(pooling_method)
        if not pooling.supports_sequential:
            raise ValueError(
                f"Pooling method '{pooling_method}' is not compatible with "
                f"token sequences. Use one of: "
                f"{[p.value for p in PoolingMethod if p.supports_sequential]}"
            )
        self._setup_camera_keys(input_keys=self.input_specification.keys)
        self.pooling_method = pooling_method
        self.backbone_name = backbone
        self.image_size = image_size
        self.intermediate_layer_index = intermediate_layer_index
        self.lora_config = lora_config
        self._build_backbone()
        self.feature_dim: int = int(self.backbone.num_features)
        self.token_pooling_head = create_token_pooling_head(
            pooling_method=pooling_method,
            input_dimension=self.feature_dim,
            num_prefix_tokens=self.backbone.num_prefix_tokens,
        )
        self.output_dim = self.token_pooling_head.output_dim
        if frozen:
            super()._freeze_weights()
        self._apply_model_dtype()

    def _build_backbone(self) -> None:
        """Build backbone using timm library."""
        pretrained_config = timm.get_pretrained_cfg(self.backbone_name)
        fixed_input_size = getattr(pretrained_config, "fixed_input_size", False)
        kwargs: dict[str, bool | int | tuple[int, int] | str] = {
            "pretrained": self.pretrained,
        }
        if self._uses_openai_clip_backbone():
            kwargs["act_layer"] = "quick_gelu"
        if self.image_size is not None:
            kwargs["img_size"] = self.image_size
        elif fixed_input_size:
            kwargs["img_size"] = pretrained_config.input_size[-1]
        backbone = timm.create_model(self.backbone_name, **kwargs)
        patch_embedding = getattr(backbone, "patch_embed", None)
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
        self.backbone = apply_lora_config(
            model=backbone,
            lora_config=self.lora_config,
            frozen=self.frozen,
        )

    def _uses_openai_clip_backbone(self) -> bool:
        """Return whether the timm backbone needs OpenAI CLIP QuickGELU."""
        return self.backbone_name in {
            FlatBackboneType.CLIP_VITL14_224_OPENAI.value,
            FlatBackboneType.CLIP_VITL14_336_OPENAI.value,
        }

    def _resolve_configured_intermediate_layer_index(self) -> int:
        """Resolve the configured ViT block index."""
        if self.intermediate_layer_index is None:
            raise RuntimeError("intermediate_layer_index is not configured.")
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None:
            raise ValueError(
                f"Backbone '{self.backbone_name}' does not expose ViT blocks for "
                "intermediate-layer extraction."
            )
        return self._resolve_intermediate_layer_index(
            intermediate_layer_index=self.intermediate_layer_index,
            output_count=len(blocks),
        )

    def _forward_backbone_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return token features from the configured backbone layer."""
        if self.intermediate_layer_index is None:
            return self.backbone.forward_features(images)
        layer_index = self._resolve_configured_intermediate_layer_index()
        if not hasattr(self.backbone, "forward_intermediates"):
            raise ValueError(
                f"Backbone '{self.backbone_name}' does not support intermediate-layer "
                "extraction."
            )
        features = self.backbone.forward_intermediates(
            images,
            indices=[layer_index],
            return_prefix_tokens=True,
            output_fmt="NLC",
            intermediates_only=True,
        )[0]
        if isinstance(features, tuple):
            patch_tokens, prefix_tokens = features
            return torch.cat([prefix_tokens, patch_tokens], dim=1)
        return features

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
        last_hidden_state = self._forward_backbone_features(images)
        return self.token_pooling_head(last_hidden_state)

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
            num_prefix_tokens=self.backbone.num_prefix_tokens,
        )
        self.output_dim = self.token_pooling_head.output_dim
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
