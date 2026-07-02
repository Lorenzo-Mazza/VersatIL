"""Lightweight geometry-aware RGBD encoder."""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from versatil.data.constants import CameraModality
from versatil.models.encoding.encoders.base import EncoderInput
from versatil.models.encoding.encoders.constants import (
    EncoderOutputKeys,
    PoolingMethod,
)
from versatil.models.encoding.encoders.image_mixin import RGBDEncoderMixin
from versatil.models.encoding.encoders.unconditional import Encoder
from versatil.models.encoding.explainability import (
    ActivationLayout,
    ExplanationTargetKind,
    VisionExplanationTarget,
)
from versatil.models.feature_meta import FeatureMetadata, infer_feature_type
from versatil.models.layers import PatchEmbedding
from versatil.models.layers.constants import AttentionDecompositionMode
from versatil.models.layers.geometric_attention.geometric_attention_encoder import (
    GeometricAttentionEncoderBlock,
)
from versatil.models.layers.patch_embedding import PatchEmbedType
from versatil.models.layers.pooling.pooling_head import (
    PoolingHead,
    create_spatial_pooling_head,
)


class GeometricRGBDEncoder(RGBDEncoderMixin, Encoder):
    """Single-layer geometry-aware RGBD encoder."""

    def __init__(
        self,
        input_keys: str | list[str],
        embedding_dimension: int = 512,
        num_heads: int = 8,
        ffn_dimension: int = 2048,
        decomposition_mode: str = AttentionDecompositionMode.SEPARABLE.value,
        initial_decay: float = 2.0,
        decay_range: float = 4.0,
        patch_size: int = 16,
        pooling_method: str = PoolingMethod.AVERAGE.value,
        pretrained: bool = False,
        frozen: bool = False,
        model_dtype: str | None = None,
    ):
        """Initialize the geometric RGBD encoder.

        Args:
            input_keys: Input keys for RGB and depth observations.
            embedding_dimension: Dimension of patch embeddings and attention.
            num_heads: Number of attention heads.
            ffn_dimension: Hidden dimension of the feed-forward network.
            decomposition_mode: Attention computation strategy (full or separable).
            initial_decay: Initial decay rate for spatial biases.
            decay_range: Range of decay rates across heads.
            patch_size: Size of image patches for the patch embedding.
            pooling_method: Feature pooling method applied after attention.
            pretrained: Whether to use pretrained weights (not supported).
            frozen: Whether to freeze encoder weights (not supported).
            model_dtype: Precision string from experiment config (e.g. ``"bf16-mixed"``).
        """
        specification = EncoderInput(
            keys=input_keys,
            exactly_one_camera_modality=[CameraModality.RGB, CameraModality.DEPTH],
            required_camera_modalities=[CameraModality.RGB, CameraModality.DEPTH],
        )
        super().__init__(
            input_specification=specification,
            pretrained=pretrained,
            frozen=frozen,
            model_dtype=model_dtype,
        )
        if pretrained:
            logging.warning(
                "GeometricRGBDEncoder does not support pretrained weights. Continuing with random initialization."
            )
        if frozen:
            raise ValueError(
                "Freezing GeometricRGBDEncoder does not make sense as it has no pretrained weights. Set frozen=False."
            )
        self._setup_camera_keys(input_keys=self.input_specification.keys)
        self.embedding_dimension = embedding_dimension
        self.decomposition_mode = AttentionDecompositionMode(decomposition_mode)
        self.pooling_method = pooling_method
        self.patch_size = patch_size
        self.patch_embed = PatchEmbedding(
            patch_size=self.patch_size,
            in_chans=3,
            embed_dim=embedding_dimension,
            embed_type=PatchEmbedType.STANDARD.value,
            norm_layer=LayerNorm,
        )

        self.attention_block = GeometricAttentionEncoderBlock(
            decomposition_mode=AttentionDecompositionMode(decomposition_mode),
            embedding_dimension=embedding_dimension,
            num_heads=num_heads,
            ffn_dimension=ffn_dimension,
            initial_decay=initial_decay,
            decay_range=decay_range,
        )
        self.pre_attention_norm = nn.LayerNorm(embedding_dimension, eps=1e-6)
        self.post_attention_norm = nn.LayerNorm(embedding_dimension, eps=1e-6)
        self.pooling_head: PoolingHead | None = None
        self.output_dim: int | tuple[int, ...] = self.embedding_dimension
        if frozen:
            super()._freeze_weights()
        self._apply_model_dtype()

    def _setup_pooling(self, spatial_height: int, spatial_width: int) -> None:
        """Create pooling head from feature map spatial dimensions.

        Args:
            spatial_height: Height of the backbone's output feature map.
            spatial_width: Width of the backbone's output feature map.
        """
        self.pooling_head = create_spatial_pooling_head(
            pooling_method=self.pooling_method,
            input_dimension=self.embedding_dimension,
            spatial_height=spatial_height,
            spatial_width=spatial_width,
        )
        self.output_dim = self.pooling_head.output_dim

    def _encode_single_image(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "GeometricRGBDEncoder processes RGB+depth jointly. Use encode() instead."
        )

    def encode_features(
        self, rgb_image: torch.Tensor, depth_map: torch.Tensor
    ) -> tuple[torch.Tensor, int, int]:
        """Encode RGB and depth into joint RGBD features using geometric attention.

        Args:
            rgb_image: RGB image tensor of shape (B, C, H, W).
            depth_map: Depth map tensor of shape (B, 1, H, W).

        Returns:
            Tuple of (features, H_patches, W_patches) where features has shape
            (B, embedding_dimension, H_patches, W_patches).
        """
        features, H_patches, W_patches = self.patch_embed(
            rgb_image, return_patch_size=True
        )  # (B, N_patches, embedding_dimension)
        features = self.pre_attention_norm(features)
        features = features.reshape(
            rgb_image.shape[0], H_patches, W_patches, self.embedding_dimension
        )
        depth_map_resized = F.interpolate(
            depth_map, size=(H_patches, W_patches), mode="bilinear", align_corners=False
        )
        features = self.attention_block(
            features, depth_map_resized
        )  # (B, H_patches, W_patches, embedding_dimension)
        features = self.post_attention_norm(features)
        features = features.permute(
            0, 3, 1, 2
        ).contiguous()  # (B, embedding_dimension, H_patches, W_patches)
        return features, H_patches, W_patches

    def encode(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Encode RGB + depth into fused features.

        Args:
            inputs: Dict with RGB as (B, C, H, W) and depth as (B, 1, H, W).

        Returns:
            Dict with RGBD features.
        """
        rgb_key = self._camera_key_for_modality(modality=CameraModality.RGB)
        depth_key = self._camera_key_for_modality(modality=CameraModality.DEPTH)
        rgb = inputs[rgb_key]
        depth = inputs[depth_key]

        if self.pooling_head is None:
            raise RuntimeError(
                "pooling_head is not initialized. Call set_image_size() before forward."
            )
        features, feature_map_height, feature_map_width = self.encode_features(
            rgb, depth
        )
        pooled_features = self.pooling_head(features)
        return {EncoderOutputKeys.RGBD.value: pooled_features}

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
            mock_rgb = torch.zeros(1, 3, image_height, image_width, dtype=probe_dtype)
            mock_depth = torch.zeros(1, 1, image_height, image_width, dtype=probe_dtype)
            _, spatial_height, spatial_width = self.encode_features(
                rgb_image=mock_rgb, depth_map=mock_depth
            )
        self._setup_pooling(spatial_height=spatial_height, spatial_width=spatial_width)
        self._apply_model_dtype()

    def get_explainability_targets(self) -> list[VisionExplanationTarget]:
        """Return the geometric attention block for spatial attribution maps.

        Returns:
            One NHWC spatial feature-map target from the RGBD attention block.
        """
        return [
            VisionExplanationTarget(
                layer=self.attention_block,
                target_kind=ExplanationTargetKind.SPATIAL_FEATURE_MAP.value,
                activation_layout=ActivationLayout.NHWC.value,
            )
        ]

    def get_output_specification(self) -> list[FeatureMetadata]:
        """Return the output feature names and dimensions for this encoder.

        Returns:
            List of FeatureMetadata with RGBD feature name and its pooled dimension.
        """
        dimension = (
            (self.output_dim,) if isinstance(self.output_dim, int) else self.output_dim
        )
        return [
            FeatureMetadata(
                key=EncoderOutputKeys.RGBD.value,
                feature_type=infer_feature_type(dimension),
                dimension=dimension,
            )
        ]
