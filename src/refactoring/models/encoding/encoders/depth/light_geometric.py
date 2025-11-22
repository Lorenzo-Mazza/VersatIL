"""Lightweight geometry-aware RGBD encoder."""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from refactoring.data.constants import Cameras
from refactoring.models.encoding.encoders.base import EncoderInput, EncoderOutput
from refactoring.models.encoding.encoders.constants import (
    EncoderOutputKeys,
    PoolingMethod,
)
from refactoring.models.encoding.encoders.unconditional import Encoder
from refactoring.models.layers import PatchEmbedding
from refactoring.models.layers.constants import AttentionDecompositionMode
from refactoring.models.layers.geometric_attention.geometric_attention_encoder import (
    GeometricAttentionEncoderBlock,
)
from refactoring.models.layers.patch_embedding import PatchEmbedType
from refactoring.models.layers.pooling.pooling_head import create_pooling_head


class LightGeometricEncoder(Encoder):
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
    ):
        specification = EncoderInput(keys=input_keys,required=[Cameras.DEPTH.value], one_of_groups=[[Cameras.LEFT.value, Cameras.RIGHT.value]])
        super().__init__(input_specification=specification, pretrained=pretrained, frozen=frozen)
        if pretrained:
            logging.warning("LightGeometricEncoder does not support pretrained weights. Continuing with random initialization.")
        if frozen:
            raise ValueError("Freezing LightGeometricEncoder does not make sense as it has no pretrained weights. Set frozen=False.")
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
        self.norm = nn.LayerNorm(embedding_dimension, eps=1e-6)
        self._setup_pooling()
        if frozen:
            super()._freeze_weights()

    def _setup_pooling(self):
        """Setup mock pooling head. The actual pooling head will be created in forward()."""
        mock_pooling_head = create_pooling_head(
            pooling_method=self.pooling_method,
            feature_channels=self.embedding_dimension,
            spatial_height=self.patch_size,
            spatial_width=self.patch_size,
        )
        self.pooling_head = None # Will be created in forward() with correct patch dimensions
        self.output_dim = mock_pooling_head.get_output_dim(self.embedding_dimension)



    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        rgb_key = [k for k in self.input_specification.keys if k in self.input_specification.one_of_groups[0]][0]
        depth_key = self.input_specification.required[0]
        rgb = inputs[rgb_key]
        depth = inputs[depth_key]
        has_time = rgb.dim() == 5
        if has_time:
            B, T, C, H, W = rgb.shape
            rgb = rgb.reshape(B * T, C, H, W)
            depth = depth.reshape(B * T, 1, H, W)
        else:
            B = rgb.shape[0]
            T = 1

        features, H_patches, W_patches = self.patch_embed(rgb, return_patch_size=True)  # (B, N_patches, embedding_dimension)
        features = self.norm(features)
        features = features.reshape(B if not has_time else B*T, H_patches, W_patches, self.embedding_dimension)
        depth_map = F.interpolate(depth, size=(H_patches, W_patches), mode='bilinear', align_corners=False)
        features = self.attention_block(features, depth_map) # (B, H_patches, W_patches, embedding_dimension)
        features = self.norm(features)
        features = features.permute(0, 3, 1, 2).contiguous() # (B, embedding_dimension, H_patches, W_patches)
        if self.pooling_head is None:
            self.pooling_head = create_pooling_head(
                pooling_method=self.pooling_method,
                feature_channels=self.embedding_dimension,
                spatial_height=H_patches,
                spatial_width=W_patches,
            )

        pooled_features = self.pooling_head(features)
        if has_time:
            pooled_features = pooled_features.reshape(B, T, *pooled_features.shape[1:])  # Batch, Time, Features

        return {EncoderOutputKeys.RGBD.value: pooled_features}


    def get_output_specification(self) -> EncoderOutput:
        return EncoderOutput(
            features=[EncoderOutputKeys.RGBD.value],
            dimensions={EncoderOutputKeys.RGBD.value: self.output_dim},
        )
