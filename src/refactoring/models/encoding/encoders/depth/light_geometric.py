"""Lightweight geometry-aware RGB+Depth encoder."""


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
    """Single-layer geometry-aware RGB+Depth encoder."""
    def __init__(
        self,
        input_keys: str | list[str],
        embedding_dimension: int = 512,
        num_heads: int = 8,
        ffn_dimension: int = 2048,
        decomposition_mode: str = AttentionDecompositionMode.SEPARABLE.value,
        initial_decay: float = 2.0,
        decay_range: float = 4.0,
        image_size: int = 224,
        pooling_method: str = PoolingMethod.AVERAGE.value,
        pretrained: bool = False,
        frozen: bool = False,
    ):
        specification = EncoderInput(keys=input_keys,required=[Cameras.DEPTH.value], one_of_groups=[[Cameras.LEFT.value, Cameras.RIGHT.value]])
        super().__init__(input_specification=specification, pretrained=pretrained, frozen=frozen)
        self.embedding_dimension = embedding_dimension
        self.decomposition_mode = AttentionDecompositionMode(decomposition_mode)
        self.pooling_method = pooling_method
        self.image_size = image_size
        self.patch_embed = PatchEmbedding(
            patch_size=16,
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
        """Setup pooling head."""
        patch_size = self.image_size // 16
        self.pooling_head = create_pooling_head(
            pooling_method=self.pooling_method,
            feature_channels=self.embedding_dimension,
            spatial_height=patch_size,
            spatial_width=patch_size,
        )
        self.output_dim = self.pooling_head.get_output_dim(self.embedding_dimension)



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

        features = self.patch_embed(rgb)
        features = self.norm(features)
        H_patches = W_patches = int(features.shape[1] ** 0.5)
        features = features.reshape(B if not has_time else B*T, H_patches, W_patches, self.embedding_dimension)
        depth_map = F.interpolate(depth, size=(H_patches, W_patches), mode='bilinear', align_corners=False)
        features = self.attention_block(features, depth_map)
        features = self.norm(features)
        features = features.permute(0, 3, 1, 2).contiguous() # (B, C, H, W)
        pooled_features = self.pooling_head(features)
        if has_time:
            pooled_features = pooled_features.reshape(B, T, *pooled_features.shape[1:])  # Batch, Time, Features

        return {EncoderOutputKeys.RGBD.value: pooled_features}


    def get_output_specification(self) -> EncoderOutput:
        return EncoderOutput(
            features=[EncoderOutputKeys.RGBD.value],
            dimensions={EncoderOutputKeys.RGBD.value: self.output_dim},
        )
