"""DFormerv2: Geometry-aware RGB+Depth encoder for imitation learning.

Based on: "DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation"
https://github.com/VCIP-RGBD/DFormer
"""

import enum

import torch
import torch.nn as nn
import torch.nn.functional as F

from versatil.data.constants import RGB_CAMERAS, Cameras
from versatil.models.encoding.encoders.base import EncoderInput, EncoderOutput
from versatil.models.encoding.encoders.constants import (
    EncoderOutputKeys,
    PoolingMethod,
)
from versatil.models.encoding.encoders.unconditional import Encoder
from versatil.models.layers import FrozenBatchNorm2d, PatchEmbedding, PatchMerging
from versatil.models.layers.constants import AttentionDecompositionMode
from versatil.models.layers.geometric_attention.geometric_attention_encoder import (
    GeometricAttentionEncoderBlock,
)
from versatil.models.layers.patch_embedding import PatchEmbedType
from versatil.models.layers.pooling.pooling_head import create_pooling_head


class DFormerVariant(enum.StrEnum):
    """Available DFormerv2 model variants."""

    SMALL = "S"
    BASE = "B"
    LARGE = "L"


class DFormerStage(nn.Module):
    """Single DFormer stage with multiple geometric attention blocks and optional downsampling."""

    def __init__(
        self,
        embedding_dimension: int,
        num_heads: int,
        num_blocks: int,
        decomposition_mode: AttentionDecompositionMode,
        drop_path_rate: float = 0.0,
        use_layer_scale: bool = False,
        layer_scale_init_value: float = 1e-5,
        initial_decay: float = 2.0,
        decay_range: float = 4.0,
        ffn_expansion_factor: int = 4,
        downsample: nn.Module | None = None,
    ):
        """Initialize DFormer stage.

        Args:
            embedding_dimension: Feature dimension for this stage
            num_heads: Number of attention heads
            num_blocks: Number of geometric attention blocks in this stage
            decomposition_mode: Attention computation strategy (full or separable)
            drop_path_rate: Stochastic depth rate
            use_layer_scale: Whether to use layer scaling
            layer_scale_init_value: Initial value for layer scale parameters
            initial_decay: Initial decay rate for spatial biases
            decay_range: Range of decay rates across heads
            ffn_expansion_factor: Expansion factor for FFN hidden dimension
            downsample: Optional downsampling module for next stage
        """
        super().__init__()
        self.embedding_dimension = embedding_dimension

        self.blocks = nn.ModuleList(
            [
                GeometricAttentionEncoderBlock(
                    decomposition_mode=decomposition_mode,
                    embedding_dimension=embedding_dimension,
                    num_heads=num_heads,
                    ffn_dimension=embedding_dimension * ffn_expansion_factor,
                    drop_path_rate=drop_path_rate,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                    initial_decay=initial_decay,
                    decay_range=decay_range,
                )
                for _ in range(num_blocks)
            ]
        )

        self.downsample = downsample
        self.norm = nn.LayerNorm(embedding_dimension, eps=1e-6)

    def forward(
        self, rgb_features: torch.Tensor, depth_map: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the stage.

        Args:
            rgb_features: RGB features of shape (B, H, W, C)
            depth_map: Depth map of shape (B, 1, H, W)

        Returns:
            output_features: Normalized features for this stage output (B, H, W, C)
            next_features: Downsampled features for next stage (B, H', W', C') or same as output if no downsample
            depth_map: Resized depth map for next stage (B, 1, H', W') or same as input if no downsample
        """
        features = rgb_features
        for block in self.blocks:
            features = block(features, depth_map)
        output_features = self.norm(features)
        if self.downsample is not None:
            next_features = self.downsample(features)
            B, H_new, W_new, C_new = next_features.shape
            depth_map = F.interpolate(
                depth_map, size=(H_new, W_new), mode="bilinear", align_corners=False
            )
        else:
            next_features = output_features
        return output_features, next_features, depth_map


class DFormerEncoder(Encoder):
    """DFormerv2 encoder for RGB+Depth fusion using geometric self-attention.

    Hierarchical encoder with multi-scale feature extraction and depth-conditioned attention.
    """

    VARIANT_CONFIGS = {
        DFormerVariant.SMALL.value: {
            "embed_dims": [64, 128, 256, 512],
            "depths": [3, 4, 18, 4],
            "num_heads": [4, 4, 8, 16],
            "decay_ranges": [4, 4, 6, 6],
            "use_layer_scales": [False, False, False, False],
        },
        DFormerVariant.BASE.value: {
            "embed_dims": [80, 160, 320, 512],
            "depths": [4, 8, 25, 8],
            "num_heads": [5, 5, 10, 16],
            "decay_ranges": [5, 5, 6, 6],
            "use_layer_scales": [False, False, True, True],
        },
        DFormerVariant.LARGE.value: {
            "embed_dims": [112, 224, 448, 640],
            "depths": [4, 8, 25, 8],
            "num_heads": [7, 7, 14, 20],
            "decay_ranges": [6, 6, 6, 6],
            "use_layer_scales": [False, False, True, True],
        },
    }

    def __init__(
        self,
        input_keys: str | list[str],
        variant: str = DFormerVariant.SMALL.value,
        decomposition_mode: str = AttentionDecompositionMode.SEPARABLE.value,
        drop_path_rate: float = 0.1,
        layer_scale_init_value: float = 1e-6,
        initial_decay: float = 2.0,
        pretrained: bool = False,
        frozen: bool = False,
        checkpoint_path: str | None = None,
        pooling_method: str = PoolingMethod.AVERAGE.value,
    ):
        """Initialize DFormer encoder.

        Args:
            input_keys: Input keys for RGB and depth
            variant: Model variant (S/B/L)
            decomposition_mode: Attention computation strategy
            pooling_method: Feature pooling method (spatial_softmax or global_average)
            drop_path_rate: Stochastic depth rate
            layer_scale_init_value: Initial value for layer scale
            initial_decay: Initial decay rate for spatial biases
            pretrained: Whether to use pretrained weights
            frozen: Whether to freeze encoder weights
            checkpoint_path: Path to checkpoint for loading weights
        """
        specification = EncoderInput(
            keys=input_keys, required=[Cameras.DEPTH.value], one_of_groups=[RGB_CAMERAS]
        )
        super().__init__(
            input_specification=specification, pretrained=pretrained, frozen=frozen
        )
        if variant not in self.VARIANT_CONFIGS:
            raise ValueError(
                f"Variant '{variant}' not supported. "
                f"Choose from: {list(self.VARIANT_CONFIGS.keys())}"
            )
        if pretrained and checkpoint_path is None:
            raise ValueError(
                "Pretrained=True requires a valid checkpoint_path for DFormerEncoder."
            )
        self.variant = variant
        self.pooling_method = pooling_method
        self.decomposition_mode = AttentionDecompositionMode(decomposition_mode)
        config = self.VARIANT_CONFIGS[variant]
        self.embed_dims: list[int] = config["embed_dims"]
        self.depths: list[int] = config["depths"]
        self.num_heads: list[int] = config["num_heads"]
        self.decay_ranges: list[int] = config["decay_ranges"]
        self.use_layer_scales: list[bool] = config["use_layer_scales"]
        self.num_stages = len(self.embed_dims)
        # Patch size is fixed at 4 to match original DFormerv2 architecture (2 stride-2 convs = 4x downsample)
        # This cannot be changed without breaking pretrained model loading
        self.patch_embed = PatchEmbedding(
            patch_size=4,
            in_chans=3,
            embed_dim=self.embed_dims[0],
            embed_type=PatchEmbedType.PROGRESSIVE.value,
            norm_layer=FrozenBatchNorm2d,
        )
        self._build_backbone(
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            initial_decay=initial_decay,
        )
        self.feature_dim = self.embed_dims[-1]
        self._setup_pooling()
        if pretrained:
            self._load_checkpoint(checkpoint_path)
        if frozen:
            super()._freeze_weights()

    def _build_backbone(
        self,
        drop_path_rate: float,
        layer_scale_init_value: float = 1e-6,
        initial_decay: float = 2.0,
    ):
        """Build DFormer backbone with multiple stages.

        Args:
            drop_path_rate: Overall stochastic depth rate
            layer_scale_init_value: Initial value for layer scale parameters
            initial_decay: Initial decay rate for spatial biases
        """
        drop_path_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        self.stages = nn.ModuleList()
        depth_idx = 0
        for stage_idx in range(self.num_stages):
            stage_drop_paths = drop_path_rates[
                depth_idx : depth_idx + self.depths[stage_idx]
            ]
            if stage_idx < self.num_stages - 1:
                downsample = PatchMerging(
                    dim=self.embed_dims[stage_idx],
                    out_dim=self.embed_dims[stage_idx + 1],
                    norm_layer=nn.LayerNorm,
                )
            else:
                downsample = None
            stage = DFormerStage(
                embedding_dimension=self.embed_dims[stage_idx],
                num_heads=self.num_heads[stage_idx],
                num_blocks=self.depths[stage_idx],
                decomposition_mode=self.decomposition_mode,
                drop_path_rate=sum(stage_drop_paths) / len(stage_drop_paths),
                use_layer_scale=self.use_layer_scales[stage_idx],
                layer_scale_init_value=layer_scale_init_value,
                initial_decay=initial_decay,
                decay_range=self.decay_ranges[stage_idx],
                downsample=downsample,
            )
            self.stages.append(stage)
            depth_idx += self.depths[stage_idx]

    def _setup_pooling(self):
        """Setup pooling head based on final feature map size."""
        with torch.no_grad():
            mock_rgb = torch.zeros(1, 3, 224, 224)
            features = self.patch_embed(mock_rgb)
            depth_mock = torch.zeros(1, 1, 224, 224)
            depth_map = F.interpolate(
                depth_mock, size=features.shape[1:3], mode="bilinear"
            )
            for stage in self.stages:
                _, features, depth_map = stage(features, depth_map)
            B, H_final, W_final, C_final = features.shape

        mock_pooling_head = create_pooling_head(
            pooling_method=self.pooling_method,
            feature_channels=self.feature_dim,
            spatial_height=H_final,
            spatial_width=W_final,
        )
        self.pooling_head = (
            None  # Will be created in forward() with correct patch dimensions
        )
        self.output_dim = mock_pooling_head.get_output_dim(self.feature_dim)

    def _load_checkpoint(self, checkpoint_path: str):
        """Load pretrained weights from checkpoint."""
        state_dict = torch.load(checkpoint_path, map_location="cpu")

        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("backbone."):
                cleaned_state_dict[key[9:]] = value
            else:
                cleaned_state_dict[key] = value

        self.load_state_dict(cleaned_state_dict, strict=False)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass through DFormer encoder.

        Args:
            inputs: Dictionary with RGB and depth inputs
                - RGB: (B, C, H, W) or (B, T, C, H, W)
                - Depth: (B, 1, H, W) or (B, T, 1, H, W)

        Returns:
            Dictionary with RGBD features of shape (B, output_dim) or (B, T, output_dim)
        """
        rgb_key = [
            k
            for k in self.input_specification.keys
            if k in self.input_specification.one_of_groups[0]
        ][0]
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
        rgb_features, H_patches, W_patches = self.patch_embed(
            rgb, return_patch_size=True
        )  # (B, H_patches, W_patches, C)
        depth_map = F.interpolate(
            depth, size=(H_patches, W_patches), mode="bilinear", align_corners=False
        )
        features = rgb_features
        for stage in self.stages:
            output_features, next_features, depth_map = stage(features, depth_map)
            features = next_features

        final_features = features.permute(
            0, 3, 1, 2
        ).contiguous()  # (B, C, H_feature_maps, W_feature_maps)
        _, _, H_feature_maps, W_feature_maps = final_features.shape
        if self.pooling_head is None:
            self.pooling_head = create_pooling_head(
                pooling_method=self.pooling_method,
                feature_channels=self.feature_dim,
                spatial_height=H_patches,
                spatial_width=W_patches,
            ).to(final_features.device)
        pooled_features = self.pooling_head(final_features)
        if has_time:
            pooled_features = pooled_features.reshape(
                B, T, *pooled_features.shape[1:]
            )  # Batch, Time, Features

        return {EncoderOutputKeys.RGBD.value: pooled_features}

    def get_output_specification(self) -> EncoderOutput:
        return EncoderOutput(
            features=[EncoderOutputKeys.RGBD.value],
            dimensions={EncoderOutputKeys.RGBD.value: self.output_dim},
        )
