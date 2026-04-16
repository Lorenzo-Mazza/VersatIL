"""DFormerv2: Geometry-aware RGB+Depth encoder for imitation learning.

Based on: "DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation"
https://github.com/VCIP-RGBD/DFormer
"""

import enum

import torch
import torch.nn as nn
import torch.nn.functional as F

from versatil.data.constants import RGB_CAMERAS, Cameras
from versatil.data.metadata import BaseMetadata, CameraMetadata
from versatil.models.encoding.encoders.base import EncoderInput
from versatil.models.encoding.encoders.constants import (
    EncoderOutputKeys,
    PoolingMethod,
)
from versatil.models.encoding.encoders.image_mixin import RGBDEncoderMixin
from versatil.models.encoding.encoders.unconditional import Encoder
from versatil.models.feature_meta import FeatureMetadata, infer_feature_type
from versatil.models.layers import FrozenBatchNorm2d, PatchEmbedding, PatchMerging
from versatil.models.layers.constants import AttentionDecompositionMode
from versatil.models.layers.geometric_attention.geometric_attention_encoder import (
    GeometricAttentionEncoderBlock,
)
from versatil.models.layers.patch_embedding import PatchEmbedType
from versatil.models.layers.pooling.pooling_head import (
    PoolingHead,
    create_spatial_pooling_head,
)


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


class DFormerEncoder(RGBDEncoderMixin, Encoder):
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
        model_dtype: str | None = None,
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
            model_dtype: Precision string from experiment config (e.g. ``"bf16-mixed"``).
        """
        specification = EncoderInput(
            keys=input_keys, required=[Cameras.DEPTH.value], one_of_groups=[RGB_CAMERAS]
        )
        super().__init__(
            input_specification=specification,
            pretrained=pretrained,
            frozen=frozen,
            model_dtype=model_dtype,
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
        self.pooling_head: PoolingHead | None = None
        self.output_dim: int | tuple[int, ...] = self.feature_dim
        if pretrained:
            self._load_checkpoint(checkpoint_path)
        if frozen:
            super()._freeze_weights()
        self._apply_model_dtype()

    def _build_backbone(
        self,
        drop_path_rate: float,
        layer_scale_init_value: float = 1e-6,
        initial_decay: float = 2.0,
    ):
        """Build the hierarchical backbone with multiple DFormer stages.

        Args:
            drop_path_rate: Overall stochastic depth rate distributed across stages.
            layer_scale_init_value: Initial value for layer scale parameters.
            initial_decay: Initial decay rate for spatial biases.
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

    def _load_checkpoint(self, checkpoint_path: str):
        """Load pretrained weights from checkpoint."""
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

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

    def _encode_single_image(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "DFormerEncoder processes RGB+depth jointly. Use encode() instead."
        )

    def encode(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Encode RGB + depth through DFormer.

        Args:
            inputs: Dict with RGB as (B, C, H, W) and depth as (B, 1, H, W).

        Returns:
            Dict with RGBD features.
        """
        rgb_key = [
            k
            for k in self.input_specification.keys
            if k in self.input_specification.one_of_groups[0]
        ][0]
        depth_key = self.input_specification.required[0]

        rgb = inputs[rgb_key]
        depth = inputs[depth_key]
        rgb_features, patch_height, patch_width = self.patch_embed(
            rgb, return_patch_size=True
        )  # (B, H_patches, W_patches, C)
        depth_map = F.interpolate(
            depth,
            size=(patch_height, patch_width),
            mode="bilinear",
            align_corners=False,
        )
        features = rgb_features
        for stage in self.stages:
            output_features, next_features, depth_map = stage(features, depth_map)
            features = next_features

        if self.pooling_head is None:
            raise RuntimeError(
                "pooling_head is not initialized. Call set_image_size() before forward."
            )
        final_features = features.permute(0, 3, 1, 2).contiguous()
        pooled_features = self.pooling_head(final_features)
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
            features = self.patch_embed(mock_rgb)
            depth_mock = torch.zeros(1, 1, image_height, image_width, dtype=probe_dtype)
            depth_map = F.interpolate(
                depth_mock, size=features.shape[1:3], mode="bilinear"
            )
            for stage in self.stages:
                _, features, depth_map = stage(features, depth_map)
            _, spatial_height, spatial_width, _ = features.shape
        self._setup_pooling(spatial_height=spatial_height, spatial_width=spatial_width)
        self._apply_model_dtype()

    def validate_input_metadata(self, key: str, metadata: BaseMetadata) -> str | None:
        """Validate that RGB keys have 3-channel metadata and depth key is single-channel.

        Args:
            key: Observation key being validated.
            metadata: Metadata from the observation space for this key.

        Returns:
            Error message if incompatible, None if valid.
        """
        if not isinstance(metadata, CameraMetadata):
            return f"Expected CameraMetadata for '{key}', got {type(metadata).__name__}"
        if key == Cameras.DEPTH.value:
            if not metadata.is_single_channel:
                return (
                    f"Expected single-channel depth for '{key}', "
                    f"got {metadata.channels} channels"
                )
        else:
            if not metadata.is_rgb:
                return (
                    f"Expected 3-channel RGB for '{key}', "
                    f"got {metadata.channels} channels"
                )
        return None

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
