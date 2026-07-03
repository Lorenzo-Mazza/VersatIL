"""DFormerv2: Geometry-aware RGB+Depth encoder for imitation learning.

Based on: "DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation"
https://github.com/VCIP-RGBD/DFormer
"""

import enum

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from versatil.data.constants import CameraModality
from versatil.models.adaptation.lora import (
    LoRAAdaptation,
    apply_lora_config,
    is_lora_enabled,
)
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


class DFormerPretrainedWeights(enum.StrEnum):
    """Pretrained checkpoint families on the HuggingFace mirror."""

    IMAGENET = "imagenet"
    NYU = "nyu"
    SUNRGBD = "sunrgbd"


DFORMER_HUGGINGFACE_REPO = "bbynku/DFormerv2"

DFORMER_PRETRAINED_FILENAMES = {
    (DFormerVariant.SMALL.value, DFormerPretrainedWeights.IMAGENET.value): (
        "DFormerv2/pretrained/DFormerv2_Small_pretrained.pth"
    ),
    (DFormerVariant.BASE.value, DFormerPretrainedWeights.IMAGENET.value): (
        "DFormerv2/pretrained/DFormerv2_Base_pretrained.pth"
    ),
    (DFormerVariant.LARGE.value, DFormerPretrainedWeights.IMAGENET.value): (
        "DFormerv2/pretrained/DFormerv2_Large_pretrained.pth"
    ),
    (DFormerVariant.SMALL.value, DFormerPretrainedWeights.NYU.value): (
        "DFormerv2/NYU/DFormerv2_Small_NYU.pth"
    ),
    (DFormerVariant.BASE.value, DFormerPretrainedWeights.NYU.value): (
        "DFormerv2/NYU/DFormerv2_Base_NYU.pth"
    ),
    (DFormerVariant.LARGE.value, DFormerPretrainedWeights.NYU.value): (
        "DFormerv2/NYU/DFormerv2_Large_NYU.pth"
    ),
    (DFormerVariant.SMALL.value, DFormerPretrainedWeights.SUNRGBD.value): (
        "DFormerv2/SUNRGBD/DFormerv2_Small_SUNRGBD.pth"
    ),
    (DFormerVariant.BASE.value, DFormerPretrainedWeights.SUNRGBD.value): (
        "DFormerv2/SUNRGBD/DFormerv2_Base_SUNRGBD.pth"
    ),
    (DFormerVariant.LARGE.value, DFormerPretrainedWeights.SUNRGBD.value): (
        "DFormerv2/SUNRGBD/DFormerv2_Large_SUNRGBD.pth"
    ),
}


class DFormerStage(nn.Module):
    """Single DFormer stage with multiple geometric attention blocks and optional downsampling."""

    def __init__(
        self,
        embedding_dimension: int,
        number_of_heads: int,
        num_blocks: int,
        decomposition_mode: AttentionDecompositionMode,
        drop_path_rate: float = 0.0,
        use_layer_scale: bool = False,
        layer_scale_init_value: float = 1e-5,
        initial_decay: float = 2.0,
        decay_range: float = 4.0,
        ffn_expansion_factor: int = 4,
        downsample: nn.Module | None = None,
        use_raster_positions: bool = False,
        use_feedforward_convolution: bool = False,
    ):
        """Initialize DFormer stage.

        Args:
            embedding_dimension: Feature dimension for this stage
            number_of_heads: Number of attention heads
            num_blocks: Number of geometric attention blocks in this stage
            decomposition_mode: Attention computation strategy (full or separable)
            drop_path_rate: Stochastic depth rate
            use_layer_scale: Whether to use layer scaling
            layer_scale_init_value: Initial value for layer scale parameters
            initial_decay: Initial decay rate for spatial biases
            decay_range: Range of decay rates across heads
            ffn_expansion_factor: Expansion factor for FFN hidden dimension
            downsample: Optional downsampling module for next stage
            use_raster_positions: Whether rotary encoding uses flattened raster
                grid positions (the DFormerv2 reference convention).
            use_feedforward_convolution: Whether blocks use the DFormerv2 FFN
                with an inner depthwise convolution instead of a plain MLP.
        """
        super().__init__()
        self.embedding_dimension = embedding_dimension

        self.blocks = nn.ModuleList(
            [
                GeometricAttentionEncoderBlock(
                    decomposition_mode=decomposition_mode,
                    embedding_dimension=embedding_dimension,
                    number_of_heads=number_of_heads,
                    ffn_dimension=embedding_dimension * ffn_expansion_factor,
                    drop_path_rate=drop_path_rate,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                    initial_decay=initial_decay,
                    decay_range=decay_range,
                    use_raster_positions=use_raster_positions,
                    use_feedforward_convolution=use_feedforward_convolution,
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
            "number_of_heads": [4, 4, 8, 16],
            "decay_ranges": [4, 4, 6, 6],
            "ffn_ratios": [4, 4, 3, 3],
            "use_layer_scales": [False, False, False, False],
        },
        DFormerVariant.BASE.value: {
            "embed_dims": [80, 160, 320, 512],
            "depths": [4, 8, 25, 8],
            "number_of_heads": [5, 5, 10, 16],
            "decay_ranges": [5, 5, 6, 6],
            "ffn_ratios": [4, 4, 3, 3],
            "use_layer_scales": [False, False, True, True],
        },
        DFormerVariant.LARGE.value: {
            "embed_dims": [112, 224, 448, 640],
            "depths": [4, 8, 25, 8],
            "number_of_heads": [7, 7, 14, 20],
            "decay_ranges": [6, 6, 6, 6],
            "ffn_ratios": [4, 4, 3, 3],
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
        pretrained_weights: str = DFormerPretrainedWeights.IMAGENET.value,
        pooling_method: str = PoolingMethod.AVERAGE.value,
        model_dtype: str | None = None,
        lora_config: LoRAAdaptation | None = None,
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
            pretrained_weights: Which checkpoint family to download from
                https://huggingface.co/bbynku/DFormerv2 when ``pretrained``
                is set: the ImageNet backbone or the NYU/SUNRGBD finetuned
                models.
            model_dtype: Precision string from experiment config (e.g. ``"bf16-mixed"``).
            lora_config: Optional LoRA adapter configuration applied to the
                stage linears.
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
        if variant not in self.VARIANT_CONFIGS:
            raise ValueError(
                f"Variant '{variant}' not supported. "
                f"Choose from: {list(self.VARIANT_CONFIGS.keys())}"
            )
        weights_key = (variant, DFormerPretrainedWeights(pretrained_weights).value)
        self._setup_camera_keys(input_keys=self.input_specification.keys)
        self.variant = variant
        self.pooling_method = pooling_method
        self.decomposition_mode = AttentionDecompositionMode(decomposition_mode)
        config = self.VARIANT_CONFIGS[variant]
        self.embed_dims: list[int] = config["embed_dims"]
        self.depths: list[int] = config["depths"]
        self.number_of_heads: list[int] = config["number_of_heads"]
        self.decay_ranges: list[int] = config["decay_ranges"]
        self.ffn_ratios: list[int] = config["ffn_ratios"]
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
            checkpoint_path = hf_hub_download(
                repo_id=DFORMER_HUGGINGFACE_REPO,
                filename=DFORMER_PRETRAINED_FILENAMES[weights_key],
            )
            self._load_checkpoint(checkpoint_path)
        self.lora_config = lora_config
        self.stages = nn.ModuleList(
            [
                apply_lora_config(model=stage, lora_config=lora_config, frozen=frozen)
                for stage in self.stages
            ]
        )
        if is_lora_enabled(lora_config=lora_config):
            # PEFT freezes the wrapped stages; the patch embedding sits outside
            # them and must freeze too so only adapters train.
            for parameter in self.patch_embed.parameters():
                parameter.requires_grad = False
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
                # The reference merges with a biased conv followed by
                # BatchNorm; LayerNorm here would reject pretrained weights.
                downsample = PatchMerging(
                    dim=self.embed_dims[stage_idx],
                    out_dim=self.embed_dims[stage_idx + 1],
                    norm_layer=FrozenBatchNorm2d,
                    bias=True,
                )
            else:
                downsample = None
            # The reference runs decomposed attention on all stages except
            # the last, which always uses full attention.
            stage_decomposition_mode = (
                self.decomposition_mode
                if stage_idx < self.num_stages - 1
                else AttentionDecompositionMode.FULL
            )
            stage = DFormerStage(
                embedding_dimension=self.embed_dims[stage_idx],
                number_of_heads=self.number_of_heads[stage_idx],
                num_blocks=self.depths[stage_idx],
                decomposition_mode=stage_decomposition_mode,
                drop_path_rate=sum(stage_drop_paths) / len(stage_drop_paths),
                use_layer_scale=self.use_layer_scales[stage_idx],
                layer_scale_init_value=layer_scale_init_value,
                initial_decay=initial_decay,
                decay_range=self.decay_ranges[stage_idx],
                ffn_expansion_factor=self.ffn_ratios[stage_idx],
                downsample=downsample,
                use_raster_positions=True,
                use_feedforward_convolution=True,
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

    REFERENCE_KEY_REPLACEMENTS = (
        ("patch_embed.proj.", "patch_embed.projection."),
        ("layers.", "stages."),
        (".Attention.q_proj.", ".attention.query_projection."),
        (".Attention.k_proj.", ".attention.key_projection."),
        (".Attention.v_proj.", ".attention.value_projection."),
        (
            ".Attention.lepe.dwconv.",
            ".attention.learned_positional_encodings.convolution.",
        ),
        (".Attention.out_proj.", ".attention.output_projection."),
        (".cnn_pos_encode.dwconv.", ".input_positional_encoding.convolution."),
        (".layer_norm1.", ".norm1."),
        (".layer_norm2.", ".norm2."),
        (".ffn.fc1.", ".mlp.fc1."),
        (".ffn.fc2.", ".mlp.fc2."),
        (".ffn.dwconv.dwconv.", ".mlp.dwconv.convolution."),
        (".gamma_1", ".gamma1"),
        (".gamma_2", ".gamma2"),
        (".Geo.angle", ".attention.geometric_bias.rotary_encoding.frequencies"),
        (".Geo.decay", ".attention.geometric_bias.spatial_decay.decay_rates"),
        (".Geo.weight", ".attention.geometric_bias.bias_weights"),
    )

    @classmethod
    def _remap_reference_keys(
        cls, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Translate reference DFormerv2 checkpoint keys to this module tree.

        Args:
            state_dict: Checkpoint state dict in the official DFormerv2 naming.

        Returns:
            State dict with keys renamed to match ``DFormerEncoder``.
        """
        remapped = {}
        for key, value in state_dict.items():
            if key.startswith("extra_norms."):
                # The reference norms the outputs of stages 1..3 only; our
                # per-stage norms hold those weights one index later.
                stage_index = int(key.split(".")[1]) + 1
                remapped[f"stages.{stage_index}.norm.{key.split('.', 2)[2]}"] = value
                continue
            for reference_name, our_name in cls.REFERENCE_KEY_REPLACEMENTS:
                key = key.replace(reference_name, our_name)
            remapped[key] = value
        return remapped

    def _load_checkpoint(self, checkpoint_path: str):
        """Load pretrained weights from an official DFormerv2 checkpoint.

        Pretrained backbones for all variants are mirrored at
        https://huggingface.co/bbynku/DFormerv2.

        Raises:
            ValueError: If checkpoint tensors are left over or module weights
                stay uninitialized beyond the documented exceptions, which
                would silently train from partial weights.
        """
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
        remapped_state_dict = self._remap_reference_keys(cleaned_state_dict)

        incompatible = self.load_state_dict(remapped_state_dict, strict=False)
        # BatchNorm batch counters have no FrozenBatchNorm2d counterpart,
        # ImageNet-pretrained checkpoints carry classification heads and a
        # final norm this encoder does not use, and the NYU/SUNRGBD
        # checkpoints carry segmentation decode heads.
        unexpected = [
            key
            for key in incompatible.unexpected_keys
            if not key.endswith("num_batches_tracked")
            and not key.startswith(
                ("head.", "aux_head.", "norm.", "proj.", "decode_head.")
            )
        ]
        # Stage output norms exist only in segmentation checkpoints
        # (extra_norms); ImageNet backbones leave them at identity. The
        # trailing LayerNorm of PatchEmbedding is unused on the progressive
        # path.
        missing = [
            key
            for key in incompatible.missing_keys
            if ".norm." not in key and not key.startswith("patch_embed.norm.")
        ]
        if unexpected or missing:
            raise ValueError(
                "Pretrained DFormerv2 checkpoint did not load cleanly. "
                f"Unmatched checkpoint tensors: {sorted(unexpected)[:8]}; "
                f"uninitialized module weights: {sorted(missing)[:8]}."
            )

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
        rgb_key = self._camera_key_for_modality(modality=CameraModality.RGB)
        depth_key = self._camera_key_for_modality(modality=CameraModality.DEPTH)

        rgb = inputs[rgb_key]
        depth_map = inputs[depth_key]
        rgb_features = self.patch_embed(rgb)  # (B, H_patches, W_patches, C)
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
            depth_map = torch.zeros(1, 1, image_height, image_width, dtype=probe_dtype)
            for stage in self.stages:
                _, features, depth_map = stage(features, depth_map)
            _, spatial_height, spatial_width, _ = features.shape
        self._setup_pooling(spatial_height=spatial_height, spatial_width=spatial_width)
        self._apply_model_dtype()

    def get_explainability_targets(self) -> list[VisionExplanationTarget]:
        """Return the final DFormer stage for spatial attribution maps.

        DFormer stages return a tuple; ``output_index=0`` selects the stage
        output before downsampling. The output is NHWC and is converted by the
        explainability package before feature-grid map computation.

        Returns:
            One NHWC spatial feature-map target for the final DFormer stage.
        """
        return [
            VisionExplanationTarget(
                layer=self.stages[-1],
                target_kind=ExplanationTargetKind.SPATIAL_FEATURE_MAP.value,
                activation_layout=ActivationLayout.NHWC.value,
                output_index=0,
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
