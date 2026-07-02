# Encoding Pipeline

The [`EncodingPipeline`][versatil.models.encoding.pipeline.EncodingPipeline] orchestrates multi-modal observation encoding. It manages a collection of encoders, runs them in sequence, and optionally fuses their outputs before passing features to the decoder.

```
Observations --> [Encoder 1] --> features_A
               [Encoder 2] --> features_B   --> [Fusion] --> fused_AB
               [Encoder 3] --> features_C
                                                Final: {features_A, features_B, fused_AB, features_C}
```

All encoder and fusion outputs persist in the output dictionary.

## Encoder Types

All encoders subclass [`Encoder`][versatil.models.encoding.encoders.unconditional.Encoder] (unconditional) or [`ConditionalEncoder`][versatil.models.encoding.encoders.conditional.ConditionalEncoder] (conditional). Both inherit from the internal [`EncodingMixin`][versatil.models.encoding.encoders.base.EncodingMixin] abstract base. Encoders implement two core methods:

- `get_output_specification()` -- returns a `list[FeatureMetadata]` declaring feature keys, types, and dimensions
- `forward(inputs)` -- processes observation tensors and returns a feature dictionary

Modality mix-ins provide shared functionality:

- **[`ImageEncoderMixin`][versatil.models.encoding.encoders.image_mixin.ImageEncoderMixin]** -- abstract base class with `_output_modality` and camera metadata routing. Handles multi-camera encoding with automatic feature naming (`modality:camera_key` e.g. `rgb:left`). Encoders declare camera modality requirements through their input specification; experiment validation checks those requirements against [`RGBCameraMetadata`][versatil.data.metadata.RGBCameraMetadata] and [`DepthCameraMetadata`][versatil.data.metadata.DepthCameraMetadata] from the observation space. The encoding pipeline injects per-camera image sizes and calls encoder `set_image_size()` hooks. Three concrete subclasses:
    - **[`RGBEncoderMixin`][versatil.models.encoding.encoders.image_mixin.RGBEncoderMixin]** -- for RGB camera encoders
    - **[`DepthEncoderMixin`][versatil.models.encoding.encoders.image_mixin.DepthEncoderMixin]** -- for depth camera encoders
    - **[`RGBDEncoderMixin`][versatil.models.encoding.encoders.image_mixin.RGBDEncoderMixin]** -- for RGB+depth cross-modal encoders (DFormer, GeometricRGBD)
- **[`LanguageEncoderMixin`][versatil.models.encoding.encoders.language_mixin.LanguageEncoderMixin]** -- tokenized text extraction, padding/truncation, attention mask construction, and output padding mask generation.

Encoders are split into two categories:

- **Unconditional** ([`Encoder`][versatil.models.encoding.encoders.unconditional.Encoder]) -- standard encoders that process inputs independently
- **Conditional** ([`ConditionalEncoder`][versatil.models.encoding.encoders.conditional.ConditionalEncoder]) -- encoders that accept a conditioning tensor from another encoder's output (e.g., FiLM conditioning from language features)

Conditional encoders always run after unconditional encoders in the pipeline.

## [`FeatureMetadata`][versatil.models.feature_meta.FeatureMetadata]

Encoders declare their outputs via [`FeatureMetadata`][versatil.models.feature_meta.FeatureMetadata], a frozen dataclass that travels from encoder through fusion to decoder validation:

```python
@dataclass(frozen=True)
class FeatureMetadata:
    key: str                   
    feature_type: str           
    dimension: tuple[int, ...] 
```

Feature types are classified by [`FeatureType`][versatil.models.feature_meta.FeatureType] enum values:

| Type | Value | Dimension | Produced When |
|---|---|---|---|
| **SPATIAL** | `"spatial"` | `(C, H, W)` | `pooling_method="none"` on [`SpatialRGBEncoder`][versatil.models.encoding.encoders.rgb.spatial.SpatialRGBEncoder] / [`SpatialDepthEncoder`][versatil.models.encoding.encoders.depth.spatial.SpatialDepthEncoder] |
| **SEQUENTIAL** | `"sequential"` | `(S, D)` | `pooling_method="none"` on [`FlatRGBEncoder`][versatil.models.encoding.encoders.rgb.flat.FlatRGBEncoder] or token/language outputs |
| **FLAT** | `"flat"` | `(D,)` | Any pooling method that produces a flat feature vector |

The decoder's [`DecoderInput`][versatil.models.decoding.decoders.base.DecoderInput] validates feature types at initialization via `required_types` and `raises_for_types`, catching configuration errors before training starts.

## Multi-Camera Encoding

[`ImageEncoderMixin`][versatil.models.encoding.encoders.image_mixin.ImageEncoderMixin] (via its subclasses [`RGBEncoderMixin`][versatil.models.encoding.encoders.image_mixin.RGBEncoderMixin], [`DepthEncoderMixin`][versatil.models.encoding.encoders.image_mixin.DepthEncoderMixin], [`RGBDEncoderMixin`][versatil.models.encoding.encoders.image_mixin.RGBDEncoderMixin]) automatically detects multi-camera setups from `input_keys` and generates output features with `modality:camera_key` naming:

| Setup | Input Keys | Output Keys |
|---|---|---|
| Single camera | `["left"]` | `rgb` |
| Multi-camera | `["left", "right"]` | `rgb:left`, `rgb:right` |


## RGB Encoders

### [`SpatialRGBEncoder`][versatil.models.encoding.encoders.rgb.spatial.SpatialRGBEncoder]

Any timm backbone that outputs `(B, C, H, W)` spatial feature maps. Covers CNNs (ResNet, EfficientNet, ConvNeXt, ConvNeXtV2, EdgeNeXt, MobileNetV4), Swin Transformers, TinyViT, and other spatial-output architectures. Handles both NCHW and NHWC output layouts transparently, and strict input size backbones.

- **Input:** RGB image `(B, 3, H, W)` or `(B, T, 3, H, W)` for temporal observations
- **Output key:** `rgb` (or `rgb:{camera}` for multi-camera)
- **Feature type:** FLAT (after pooling) or SPATIAL (without pooling)
- **Pooling:** Average, Max, Spatial Softmax, Learned Aggregation, or None

```python
SpatialRGBEncoder(
    input_keys="left",
    backbone="timm/resnet18.a1_in1k",
    pooling_method="average_pooling",
    batch_norm_handling="frozen",
    pretrained=True,
)
```

!!! info "BatchNorm handling"
    BatchNorm is problematic with temporal data: reshaping `(B, T, C, H, W)` to `(B*T, C, H, W)` causes batch statistics to mix frames across time. Options: `frozen` (preserves pretrained stats), `groupnorm` (per-sample stats), or `default` (keep as-is).

### [`FlatRGBEncoder`][versatil.models.encoding.encoders.rgb.flat.FlatRGBEncoder]

Backbones that output `(B, S, D)` flat token sequences (ViT, DINOv2, DINOv3, DeiT, CLIP ViT). Uses timm `forward_features()`.

- **Input:** RGB image `(B, 3, H, W)`
- **Output key:** `rgb` (or `rgb:{camera}` for multi-camera)
- **Feature type:** FLAT (with pooling) or SEQUENTIAL (without pooling, returns patch tokens)
- **Supports:** Dynamic image sizes

```python
FlatRGBEncoder(
    input_keys="left",
    backbone="timm/vit_base_patch14_dinov2.lvd142m",
    pooling_method="default",  # Uses CLS token
    pretrained=True,
    frozen=True,
)
```

### [`DinoV2SigLIPRGBEncoder`][versatil.models.encoding.encoders.rgb.dinov2_siglip.DinoV2SigLIPRGBEncoder]

Paired DINOv2+SigLIP RGB encoder that runs two timm flat vision towers, applies
the tower-specific image standardization, and concatenates their patch tokens.

- **Input:** RGB image `(B, 3, H, W)`
- **Output key:** `rgb` (or `rgb:{camera}` for multi-camera)
- **Feature type:** SEQUENTIAL
- **Supported paired backbones:** 224px and 384px DINOv2+SigLIP variants

### [`ConditionalCNNEncoder`][versatil.models.encoding.encoders.rgb.conditional_cnn.ConditionalCNNEncoder]

ResNet with FiLM (Feature-wise Linear Modulation) conditioning. Each residual block receives a conditioning vector that modulates feature maps via learned affine transformations.

- **Input:** RGB image + conditioning tensor from another encoder
- **Output key:** `rgb`
- **Feature type:** FLAT (after pooling)
- **Supported backbones:** ResNet18, ResNet34 only

```python
ConditionalCNNEncoder(
    input_keys="left",
    condition_key="language_encoder_language",  # Feature from language encoder
    condition_dim=768,
    backbone="timm/resnet18.a1_in1k",
    pooling_method="spatial_softmax",
)
```


## Depth Encoders

### [`SpatialDepthEncoder`][versatil.models.encoding.encoders.depth.spatial.SpatialDepthEncoder]

Adapts timm spatial backbones for single-channel depth images by setting `in_chans=1`. Same architecture support as [`SpatialRGBEncoder`][versatil.models.encoding.encoders.rgb.spatial.SpatialRGBEncoder].

- **Input:** Depth image `(B, 1, H, W)`
- **Output key:** `depth`
- **Feature type:** FLAT (after pooling) or SPATIAL (without pooling)

```python
SpatialDepthEncoder(
    input_keys="depth",
    backbone="timm/resnet18.a1_in1k",
    pooling_method="average_pooling",
)
```

## Proprioceptive Encoder

### [`ProprioceptiveEncoder`][versatil.models.encoding.encoders.proprioceptive.base.ProprioceptiveEncoder]

MLP-based encoder for robot state vectors (joint positions, velocities, gripper state, etc.).

- **Input:** State vector `(B, D)` or `(B, T, D)`
- **Output key:** `proprio`
- **Feature type:** FLAT

Multiple proprioceptive keys are concatenated along the last dimension before encoding.

```python
ProprioceptiveEncoder(
    input_keys=["proprio_robot_frame", "gripper_state_obs"],
    output_dim=64,
    hidden_dims=[128],
    activation="relu",
)
```

## Language Encoder

### [`LanguageEncoder`][versatil.models.encoding.encoders.language.language.LanguageEncoder]

Text encoder using HuggingFace Transformers models. Requires tokenized input from the data pipeline.

- **Input:** Tokenized text + attention mask
- **Output keys:** `language` and `language_padding_mask` (always both, with dimensions depending on pooling method)
- **Feature type:** FLAT (with pooling) or SEQUENTIAL (without pooling)
- **Supports:** Embedding-only mode for lightweight token embeddings

```python
LanguageEncoder(
    pretrained=True,
    frozen=True,
    model_name="bert-base-uncased",
    pooling_method="default",  # Uses CLS token
)
```
## Cross-Modal Encoders

## RGBD Encoders

#### [`DFormerEncoder`][versatil.models.encoding.encoders.cross_modal.rgbd.dformerv2.DFormerEncoder]

Geometry-aware RGB+Depth encoder using geometric self-attention. Based on [DFormerV2](https://arxiv.org/abs/2504.04701).

- **Input:** RGB image + Depth image
- **Output key:** `rgbd`
- **Feature type:** FLAT (after pooling) or SPATIAL
- **Variants:** Small, Base, Large

Processes RGB and depth through parallel patch embedding streams and fuses them via geometric attention blocks that use depth-derived spatial relationships.

Pretrained backbone checkpoints (S/B/L, from the official
[DFormer repository](https://github.com/VCIP-RGBD/DFormer)) are mirrored at
[bbynku/DFormerv2](https://huggingface.co/bbynku/DFormerv2) on the HuggingFace
Hub. With `pretrained: true` the selected checkpoint is downloaded into the
HuggingFace cache automatically; `pretrained_weights` picks between the
ImageNet backbone (default) and the NYU/SUNRGBD finetuned models. LoRA
adapters can be enabled through `lora_config`, like the other encoders.

#### [`GeometricRGBDEncoder`][versatil.models.encoding.encoders.cross_modal.rgbd.geometric_rgbd.GeometricRGBDEncoder]

Single-layer geometry-aware RGBD encoder. A lightweight alternative to DFormerV2 with a single geometric attention block.

- **Input:** RGB image + Depth image (requires both)
- **Output key:** `rgbd`
- **Feature type:** FLAT (after pooling) or SPATIAL (without pooling)

```python
GeometricRGBDEncoder(
    input_keys=["left", "depth"],
    embedding_dimension=512,
    num_heads=8,
    decomposition_mode="separable",
)
```

### Vision-Language-Model(VLM) Encoders

#### [`VLMEncoder`][versatil.models.encoding.encoders.cross_modal.vision_language.vlm_encoder.VLMEncoder]

CLIP-style VLM encoder with separate vision and language pathways. Produces independent features for each modality.

- **Input:** RGB image(s) + tokenized text
- **Output keys:** `rgb` (or `rgb:{camera}` for multi-camera), `language`, `language_padding_mask`
- **Feature type:** Per-output (FLAT or SEQUENTIAL depending on pooling)

```python
VLMEncoder(
    input_keys=["left"],
    pretrained=True,
    frozen=True,
    model_name="openai/clip-vit-base-patch32",
    pooling_method="default",
)
```

VLM encoder configs list only vision keys. The observation tokenizer routes
language automatically through the internal `tokenized_observations` key.
Since VLM encoders can produce multiple outputs, fusion and decoder configs
select the pipeline-prefixed feature names, e.g. `left_rgb` and
`left_language`.

!!! info "Generative Language Models"
    PaliGemma, Prismatic, SmolVLM, and similar causal vision-language models are not encoding-pipeline encoders. They live under `versatil.models.decoding.generative_language_models` and are owned by VLA decoders through `policy.decoder.vlm_backbone`. See [VLA decoders](decoders.md#vla-decoders).

## Available Backbones

### Spatial Backbones ([`SpatialBackboneType`][versatil.models.encoding.encoders.constants.SpatialBackboneType])

| Enum | Model ID |
|---|---|
| `RESNET18` | `resnet18.a1_in1k` |
| `RESNET34` | `resnet34.a1_in1k` |
| `RESNET50` | `resnet50.a1_in1k` |
| `EFFICIENTNET_B0` | `efficientnet_b0.ra_in1k` |
| `EFFICIENTNET_B2` | `efficientnet_b2.ra_in1k` |
| `EDGENEXT_XX_SMALL` | `edgenext_xx_small.in1k` |
| `EDGENEXT_X_SMALL` | `edgenext_x_small.in1k` |
| `EDGENEXT_SMALL` | `edgenext_small.usi_in1k` |
| `EDGENEXT_BASE` | `edgenext_base.usi_in1k` |
| `MOBILENETV4_SMALL_050` | `mobilenetv4_conv_small_050.e3000_r224_in1k` |
| `CONVNEXT_NANO` | `convnext_nano.in12k_ft_in1k` |
| `CONVNEXT_TINY` | `convnext_tiny.fb_in22k_ft_in1k` |
| `CONVNEXT_BASE` | `convnext_base.fb_in22k_ft_in1k` |
| `CONVNEXTV2_NANO` | `convnextv2_nano.fcmae_ft_in22k_in1k` |
| `TINYVIT_21M` | `tiny_vit_21m_224.dist_in22k_ft_in1k` |
| `SWIN_TINY` | `swin_tiny_patch4_window7_224.ms_in22k_ft_in1k` |
| `SWIN_BASE` | `swin_base_patch4_window7_224.ms_in22k_ft_in1k` |
| `DINOV3_CONVNEXT_SMALL` | `convnext_small.dinov3_lvd1689m` |

### Flat Backbones ([`FlatBackboneType`][versatil.models.encoding.encoders.constants.FlatBackboneType])

| Enum | Model ID |
|---|---|
| `VIT_BASE` | `vit_base_patch16_clip_224.laion2b_ft_in12k_in1k` |
| `CLIP_VITL14_224_OPENAI` | `vit_large_patch14_clip_224.openai` |
| `CLIP_VITL14_336_OPENAI` | `vit_large_patch14_clip_336.openai` |
| `DINOV2_VITS14` | `vit_small_patch14_dinov2.lvd142m` |
| `DINOV2_VITB14` | `vit_base_patch14_dinov2.lvd142m` |
| `DINOV2_VITL14` | `vit_large_patch14_dinov2.lvd142m` |
| `DINOV2_VITL14_REG4` | `vit_large_patch14_reg4_dinov2.lvd142m` |
| `IN1K_VITL16_224` | `vit_large_patch16_224.augreg_in21k_ft_in1k` |
| `DINOV3_VITS16` | `vit_small_patch16_dinov3.lvd1689m` |
| `DINOV3_VITS16PLUS` | `vit_small_plus_patch16_dinov3.lvd1689m` |
| `DINOV3_VITB16` | `vit_base_patch16_dinov3.lvd1689m` |
| `DEIT_TINY` | `deit_tiny_patch16_224.fb_in1k` |
| `DEIT_SMALL` | `deit_small_patch16_224.fb_in1k` |
| `DEIT_BASE` | `deit_base_patch16_224.fb_in1k` |
| `SIGLIP_BASE_B16_224` | `vit_base_patch16_siglip_224` |
| `SIGLIP_BASE_B16_256` | `vit_base_patch16_siglip_256` |
| `SIGLIP_BASE_B16_384` | `vit_base_patch16_siglip_384` |
| `SIGLIP_SO400M_224` | `vit_so400m_patch14_siglip_224` |
| `SIGLIP_SO400M_384` | `vit_so400m_patch14_siglip_384` |

### DINOv2+SigLIP Paired Backbones ([`DinoV2SigLIPBackboneType`][versatil.models.encoding.encoders.constants.DinoV2SigLIPBackboneType])

| Enum | Model ID |
|---|---|
| `DINOV2_SIGLIP_VIT_SO_224PX` | `dinosiglip-vit-so-224px` |
| `DINOV2_SIGLIP_VIT_SO_384PX` | `dinosiglip-vit-so-384px` |

### Language Encoder Models ([`LanguageEncoderType`][versatil.models.encoding.encoders.constants.LanguageEncoderType])

| Enum | Model ID |
|---|---|
| `BERT_BASE` | `bert-base-uncased` |
| `DISTILBERT_BASE` | `distilbert-base-uncased` |
| `MINI_LM_L6` | `sentence-transformers/all-MiniLM-L6-v2` |
| `MINI_LM_L12` | `sentence-transformers/all-MiniLM-L12-v2` |
| `EMBEDDINGGEMMA_300M` | `google/embeddinggemma-300m` |
| `QWEN_3_EMBEDDING_0_6B` | `Qwen/Qwen3-Embedding-0.6B` |
| `BGE_BASE_EN_V1_5` | `BAAI/bge-base-en-v1.5` |
| `LLAMA_EMBED_NEMOTRON_8B` | `nvidia/llama-embed-nemotron-8b` |
| `LLAMA_NEMOTRON_EMBED_1B_V2` | `nvidia/llama-nemotron-embed-1b-v2` |
| `GTE_QWEN2_1_5B_INSTRUCT` | `Alibaba-NLP/gte-Qwen2-1.5B-instruct` |
| `JINA_EMBEDDINGS_V3` | `jinaai/jina-embeddings-v3` |
| `E5_BASE` | `intfloat/e5-base` |
| `ALBERT_BASE` | `albert-base-v2` |
| `ROBERTA_BASE` | `roberta-base` |
| `DEBERTA_V3_BASE` | `microsoft/deberta-v3-base` |
| `DISTIL_ROBERTA_BASE` | `distilbert/distilroberta-base` |

### VLM Encoder Backbones ([`ImageTextModelType`][versatil.models.encoding.encoders.constants.ImageTextModelType])

| Enum | Model ID |
|---|---|
| `CLIP_VITB32` | `openai/clip-vit-base-patch32` |
| `CLIP_VITB16` | `openai/clip-vit-base-patch16` |
| `CLIP_VITL14` | `openai/clip-vit-large-patch14` |
| `SIGLIP_BASE_PATCH16` | `google/siglip2-base-patch16-naflex` |
| `SIGLIP_SO400M` | `google/siglip-so400m-patch14-384` |

Backbones are extended by adding new enum values in `src/versatil/models/encoding/encoders/constants.py` that map to timm or HuggingFace model identifiers.

## Pooling Methods

All vision and language encoders support configurable pooling via [`PoolingMethod`][versatil.models.encoding.encoders.constants.PoolingMethod]:

| Method | Enum Value | Description |
|---|---|---|
| Default | `default` | CLS token ([`FlatRGBEncoder`][versatil.models.encoding.encoders.rgb.flat.FlatRGBEncoder]), max pooling ([`SpatialRGBEncoder`][versatil.models.encoding.encoders.rgb.spatial.SpatialRGBEncoder]), pooled output (VLM) |
| Average | `average_pooling` | Global Average Pooling (spatial encoders) or mean pooling (flat/sequential encoders) |
| Max | `max_pooling` | Global Max Pooling for spatial feature maps |
| Spatial Softmax | `spatial_softmax` | Spatial Softmax pooling for spatial feature maps |
| Learned Aggregation | `learned_aggregation` | Learned attention aggregation of patch tokens |
| None | `none` | Return full spatial/sequential features without pooling |

Setting pooling to `none` preserves spatial or sequential structure, producing SPATIAL `(C, H, W)` or SEQUENTIAL `(S, D)` features instead of FLAT.

## Fusion Modules

Fusion modules combine features from multiple encoders into a single representation. All fusion modules inherit from [`FusionModule`][versatil.models.encoding.fusion.base.FusionModule] and are set up lazily -- their layers are built after encoder output dimensions are known.

### [`ConcatFusion`][versatil.models.encoding.fusion.concat.ConcatFusion]

Projects each input feature to a shared `hidden_dim`, then concatenates along the last dimension.

- **Output dimension:** `hidden_dim * num_inputs`
- **Feature type:** FLAT or SEQUENTIAL (preserves input structure)

```python
ConcatFusion(
    input_features=["left_encoder_rgb", "right_encoder_rgb"],
    output_name="fused_visual",
    hidden_dim=256,
)
# Output dim: 256 * 2 = 512
```

### [`MLPFusion`][versatil.models.encoding.fusion.mlp.MLPFusion]

Projects, concatenates, then applies an MLP for non-linear fusion.

- **Output dimension:** last element of `mlp_hidden_dims`
- **Feature type:** FLAT or SEQUENTIAL

```python
MLPFusion(
    input_features=["rgb_encoder_rgb", "state_encoder_proprio"],
    output_name="fused_obs",
    hidden_dim=256,
    mlp_hidden_dims=[512, 256],
    activation_name="gelu",
    dropout=0.1,
)
# Output dim: 256
```

### [`AttentionFusion`][versatil.models.encoding.fusion.attention.AttentionFusion]

Projects features to a shared dimension and applies multi-head cross-attention. One feature serves as the query, the rest as key-value pairs.

- **Output dimension:** `hidden_dim`
- **Feature type:** FLAT or SEQUENTIAL

```python
AttentionFusion(
    input_features=["left_encoder_rgb", "right_encoder_rgb", "depth_encoder_depth"],
    output_name="fused_visual",
    hidden_dim=256,
    input_feature_query="left_encoder_rgb",  # Uses left camera as query
    num_heads=8,
    use_residual=True,
    use_norm=True,
)
# Output dim: 256
```

If `input_feature_query` is not specified, the first feature in the list is used as the query.

## Feature Naming

### Encoder Output Keys

Each encoder type uses a specific output key from [`EncoderOutputKeys`][versatil.models.encoding.encoders.constants.EncoderOutputKeys]:

| Output Key | Value | Used By |
|---|---|---|
| `RGB` | `rgb` | [`SpatialRGBEncoder`][versatil.models.encoding.encoders.rgb.spatial.SpatialRGBEncoder], [`FlatRGBEncoder`][versatil.models.encoding.encoders.rgb.flat.FlatRGBEncoder], [`ConditionalCNNEncoder`][versatil.models.encoding.encoders.rgb.conditional_cnn.ConditionalCNNEncoder], [`VLMEncoder`][versatil.models.encoding.encoders.cross_modal.vision_language.vlm_encoder.VLMEncoder] |
| `DEPTH` | `depth` | [`SpatialDepthEncoder`][versatil.models.encoding.encoders.depth.spatial.SpatialDepthEncoder] |
| `RGBD` | `rgbd` | [`DFormerEncoder`][versatil.models.encoding.encoders.cross_modal.rgbd.dformerv2.DFormerEncoder], [`GeometricRGBDEncoder`][versatil.models.encoding.encoders.cross_modal.rgbd.geometric_rgbd.GeometricRGBDEncoder] |
| `PROPRIOCEPTIVE` | `proprio` | [`ProprioceptiveEncoder`][versatil.models.encoding.encoders.proprioceptive.base.ProprioceptiveEncoder] |
| `LANGUAGE` | `language` | [`LanguageEncoder`][versatil.models.encoding.encoders.language.language.LanguageEncoder], [`VLMEncoder`][versatil.models.encoding.encoders.cross_modal.vision_language.vlm_encoder.VLMEncoder] |
| `PADDING_MASK` | `padding_mask` | [`LanguageEncoder`][versatil.models.encoding.encoders.language.language.LanguageEncoder], [`VLMEncoder`][versatil.models.encoding.encoders.cross_modal.vision_language.vlm_encoder.VLMEncoder] |

!!! note "Multi-camera naming"
    For multi-camera encoders, output keys use the format `modality:camera_key`.

!!! note "Pipeline prefixing"
    The encoding pipeline always prepends each encoder's output key with the encoder name. Encoders return raw output keys (e.g., `rgb` or `rgb:left`), the pipeline produces prefixed keys (e.g., `eye_rgb`, `eye_rgb:left`).

## Adding a New Encoder

### 1. Define the config dataclass

```python
# src/versatil/configs/encoding/encoder.py
@dataclass
class MyEncoderConfig(EncoderConfig):
    _target_: str = "versatil.models.encoding.encoders.my_module.MyEncoder"
    feature_dim: int = 256
```

### 2. Implement the encoder

```python
# src/versatil/models/encoding/encoders/my_module.py
from versatil.models.encoding.encoders.base import EncoderInput
from versatil.models.encoding.encoders.unconditional import Encoder
from versatil.models.feature_meta import FeatureMetadata, FeatureType


class MyEncoder(Encoder):

    def __init__(
        self,
        input_keys: str | list[str],
        feature_dim: int = 256,
        pretrained: bool = False,
        frozen: bool = False,
    ):
        specification = EncoderInput(keys=input_keys)
        super().__init__(
            input_specification=specification,
            pretrained=pretrained,
            frozen=frozen,
        )
        self.feature_dim = feature_dim
        # Build your network layers here

    def get_output_specification(self) -> list[FeatureMetadata]:
        return [
            FeatureMetadata(
                key="my_feature",
                feature_type=FeatureType.FLAT.value,
                dimension=(self.feature_dim,),
            )
        ]

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        x = inputs[self.input_specification.keys[0]]
        encoded = self.encode(x)  # Your encoding logic
        return {"my_feature": encoded}
```

### 3. Register in the config store and add tests

- Register the config dataclass in `src/versatil/configs/__init__.py`
- Add a YAML config in `hydra_configs/policy/encoding_pipeline/`
- Write tests in `tests/models/encoding/`

For conditional encoders, subclass [`ConditionalEncoder`][versatil.models.encoding.encoders.conditional.ConditionalEncoder] instead and implement `encode(inputs, conditioning)`. The base `forward()` handles temporal flattening/unflattening and delegates to `encode()`.
