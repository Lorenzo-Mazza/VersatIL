# Encoding Pipeline

The `EncodingPipeline` orchestrates multi-modal observation encoding. It manages a collection of encoders, runs them in sequence, and optionally fuses their outputs before passing features to the decoder.

```
Observations --> [Encoder 1] --> features_A
               [Encoder 2] --> features_B   --> [Fusion] --> fused_AB
               [Encoder 3] --> features_C
                                                Final: {features_A, features_B, fused_AB, features_C}
```

All encoder and fusion outputs persist in the output dictionary.

## Encoder Types

All encoders subclass `Encoder` (unconditional) or `ConditionalEncoder` (conditional). Both inherit from the internal `EncodingMixin` abstract base. Encoders implement two core methods:

- `get_output_specification()` -- returns a `list[FeatureMetadata]` declaring feature keys, types, and dimensions
- `forward(inputs)` -- processes observation tensors and returns a feature dictionary

Modality mix-ins provide shared functionality:

- **`ImageEncoderMixin`** -- abstract base class with `_output_modality` and `_camera_group` abstract properties. Handles multi-camera encoding with automatic feature naming (`modality.camera_key` e.g. `rgb.left`). Encoders implement `_encode_single_image()`; the mixin handles iteration, resize, and feature registration. Per-camera image sizes are set from `CameraMetadata` via `set_image_size()`, not hardcoded in encoder configs. Three concrete subclasses:
    - **`RGBEncoderMixin(ImageEncoderMixin)`** -- for RGB camera encoders
    - **`DepthEncoderMixin(ImageEncoderMixin)`** -- for depth camera encoders
    - **`RGBDEncoderMixin(ImageEncoderMixin)`** -- for RGB+depth cross-modal encoders (DFormer, GeometricRGBD)
- **`LanguageEncoderMixin`** -- tokenized text extraction, padding/truncation, attention mask construction, and output padding mask generation.

Encoders are split into two categories:

- **Unconditional** (`Encoder`) -- standard encoders that process inputs independently
- **Conditional** (`ConditionalEncoder`) -- encoders that accept a conditioning tensor from another encoder's output (e.g., FiLM conditioning from language features)

Conditional encoders always run after unconditional encoders in the pipeline.

## FeatureMetadata

Encoders declare their outputs via `FeatureMetadata`, a frozen dataclass that travels from encoder through fusion to decoder validation:

```python
@dataclass(frozen=True)
class FeatureMetadata:
    key: str                   
    feature_type: str           
    dimension: tuple[int, ...] 
```

Feature types are classified by `FeatureType` enum values:

| Type | Value | Dimension | Produced When |
|---|---|---|---|
| **SPATIAL** | `"spatial"` | `(C, H, W)` | `pooling_method="none"` on SpatialRGBEncoder / SpatialDepthEncoder |
| **SEQUENTIAL** | `"sequential"` | `(S, D)` | `pooling_method="none"` on FlatRGBEncoder, or generative VLM output |
| **FLAT** | `"flat"` | `(D,)` | Any pooling method that produces a flat feature vector |

The decoder's `DecoderInput` validates feature types at initialization via `required_types` and `raises_for_types`, catching configuration errors before training starts.

## Multi-Camera Encoding

`ImageEncoderMixin` (via its subclasses `RGBEncoderMixin`, `DepthEncoderMixin`, `RGBDEncoderMixin`) automatically detects multi-camera setups from `input_keys` and generates output features with dotted naming:

| Setup | Input Keys | Output Keys |
|---|---|---|
| Single camera | `["left"]` | `rgb` |
| Multi-camera | `["left", "right"]` | `rgb.left`, `rgb.right` |


## RGB Encoders

### SpatialRGBEncoder

Any timm backbone that outputs `(B, C, H, W)` spatial feature maps. Covers CNNs (ResNet, EfficientNet, ConvNeXt, ConvNeXtV2, EdgeNeXt, MobileNetV4), Swin Transformers, TinyViT, and other spatial-output architectures. Handles both NCHW and NHWC output layouts transparently, and strict input size backbones.

- **Input:** RGB image `(B, 3, H, W)` or `(B, T, 3, H, W)` for temporal observations
- **Output key:** `rgb` (or `rgb.{camera}` for multi-camera)
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

### FlatRGBEncoder

Backbones that output `(B, S, D)` flat token sequences (ViT, DINOv2, DINOv3, DeiT, CLIP ViT). Uses timm `forward_features()`.

- **Input:** RGB image `(B, 3, H, W)`
- **Output key:** `rgb` (or `rgb.{camera}` for multi-camera)
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

### ConditionalCNNEncoder

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

### SpatialDepthEncoder

Adapts timm spatial backbones for single-channel depth images by setting `in_chans=1`. Same architecture support as `SpatialRGBEncoder`.

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

### ProprioceptiveEncoder

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

### LanguageEncoder

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

#### DFormerEncoder

Geometry-aware RGB+Depth encoder using geometric self-attention. Based on [DFormerV2](https://arxiv.org/abs/2504.04701).

- **Input:** RGB image + Depth image
- **Output key:** `rgbd`
- **Feature type:** FLAT (after pooling) or SPATIAL
- **Variants:** Small, Base, Large

Processes RGB and depth through parallel patch embedding streams and fuses them via geometric attention blocks that use depth-derived spatial relationships.

#### GeometricRGBDEncoder

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

### Vision-Language Encoders

#### TwoTowerVLMEncoder

CLIP-style dual-tower encoder with separate vision and language pathways. Produces independent features for each modality.

- **Input:** RGB image(s) + tokenized text
- **Output keys:** `rgb` (or `rgb.{camera}` for multi-camera), `language`, `language_padding_mask`
- **Feature type:** Per-output (FLAT or SEQUENTIAL depending on pooling)

```python
TwoTowerVLMEncoder(
    input_keys=["left", "tokenized_observations"],
    pretrained=True,
    frozen=True,
    model_name="openai/clip-vit-base-patch32",
    pooling_method="default",
)
```

Since two-tower VLMs always produce multiple outputs, fusion and decoder configs use dot notation to select features: `vlm_encoder.rgb`, `vlm_encoder.language`.

#### GenerativeVLMEncoder

Abstract base for single-stream generative VLMs that fuse vision and language in a single language model pass. The common flow: embed images, embed text, concatenate, run LM. Subclasses only implement model-specific image embedding.

- **Input:** RGB image(s) + tokenized text
- **Output keys:** `fused_rgb_language`, `fused_rgb_language_padding_mask`
- **Feature type:** SEQUENTIAL (fused image + text token sequence)

Two concrete subclasses:

##### PaliGemmaEncoder

PaliGemma2 models (Gemma2-based). Processes cameras individually.

```python
PaliGemmaEncoder(
    input_keys=["left", "tokenized_observations"],
    pretrained=True,
    frozen=True,
    model_name="google/paligemma2-3b-pt-224",
    use_embeddings_only=True,  # Return raw embeddings without running LM layers
)
```

##### SmolVLMEncoder

SmolVLM models (Llama-based). Stacks all camera images along the `num_images` dimension for joint processing.

```python
SmolVLMEncoder(
    input_keys=["left", "right", "tokenized_observations"],
    pretrained=True,
    frozen=True,
    model_name="HuggingFaceTB/SmolVLM-256M-Instruct",
    use_embeddings_only=True,
)
```

!!! info "Embeddings-only mode"
    When `use_embeddings_only=True`, the encoder returns raw image + text embeddings without running them through the LM layers. The LM layers remain available for interleaved decoders (Pi0, SmolVLA) via `get_backbone_layers()`.

## Available Backbones

### Spatial Backbones (`SpatialBackboneType`)

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
| `DINOV3_CONVNEXT_SMALL` | `convnext_small_dinov3.lvd1689m` |

### Flat Backbones (`FlatBackboneType`)

| Enum | Model ID |
|---|---|
| `VIT_BASE` | `vit_base_patch16_clip_224.laion2b_ft_in12k_in1k` |
| `DINOV2_VITS14` | `vit_small_patch14_dinov2.lvd142m` |
| `DINOV2_VITB14` | `vit_base_patch14_dinov2.lvd142m` |
| `DINOV2_VITL14` | `vit_large_patch14_dinov2.lvd142m` |
| `DINOV3_VITS16` | `vit_small_patch16_dinov3.lvd1689m` |
| `DINOV3_VITS16PLUS` | `vit_small_plus_patch16_dinov3.lvd1689m` |
| `DINOV3_VITB16` | `vit_base_patch16_dinov3.lvd1689m` |
| `DEIT_TINY` | `deit_tiny_patch16_224.fb_in1k` |
| `DEIT_SMALL` | `deit_small_patch16_224.fb_in1k` |
| `DEIT_BASE` | `deit_base_patch16_224.fb_in1k` |

### Language Models (`LanguageEncoderType`)

| Enum | Model ID |
|---|---|
| `BERT_BASE` | `bert-base-uncased` |
| `DISTILBERT_BASE` | `distilbert-base-uncased` |
| `MINI_LM_L6` | `sentence-transformers/all-MiniLM-L6-v2` |
| `GEMMA_2B` | `google/gemma-2b` |
| `QWEN_2_0_5B` | `Qwen/Qwen2-0.5B` |
| `QWEN_2_1_5B` | `Qwen/Qwen2-1.5B` |
| `ALBERT_BASE` | `albert-base-v2` |
| `ROBERTA_BASE` | `roberta-base` |
| `GPT2` | `gpt2` |
| `DEBERTA_V3_BASE` | `microsoft/deberta-v3-base` |
| `PHI_2` | `microsoft/phi-2` |
| `LLAMA_3_2_1B` | `meta-llama/Llama-3.2-1B` |

### Two-Tower VLMs (`ImageTextModelType`)

| Enum | Model ID |
|---|---|
| `CLIP_VITB32` | `openai/clip-vit-base-patch32` |
| `CLIP_VITB16` | `openai/clip-vit-base-patch16` |
| `CLIP_VITL14` | `openai/clip-vit-large-patch14` |
| `SIGLIP_BASE_PATCH16` | `google/siglip2-base-patch16-naflex` |
| `SIGLIP_SO400M` | `google/siglip-so400m-patch14-384` |

### Generative VLMs

**PaliGemma (`PaliGemmaModelType`):**

| Enum | Model ID |
|---|---|
| `PALIGEMMA2_3B_224` | `google/paligemma2-3b-pt-224` |
| `PALIGEMMA2_3B_448` | `google/paligemma2-3b-pt-448` |
| `PALIGEMMA2_3B_896` | `google/paligemma2-3b-pt-896` |

**SmolVLM (`SmolVLMModelType`):**

| Enum | Model ID |
|---|---|
| `SMOLVLM_256M` | `HuggingFaceTB/SmolVLM-256M-Instruct` |
| `SMOLVLM_500M` | `HuggingFaceTB/SmolVLM-500M-Instruct` |
| `SMOLVLM_2_2B` | `HuggingFaceTB/SmolVLM-2.2B-Instruct` |

Backbones are extended by adding new enum values in `src/versatil/models/encoding/encoders/constants.py` that map to timm or HuggingFace model identifiers.

## Pooling Methods

All vision and language encoders support configurable pooling via `PoolingMethod`:

| Method | Enum Value | Description |
|---|---|---|
| Default | `default` | CLS token (FlatRGBEncoder), max pooling (SpatialRGBEncoder), pooled output (VLM) |
| Average | `average_pooling` | Global Average Pooling (spatial encoders) or mean pooling (flat/sequential encoders) |
| Max | `max_pooling` | Global Max Pooling for spatial feature maps |
| Spatial Softmax | `spatial_softmax` | Spatial Softmax pooling for spatial feature maps |
| Learned Aggregation | `learned_aggregation` | Learned attention aggregation of patch tokens |
| None | `none` | Return full spatial/sequential features without pooling |

Setting pooling to `none` preserves spatial or sequential structure, producing SPATIAL `(C, H, W)` or SEQUENTIAL `(T, D)` features instead of FLAT.

## Fusion Modules

Fusion modules combine features from multiple encoders into a single representation. All fusion modules inherit from `FusionModule` and are set up lazily -- their layers are built after encoder output dimensions are known.

### ConcatFusion

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

### MLPFusion

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

### AttentionFusion

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

Each encoder type uses a specific output key from `EncoderOutputKeys`:

| Output Key | Value | Used By |
|---|---|---|
| `RGB` | `rgb` | SpatialRGBEncoder, FlatRGBEncoder, ConditionalCNNEncoder, TwoTowerVLMEncoder |
| `DEPTH` | `depth` | SpatialDepthEncoder |
| `RGBD` | `rgbd` | DFormerEncoder, GeometricRGBDEncoder |
| `PROPRIOCEPTIVE` | `proprio` | ProprioceptiveEncoder |
| `LANGUAGE` | `language` | LanguageEncoder, TwoTowerVLMEncoder |
| `FUSED_RGB_LANGUAGE` | `fused_rgb_language` | PaliGemmaEncoder, SmolVLMEncoder |
| `PADDING_MASK` | `padding_mask` | LanguageEncoder, TwoTowerVLMEncoder, PaliGemmaEncoder, SmolVLMEncoder |

!!! note "Multi-camera naming"
    For multi-camera encoders, output keys use dotted notation: `modality.camera_key`.

!!! note "Pipeline prefixing"
    The encoding pipeline always prepends each encoder's output key with the encoder name. Encoders return raw output keys (e.g., `rgb` or `rgb.left`), the pipeline produces prefixed keys (e.g., `eye_rgb`, `eye_rgb.left`).

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

For conditional encoders, subclass `ConditionalEncoder` instead and implement `encode(inputs, conditioning)`. The base `forward()` handles temporal flattening/unflattening and delegates to `encode()`.