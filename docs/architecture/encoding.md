# Encoding Pipeline

The `EncodingPipeline` orchestrates multi-modal observation encoding. It manages a collection of encoders, runs them in sequence, and optionally fuses their outputs before passing features to the decoder.

```
Observations ─→ [Encoder 1] ─→ features_A
               [Encoder 2] ─→ features_B   ─→ [Fusion] ─→ fused_AB
               [Encoder 3] ─→ features_C
                                                Final: {fused_AB, features_C}
```

For the full API, see the [EncodingPipeline reference](../reference/versatil/models/encoding/pipeline.md).

## Encoder Types

All encoders subclass `Encoder` (unconditional) or `ConditionalEncoder` (conditional). Both inherit from the internal `EncodingMixin` abstract base. Encoders implement two core methods:

- `get_output_specification()` -- returns an `EncoderOutput` declaring feature names and dimensions
- `forward(inputs)` -- processes observation tensors and returns a feature dictionary

Encoders are split into two categories:

- **Unconditional** (`Encoder`) -- standard encoders that process inputs independently
- **Conditional** (`ConditionalEncoder`) -- encoders that accept a conditioning tensor from another encoder's output (e.g., FiLM conditioning from language features)

Conditional encoders always run after unconditional encoders in the pipeline.

### RGB Encoders

#### CNNEncoder

Convolutional encoder supporting any CNN backbone from the [timm](https://github.com/huggingface/pytorch-image-models) library.

- **Input:** RGB image `(B, 3, H, W)` or `(B, T, 3, H, W)` for temporal observations
- **Output key:** `rgb`
- **Feature type:** FLAT (after pooling) or SPATIAL (without pooling)
- **Pooling:** Average, Max, Spatial Softmax, Learned Aggregation, or None

```python
CNNEncoder(
    input_keys="left",
    backbone="timm/resnet18.a1_in1k",
    pooling_method="average_pooling",
    batch_norm_handling="frozen",
    pretrained=True,
)
```

!!! info "BatchNorm handling"
    BatchNorm is problematic with temporal data: reshaping `(B, T, C, H, W)` to `(B*T, C, H, W)` causes batch statistics to mix frames across time. Options: `frozen` (preserves pretrained stats), `groupnorm` (per-sample stats), or `default` (keep as-is).

#### ViTEncoder

Vision Transformer encoder using timm models via HuggingFace Transformers.

- **Input:** RGB image `(B, 3, H, W)`
- **Output key:** `rgb`
- **Feature type:** FLAT (with pooling) or SEQUENTIAL (without pooling, returns patch tokens)
- **Supports:** Dynamic image sizes

```python
ViTEncoder(
    input_keys="left",
    backbone="timm/vit_base_patch14_dinov2.lvd142m",
    pooling_method="default",  # Uses CLS token
    pretrained=True,
    frozen=True,
)
```

#### ConditionalCNNEncoder

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

### Depth Encoders

#### DepthCNNEncoder

Adapts timm CNN backbones for single-channel depth images by setting `num_channels=1`.

- **Input:** Depth image `(B, 1, H, W)`
- **Output key:** `depth`
- **Feature type:** FLAT (after pooling) or SPATIAL (without pooling)

```python
DepthCNNEncoder(
    input_keys="depth",
    backbone="timm/resnet18.a1_in1k",
    pooling_method="average_pooling",
)
```

#### DFormerEncoder

Geometry-aware RGB+Depth encoder (`DFormerEncoder`) using geometric self-attention. Based on the [DFormerv2 paper](https://arxiv.org/abs/2504.04701).

- **Input:** RGB image + Depth image
- **Output key:** `rgbd`
- **Feature type:** FLAT (after pooling) or SPATIAL
- **Variants:** Small, Base, Large

Processes RGB and depth through parallel patch embedding streams and fuses them via geometric attention blocks that use depth-derived spatial relationships.

#### LightGeometricEncoder

Single-layer geometry-aware RGBD encoder. A lightweight alternative to DFormerV2 with a single geometric attention block.

- **Input:** RGB image + Depth image (requires both)
- **Output key:** `rgbd`
- **Feature type:** FLAT (after pooling) or SPATIAL (without pooling)

```python
LightGeometricEncoder(
    input_keys=["left", "depth"],
    embedding_dimension=512,
    num_heads=8,
    decomposition_mode="separable",
)
```

### Proprioceptive Encoder

#### ProprioceptiveEncoder

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

### Language Encoder

#### LanguageEncoder

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

### VLM Encoder

#### VLMEncoder

Vision-Language Model encoder (CLIP, SigLIP) that produces both image and text features from a shared model. This is a multi-output encoder.

- **Input:** RGB image + tokenized text
- **Output keys:** `rgb`, `language`, `language_padding_mask`
- **Feature type:** FLAT (with pooling) or SEQUENTIAL (without pooling) per output

```python
VLMEncoder(
    input_keys=["left", "tokenized_observations"],
    pretrained=True,
    frozen=True,
    model_name="openai/clip-vit-base-patch32",
    pooling_method="default",
)
```

Since VLMs produce multiple outputs, fusion and decoder configs use dot notation to select features: `vlm_encoder.rgb`, `vlm_encoder.language`.

## Available Backbones

### RGB Backbones (`RGBBackboneType`)

| Enum | Model ID | Type |
|---|---|---|
| `RESNET18` | `timm/resnet18.a1_in1k` | CNN |
| `RESNET34` | `timm/resnet34.a1_in1k` | CNN |
| `RESNET50` | `timm/resnet50.a1_in1k` | CNN |
| `EFFICIENTNET_B0` | `timm/efficientnet_b0.ra_in1k` | CNN |
| `EDGENEXT_XX_SMALL` | `timm/edgenext_xx_small.in1k` | CNN |
| `EDGENEXT_X_SMALL` | `timm/edgenext_x_small.in1k` | CNN |
| `EDGENEXT_SMALL` | `timm/edgenext_small.usi_in1k` | CNN |
| `EDGENEXT_BASE` | `timm/edgenext_base.usi_in1k` | CNN |
| `MOBILENETV4_SMALL_050` | `timm/mobilenetv4_conv_small_050.e3000_r224_in1k` | CNN |
| `VIT_BASE` | `timm/vit_base_patch16_clip_224.laion2b_ft_in12k_in1k` | ViT |
| `DINOV2_VITS14` | `timm/vit_small_patch14_dinov2.lvd142m` | ViT |
| `DINOV2_VITB14` | `timm/vit_base_patch14_dinov2.lvd142m` | ViT |
| `DINOV2_VITL14` | `timm/vit_large_patch14_dinov2.lvd142m` | ViT |
| `DINOV3_VITS16` | `timm/vit_small_patch16_dinov3.lvd1689m` | ViT |
| `DINOV3_VITS16PLUS` | `timm/vit_small_plus_patch16_dinov3.lvd1689m` | ViT |
| `DINOV3_VITB16` | `timm/vit_base_patch16_dinov3.lvd1689m` | ViT |

### Language Models (`LanguageEncoderType`)

| Enum | Model ID |
|---|---|
| `BERT_BASE` | `bert-base-uncased` |
| `DISTILBERT_BASE` | `distilbert-base-uncased` |
| `MINI_LM_L6` | `sentence-transformers/all-MiniLM-L6-v2` |
| `GEMMA_2B` | `google/gemma-2b` |
| `QWEN_2_1_5B` | `Qwen/Qwen2-1.5B` |
| `ALBERT_BASE` | `albert-base-v2` |

### VLM Models (`ImageTextModelType`)

| Enum | Model ID |
|---|---|
| `CLIP_VITB32` | `openai/clip-vit-base-patch32` |
| `CLIP_VITB16` | `openai/clip-vit-base-patch16` |
| `SIGLIP_BASE_PATCH16` | `google/siglip2-base-patch16-naflex` |

Backbones are extended by adding new enum values in `src/versatil/models/encoding/encoders/constants.py` that map to timm or HuggingFace model identifiers.

## Pooling Methods

All vision and language encoders support configurable pooling via `PoolingMethod`:

| Method | Enum Value | Description |
|---|---|---|
| Default | `default` | CLS token (ViT), max pooling (CNN), pooled output (VLM) |
| Average | `average_pooling` | Global Average Pooling (CNN) or mean pooling (Transformer) |
| Max | `max_pooling` | Global Max Pooling for CNN feature maps |
| Spatial Softmax | `spatial_softmax` | Spatial Softmax pooling for CNN feature maps |
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

### Feature Consumption

Fusion modules consume their input features. After fusion, only the fusion output name exists in the feature dictionary -- the original encoder features that were fused are removed.

```
Before fusion: {"left_rgb": ..., "right_rgb": ..., "proprio": ...}
Fusion(left_rgb, right_rgb) → fused_visual
After fusion:  {"fused_visual": ..., "proprio": ...}
```

This prevents feature duplication and ensures the decoder receives only semantically meaningful final features.

## Feature Naming

### Encoder Output Keys

Each encoder type uses a specific output key from `EncoderOutputKeys`:

| Output Key | Value | Used By |
|---|---|---|
| `RGB` | `rgb` | CNNEncoder, ViTEncoder, ConditionalCNNEncoder, VLMEncoder |
| `DEPTH` | `depth` | DepthCNNEncoder |
| `RGBD` | `rgbd` | DFormerEncoder, LightGeometricEncoder |
| `PROPRIOCEPTIVE` | `proprio` | ProprioceptiveEncoder |
| `LANGUAGE` | `language` | LanguageEncoder, VLMEncoder |
| `PADDING_MASK` | `padding_mask` | LanguageEncoder, VLMEncoder (combined as `language_padding_mask`) |

### Spatial vs Flat Features

Features are classified by their dimension shape:

| Classification | Dimension | Produced When |
|---|---|---|
| **SPATIAL** | `(C, H, W)` tuple of length 3 | `pooling_method="none"` on CNN encoders |
| **SEQUENTIAL** | `(T, D)` tuple of length 2 | `pooling_method="none"` on Transformer encoders |
| **FLAT** | `int` or `(D,)` | Any pooling method other than `none` |

Spatial features are identified by `EncoderOutputKeys.RGB`, `DEPTH`, and `RGBD`. The decoder's `DecoderInput` can enforce or reject specific feature types via `required_types` and `raises_for_types`.

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
from versatil.models.encoding.encoders.base import EncoderInput, EncoderOutput
from versatil.models.encoding.encoders.unconditional import Encoder


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

    def get_output_specification(self) -> EncoderOutput:
        return EncoderOutput(
            features=["my_feature"],
            dimensions={"my_feature": self.feature_dim},
        )

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
- Add a YAML config in `hydra_configs/policy/encoder/`
- Write tests in `tests/models/encoding/`

For conditional encoders, subclass `ConditionalEncoder` instead and implement `forward(inputs, conditioning)`.
