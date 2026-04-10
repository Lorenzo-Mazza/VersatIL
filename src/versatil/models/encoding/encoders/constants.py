"""Constants for encoder configurations, backbone types, and output keys."""

import enum


class SpatialBackboneType(enum.StrEnum):
    """Backbones producing (B, C, H, W) spatial feature maps via timm features_only."""

    RESNET18 = "resnet18.a1_in1k"  # https://huggingface.co/timm/resnet18.a1_in1k
    RESNET34 = "resnet34.a1_in1k"  # https://huggingface.co/timm/resnet34.a1_in1k
    RESNET50 = "resnet50.a1_in1k"  # https://huggingface.co/timm/resnet50.a1_in1k
    EFFICIENTNET_B0 = (
        "efficientnet_b0.ra_in1k"  # https://huggingface.co/timm/efficientnet_b0.ra_in1k
    )
    EFFICIENTNET_B2 = (
        "efficientnet_b2.ra_in1k"  # https://huggingface.co/timm/efficientnet_b2.ra_in1k
    )
    EDGENEXT_XX_SMALL = (
        "edgenext_xx_small.in1k"  # https://huggingface.co/timm/edgenext_xx_small.in1k
    )
    EDGENEXT_X_SMALL = (
        "edgenext_x_small.in1k"  # https://huggingface.co/timm/edgenext_x_small.in1k
    )
    EDGENEXT_SMALL = (
        "edgenext_small.usi_in1k"  # https://huggingface.co/timm/edgenext_small.usi_in1k
    )
    EDGENEXT_BASE = (
        "edgenext_base.usi_in1k"  # https://huggingface.co/timm/edgenext_base.usi_in1k
    )
    MOBILENETV4_SMALL_050 = "mobilenetv4_conv_small_050.e3000_r224_in1k"  # https://huggingface.co/timm/mobilenetv4_conv_small_050.e3000_r224_in1k
    CONVNEXT_NANO = "convnext_nano.in12k_ft_in1k"  # https://huggingface.co/timm/convnext_nano.in12k_ft_in1k
    CONVNEXT_TINY = "convnext_tiny.fb_in22k_ft_in1k"  # https://huggingface.co/timm/convnext_tiny.fb_in22k_ft_in1k
    CONVNEXT_BASE = "convnext_base.fb_in22k_ft_in1k"  # https://huggingface.co/timm/convnext_base.fb_in22k_ft_in1k
    CONVNEXTV2_NANO = "convnextv2_nano.fcmae_ft_in22k_in1k"  # https://huggingface.co/timm/convnextv2_nano.fcmae_ft_in22k_in1k
    DINOV3_CONVNEXT_SMALL = "convnext_small.dinov3_lvd1689m"  # https://huggingface.co/timm/convnext_small.dinov3_lvd1689m
    TINY_VIT_21M = "tiny_vit_21m_224.dist_in22k_ft_in1k"  # https://huggingface.co/timm/tiny_vit_21m_224.dist_in22k_ft_in1k
    SWIN_TINY = "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k"  # https://huggingface.co/timm/swin_tiny_patch4_window7_224.ms_in22k_ft_in1k
    SWIN_BASE = "swin_base_patch4_window7_224.ms_in22k_ft_in1k"  # https://huggingface.co/timm/swin_base_patch4_window7_224.ms_in22k_ft_in1k


class FlatBackboneType(enum.StrEnum):
    """Backbones producing (B, S, D) token sequences via timm forward_features."""

    VIT_BASE = "vit_base_patch16_clip_224.laion2b_ft_in12k_in1k"  # https://huggingface.co/timm/vit_base_patch16_clip_224.laion2b_ft_in12k_in1k
    DINOV2_VITS14 = "vit_small_patch14_dinov2.lvd142m"  # https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m
    DINOV2_VITB14 = "vit_base_patch14_dinov2.lvd142m"  # https://huggingface.co/timm/vit_base_patch14_dinov2.lvd142m
    DINOV2_VITL14 = "vit_large_patch14_dinov2.lvd142m"  # https://huggingface.co/timm/vit_large_patch14_dinov2.lvd142m
    DINOV3_VITS16 = "vit_small_patch16_dinov3.lvd1689m"  # https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m
    DINOV3_VITS16PLUS = "vit_small_plus_patch16_dinov3.lvd1689m"  # https://huggingface.co/timm/vit_small_plus_patch16_dinov3.lvd1689m
    DINOV3_VITB16 = "vit_base_patch16_dinov3.lvd1689m"  # https://huggingface.co/timm/vit_base_patch16_dinov3.lvd1689m
    DEIT_TINY = "deit_tiny_patch16_224.fb_in1k"  # https://huggingface.co/timm/deit_tiny_patch16_224.fb_in1k
    DEIT_SMALL = "deit_small_patch16_224.fb_in1k"  # https://huggingface.co/timm/deit_small_patch16_224.fb_in1k
    DEIT_BASE = "deit_base_patch16_224.fb_in1k"  # https://huggingface.co/timm/deit_base_patch16_224.fb_in1k


# Union type for all RGB backbones (Spatial + Flat)
RGBBackboneType = enum.StrEnum(
    "RGBBackboneType",
    {e.name: e.value for e in list(SpatialBackboneType) + list(FlatBackboneType)},
)


class ImageTextModelType(enum.StrEnum):
    """Available image+text multimodal encoders."""

    CLIP_VITB32 = "openai/clip-vit-base-patch32"
    CLIP_VITB16 = "openai/clip-vit-base-patch16"
    SIGLIP_BASE_PATCH16 = "google/siglip2-base-patch16-naflex"
    CLIP_VITL14 = "openai/clip-vit-large-patch14"
    SIGLIP_SO400M = "google/siglip-so400m-patch14-384"


class PaliGemmaModelType(enum.StrEnum):
    """Available PaliGemma model names."""

    PALIGEMMA2_3B_224 = "google/paligemma2-3b-pt-224"
    PALIGEMMA2_3B_448 = "google/paligemma2-3b-pt-448"
    PALIGEMMA2_3B_896 = "google/paligemma2-3b-pt-896"


class SmolVLMModelType(enum.StrEnum):
    """Available SmolVLM/Idefics3 model names."""

    SMOLVLM_256M = "HuggingFaceTB/SmolVLM-256M-Instruct"
    SMOLVLM_500M = "HuggingFaceTB/SmolVLM-500M-Instruct"
    SMOLVLM_2_2B = "HuggingFaceTB/SmolVLM-2.2B-Instruct"


class AttentionImplementation(enum.StrEnum):
    """Attention implementation types."""

    EAGER = "eager"  # Standard PyTorch attention
    SDPA = "sdpa"  # Scaled Dot-Product Attention, automatically enables Torch 2.10 Flash Attention kernels


class PoolingMethod(enum.StrEnum):
    """Feature pooling methods for spatial and flat RGB encoders."""

    LEARNED_AGGREGATION = "learned_aggregation"  # learned attention aggregation of patch tokens/feature channels
    DEFAULT = "default"  # use [CLS] token for flat encoders, max pooling for spatial encoders, or pooled output in VLMs
    SPATIAL_SOFTMAX = (
        "spatial_softmax"  # Spatial Softmax pooling for spatial feature maps
    )
    AVERAGE = "average_pooling"  # Global Average Pooling (GAP) for spatial feature maps or mean pooling for token sequences
    MAX = "max_pooling"  # Global Max Pooling for spatial feature maps
    NONE = "none"  # Return full spatial features or last hidden state tokens without pooling

    @property
    def supports_spatial(self) -> bool:
        """Whether this pooling method works with spatial (B, C, H, W) feature maps."""
        return True

    @property
    def supports_sequential(self) -> bool:
        """Whether this pooling method works with sequential (B, S, D) token sequences."""
        return self not in {PoolingMethod.SPATIAL_SOFTMAX, PoolingMethod.MAX}


class BatchNormHandling(enum.StrEnum):
    """How to handle BatchNorm layers in spatial RGB backbones.

    BatchNorm is problematic for temporal data: when reshaping (B,T,C,H,W) to (B*T,C,H,W),
    batch statistics mix frames across time, leaking future information into each frame's
    representation. This causes train/test mismatch since the signal vanishes at inference.
    """

    DEFAULT = "default"  # Keep BatchNorm as-is.
    FROZEN = "frozen"  # Freeze BN: preserves pretrained stats, no batch dependency
    CONVERT_TO_GROUPNORM = "groupnorm"  # Replace with GroupNorm (per-sample stats, but loses pretrained weights benefits)


class LanguageEncoderType(enum.StrEnum):
    """Available language encoders."""

    BERT_BASE = "bert-base-uncased"
    DISTILBERT_BASE = "distilbert-base-uncased"
    MINI_LM_L6 = "sentence-transformers/all-MiniLM-L6-v2"
    GEMMA_2B = "google/gemma-2b"
    QWEN_2_0_5B = "Qwen/Qwen2-0.5B"
    QWEN_2_1_5B = "Qwen/Qwen2-1.5B"
    ALBERT_BASE = "albert-base-v2"
    ROBERTA_BASE = "roberta-base"
    GPT2 = "gpt2"
    DEBERTA_V3_BASE = "microsoft/deberta-v3-base"
    PHI_2 = "microsoft/phi-2"
    LLAMA_3_2_1B = "meta-llama/Llama-3.2-1B"


class EncoderOutputKeys(enum.StrEnum):
    """Types of encoder output keys to use for extracting an output feature from an encoder."""

    RGB = "rgb"
    RGBD = "rgbd"
    LANGUAGE = "language"
    DEPTH = "depth"
    PROPRIOCEPTIVE = "proprio"
    FUSED_RGB_LANGUAGE = "fused_rgb_language"
    PADDING_MASK = "padding_mask"
