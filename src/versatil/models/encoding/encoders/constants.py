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
    CLIP_VITL14_224_OPENAI = "vit_large_patch14_clip_224.openai"  # https://huggingface.co/timm/vit_large_patch14_clip_224.openai
    CLIP_VITL14_336_OPENAI = "vit_large_patch14_clip_336.openai"  # https://huggingface.co/timm/vit_large_patch14_clip_336.openai
    DINOV2_VITS14 = "vit_small_patch14_dinov2.lvd142m"  # https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m
    DINOV2_VITB14 = "vit_base_patch14_dinov2.lvd142m"  # https://huggingface.co/timm/vit_base_patch14_dinov2.lvd142m
    DINOV2_VITL14 = "vit_large_patch14_dinov2.lvd142m"  # https://huggingface.co/timm/vit_large_patch14_dinov2.lvd142m
    DINOV2_VITL14_REG4 = "vit_large_patch14_reg4_dinov2.lvd142m"  # https://huggingface.co/timm/vit_large_patch14_reg4_dinov2.lvd142m
    IN1K_VITL16_224 = "vit_large_patch16_224.augreg_in21k_ft_in1k"  # https://huggingface.co/timm/vit_large_patch16_224.augreg_in21k_ft_in1k
    DINOV3_VITS16 = "vit_small_patch16_dinov3.lvd1689m"  # https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m
    DINOV3_VITS16PLUS = "vit_small_plus_patch16_dinov3.lvd1689m"  # https://huggingface.co/timm/vit_small_plus_patch16_dinov3.lvd1689m
    DINOV3_VITB16 = "vit_base_patch16_dinov3.lvd1689m"  # https://huggingface.co/timm/vit_base_patch16_dinov3.lvd1689m
    DEIT_TINY = "deit_tiny_patch16_224.fb_in1k"  # https://huggingface.co/timm/deit_tiny_patch16_224.fb_in1k
    DEIT_SMALL = "deit_small_patch16_224.fb_in1k"  # https://huggingface.co/timm/deit_small_patch16_224.fb_in1k
    DEIT_BASE = "deit_base_patch16_224.fb_in1k"  # https://huggingface.co/timm/deit_base_patch16_224.fb_in1k
    SIGLIP_BASE_B16_224 = "vit_base_patch16_siglip_224"  # https://huggingface.co/timm/vit_base_patch16_siglip_224
    SIGLIP_BASE_B16_256 = "vit_base_patch16_siglip_256"  # https://huggingface.co/timm/vit_base_patch16_siglip_256
    SIGLIP_BASE_B16_384 = "vit_base_patch16_siglip_384"  # https://huggingface.co/timm/vit_base_patch16_siglip_384
    SIGLIP_SO400M_224 = "vit_so400m_patch14_siglip_224"  # https://huggingface.co/timm/vit_so400m_patch14_siglip_224
    SIGLIP_SO400M_384 = "vit_so400m_patch14_siglip_384"  # https://huggingface.co/timm/vit_so400m_patch14_siglip_384


class DinoV2SigLIPBackboneType(enum.StrEnum):
    """DINOv2+SigLIP paired RGB vision backbone identifiers."""

    DINOV2_SIGLIP_VIT_SO_224PX = "dinosiglip-vit-so-224px"
    DINOV2_SIGLIP_VIT_SO_384PX = "dinosiglip-vit-so-384px"


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
    MINI_LM_L12 = "sentence-transformers/all-MiniLM-L12-v2"
    EMBEDDINGGEMMA_300M = "google/embeddinggemma-300m"
    QWEN_3_EMBEDDING_0_6B = "Qwen/Qwen3-Embedding-0.6B"
    BGE_BASE_EN_V1_5 = "BAAI/bge-base-en-v1.5"
    LLAMA_EMBED_NEMOTRON_8B = "nvidia/llama-embed-nemotron-8b"
    LLAMA_NEMOTRON_EMBED_1B_V2 = "nvidia/llama-nemotron-embed-1b-v2"
    GTE_QWEN2_1_5B_INSTRUCT = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
    JINA_EMBEDDINGS_V3 = "jinaai/jina-embeddings-v3"
    E5_BASE = "intfloat/e5-base"
    ALBERT_BASE = "albert-base-v2"
    ROBERTA_BASE = "roberta-base"
    DEBERTA_V3_BASE = "microsoft/deberta-v3-base"
    DISTIL_ROBERTA_BASE = "distilbert/distilroberta-base"


class EncoderOutputKeys(enum.StrEnum):
    """Types of encoder output keys to use for extracting an output feature from an encoder."""

    RGB = "rgb"
    RGBD = "rgbd"
    LANGUAGE = "language"
    DEPTH = "depth"
    PROPRIOCEPTIVE = "proprio"
    FUSED_RGB_LANGUAGE = "fused_rgb_language"
    PADDING_MASK = "padding_mask"
