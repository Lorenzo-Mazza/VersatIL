import enum


class RGBBackboneType(str, enum.Enum):
    """Available RGB image encoder backbones."""

    # Convolutional Neural Networks (CNNs)
    RESNET18 = "timm/resnet18.a1_in1k"  # https://huggingface.co/timm/resnet18.a1_in1k
    RESNET34 = "timm/resnet34.a1_in1k"  # https://huggingface.co/timm/resnet34.a1_in1k
    RESNET50 = "timm/resnet50.a1_in1k"  # https://huggingface.co/timm/resnet50.a1_in1k
    EFFICIENTNET_B0 = "timm/efficientnet_b0.ra_in1k"  # https://huggingface.co/timm/efficientnet_b0.ra_in1k
    EDGENEXT_XX_SMALL = "timm/edgenext_xx_small.in1k"  # https://huggingface.co/timm/edgenext_xx_small.in1k
    EDGENEXT_X_SMALL = "timm/edgenext_x_small.in1k"  # https://huggingface.co/timm/edgenext_x_small.in1k
    EDGENEXT_SMALL = "timm/edgenext_small.usi_in1k"  # https://huggingface.co/timm/edgenext_small.usi_in1k
    EDGENEXT_BASE = "timm/edgenext_base.usi_in1k"  # https://huggingface.co/timm/edgenext_base.usi_in1k
    MOBILENETV4_SMALL_050 = "timm/mobilenetv4_conv_small_050.e3000_r224_in1k"  # https://huggingface.co/timm/mobilenetv4_conv_small_050.e3000_r224_in1k
    # Vision Transformers (ViT)
    VIT_BASE = "timm/vit_base_patch16_clip_224.laion2b_ft_in12k_in1k"  # https://huggingface.co/timm/vit_base_patch16_clip_224.laion2b_ft_in12k_in1k
    DINOV2_VITS14 = "timm/vit_small_patch14_dinov2.lvd142m"  # https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m
    DINOV2_VITB14 = "timm/vit_base_patch14_dinov2.lvd142m"  # https://huggingface.co/timm/vit_base_patch14_dinov2.lvd142m
    DINOV2_VITL14 = "timm/vit_large_patch14_dinov2.lvd142m"  # https://huggingface.co/timm/vit_large_patch14_dinov2.lvd142m
    DINOV3_VITS16 = "timm/vit_small_patch16_dinov3.lvd1689m"  # https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m
    DINOV3_VITS16PLUS = "timm/vit_small_plus_patch16_dinov3.lvd1689m"  # https://huggingface.co/timm/vit_small_plus_patch16_dinov3.lvd1689m
    DINOV3_VITB16 = "timm/vit_base_patch16_dinov3.lvd1689m"  # https://huggingface.co/timm/vit_base_patch16_dinov3.lvd1689m


class ImageTextModelType(str, enum.Enum):
    """Available image+text multimodal encoders."""

    # CLIP models (OpenAI)
    CLIP_VITB32 = "openai/clip-vit-base-patch32"
    CLIP_VITB16 = "openai/clip-vit-base-patch16"
    SIGLIP_BASE_PATCH16 = "google/siglip2-base-patch16-naflex"


class AttentionImplementation(str, enum.Enum):
    """Attention implementation types."""

    EAGER = "eager"  # Standard PyTorch attention
    SDPA = "sdpa"  # Scaled Dot-Product Attention
    FLASH_ATTENTION_2 = "flash_attention_2"  # using Dao-AILab/flash-attention


class PoolingMethod(str, enum.Enum):
    """Feature pooling methods for Convolutional and Transformer encoders."""

    LEARNED_AGGREGATION = "learned_aggregation"  # learned attention aggregation of patch tokens/feature channels
    DEFAULT = "default"  # use [CLS] token in ViT, max pooling in CNNs or pooled output in VLMs
    SPATIAL_SOFTMAX = "spatial_softmax"  # Spatial Softmax pooling for CNN feature maps
    AVERAGE = "average_pooling"  # Global Average Pooling (GAP) for CNN feature maps or mean pooling for Transformer tokens
    MAX = "max_pooling"  # Global Max Pooling for CNN feature maps
    NONE = "none"  # Return full spatial features or last hidden state tokens without pooling


class BatchNormHandling(str, enum.Enum):
    """How to handle BatchNorm layers in CNN backbones.

    BatchNorm is problematic for temporal data: when reshaping (B,T,C,H,W) to (B*T,C,H,W),
    batch statistics mix frames across time, leaking future information into each frame's
    representation. This causes train/test mismatch since the signal vanishes at inference.
    """

    DEFAULT = "default"  # Keep BatchNorm as-is.
    FROZEN = "frozen"  # Freeze BN: preserves pretrained stats, no batch dependency
    CONVERT_TO_GROUPNORM = "groupnorm"  # Replace with GroupNorm (per-sample stats, but loses pretrained weights benefits)


class LanguageEncoderType(str, enum.Enum):
    """Available language encoders."""

    BERT_BASE = "bert-base-uncased"  
    DISTILBERT_BASE = "distilbert-base-uncased" 
    MINI_LM_L6 = "sentence-transformers/all-MiniLM-L6-v2"
    GEMMA_2B = "google/gemma-2b"
    QWEN_2_1_5B = "Qwen/Qwen2-1.5B"
    ALBERT_BASE = "albert-base-v2" 

class EncoderOutputKeys(str, enum.Enum):
    """Types of encoder output keys to use for extracting an output feature from an encoder."""

    RGB = "rgb"
    RGBD = "rgbd"
    LANGUAGE = "language"
    DEPTH = "depth"
    PROPRIOCEPTIVE = "proprio"
    PADDING_MASK = "padding_mask"


SPATIAL_FEATURES = [
    EncoderOutputKeys.RGB.value,
    EncoderOutputKeys.DEPTH.value,
    EncoderOutputKeys.RGBD.value,
]
