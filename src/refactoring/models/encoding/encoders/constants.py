import enum


class EncoderType(str, enum.Enum):
    """Available encoder types."""
    RGB_IMAGE = "rgb_encoder"
    DEPTH = "depth_encoder"
    LANGUAGE = "language_encoder"
    PROPRIOCEPTIVE = "proprioceptive_encoder"
    IMAGE_TEXT = "image_text_encoder"
    RGB_DEPTH = "rgb_depth_encoder"

class RGBBackboneType(str, enum.Enum):
    """Available RGB image encoder backbones."""
    #Convolutional Neural Networks (CNNs)
    RESNET18 = "timm/resnet18.a1_in1k"  # https://huggingface.co/timm/resnet18.a1_in1k
    RESNET34 = "timm/resnet34.a1_in1k"  # https://huggingface.co/timm/resnet34.a1_in1k
    RESNET50 = "timm/resnet50.a1_in1k"  # https://huggingface.co/timm/resnet50.a1_in1k
    EFFICIENTNET_B0 = "timm/efficientnet_b0.ra_in1k" # https://huggingface.co/timm/efficientnet_b0.ra_in1k
    EDGENEXT_XX_SMALL = "timm/edgenext_xx_small.in1k" # https://huggingface.co/timm/edgenext_xx_small.in1k
    EDGENEXT_X_SMALL = "timm/edgenext_x_small.in1k" # https://huggingface.co/timm/edgenext_x_small.in1k
    EDGENEXT_SMALL = "timm/edgenext_small.usi_in1k" # https://huggingface.co/timm/edgenext_small.usi_in1k
    EDGENEXT_BASE = "timm/edgenext_base.usi_in1k" # https://huggingface.co/timm/edgenext_base.usi_in1k
    #Vision Transformers (ViT)
    VIT_BASE = "timm/vit_base_patch16_clip_224.laion2b_ft_in12k_in1k" # https://huggingface.co/timm/vit_base_patch16_clip_224.laion2b_ft_in12k_in1k
    DINOV2_VITS14 = "timm/vit_small_patch14_dinov2.lvd142m" # https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m
    DINOV2_VITB14 = "timm/vit_base_patch14_dinov2.lvd142m" # https://huggingface.co/timm/vit_base_patch14_dinov2.lvd142m
    DINOV2_VITL14 = "timm/vit_large_patch14_dinov2.lvd142m" # https://huggingface.co/timm/vit_large_patch14_dinov2.lvd142m
    DINOV3_VITS16 = "timm/vit_small_patch16_dinov3.lvd1689m" # https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m
    DINOV3_VITS16PLUS = "timm/vit_small_plus_patch16_dinov3.lvd1689m" # https://huggingface.co/timm/vit_small_plus_patch16_dinov3.lvd1689m
    DINOV3_VITB16 = "timm/vit_base_patch16_dinov3.lvd1689m" # https://huggingface.co/timm/vit_base_patch16_dinov3.lvd1689m


class ImageTextModelType(str, enum.Enum):
    """Available image+text multimodal encoders."""
    # CLIP models (OpenAI)
    CLIP_VITB32 = "openai/clip-vit-base-patch32"
    CLIP_VITB16 = "openai/clip-vit-base-patch16"
    SIGLIP_BASE_PATCH16 = "google/siglip2-base-patch16-naflex"


class AttentionImplementation(str, enum.Enum):
    """Attention implementation types."""
    EAGER = "eager" # Standard PyTorch attention
    SDPA = "sdpa" # Scaled Dot-Product Attention
    FLASH_ATTENTION_2 = "flash_attention_2" # using Dao-AILab/flash-attention


class FeatureExtractionMethod(str, enum.Enum):
    """Methods for extracting features from ViT encoders."""
    CLS_TOKEN = "cls_token"
    AVERAGE_PATCH_TOKENS = "average_patch_tokens"
    LEARNED_AGGREGATION = "learned_aggregation"  # learned weighted aggregation of patch tokens


class PoolingMethod(str, enum.Enum):
    """Feature pooling methods for CNN encoders."""
    SPATIAL_SOFTMAX = "spatial_softmax"
    GLOBAL_AVERAGE = "global_average"
    NONE = "none" # Return full spatial features without pooling


class LanguageEncoderType(str, enum.Enum):
    """Available language encoders."""
    BERT_BASE = "bert-base-uncased"  # supports AttentionImplementation.SDPA
    DISTILBERT_BASE = "distilbert-base-uncased" # supports AttentionImplementation.SDPA
    MINI_LM_L6 = "sentence-transformers/all-MiniLM-L6-v2"



class EncoderOutputKeys(str, enum.Enum):
    """Types of encoder output keys to use for extracting an output feature from an encoder."""
    RGB = "image"
    RGBD = "rgbd"
    LANGUAGE = "language"
    DEPTH = "depth"
    PROPRIOCEPTIVE = "proprio"
    DEFAULT = "default"

SPATIAL_FEATURES = [
    EncoderOutputKeys.RGB.value,
    EncoderOutputKeys.DEPTH.value,
    EncoderOutputKeys.RGBD.value,
]

