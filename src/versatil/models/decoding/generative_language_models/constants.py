"""Constants for generative (vision-) language-model components."""

import enum

from versatil.models.encoding.encoders.constants import FlatBackboneType


class PaliGemmaModelType(enum.StrEnum):
    """Available PaliGemma model names."""

    PALIGEMMA2_3B_224 = "google/paligemma2-3b-pt-224"
    PALIGEMMA2_3B_448 = "google/paligemma2-3b-pt-448"
    PALIGEMMA2_3B_896 = "google/paligemma2-3b-pt-896"


class SmolVLMModelType(enum.StrEnum):
    """Available SmolVLM/Idefics3 model names."""

    SMOLVLM_256M = "HuggingFaceTB/SmolVLM-256M-Instruct"
    SMOLVLM_500M = "HuggingFaceTB/SmolVLM-500M-Instruct"
    SMOLVLM_2_2B = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"


class PrismaticModelType(enum.StrEnum):
    """Available Prismatic model folders supported by the raw loader."""

    CLIP_224PX_7B = "clip-224px+7b"
    CLIP_336PX_RESIZE_CROP_7B = "clip-336px-resize-crop+7b"
    CLIP_336PX_RESIZE_NAIVE_7B = "clip-336px-resize-naive+7b"
    DINOCLIP_336PX_LETTERBOX_7B = "dinoclip-336px-letterbox+7b"
    DINOCLIP_336PX_RESIZE_NAIVE_7B = "dinoclip-336px-resize-naive+7b"
    DINOSIGLIP_384PX_LETTERBOX_7B = "dinosiglip-384px-letterbox+7b"
    DINOSIGLIP_384PX_RESIZE_NAIVE_7B = "dinosiglip-384px-resize-naive+7b"
    DINOV2_224PX_7B = "dinov2-224px+7b"
    FULL_FT_MULTI_STAGE_7B = "full-ft-multi-stage+7b"
    FULL_FT_ONE_STAGE_7B = "full-ft-one-stage+7b"
    IN1K_224PX_7B = "in1k-224px+7b"
    LLAMA2_7B = "llama2+7b"
    LLAMA2_13B = "llama2+13b"
    LLAMA2_CHAT_7B = "llama2-chat+7b"
    LLAMA2_CHAT_13B = "llama2-chat+13b"
    LLAMA2_NO_COTRAINING_7B = "llama2-no-cotraining+7b"
    LLAVA_LRV_7B = "llava-lrv+7b"
    LLAVA_LVIS4V_7B = "llava-lvis4v+7b"
    LLAVA_LVIS4V_LRV_7B = "llava-lvis4v-lrv+7b"
    MISTRAL_INSTRUCT_V0_1_7B = "mistral-instruct-v0.1+7b"
    MISTRAL_V0_1_7B = "mistral-v0.1+7b"
    ONE_STAGE_7B = "one-stage+7b"
    ONE_STAGE_13B = "one-stage+13b"
    PHI_2_3B = "phi-2+3b"
    PRISM_CLIP_7B = "prism-clip+7b"
    PRISM_CLIP_13B = "prism-clip+13b"
    PRISM_CLIP_CONTROLLED_7B = "prism-clip-controlled+7b"
    PRISM_CLIP_CONTROLLED_13B = "prism-clip-controlled+13b"
    PRISM_DINOSIGLIP_7B = "prism-dinosiglip+7b"
    PRISM_DINOSIGLIP_13B = "prism-dinosiglip+13b"
    PRISM_DINOSIGLIP_224PX_7B = "prism-dinosiglip-224px+7b"
    PRISM_DINOSIGLIP_224PX_CONTROLLED_7B = "prism-dinosiglip-224px-controlled+7b"
    PRISM_DINOSIGLIP_CONTROLLED_7B = "prism-dinosiglip-controlled+7b"
    PRISM_DINOSIGLIP_CONTROLLED_13B = "prism-dinosiglip-controlled+13b"
    PRISM_SIGLIP_7B = "prism-siglip+7b"
    PRISM_SIGLIP_13B = "prism-siglip+13b"
    PRISM_SIGLIP_CONTROLLED_7B = "prism-siglip-controlled+7b"
    PRISM_SIGLIP_CONTROLLED_13B = "prism-siglip-controlled+13b"
    REPRODUCTION_LLAVA_V15_7B = "reproduction-llava-v15+7b"
    REPRODUCTION_LLAVA_V15_13B = "reproduction-llava-v15+13b"
    SIGLIP_224PX_7B = "siglip-224px+7b"
    SIGLIP_384PX_LETTERBOX_7B = "siglip-384px-letterbox+7b"
    SIGLIP_384PX_RESIZE_CROP_7B = "siglip-384px-resize-crop+7b"
    SIGLIP_384PX_RESIZE_NAIVE_7B = "siglip-384px-resize-naive+7b"
    TRAIN_1P25_EPOCHS_7B = "train-1.25-epochs+7b"
    TRAIN_1P5_EPOCHS_7B = "train-1.5-epochs+7b"
    TRAIN_2_EPOCHS_7B = "train-2-epochs+7b"
    TRAIN_3_EPOCHS_7B = "train-3-epochs+7b"
    VICUNA_NO_COTRAINING_7B = "vicuna-no-cotraining+7b"


class PrismaticVisionBackboneType(enum.StrEnum):
    """Available Prismatic vision backbone identifiers."""

    CLIP_VIT_L = "clip-vit-l"
    CLIP_VIT_L_336PX = "clip-vit-l-336px"
    DINOCLIP_VIT_L_336PX = "dinoclip-vit-l-336px"
    DINOV2_VIT_L = "dinov2-vit-l"
    DINOSIGLIP_VIT_SO_224PX = "dinosiglip-vit-so-224px"
    DINOSIGLIP_VIT_SO_384PX = "dinosiglip-vit-so-384px"
    IN1K_VIT_L = "in1k-vit-l"
    SIGLIP_VIT_SO400M = "siglip-vit-so400m"
    SIGLIP_VIT_SO400M_384PX = "siglip-vit-so400m-384px"


class PrismaticLLMBackboneType(enum.StrEnum):
    """Available Prismatic language-model backbone identifiers."""

    LLAMA2_7B_PURE = "llama2-7b-pure"
    LLAMA2_13B_PURE = "llama2-13b-pure"
    LLAMA2_7B_CHAT = "llama2-7b-chat"
    LLAMA2_13B_CHAT = "llama2-13b-chat"
    VICUNA_V15_7B = "vicuna-v15-7b"
    VICUNA_V15_13B = "vicuna-v15-13b"
    MISTRAL_V0_1_7B_PURE = "mistral-v0.1-7b-pure"
    MISTRAL_V0_1_7B_INSTRUCT = "mistral-v0.1-7b-instruct"
    PHI_2_3B = "phi-2-3b"


PRISMATIC_REPOSITORY_ID = "TRI-ML/prismatic-vlms"
PRISMATIC_CHECKPOINT_FILENAME = "checkpoints/latest-checkpoint.pt"
PRISMATIC_CONFIG_FILENAME = "config.json"
PRISMATIC_PAD_TO_MULTIPLE_OF = 64

PRISMATIC_VISION_CHECKPOINT_KEY_RENAMES = {
    "dino_featurizer": "0.backbone",
    "clip_featurizer": "1.backbone",
    "siglip_featurizer": "1.backbone",
    "featurizer": "0.backbone",
}

PRISMATIC_VISION_BACKBONES: dict[
    PrismaticVisionBackboneType, tuple[FlatBackboneType, ...]
] = {
    PrismaticVisionBackboneType.CLIP_VIT_L: (FlatBackboneType.CLIP_VITL14_224_OPENAI,),
    PrismaticVisionBackboneType.CLIP_VIT_L_336PX: (
        FlatBackboneType.CLIP_VITL14_336_OPENAI,
    ),
    PrismaticVisionBackboneType.DINOCLIP_VIT_L_336PX: (
        FlatBackboneType.DINOV2_VITL14_REG4,
        FlatBackboneType.CLIP_VITL14_336_OPENAI,
    ),
    PrismaticVisionBackboneType.DINOV2_VIT_L: (FlatBackboneType.DINOV2_VITL14_REG4,),
    PrismaticVisionBackboneType.DINOSIGLIP_VIT_SO_224PX: (
        FlatBackboneType.DINOV2_VITL14_REG4,
        FlatBackboneType.SIGLIP_SO400M_224,
    ),
    PrismaticVisionBackboneType.DINOSIGLIP_VIT_SO_384PX: (
        FlatBackboneType.DINOV2_VITL14_REG4,
        FlatBackboneType.SIGLIP_SO400M_384,
    ),
    PrismaticVisionBackboneType.IN1K_VIT_L: (FlatBackboneType.IN1K_VITL16_224,),
    PrismaticVisionBackboneType.SIGLIP_VIT_SO400M: (
        FlatBackboneType.SIGLIP_SO400M_224,
    ),
    PrismaticVisionBackboneType.SIGLIP_VIT_SO400M_384PX: (
        FlatBackboneType.SIGLIP_SO400M_384,
    ),
}

PRISMATIC_VISION_IMAGE_SIZES: dict[PrismaticVisionBackboneType, int] = {
    PrismaticVisionBackboneType.CLIP_VIT_L: 224,
    PrismaticVisionBackboneType.CLIP_VIT_L_336PX: 336,
    PrismaticVisionBackboneType.DINOCLIP_VIT_L_336PX: 336,
    PrismaticVisionBackboneType.DINOV2_VIT_L: 224,
    PrismaticVisionBackboneType.DINOSIGLIP_VIT_SO_224PX: 224,
    PrismaticVisionBackboneType.DINOSIGLIP_VIT_SO_384PX: 384,
    PrismaticVisionBackboneType.IN1K_VIT_L: 224,
    PrismaticVisionBackboneType.SIGLIP_VIT_SO400M: 224,
    PrismaticVisionBackboneType.SIGLIP_VIT_SO400M_384PX: 384,
}

PRISMATIC_LLM_BACKBONES: dict[PrismaticLLMBackboneType, str] = {
    PrismaticLLMBackboneType.LLAMA2_7B_PURE: "meta-llama/Llama-2-7b-hf",
    PrismaticLLMBackboneType.LLAMA2_13B_PURE: "meta-llama/Llama-2-13b-hf",
    PrismaticLLMBackboneType.LLAMA2_7B_CHAT: "meta-llama/Llama-2-7b-chat-hf",
    PrismaticLLMBackboneType.LLAMA2_13B_CHAT: "meta-llama/Llama-2-13b-chat-hf",
    PrismaticLLMBackboneType.VICUNA_V15_7B: "lmsys/vicuna-7b-v1.5",
    PrismaticLLMBackboneType.VICUNA_V15_13B: "lmsys/vicuna-13b-v1.5",
    PrismaticLLMBackboneType.MISTRAL_V0_1_7B_PURE: "mistralai/Mistral-7B-v0.1",
    PrismaticLLMBackboneType.MISTRAL_V0_1_7B_INSTRUCT: (
        "mistralai/Mistral-7B-Instruct-v0.1"
    ),
    PrismaticLLMBackboneType.PHI_2_3B: "microsoft/phi-2",
}
