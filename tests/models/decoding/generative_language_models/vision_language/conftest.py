"""Shared fixtures for VLM backbone tests."""

from collections.abc import Callable
from unittest.mock import patch

import pytest
from transformers import (
    Gemma2Config,
    Idefics3Config,
    LlamaConfig,
    PaliGemmaConfig,
    SiglipVisionConfig,
)

from versatil.data.constants import Cameras
from versatil.models.adaptation.lora import LoRAAdaptation
from versatil.models.decoding.generative_language_models.constants import (
    PaliGemmaModelType,
    SmolVLMModelType,
)
from versatil.models.decoding.generative_language_models.vision_language.paligemma import (
    PaliGemmaVLM,
)
from versatil.models.decoding.generative_language_models.vision_language.smolvlm import (
    SmolVLM,
)
from versatil.training.constants import PrecisionType

TINY_HIDDEN_DIM = 32

VLM_INPUT_KEYS = [
    Cameras.LEFT.value,
]


def _lora_cache_key(
    lora_config: LoRAAdaptation | None,
) -> tuple[bool, int, int, float, str, tuple[str, ...], str]:
    """Return a stable cache key for optional LoRA settings."""
    if lora_config is None:
        return False, 0, 0, 0.0, "", (), ""
    exclude_modules = tuple(lora_config.exclude_modules or [])
    return (
        lora_config.enabled,
        lora_config.rank,
        lora_config.alpha,
        lora_config.dropout,
        lora_config.target_modules,
        exclude_modules,
        lora_config.bias,
    )


def make_tiny_smolvlm_config() -> Idefics3Config:
    """Create a tiny SmolVLM config without network access."""
    text_config = LlamaConfig(
        num_hidden_layers=1,
        hidden_size=TINY_HIDDEN_DIM,
        intermediate_size=TINY_HIDDEN_DIM * 2,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    vision_config = SiglipVisionConfig(
        hidden_size=TINY_HIDDEN_DIM,
        intermediate_size=TINY_HIDDEN_DIM * 2,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=56,
        patch_size=14,
    )
    return Idefics3Config(
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        scale_factor=4,
    )


def make_tiny_paligemma_config() -> PaliGemmaConfig:
    """Create a tiny PaliGemma config without network access."""
    text_config = Gemma2Config(
        num_hidden_layers=1,
        hidden_size=TINY_HIDDEN_DIM,
        intermediate_size=TINY_HIDDEN_DIM * 2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=TINY_HIDDEN_DIM // 2,
        vocab_size=1000,
    )
    vision_config = SiglipVisionConfig(
        hidden_size=TINY_HIDDEN_DIM,
        intermediate_size=TINY_HIDDEN_DIM * 2,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=56,
        patch_size=14,
    )
    config = PaliGemmaConfig(
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        projection_dim=TINY_HIDDEN_DIM,
    )
    # PaliGemma-specific: SiglipVisionConfig doesn't have this field
    config.vision_config.num_image_tokens = 16
    return config


@pytest.fixture(scope="session")
def real_smolvlm_backbone() -> Callable[..., SmolVLM]:
    """Factory for a real but tiny SmolVLM backbone, cached per dtype."""
    tiny_config = make_tiny_smolvlm_config()
    cache: dict[
        tuple[
            str,
            bool,
            tuple[bool, int, int, float, str, tuple[str, ...], str],
        ],
        SmolVLM,
    ] = {}

    def factory(
        model_dtype: str = PrecisionType.FP32.value,
        frozen: bool = False,
        lora_config: LoRAAdaptation | None = None,
    ) -> SmolVLM:
        cache_key = (model_dtype, frozen, _lora_cache_key(lora_config=lora_config))
        if cache_key not in cache:
            with patch(
                "versatil.models.decoding.generative_language_models.vision_language.huggingface.AutoConfig.from_pretrained",
                return_value=tiny_config,
            ):
                cache[cache_key] = SmolVLM(
                    input_keys=VLM_INPUT_KEYS,
                    pretrained=False,
                    frozen=frozen,
                    model_name=SmolVLMModelType.SMOLVLM_256M.value,
                    model_dtype=model_dtype,
                    lora_config=lora_config,
                )
        return cache[cache_key]

    return factory


@pytest.fixture(scope="session")
def real_paligemma_backbone() -> Callable[..., PaliGemmaVLM]:
    """Factory for a real but tiny PaliGemma backbone, cached per dtype."""
    tiny_config = make_tiny_paligemma_config()
    cache: dict[
        tuple[
            str,
            bool,
            tuple[bool, int, int, float, str, tuple[str, ...], str],
        ],
        PaliGemmaVLM,
    ] = {}

    def factory(
        model_dtype: str = PrecisionType.FP32.value,
        frozen: bool = True,
        lora_config: LoRAAdaptation | None = None,
    ) -> PaliGemmaVLM:
        cache_key = (model_dtype, frozen, _lora_cache_key(lora_config=lora_config))
        if cache_key not in cache:
            with patch(
                "versatil.models.decoding.generative_language_models.vision_language.huggingface.AutoConfig.from_pretrained",
                return_value=tiny_config,
            ):
                cache[cache_key] = PaliGemmaVLM(
                    input_keys=VLM_INPUT_KEYS,
                    pretrained=False,
                    frozen=frozen,
                    model_name=PaliGemmaModelType.PALIGEMMA2_3B_224.value,
                    model_dtype=model_dtype,
                    lora_config=lora_config,
                )
        return cache[cache_key]

    return factory
