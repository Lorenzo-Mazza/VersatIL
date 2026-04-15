"""Shared fixtures for VLM encoder tests."""

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

from versatil.data.constants import Cameras, SampleKey
from versatil.models.encoding.encoders.constants import (
    PaliGemmaModelType,
    SmolVLMModelType,
)
from versatil.models.encoding.encoders.cross_modal.vision_language.paligemma import (
    PaliGemmaEncoder,
)
from versatil.models.encoding.encoders.cross_modal.vision_language.smolvlm import (
    SmolVLMEncoder,
)
from versatil.training.constants import PrecisionType

TINY_HIDDEN_DIM = 32

VLM_INPUT_KEYS = [
    Cameras.LEFT.value,
    SampleKey.TOKENIZED_OBSERVATIONS.value,
]


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
def real_smolvlm_encoder() -> Callable[..., SmolVLMEncoder]:
    """Factory for a real but tiny SmolVLM encoder, cached per dtype."""
    tiny_config = make_tiny_smolvlm_config()
    cache: dict[str, SmolVLMEncoder] = {}

    def factory(
        model_dtype: str = PrecisionType.FP32.value,
    ) -> SmolVLMEncoder:
        if model_dtype not in cache:
            with patch(
                "versatil.models.encoding.encoders.cross_modal.vision_language.generative_vlm.AutoConfig.from_pretrained",
                return_value=tiny_config,
            ):
                cache[model_dtype] = SmolVLMEncoder(
                    input_keys=VLM_INPUT_KEYS,
                    pretrained=False,
                    frozen=False,
                    model_name=SmolVLMModelType.SMOLVLM_256M.value,
                    model_dtype=model_dtype,
                )
        return cache[model_dtype]

    return factory


@pytest.fixture(scope="session")
def real_paligemma_encoder() -> Callable[..., PaliGemmaEncoder]:
    """Factory for a real but tiny PaliGemma encoder, cached per dtype."""
    tiny_config = make_tiny_paligemma_config()
    cache: dict[str, PaliGemmaEncoder] = {}

    def factory(
        model_dtype: str = PrecisionType.FP32.value,
    ) -> PaliGemmaEncoder:
        if model_dtype not in cache:
            with patch(
                "versatil.models.encoding.encoders.cross_modal.vision_language.generative_vlm.AutoConfig.from_pretrained",
                return_value=tiny_config,
            ):
                cache[model_dtype] = PaliGemmaEncoder(
                    input_keys=VLM_INPUT_KEYS,
                    pretrained=False,
                    frozen=True,
                    model_name=PaliGemmaModelType.PALIGEMMA2_3B_224.value,
                    model_dtype=model_dtype,
                )
        return cache[model_dtype]

    return factory
