"""Shared fixtures for VLM backbone tests."""

from collections.abc import Callable

import pytest

from versatil.data.constants import Cameras
from versatil.models.adaptation.lora import LoRAAdaptation
from versatil.models.decoding.generative_language_models.vision_language.paligemma import (
    PaliGemmaVLM,
)
from versatil.models.decoding.generative_language_models.vision_language.smolvlm import (
    SmolVLM,
)
from versatil.training.constants import PrecisionType

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


@pytest.fixture(scope="session")
def real_smolvlm_backbone(
    tiny_smolvlm_backbone_factory: Callable[..., SmolVLM],
) -> Callable[..., SmolVLM]:
    """Factory for a real but tiny SmolVLM backbone, cached per dtype."""
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
            cache[cache_key] = tiny_smolvlm_backbone_factory(
                input_keys=VLM_INPUT_KEYS,
                frozen=frozen,
                model_dtype=model_dtype,
                lora_config=lora_config,
            )
        return cache[cache_key]

    return factory


@pytest.fixture(scope="session")
def real_paligemma_backbone(
    tiny_paligemma_backbone_factory: Callable[..., PaliGemmaVLM],
) -> Callable[..., PaliGemmaVLM]:
    """Factory for a real but tiny PaliGemma backbone, cached per dtype."""
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
            cache[cache_key] = tiny_paligemma_backbone_factory(
                input_keys=VLM_INPUT_KEYS,
                frozen=frozen,
                model_dtype=model_dtype,
                lora_config=lora_config,
            )
        return cache[cache_key]

    return factory
