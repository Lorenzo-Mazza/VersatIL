"""Shared fixtures for VLM backbone tests."""

from collections.abc import Callable

import pytest
import torch

from versatil.data.constants import Cameras
from versatil.models.adaptation.lora import LoRAAdaptation
from versatil.models.decoding.generative_language_models.vision_language.smolvlm import (
    SmolVLM,
)
from versatil.training.constants import PrecisionType

VLM_INPUT_KEYS = [
    Cameras.LEFT.value,
]


@pytest.fixture
def language_input_factory(
    padding_mask_factory: Callable[..., torch.Tensor],
) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    """Create token batches and attention masks.

    Note:
        B is batch size and S is the number of token IDs.
    """

    def factory(
        token_ids: list[int],
        batch_size: int = 1,
        padded_from: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.tensor(token_ids, dtype=torch.long).repeat(
            batch_size, 1
        )  # (B, S)
        padding_mask = padding_mask_factory(
            batch_size=batch_size,
            sequence_length=len(token_ids),
            padded_from=padded_from,
        )  # (B, S)
        return tokens, (~padding_mask).to(dtype=torch.long)  # (B, S), (B, S)

    return factory


@pytest.fixture(scope="session")
def real_smolvlm_backbone(
    tiny_smolvlm_backbone_factory: Callable[..., SmolVLM],
    lora_cache_key_factory: Callable[
        [LoRAAdaptation | None], tuple[bool, int, int, float, str, tuple[str, ...], str]
    ],
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
        cache_key = (
            model_dtype,
            frozen,
            lora_cache_key_factory(lora_config=lora_config),
        )
        if cache_key not in cache:
            cache[cache_key] = tiny_smolvlm_backbone_factory(
                input_keys=VLM_INPUT_KEYS,
                frozen=frozen,
                model_dtype=model_dtype,
                lora_config=lora_config,
            )
        return cache[cache_key]

    return factory
