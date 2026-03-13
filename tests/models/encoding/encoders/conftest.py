"""Encoder test fixtures."""
from collections.abc import Callable

import numpy as np
import pytest
import torch

from versatil.models.encoding.encoders.base import EncoderInput


@pytest.fixture
def encoder_input_factory() -> Callable[..., EncoderInput]:
    """Factory for EncoderInput instances with configurable fields."""
    def factory(
        keys: str | list[str] = "left",
        required: list[str] | None = None,
        one_of_groups: list[list[str]] | None = None,
        at_least_one_of_groups: list[list[str]] | None = None,
        conditioning_key: str | None = None,
        conditioning_required: list[str] | None = None,
        conditioning_one_of_groups: list[list[str]] | None = None,
        requires_tokenized: bool = False,
    ) -> EncoderInput:
        return EncoderInput(
            keys=keys,
            required=required if required is not None else [],
            one_of_groups=one_of_groups if one_of_groups is not None else [],
            at_least_one_of_groups=at_least_one_of_groups if at_least_one_of_groups is not None else [],
            conditioning_key=conditioning_key,
            conditioning_required=conditioning_required if conditioning_required is not None else [],
            conditioning_one_of_groups=conditioning_one_of_groups if conditioning_one_of_groups is not None else [],
            requires_tokenized=requires_tokenized,
        )
    return factory


@pytest.fixture
def image_input_factory(
    rng: np.random.Generator,
) -> Callable[..., dict[str, torch.Tensor]]:
    """Factory for image input tensors."""
    def factory(
        key: str = "left",
        batch_size: int = 2,
        channels: int = 3,
        height: int = 224,
        width: int = 224,
        time_steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if time_steps is not None:
            shape = (batch_size, time_steps, channels, height, width)
        else:
            shape = (batch_size, channels, height, width)
        tensor = torch.from_numpy(
            rng.standard_normal(shape).astype(np.float32)
        )
        return {key: tensor}
    return factory