"""Encoding package test fixtures: mock factories for encoder and fusion dependencies."""
from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from versatil.models.encoding.encoders.base import EncoderInput, EncoderOutput, EncodingMixin
from versatil.models.encoding.encoders.conditional import ConditionalEncoder
from versatil.models.encoding.fusion.base import FusionModule


@pytest.fixture
def encoder_mock_factory(rng: np.random.Generator) -> Callable[..., MagicMock]:
    """Factory for mock encoders compatible with EncodingPipeline setup."""
    def factory(
        output_features: list[str] | None = None,
        output_dimensions: dict[str, int | tuple[int, ...]] | None = None,
        input_keys: list[str] | None = None,
        requires_tokenized: bool = False,
        vocab_size: int | None = None,
        forward_return: dict[str, torch.Tensor] | None = None,
        batch_size: int = 2,
    ) -> MagicMock:
        if output_features is None:
            output_features = ["embedding"]
        if output_dimensions is None:
            output_dimensions = {feat: 64 for feat in output_features}
        if input_keys is None:
            input_keys = ["left"]
        encoder = MagicMock(spec=EncodingMixin)
        encoder.get_output_specification.return_value = EncoderOutput(
            features=output_features,
            dimensions=output_dimensions,
        )
        encoder.input_specification = EncoderInput(
            keys=input_keys,
            requires_tokenized=requires_tokenized,
        )
        encoder.get_vocab_size.return_value = vocab_size
        if forward_return is None:
            forward_return = {
                feat: torch.from_numpy(
                    rng.standard_normal((batch_size, dim) if isinstance(dim, int) else (batch_size, *dim)).astype(
                        np.float32
                    )
                )
                for feat, dim in output_dimensions.items()
            }
        encoder.return_value = forward_return
        return encoder
    return factory


@pytest.fixture
def conditional_encoder_mock_factory(
    rng: np.random.Generator,
) -> Callable[..., MagicMock]:
    """Factory for mock conditional encoders compatible with EncodingPipeline setup."""
    def factory(
        output_features: list[str] | None = None,
        output_dimensions: dict[str, int | tuple[int, ...]] | None = None,
        input_keys: list[str] | None = None,
        condition_key: str = "rgb_encoder_embedding",
        forward_return: dict[str, torch.Tensor] | None = None,
        batch_size: int = 2,
    ) -> MagicMock:
        if output_features is None:
            output_features = ["embedding"]
        if output_dimensions is None:
            output_dimensions = {feat: 64 for feat in output_features}
        if input_keys is None:
            input_keys = ["right"]
        encoder = MagicMock(spec=ConditionalEncoder)
        encoder.get_output_specification.return_value = EncoderOutput(
            features=output_features,
            dimensions=output_dimensions,
        )
        encoder.input_specification = EncoderInput(
            keys=input_keys,
            conditioning_key=condition_key,
            requires_tokenized=False,
        )
        encoder.condition_key = condition_key
        encoder.get_vocab_size.return_value = None
        if forward_return is None:
            forward_return = {
                feat: torch.from_numpy(
                    rng.standard_normal((batch_size, dim) if isinstance(dim, int) else (batch_size, *dim)).astype(
                        np.float32
                    )
                )
                for feat, dim in output_dimensions.items()
            }
        encoder.return_value = forward_return
        return encoder
    return factory


@pytest.fixture
def fusion_module_mock_factory(
    rng: np.random.Generator,
) -> Callable[..., MagicMock]:
    """Factory for mock fusion modules compatible with EncodingPipeline setup."""
    def factory(
        input_features: list[str] | None = None,
        output_name: str = "fused",
        output_dimension: int = 128,
        forward_return: torch.Tensor | None = None,
        batch_size: int = 2,
    ) -> MagicMock:
        if input_features is None:
            input_features = ["rgb_encoder_embedding"]
        fusion = MagicMock(spec=FusionModule)
        fusion.input_features = input_features
        fusion.output_name = output_name
        fusion.get_output_dim.return_value = output_dimension
        if forward_return is None:
            forward_return = torch.from_numpy(
                rng.standard_normal((batch_size, output_dimension)).astype(np.float32)
            )
        fusion.return_value = forward_return
        return fusion
    return factory