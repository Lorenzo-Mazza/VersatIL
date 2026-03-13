"""Tests for versatil.models.encoding.encoders.unconditional module."""
from collections.abc import Callable

import pytest

from versatil.models.encoding.encoders.base import EncoderInput, EncoderOutput, EncodingMixin
from versatil.models.encoding.encoders.unconditional import Encoder


class ConcreteEncoder(Encoder):
    """Minimal concrete implementation for testing."""

    def get_output_specification(self) -> EncoderOutput:
        return EncoderOutput(features=["test"], dimensions={"test": 64})

    def forward(self, inputs):
        return {}


@pytest.fixture
def concrete_encoder_factory(
    encoder_input_factory: Callable[..., EncoderInput],
) -> Callable[..., ConcreteEncoder]:
    """Factory for ConcreteEncoder instances."""
    def factory(
        keys: str | list[str] = "left",
        pretrained: bool = False,
        frozen: bool = False,
        device: str | None = "cpu",
    ) -> ConcreteEncoder:
        input_specification = encoder_input_factory(keys=keys)
        return ConcreteEncoder(
            input_specification=input_specification,
            pretrained=pretrained,
            frozen=frozen,
            device=device,
        )
    return factory


class TestEncoderInitialization:

    def test_inherits_from_encoding_mixin(
        self,
        concrete_encoder_factory: Callable[..., ConcreteEncoder],
    ):
        encoder = concrete_encoder_factory()
        assert isinstance(encoder, EncodingMixin)