"""Tests for versatil.models.encoding.encoders.conditional module."""
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise

import pytest

from versatil.models.encoding.encoders.base import EncoderInput, EncoderOutput, EncodingMixin
from versatil.models.encoding.encoders.conditional import ConditionalEncoder


class ConcreteConditionalEncoder(ConditionalEncoder):
    """Minimal concrete implementation for testing."""

    def get_output_specification(self) -> EncoderOutput:
        return EncoderOutput(features=["test"], dimensions={"test": 64})

    def forward(self, inputs, conditioning):
        return {}


@pytest.fixture
def concrete_conditional_encoder_factory(
    encoder_input_factory: Callable[..., EncoderInput],
) -> Callable[..., ConcreteConditionalEncoder]:
    """Factory for ConcreteConditionalEncoder instances."""
    def factory(
        keys: str | list[str] = "right",
        conditioning_key: str = "rgb_embedding",
        pretrained: bool = False,
        frozen: bool = False,
        device: str | None = "cpu",
    ) -> ConcreteConditionalEncoder:
        input_specification = encoder_input_factory(
            keys=keys,
            conditioning_key=conditioning_key,
        )
        return ConcreteConditionalEncoder(
            input_specification=input_specification,
            pretrained=pretrained,
            frozen=frozen,
            device=device,
        )
    return factory


class TestConditionalEncoderInitialization:

    @pytest.mark.parametrize("conditioning_key, expectation", [
        ("rgb_embedding", does_not_raise()),
        (None, pytest.raises(ValueError, match="requires conditioning_key")),
    ])
    def test_conditioning_key_validation(
        self,
        encoder_input_factory: Callable[..., EncoderInput],
        conditioning_key: str | None,
        expectation,
    ):
        input_specification = encoder_input_factory(conditioning_key=conditioning_key)
        with expectation:
            ConcreteConditionalEncoder(
                input_specification=input_specification,
                device="cpu",
            )

    def test_stores_condition_key_attribute(
        self,
        concrete_conditional_encoder_factory: Callable[..., ConcreteConditionalEncoder],
    ):
        encoder = concrete_conditional_encoder_factory(conditioning_key="rgb_embedding")
        assert encoder.condition_key == "rgb_embedding"

    def test_inherits_from_encoding_mixin(
        self,
        concrete_conditional_encoder_factory: Callable[..., ConcreteConditionalEncoder],
    ):
        encoder = concrete_conditional_encoder_factory()
        assert isinstance(encoder, EncodingMixin)