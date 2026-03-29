"""Tests for versatil.models.encoding.encoders.conditional module."""

import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import torch

from versatil.models.encoding.encoders.base import EncoderInput
from versatil.models.encoding.encoders.conditional import ConditionalEncoder
from versatil.models.feature_meta import FeatureMetadata, FeatureType


class ConcreteConditionalEncoder(ConditionalEncoder):
    def get_output_specification(self) -> list[FeatureMetadata]:
        return [
            FeatureMetadata(
                key="test", feature_type=FeatureType.FLAT.value, dimension=(64,)
            )
        ]

    def encode(self, inputs, conditioning):
        first = next(iter(inputs.values()))
        batch_size = first.shape[0]
        return {"test": torch.zeros(batch_size, 64)}


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
    @pytest.mark.parametrize(
        "conditioning_key, expectation",
        [
            ("rgb_embedding", does_not_raise()),
            (
                None,
                pytest.raises(
                    ValueError,
                    match="Conditional encoder requires conditioning_key",
                ),
            ),
        ],
    )
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

    @pytest.mark.parametrize(
        "pretrained, frozen",
        [
            (False, False),
            (True, True),
        ],
    )
    def test_stores_base_attributes(
        self,
        concrete_conditional_encoder_factory: Callable[..., ConcreteConditionalEncoder],
        pretrained: bool,
        frozen: bool,
    ):
        encoder = concrete_conditional_encoder_factory(
            pretrained=pretrained, frozen=frozen
        )
        assert encoder.pretrained is pretrained
        assert encoder.frozen is frozen
        assert encoder.condition_key == "rgb_embedding"


class TestConditionalEncoderTemporalHandling:
    @pytest.mark.parametrize("time_steps", [1, 3])
    def test_forward_flattens_temporal_and_restores(
        self,
        concrete_conditional_encoder_factory: Callable[..., ConcreteConditionalEncoder],
        rng: np.random.Generator,
        time_steps: int,
    ):
        encoder = concrete_conditional_encoder_factory()
        batch_size = 2
        inputs = {
            "right": torch.from_numpy(
                rng.standard_normal((batch_size, time_steps, 3, 8, 8)).astype(
                    np.float32
                )
            )
        }
        conditioning = torch.from_numpy(
            rng.standard_normal((batch_size, time_steps, 64)).astype(np.float32)
        )
        output = encoder.forward(inputs=inputs, conditioning=conditioning)
        assert output["test"].shape == (batch_size, time_steps, 64)

    def test_2d_conditioning_replicated_across_time(
        self,
        concrete_conditional_encoder_factory: Callable[..., ConcreteConditionalEncoder],
        rng: np.random.Generator,
    ):
        encoder = concrete_conditional_encoder_factory()
        batch_size = 2
        time_steps = 3
        inputs = {
            "right": torch.from_numpy(
                rng.standard_normal((batch_size, time_steps, 3, 8, 8)).astype(
                    np.float32
                )
            )
        }
        # 2D conditioning (B, D) — no temporal dim
        conditioning = torch.from_numpy(
            rng.standard_normal((batch_size, 64)).astype(np.float32)
        )
        output = encoder.forward(inputs=inputs, conditioning=conditioning)
        assert output["test"].shape == (batch_size, time_steps, 64)

    def test_raises_for_non_tensor_input(
        self,
        concrete_conditional_encoder_factory: Callable[..., ConcreteConditionalEncoder],
    ):
        encoder = concrete_conditional_encoder_factory()
        conditioning = torch.zeros(2, 64)
        with pytest.raises(
            ValueError,
            match=re.escape("Encoder input 'right' must be a torch.Tensor, got list."),
        ):
            encoder.forward(inputs={"right": [1, 2, 3]}, conditioning=conditioning)
