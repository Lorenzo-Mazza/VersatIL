"""Tests for versatil.models.encoding.encoders.unconditional module."""

import re
from collections.abc import Callable

import numpy as np
import pytest
import torch

from versatil.models.encoding.encoders.base import EncoderInput
from versatil.models.encoding.encoders.unconditional import Encoder
from versatil.models.feature_meta import FeatureMetadata, FeatureType


class ConcreteEncoder(Encoder):
    def get_output_specification(self) -> list[FeatureMetadata]:
        return [
            FeatureMetadata(
                key="test", feature_type=FeatureType.FLAT.value, dimension=(64,)
            )
        ]

    def encode(self, inputs):
        first = next(iter(inputs.values()))
        batch_size = first.shape[0]
        return {"test": torch.zeros(batch_size, 64)}


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
    @pytest.mark.parametrize(
        "pretrained, frozen",
        [
            (False, False),
            (True, True),
        ],
    )
    def test_stores_base_attributes(
        self,
        concrete_encoder_factory: Callable[..., ConcreteEncoder],
        pretrained: bool,
        frozen: bool,
    ):
        encoder = concrete_encoder_factory(pretrained=pretrained, frozen=frozen)
        assert encoder.pretrained is pretrained
        assert encoder.frozen is frozen
        assert encoder.input_specification.keys == ["left"]

    def test_get_output_specification_returns_feature_metadata_list(
        self,
        concrete_encoder_factory: Callable[..., ConcreteEncoder],
    ):
        encoder = concrete_encoder_factory()
        spec = encoder.get_output_specification()
        assert len(spec) == 1
        assert spec[0].key == "test"
        assert spec[0].dimension == (64,)
        assert spec[0].feature_type == FeatureType.FLAT.value


class TestValidateInputs:
    def test_raises_for_non_tensor_input(
        self,
        concrete_encoder_factory: Callable[..., ConcreteEncoder],
    ):
        encoder = concrete_encoder_factory()
        with pytest.raises(
            ValueError,
            match=re.escape("Encoder input 'left' must be a torch.Tensor, got list."),
        ):
            encoder.forward({"left": [1, 2, 3]})

    @pytest.mark.parametrize("ndim", [1, 2])
    def test_raises_for_tensor_without_temporal_dimension(
        self,
        concrete_encoder_factory: Callable[..., ConcreteEncoder],
        ndim: int,
    ):
        encoder = concrete_encoder_factory()
        shape = tuple(range(2, 2 + ndim))
        tensor = torch.zeros(shape)
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Encoder input 'left' has shape {tuple(tensor.shape)} "
                f"but all inputs must have a temporal dimension (B, T, ...)."
            ),
        ):
            encoder.forward({"left": tensor})


class TestTemporalFlattenUnflatten:
    def test_forward_flattens_temporal_and_restores(
        self,
        concrete_encoder_factory: Callable[..., ConcreteEncoder],
        rng: np.random.Generator,
    ):
        encoder = concrete_encoder_factory()
        batch_size = 2
        time_steps = 3
        inputs = {
            "left": torch.from_numpy(
                rng.standard_normal((batch_size, time_steps, 3, 8, 8)).astype(
                    np.float32
                )
            )
        }
        output = encoder.forward(inputs)
        assert output["test"].shape == (batch_size, time_steps, 64)

    def test_forward_with_single_timestep(
        self,
        concrete_encoder_factory: Callable[..., ConcreteEncoder],
        rng: np.random.Generator,
    ):
        encoder = concrete_encoder_factory()
        batch_size = 4
        inputs = {
            "left": torch.from_numpy(
                rng.standard_normal((batch_size, 1, 3, 8, 8)).astype(np.float32)
            )
        }
        output = encoder.forward(inputs)
        assert output["test"].shape == (batch_size, 1, 64)
