"""Tests for versatil.models.encoding.encoders.base module."""

from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from versatil.models.encoding.encoders.base import (
    EncoderInput,
    EncoderOutput,
    EncodingMixin,
)


class ConcreteEncodingMixin(EncodingMixin):
    """Minimal concrete implementation for testing the abstract base."""

    def __init__(
        self,
        input_specification: EncoderInput,
        pretrained: bool = False,
        frozen: bool = False,
        device: str | None = None,
    ):
        super().__init__(
            input_specification=input_specification,
            pretrained=pretrained,
            frozen=frozen,
            device=device,
        )
        self.linear = nn.Linear(64, 64)

    def get_output_specification(self) -> EncoderOutput:
        return EncoderOutput(features=["test"], dimensions={"test": 64})


@pytest.fixture
def concrete_encoder_factory(
    encoder_input_factory: Callable[..., EncoderInput],
) -> Callable[..., ConcreteEncodingMixin]:
    """Factory for ConcreteEncodingMixin instances."""

    def factory(
        keys: str | list[str] = "left",
        pretrained: bool = False,
        frozen: bool = False,
        device: str | None = "cpu",
    ) -> ConcreteEncodingMixin:
        input_specification = encoder_input_factory(keys=keys)
        return ConcreteEncodingMixin(
            input_specification=input_specification,
            pretrained=pretrained,
            frozen=frozen,
            device=device,
        )

    return factory


class TestEncoderOutput:
    @pytest.mark.parametrize(
        "features, expected_is_multi_output",
        [
            (["embedding"], False),
            (["language", "visual"], True),
        ],
    )
    def test_is_multi_output_property(
        self,
        features: list[str],
        expected_is_multi_output: bool,
    ):
        output = EncoderOutput(
            features=features,
            dimensions=dict.fromkeys(features, 64),
        )
        assert output.is_multi_output == expected_is_multi_output


class TestEncoderInputPostInit:
    @pytest.mark.parametrize(
        "keys, expected_keys",
        [
            ("left", ["left"]),
            (["left", "right"], ["left", "right"]),
        ],
    )
    def test_normalizes_keys_to_list(
        self,
        keys: str | list[str],
        expected_keys: list[str],
    ):
        input_specification = EncoderInput(keys=keys)
        assert input_specification.keys == expected_keys


class TestEncoderInputValidation:
    @pytest.mark.parametrize(
        "keys, required, expectation",
        [
            (["left", "right"], ["left"], does_not_raise()),
            (
                ["left"],
                ["left", "right"],
                pytest.raises(ValueError, match="Missing required"),
            ),
        ],
    )
    def test_required_keys_validation(
        self,
        encoder_input_factory: Callable[..., EncoderInput],
        keys: list[str],
        required: list[str],
        expectation,
    ):
        input_specification = encoder_input_factory(keys=keys, required=required)
        with expectation:
            input_specification.validate()

    @pytest.mark.parametrize(
        "keys, one_of_groups, expectation",
        [
            (["left"], [["left", "right"]], does_not_raise()),
            (
                ["left", "right"],
                [["left", "right"]],
                pytest.raises(ValueError, match="Exactly one"),
            ),
            (
                ["depth"],
                [["left", "right"]],
                pytest.raises(ValueError, match="Exactly one"),
            ),
        ],
    )
    def test_one_of_groups_validation(
        self,
        encoder_input_factory: Callable[..., EncoderInput],
        keys: list[str],
        one_of_groups: list[list[str]],
        expectation,
    ):
        input_specification = encoder_input_factory(
            keys=keys, one_of_groups=one_of_groups
        )
        with expectation:
            input_specification.validate()

    @pytest.mark.parametrize(
        "keys, at_least_one_of_groups, expectation",
        [
            (["left"], [["left", "right"]], does_not_raise()),
            (["left", "right"], [["left", "right"]], does_not_raise()),
            (
                ["depth"],
                [["left", "right"]],
                pytest.raises(ValueError, match="At least one"),
            ),
        ],
    )
    def test_at_least_one_of_groups_validation(
        self,
        encoder_input_factory: Callable[..., EncoderInput],
        keys: list[str],
        at_least_one_of_groups: list[list[str]],
        expectation,
    ):
        input_specification = encoder_input_factory(
            keys=keys,
            at_least_one_of_groups=at_least_one_of_groups,
        )
        with expectation:
            input_specification.validate()

    @pytest.mark.parametrize(
        "conditioning_key, conditioning_required, expectation",
        [
            ("rgb_embedding", ["rgb_embedding"], does_not_raise()),
            (
                "rgb_embedding",
                ["missing_key"],
                pytest.raises(ValueError, match="Missing required conditioning"),
            ),
        ],
    )
    def test_conditioning_required_validation(
        self,
        encoder_input_factory: Callable[..., EncoderInput],
        conditioning_key: str,
        conditioning_required: list[str],
        expectation,
    ):
        input_specification = encoder_input_factory(
            conditioning_key=conditioning_key,
            conditioning_required=conditioning_required,
        )
        with expectation:
            input_specification.validate()

    @pytest.mark.parametrize(
        "conditioning_key, conditioning_one_of_groups, expectation",
        [
            ("rgb_embedding", [["rgb_embedding", "depth_embedding"]], does_not_raise()),
            (
                "other_key",
                [["rgb_embedding", "depth_embedding"]],
                pytest.raises(ValueError, match="Exactly one"),
            ),
        ],
    )
    def test_conditioning_one_of_groups_validation(
        self,
        encoder_input_factory: Callable[..., EncoderInput],
        conditioning_key: str,
        conditioning_one_of_groups: list[list[str]],
        expectation,
    ):
        input_specification = encoder_input_factory(
            conditioning_key=conditioning_key,
            conditioning_one_of_groups=conditioning_one_of_groups,
        )
        with expectation:
            input_specification.validate()

    def test_skips_conditioning_validation_without_conditioning_key(
        self,
        encoder_input_factory: Callable[..., EncoderInput],
    ):
        input_specification = encoder_input_factory(
            conditioning_required=["some_key"],
        )
        input_specification.validate()


class TestEncodingMixinInitialization:
    @pytest.mark.parametrize("pretrained", [True, False])
    @pytest.mark.parametrize("frozen", [True, False])
    @pytest.mark.parametrize("keys", ["left", ["left", "right"]])
    def test_stores_configuration(
        self,
        concrete_encoder_factory: Callable[..., ConcreteEncodingMixin],
        pretrained: bool,
        frozen: bool,
        keys: str | list[str],
    ):
        encoder = concrete_encoder_factory(
            pretrained=pretrained,
            frozen=frozen,
            keys=keys,
            device="cpu",
        )
        assert encoder.pretrained == pretrained
        assert encoder.frozen == frozen
        assert encoder.device == torch.device("cpu")
        expected_keys = [keys] if isinstance(keys, str) else keys
        assert encoder.input_specification.keys == expected_keys

    @pytest.mark.parametrize(
        "cuda_available, expected_device_type",
        [
            (False, "cpu"),
            (True, "cuda"),
        ],
    )
    def test_device_defaults_based_on_cuda_availability(
        self,
        encoder_input_factory: Callable[..., EncoderInput],
        cuda_available: bool,
        expected_device_type: str,
    ):
        input_specification = encoder_input_factory()
        with patch(
            "versatil.models.encoding.encoders.base.torch.cuda.is_available",
            return_value=cuda_available,
        ):
            encoder = ConcreteEncodingMixin(
                input_specification=input_specification,
                device=None,
            )
        assert encoder.device.type == expected_device_type

    def test_validates_input_specification_on_init(
        self,
        encoder_input_factory: Callable[..., EncoderInput],
    ):
        invalid_specification = encoder_input_factory(
            keys=["left"],
            required=["left", "missing_key"],
        )
        with pytest.raises(ValueError, match="Missing required"):
            ConcreteEncodingMixin(
                input_specification=invalid_specification,
                device="cpu",
            )


class TestEncodingMixinFreezeWeights:
    def test_sets_requires_grad_false_for_all_parameters(
        self,
        concrete_encoder_factory: Callable[..., ConcreteEncodingMixin],
    ):
        encoder = concrete_encoder_factory()
        for parameter in encoder.parameters():
            assert parameter.requires_grad is True
        encoder._freeze_weights()
        for parameter in encoder.parameters():
            assert parameter.requires_grad is False


class TestEncodingMixinGetVocabSize:
    def test_returns_none_by_default(
        self,
        concrete_encoder_factory: Callable[..., ConcreteEncodingMixin],
    ):
        encoder = concrete_encoder_factory()
        assert encoder.get_vocab_size() is None
