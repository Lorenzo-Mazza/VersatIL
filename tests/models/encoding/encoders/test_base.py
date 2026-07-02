"""Tests for versatil.models.encoding.encoders.base module."""

import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from versatil.models.encoding.encoders.base import (
    EncoderInput,
    EncodingMixin,
)
from versatil.models.feature_meta import FeatureMetadata, FeatureType
from versatil.training.constants import PrecisionType


class ConcreteEncodingMixin(EncodingMixin):
    """Minimal concrete implementation for testing the abstract base."""

    def __init__(
        self,
        input_specification: EncoderInput,
        pretrained: bool = False,
        frozen: bool = False,
        device: str | None = None,
        model_dtype: str | None = None,
    ):
        super().__init__(
            input_specification=input_specification,
            pretrained=pretrained,
            frozen=frozen,
            device=device,
            model_dtype=model_dtype,
        )
        self.linear = nn.Linear(64, 64)
        self._apply_model_dtype()

    def rebuild(self) -> None:
        """Simulates a deferred rebuild (e.g. set_image_size / _build_network)."""
        self.linear = nn.Linear(64, 64)
        self._apply_model_dtype()

    def get_output_specification(self) -> list[FeatureMetadata]:
        return [
            FeatureMetadata(
                key="test",
                feature_type=FeatureType.FLAT.value,
                dimension=(64,),
            )
        ]


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
        model_dtype: str | None = None,
    ) -> ConcreteEncodingMixin:
        input_specification = encoder_input_factory(keys=keys)
        return ConcreteEncodingMixin(
            input_specification=input_specification,
            pretrained=pretrained,
            frozen=frozen,
            device=device,
            model_dtype=model_dtype,
        )

    return factory


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
                pytest.raises(
                    ValueError,
                    match=re.escape("Missing required inputs: {'right'}"),
                ),
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
        "conditioning_key, conditioning_required, expectation",
        [
            ("rgb_embedding", ["rgb_embedding"], does_not_raise()),
            (
                "rgb_embedding",
                ["missing_key"],
                pytest.raises(
                    ValueError,
                    match=re.escape("Missing required conditioning: {'missing_key'}"),
                ),
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
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "Exactly one from ['rgb_embedding', 'depth_embedding'] "
                        "required for conditioning"
                    ),
                ),
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
        "cuda_available",
        [False, True],
    )
    def test_device_defaults_based_on_cuda_runtime_support(
        self,
        encoder_input_factory: Callable[..., EncoderInput],
        cuda_available: bool,
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
        expected_device_type = (
            "cuda" if cuda_available and torch.version.cuda is not None else "cpu"
        )
        assert encoder.device.type == expected_device_type

    def test_defaults_to_no_explainability_targets(
        self,
        concrete_encoder_factory: Callable[..., ConcreteEncodingMixin],
    ):
        encoder = concrete_encoder_factory(device="cpu")
        assert encoder.get_explainability_targets() == []
        assert not encoder.is_vision_encoder()

    def test_validates_input_specification_on_init(
        self,
        encoder_input_factory: Callable[..., EncoderInput],
    ):
        invalid_specification = encoder_input_factory(
            keys=["left"],
            required=["left", "missing_key"],
        )
        with pytest.raises(
            ValueError,
            match=re.escape("Missing required inputs: {'missing_key'}"),
        ):
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

    def test_fully_frozen_encoder_stays_in_eval_mode_when_train_called(
        self,
        concrete_encoder_factory: Callable[..., ConcreteEncodingMixin],
    ):
        encoder = concrete_encoder_factory(frozen=True)
        encoder._freeze_weights()

        encoder.train()

        assert encoder.training is False
        assert encoder.linear.training is False

    def test_partially_unfrozen_encoder_can_enter_train_mode(
        self,
        concrete_encoder_factory: Callable[..., ConcreteEncodingMixin],
    ):
        encoder = concrete_encoder_factory(frozen=True)
        encoder._freeze_weights()
        next(encoder.parameters()).requires_grad_(True)

        encoder.train()

        assert encoder.training is True
        assert encoder.linear.training is True


class TestEncodingMixinGetVocabSize:
    def test_returns_none_by_default(
        self,
        concrete_encoder_factory: Callable[..., ConcreteEncodingMixin],
    ):
        encoder = concrete_encoder_factory()
        assert encoder.get_vocab_size() is None


class TestEncodingMixinModelDtype:
    @pytest.mark.parametrize(
        "model_dtype, expected_dtype, expectation",
        [
            (None, None, does_not_raise()),
            (
                PrecisionType.FP32.value,
                torch.float32,
                does_not_raise(),
            ),
            (
                PrecisionType.BF16_MIXED.value,
                torch.bfloat16,
                does_not_raise(),
            ),
            (
                PrecisionType.FP16_MIXED.value,
                torch.float16,
                does_not_raise(),
            ),
            (
                "invalid_precision",
                None,
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "Invalid model_dtype 'invalid_precision'. Must be one of: "
                        f"{[p.value for p in PrecisionType]}"
                    ),
                ),
            ),
        ],
    )
    def test_resolves_precision_string_to_torch_dtype(
        self,
        concrete_encoder_factory: Callable[..., ConcreteEncodingMixin],
        model_dtype: str | None,
        expected_dtype: torch.dtype | None,
        expectation,
    ):
        with expectation:
            encoder = concrete_encoder_factory(model_dtype=model_dtype)
            assert encoder.model_dtype == expected_dtype


class TestEncodingMixinApplyModelDtype:
    @pytest.mark.parametrize(
        "model_dtype, expected_dtype",
        [
            (None, torch.float32),
            (PrecisionType.FP32.value, torch.float32),
            (PrecisionType.BF16_MIXED.value, torch.bfloat16),
            (PrecisionType.BF16_TRUE.value, torch.bfloat16),
            (PrecisionType.FP16_MIXED.value, torch.float16),
        ],
    )
    def test_all_parameters_share_model_dtype_after_init(
        self,
        concrete_encoder_factory: Callable[..., ConcreteEncodingMixin],
        model_dtype: str | None,
        expected_dtype: torch.dtype,
    ):
        encoder = concrete_encoder_factory(model_dtype=model_dtype)
        for parameter in encoder.parameters():
            assert parameter.dtype == expected_dtype

    @pytest.mark.parametrize(
        "model_dtype, expected_dtype",
        [
            (PrecisionType.FP32.value, torch.float32),
            (PrecisionType.BF16_MIXED.value, torch.bfloat16),
        ],
    )
    def test_rebuild_preserves_model_dtype(
        self,
        concrete_encoder_factory: Callable[..., ConcreteEncodingMixin],
        model_dtype: str,
        expected_dtype: torch.dtype,
    ):
        encoder = concrete_encoder_factory(model_dtype=model_dtype)
        encoder.rebuild()
        for parameter in encoder.parameters():
            assert parameter.dtype == expected_dtype

    def test_no_op_when_model_dtype_is_none(
        self,
        concrete_encoder_factory: Callable[..., ConcreteEncodingMixin],
    ):
        encoder = concrete_encoder_factory(model_dtype=None)
        # Preserve PyTorch's default parameter dtype (fp32) without any cast.
        for parameter in encoder.parameters():
            assert parameter.dtype == torch.float32
