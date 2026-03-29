"""Tests for versatil.models.encoding.encoders.language_mixin module."""

import logging
import re
from collections.abc import Callable

import numpy as np
import pytest
import torch

from versatil.data.constants import SampleKey
from versatil.models.encoding.encoders.constants import EncoderOutputKeys, PoolingMethod
from versatil.models.encoding.encoders.language_mixin import LanguageEncoderMixin


class ConcreteLanguageEncoder(LanguageEncoderMixin):
    def __init__(self, output_modality: str):
        self._setup_language_keys(output_modality=output_modality)


@pytest.fixture
def language_mixin_factory() -> Callable[..., ConcreteLanguageEncoder]:
    def factory(
        output_modality: str = EncoderOutputKeys.LANGUAGE.value,
    ) -> ConcreteLanguageEncoder:
        return ConcreteLanguageEncoder(output_modality=output_modality)

    return factory


@pytest.fixture
def token_ids_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    def factory(
        batch_size: int = 2,
        sequence_length: int = 10,
        vocab_size: int = 1000,
    ) -> torch.Tensor:
        return torch.from_numpy(
            rng.integers(
                low=0, high=vocab_size, size=(batch_size, sequence_length)
            ).astype(np.int64)
        )

    return factory


class TestSetupLanguageKeys:
    @pytest.mark.parametrize(
        "output_modality, expected_mask_name",
        [
            (
                EncoderOutputKeys.LANGUAGE.value,
                f"{EncoderOutputKeys.LANGUAGE.value}_{EncoderOutputKeys.PADDING_MASK.value}",
            ),
            (
                EncoderOutputKeys.FUSED_RGB_LANGUAGE.value,
                f"{EncoderOutputKeys.FUSED_RGB_LANGUAGE.value}_{EncoderOutputKeys.PADDING_MASK.value}",
            ),
        ],
    )
    def test_sets_language_key_and_padding_mask_name(
        self,
        output_modality: str,
        expected_mask_name: str,
    ):
        encoder = ConcreteLanguageEncoder(output_modality=output_modality)
        assert encoder.language_key == SampleKey.TOKENIZED_OBSERVATIONS.value
        assert encoder.padding_mask_name == expected_mask_name


class TestExtractTextInputs:
    def test_extracts_token_ids_and_mask(
        self,
        language_mixin_factory: Callable[..., ConcreteLanguageEncoder],
        token_ids_factory: Callable[..., torch.Tensor],
    ):
        encoder = language_mixin_factory()
        token_ids = token_ids_factory()
        mask = torch.zeros(2, 10, dtype=torch.bool)
        inputs = {
            SampleKey.TOKENIZED_OBSERVATIONS.value: token_ids,
            SampleKey.IS_PAD_OBSERVATION.value: mask,
        }
        result_ids, result_mask = encoder._extract_text_inputs(inputs=inputs)
        assert torch.equal(result_ids, token_ids)
        assert torch.equal(result_mask, mask)

    def test_returns_none_mask_when_absent(
        self,
        language_mixin_factory: Callable[..., ConcreteLanguageEncoder],
        token_ids_factory: Callable[..., torch.Tensor],
    ):
        encoder = language_mixin_factory()
        token_ids = token_ids_factory()
        inputs = {SampleKey.TOKENIZED_OBSERVATIONS.value: token_ids}
        _, result_mask = encoder._extract_text_inputs(inputs=inputs)
        assert result_mask is None

    def test_raises_when_language_key_missing(
        self,
        language_mixin_factory: Callable[..., ConcreteLanguageEncoder],
    ):
        encoder = language_mixin_factory()
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"ConcreteLanguageEncoder expects pre-tokenized input. "
                f"Expected key '{SampleKey.TOKENIZED_OBSERVATIONS.value}' "
                f"not found in inputs. "
                f"Ensure tokenization is enabled in DataloaderConfig."
            ),
        ):
            encoder._extract_text_inputs(inputs={"wrong_key": torch.zeros(2, 10)})


class TestPadTextInputs:
    def test_truncates_longer_sequence(
        self,
        language_mixin_factory: Callable[..., ConcreteLanguageEncoder],
        token_ids_factory: Callable[..., torch.Tensor],
        caplog,
    ):
        encoder = language_mixin_factory()
        token_ids = token_ids_factory(sequence_length=20)
        mask = torch.zeros(2, 20, dtype=torch.bool)
        with caplog.at_level(logging.WARNING):
            result_ids, result_mask = encoder._pad_text_inputs(
                text_input_ids=token_ids, language_mask=mask, max_length=10
            )
        assert result_ids.shape[1] == 10
        assert result_mask.shape[1] == 10
        assert torch.equal(result_ids, token_ids[:, :10])
        assert "Input text length 20 exceeds max_length 10" in caplog.text

    def test_pads_shorter_sequence(
        self,
        language_mixin_factory: Callable[..., ConcreteLanguageEncoder],
        token_ids_factory: Callable[..., torch.Tensor],
    ):
        encoder = language_mixin_factory()
        token_ids = token_ids_factory(sequence_length=5)
        mask = torch.zeros(2, 5, dtype=torch.bool)
        result_ids, result_mask = encoder._pad_text_inputs(
            text_input_ids=token_ids, language_mask=mask, max_length=10
        )
        assert result_ids.shape[1] == 10
        assert result_mask.shape[1] == 10
        assert torch.equal(result_ids[:, :5], token_ids)
        assert torch.all(result_ids[:, 5:] == 0)
        # Padded positions marked True in mask
        assert torch.all(result_mask[:, 5:])

    def test_exact_length_unchanged(
        self,
        language_mixin_factory: Callable[..., ConcreteLanguageEncoder],
        token_ids_factory: Callable[..., torch.Tensor],
    ):
        encoder = language_mixin_factory()
        token_ids = token_ids_factory(sequence_length=10)
        mask = torch.zeros(2, 10, dtype=torch.bool)
        result_ids, result_mask = encoder._pad_text_inputs(
            text_input_ids=token_ids, language_mask=mask, max_length=10
        )
        assert torch.equal(result_ids, token_ids)
        assert torch.equal(result_mask, mask)

    def test_none_mask_stays_none_on_truncate(
        self,
        language_mixin_factory: Callable[..., ConcreteLanguageEncoder],
        token_ids_factory: Callable[..., torch.Tensor],
    ):
        encoder = language_mixin_factory()
        token_ids = token_ids_factory(sequence_length=20)
        result_ids, result_mask = encoder._pad_text_inputs(
            text_input_ids=token_ids, language_mask=None, max_length=10
        )
        assert result_ids.shape[1] == 10
        assert result_mask is None

    def test_none_mask_stays_none_on_pad(
        self,
        language_mixin_factory: Callable[..., ConcreteLanguageEncoder],
        token_ids_factory: Callable[..., torch.Tensor],
    ):
        encoder = language_mixin_factory()
        token_ids = token_ids_factory(sequence_length=5)
        result_ids, result_mask = encoder._pad_text_inputs(
            text_input_ids=token_ids, language_mask=None, max_length=10
        )
        assert result_ids.shape[1] == 10
        assert result_mask is None


class TestBuildAttentionMask:
    def test_inverts_language_mask(
        self,
        language_mixin_factory: Callable[..., ConcreteLanguageEncoder],
        token_ids_factory: Callable[..., torch.Tensor],
    ):
        encoder = language_mixin_factory()
        token_ids = token_ids_factory()
        language_mask = torch.tensor(
            [[False, False, True, True, True] + [True] * 5] * 2
        )
        result = encoder._build_attention_mask(
            language_mask=language_mask, text_input_ids=token_ids
        )
        assert result.dtype == torch.long
        assert result[0, 0].item() == 1
        assert result[0, 2].item() == 0

    def test_all_ones_when_mask_is_none(
        self,
        language_mixin_factory: Callable[..., ConcreteLanguageEncoder],
        token_ids_factory: Callable[..., torch.Tensor],
    ):
        encoder = language_mixin_factory()
        token_ids = token_ids_factory(sequence_length=8)
        result = encoder._build_attention_mask(
            language_mask=None, text_input_ids=token_ids
        )
        assert result.dtype == torch.long
        assert result.shape == (2, 8)
        assert torch.all(result == 1)


class TestBuildOutputPaddingMask:
    def test_none_pooling_returns_inverted_attention_mask(
        self,
        language_mixin_factory: Callable[..., ConcreteLanguageEncoder],
    ):
        encoder = language_mixin_factory()
        attention_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long)
        result = encoder._build_output_padding_mask(
            attention_mask=attention_mask,
            pooling_method=PoolingMethod.NONE.value,
            batch_size=1,
            device=torch.device("cpu"),
        )
        assert result.dtype == torch.bool
        expected = torch.tensor([[False, False, False, True, True]])
        assert torch.equal(result, expected)

    @pytest.mark.parametrize(
        "pooling_method",
        [PoolingMethod.DEFAULT.value, PoolingMethod.AVERAGE.value],
    )
    def test_pooled_output_returns_scalar_false(
        self,
        language_mixin_factory: Callable[..., ConcreteLanguageEncoder],
        pooling_method: str,
    ):
        encoder = language_mixin_factory()
        attention_mask = torch.ones(3, 10, dtype=torch.long)
        result = encoder._build_output_padding_mask(
            attention_mask=attention_mask,
            pooling_method=pooling_method,
            batch_size=3,
            device=torch.device("cpu"),
        )
        assert result.shape == (3,)
        assert not result.any()
