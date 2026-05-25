"""Tests for versatil.data.tokenization.action_tokenizer."""

import re
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from versatil.data.constants import SampleKey
from versatil.data.tokenization.action_discretizer import (
    BinnedActionDiscretizer,
    FastActionDiscretizer,
)
from versatil.data.tokenization.action_token_id_mapping import (
    IdentityActionTokenIdMapping,
    LanguageVocabularyActionTokenIdMapping,
)
from versatil.data.tokenization.action_tokenizer import ActionTokenizer


@pytest.fixture
def mock_auto_processor():
    """Patches load_fast_processor in action_tokenizer module."""
    with patch(
        "versatil.data.tokenization.action_discretizer.load_fast_processor"
    ) as mock:
        mock.return_value.time_horizon = None
        mock.return_value.action_dim = None
        yield mock


@pytest.fixture
def action_tokenizer_factory(mock_auto_processor):
    """Factory for ActionTokenizer with load_fast_processor mocked."""

    def factory(
        action_discretizer=None,
        token_id_mapping=None,
        use_pretrained: bool = True,
        language_tokenizer_model: str | None = None,
        fast_tokenizer_model: str = "physical-intelligence/fast",
        num_special_tokens_to_skip: int = 128,
        max_token_len: int = 256,
        pad_token_id: int = 0,
        device: torch.device | None = None,
    ) -> ActionTokenizer:
        if action_discretizer is None:
            action_discretizer = FastActionDiscretizer(
                use_pretrained=use_pretrained,
                tokenizer_model=fast_tokenizer_model,
            )
        if token_id_mapping is None:
            if language_tokenizer_model is None:
                token_id_mapping = IdentityActionTokenIdMapping()
            else:
                token_id_mapping = LanguageVocabularyActionTokenIdMapping(
                    language_tokenizer_model=language_tokenizer_model,
                    num_special_tokens_to_skip=num_special_tokens_to_skip,
                )
        return ActionTokenizer(
            action_discretizer=action_discretizer,
            token_id_mapping=token_id_mapping,
            max_token_len=max_token_len,
            pad_token_id=pad_token_id,
            device=device,
        )

    return factory


class TestActionTokenizerInit:
    def test_stores_action_discretizer(self, action_tokenizer_factory):
        action_discretizer = FastActionDiscretizer(use_pretrained=True)
        tokenizer = action_tokenizer_factory(action_discretizer=action_discretizer)
        assert tokenizer.action_discretizer is action_discretizer

    def test_stores_use_pretrained(self, action_tokenizer_factory):
        tokenizer = action_tokenizer_factory(use_pretrained=True)
        assert tokenizer.action_discretizer.use_pretrained is True

    def test_pretrained_fast_sets_vocab_size(self, action_tokenizer_factory):
        expected_fast_token_count = 2048
        expected_eos_token_id = expected_fast_token_count
        expected_tokenizer_vocab_size = expected_fast_token_count + 1
        tokenizer = action_tokenizer_factory(use_pretrained=True)
        assert tokenizer.action_discretizer.token_count == expected_fast_token_count
        assert tokenizer.eos_token_id == expected_eos_token_id
        assert tokenizer.vocab_size == expected_tokenizer_vocab_size

    def test_custom_fast_sets_vocab_size_1024(self, action_tokenizer_factory):
        expected_fast_token_count = 1024
        tokenizer = action_tokenizer_factory(use_pretrained=False)
        assert tokenizer.action_discretizer.token_count == expected_fast_token_count

    def test_pretrained_fast_is_fitted_on_init(self, action_tokenizer_factory):
        tokenizer = action_tokenizer_factory(use_pretrained=True)
        assert tokenizer._is_fitted is True

    def test_custom_fast_not_fitted_on_init(self, action_tokenizer_factory):
        tokenizer = action_tokenizer_factory(use_pretrained=False)
        assert tokenizer._is_fitted is False
        assert tokenizer.vocab_size is None
        assert tokenizer.eos_token_id is None

    @pytest.mark.parametrize("max_token_len", [64, 128, 256, 512])
    def test_stores_max_token_len(self, action_tokenizer_factory, max_token_len):
        tokenizer = action_tokenizer_factory(max_token_len=max_token_len)
        assert tokenizer.max_token_len == max_token_len

    @pytest.mark.parametrize("pad_token_id", [0, 1, 3])
    def test_stores_pad_token_id(self, action_tokenizer_factory, pad_token_id):
        tokenizer = action_tokenizer_factory(pad_token_id=pad_token_id)
        assert tokenizer.pad_token_id == pad_token_id

    def test_default_device_is_cpu(self, action_tokenizer_factory):
        tokenizer = action_tokenizer_factory()
        assert tokenizer.device == torch.device("cpu")

    def test_stores_explicit_device(self, action_tokenizer_factory, device):
        tokenizer = action_tokenizer_factory(device=device)
        assert tokenizer.device == device


class TestActionTokenizerBuildTokenizers:
    def test_loads_fast_processor(self, mock_auto_processor):
        ActionTokenizer(action_discretizer=FastActionDiscretizer(use_pretrained=True))
        mock_auto_processor.assert_called_once_with("physical-intelligence/fast")

    def test_custom_fast_model_name(self, mock_auto_processor):
        ActionTokenizer(
            action_discretizer=FastActionDiscretizer(
                use_pretrained=True,
                tokenizer_model="custom/model",
            ),
        )
        mock_auto_processor.assert_called_once_with("custom/model")

    @patch("versatil.data.tokenization.action_token_id_mapping.AutoTokenizer")
    def test_language_mapping_loads_language_tokenizer(
        self, mock_auto_tokenizer, action_tokenizer_factory
    ):
        mock_lang_tok = MagicMock()
        mock_lang_tok.vocab_size = 32000
        mock_lang_tok.pad_token = "[PAD]"
        mock_auto_tokenizer.from_pretrained.return_value = mock_lang_tok
        tokenizer = action_tokenizer_factory(
            language_tokenizer_model="some-model",
        )
        mock_auto_tokenizer.from_pretrained.assert_called_once_with("some-model")
        assert tokenizer.vocab_size == 32001

    @patch("versatil.data.tokenization.action_token_id_mapping.AutoTokenizer")
    def test_small_language_vocab_raises(
        self, mock_auto_tokenizer, mock_auto_processor
    ):
        mock_lang_tok = MagicMock()
        mock_lang_tok.vocab_size = 100
        mock_lang_tok.pad_token = "[PAD]"
        mock_auto_tokenizer.from_pretrained.return_value = mock_lang_tok
        expected_message = (
            "Language tokenizer token count (100) is too small to hold action "
            "tokens (2048) plus skipped special tokens (128). Required: 2176"
        )
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            ActionTokenizer(
                action_discretizer=FastActionDiscretizer(use_pretrained=True),
                token_id_mapping=LanguageVocabularyActionTokenIdMapping(
                    language_tokenizer_model="tiny-model"
                ),
            )

    @patch("versatil.data.tokenization.action_token_id_mapping.AutoTokenizer")
    def test_sets_pad_token_from_eos_when_none(
        self, mock_auto_tokenizer, mock_auto_processor
    ):
        mock_lang_tok = MagicMock()
        mock_lang_tok.vocab_size = 32000
        mock_lang_tok.pad_token = None
        mock_lang_tok.eos_token = "<eos>"
        mock_auto_tokenizer.from_pretrained.return_value = mock_lang_tok
        ActionTokenizer(
            action_discretizer=FastActionDiscretizer(use_pretrained=True),
            token_id_mapping=LanguageVocabularyActionTokenIdMapping(
                language_tokenizer_model="some-model"
            ),
        )
        assert mock_lang_tok.pad_token == "<eos>"


class TestActionTokenizerFit:
    def test_fit_raises_when_pretrained(
        self, action_tokenizer_factory, action_chunk_factory
    ):
        tokenizer = action_tokenizer_factory(use_pretrained=True)
        data = action_chunk_factory(batch_size=10)
        expected_message = (
            "Cannot fit a pretrained FAST action discretizer. "
            "Set use_pretrained=False to fit FAST on local data."
        )
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            tokenizer.fit(data)

    def test_fit_calls_processor_fit(
        self, action_tokenizer_factory, action_chunk_factory
    ):
        tokenizer = action_tokenizer_factory(use_pretrained=False)
        tokenizer.action_discretizer.processor.fit.return_value = (
            tokenizer.action_discretizer.processor
        )
        data = action_chunk_factory(batch_size=10)
        tokenizer.fit(data)
        tokenizer.action_discretizer.processor.fit.assert_called_once()
        assert tokenizer._is_fitted is True

    def test_fit_sets_vocab_size_without_language_tokenizer(
        self, action_tokenizer_factory, action_chunk_factory
    ):
        tokenizer = action_tokenizer_factory(use_pretrained=False)
        tokenizer.action_discretizer.processor.fit.return_value = (
            tokenizer.action_discretizer.processor
        )
        data = action_chunk_factory(batch_size=10)
        tokenizer.fit(data)
        assert tokenizer.eos_token_id == 1024
        assert tokenizer.vocab_size == 1025

    def test_fit_raises_when_processor_none(
        self, action_tokenizer_factory, action_chunk_factory
    ):
        tokenizer = action_tokenizer_factory(use_pretrained=False)
        tokenizer.action_discretizer.processor = None
        data = action_chunk_factory(batch_size=10)
        with pytest.raises(
            RuntimeError,
            match=re.escape("FAST processor not initialized"),
        ):
            tokenizer.fit(data)

    def test_fit_logs_info(self, action_tokenizer_factory, action_chunk_factory):
        tokenizer = action_tokenizer_factory(use_pretrained=False)
        tokenizer.action_discretizer.processor.fit.return_value = (
            tokenizer.action_discretizer.processor
        )
        data = action_chunk_factory(batch_size=10)
        with patch(
            "versatil.data.tokenization.action_tokenizer.logging"
        ) as mock_logging:
            tokenizer.fit(data)
            assert mock_logging.info.call_count == 2
            first_log = str(mock_logging.info.call_args_list[0])
            assert "10 chunks" in first_log
            second_log = str(mock_logging.info.call_args_list[1])
            assert "Fitted action tokenizer" in second_log


class TestLanguageVocabularyActionTokenIdMapping:
    @patch("versatil.data.tokenization.action_token_id_mapping.AutoTokenizer")
    def test_mapping_formula(self, mock_auto_tokenizer, action_tokenizer_factory):
        mock_lang_tok = MagicMock()
        mock_lang_tok.vocab_size = 32000
        mock_lang_tok.pad_token = "[PAD]"
        mock_auto_tokenizer.from_pretrained.return_value = mock_lang_tok
        token_id_mapping = LanguageVocabularyActionTokenIdMapping(
            language_tokenizer_model="model",
            num_special_tokens_to_skip=128,
        )
        local_tokens = np.array([0, 1, 2])
        mapped = token_id_mapping.encode(local_tokens)
        expected = np.array(
            [
                32000 - 1 - 128 - 0,
                32000 - 1 - 128 - 1,
                32000 - 1 - 128 - 2,
            ]
        )
        np.testing.assert_array_equal(mapped, expected)

    @patch("versatil.data.tokenization.action_token_id_mapping.AutoTokenizer")
    def test_mapping_accepts_list_input(
        self, mock_auto_tokenizer, action_tokenizer_factory
    ):
        mock_lang_tok = MagicMock()
        mock_lang_tok.vocab_size = 32000
        mock_lang_tok.pad_token = "[PAD]"
        mock_auto_tokenizer.from_pretrained.return_value = mock_lang_tok
        token_id_mapping = LanguageVocabularyActionTokenIdMapping(
            language_tokenizer_model="model",
        )
        mapped = token_id_mapping.encode([0, 1, 2])
        expected = np.array([31871, 31870, 31869])
        np.testing.assert_array_equal(mapped, expected)

    @patch("versatil.data.tokenization.action_token_id_mapping.AutoTokenizer")
    def test_roundtrip_map_unmap(self, mock_auto_tokenizer, action_tokenizer_factory):
        mock_lang_tok = MagicMock()
        mock_lang_tok.vocab_size = 32000
        mock_lang_tok.pad_token = "[PAD]"
        mock_auto_tokenizer.from_pretrained.return_value = mock_lang_tok
        token_id_mapping = LanguageVocabularyActionTokenIdMapping(
            language_tokenizer_model="model",
            num_special_tokens_to_skip=128,
        )
        original = np.array([0, 5, 100, 2047])
        mapped = token_id_mapping.encode(original)
        unmapped = token_id_mapping.decode(mapped)
        np.testing.assert_array_equal(unmapped, original)

    @patch("versatil.data.tokenization.action_token_id_mapping.AutoTokenizer")
    def test_unmap_accepts_torch_tensor(
        self, mock_auto_tokenizer, action_tokenizer_factory
    ):
        mock_lang_tok = MagicMock()
        mock_lang_tok.vocab_size = 32000
        mock_lang_tok.pad_token = "[PAD]"
        mock_auto_tokenizer.from_pretrained.return_value = mock_lang_tok
        token_id_mapping = LanguageVocabularyActionTokenIdMapping(
            language_tokenizer_model="model",
        )
        lang_tokens = torch.tensor([31871, 31870, 31869])
        local_tokens = token_id_mapping.decode(lang_tokens)
        expected = np.array([0, 1, 2])
        np.testing.assert_array_equal(local_tokens, expected)

    @patch("versatil.data.tokenization.action_token_id_mapping.AutoTokenizer")
    def test_load_state_dict_restores_skip_count(self, mock_auto_tokenizer):
        mock_lang_tok = MagicMock()
        mock_lang_tok.vocab_size = 32000
        mock_lang_tok.pad_token = "[PAD]"
        mock_auto_tokenizer.from_pretrained.return_value = mock_lang_tok
        token_id_mapping = LanguageVocabularyActionTokenIdMapping(
            language_tokenizer_model="model",
            num_special_tokens_to_skip=128,
        )
        token_id_mapping.load_state_dict(
            {
                "type": "language_vocabulary",
                "language_tokenizer_model": "model",
                "num_special_tokens_to_skip": 256,
            }
        )
        assert token_id_mapping.num_special_tokens_to_skip == 256


class TestActionTokenizerEncodeChunk:
    def test_encode_chunk_raises_when_not_fitted(
        self, action_tokenizer_factory, action_chunk_factory
    ):
        tokenizer = action_tokenizer_factory(use_pretrained=False)
        chunk = action_chunk_factory()
        with pytest.raises(
            RuntimeError,
            match=re.escape("Tokenizer must be fitted or loaded before encoding"),
        ):
            tokenizer.encode_chunk(chunk)

    def test_encode_chunk_returns_correct_keys(
        self, action_tokenizer_factory, action_chunk_factory
    ):
        tokenizer = action_tokenizer_factory(max_token_len=8)
        tokenizer.action_discretizer.processor.side_effect = lambda x: [[10, 20, 30]]
        chunk = action_chunk_factory()
        result = tokenizer.encode_chunk(chunk)
        expected_tokens = torch.tensor([10, 20, 30, tokenizer.eos_token_id, 0, 0, 0, 0])
        expected_mask = torch.tensor(
            [False, False, False, False, True, True, True, True]
        )
        assert torch.equal(result[SampleKey.TOKENIZED_ACTIONS.value], expected_tokens)
        assert torch.equal(result[SampleKey.IS_PAD_ACTION.value], expected_mask)

    @pytest.mark.parametrize(
        "max_token_len, num_tokens",
        [(8, 3), (16, 5), (32, 1)],
    )
    def test_encode_chunk_pads_to_max_token_len(
        self, action_tokenizer_factory, action_chunk_factory, max_token_len, num_tokens
    ):
        mock_tokens = list(range(10, 10 + num_tokens))
        tokenizer = action_tokenizer_factory(
            max_token_len=max_token_len, pad_token_id=0
        )
        tokenizer.action_discretizer.processor.side_effect = lambda x: [mock_tokens]
        chunk = action_chunk_factory()
        result = tokenizer.encode_chunk(chunk)
        tokens = result[SampleKey.TOKENIZED_ACTIONS.value]
        is_pad = result[SampleKey.IS_PAD_ACTION.value]
        assert tokens.shape == (max_token_len,)
        assert is_pad.shape == (max_token_len,)
        # EOS is appended after action tokens and is NOT marked as pad
        sequence_len = num_tokens + 1  # action tokens + EOS
        assert is_pad[:sequence_len].all() == False  # noqa: E712
        assert is_pad[sequence_len:].all() == True  # noqa: E712
        assert tokens[num_tokens].item() == tokenizer.eos_token_id

    def test_encode_chunk_with_pad_mask_filters_valid_actions(
        self, action_tokenizer_factory, action_chunk_factory, pad_mask_factory
    ):
        tokenizer = action_tokenizer_factory(max_token_len=8)
        tokenizer.action_discretizer.processor.side_effect = lambda x: [[10, 20]]
        chunk = action_chunk_factory()
        pad_mask = pad_mask_factory(total=5, num_valid=2)
        tokenizer.encode_chunk(chunk, is_pad_mask=pad_mask)
        called_data = tokenizer.action_discretizer.processor.call_args[0][0]
        assert called_data.shape[0] == 2

    def test_encode_chunk_with_torch_pad_mask(
        self, action_tokenizer_factory, action_chunk_factory, pad_mask_factory
    ):
        tokenizer = action_tokenizer_factory(max_token_len=8)
        tokenizer.action_discretizer.processor.side_effect = lambda x: [[10, 20]]
        chunk = action_chunk_factory(as_torch=True)
        pad_mask = pad_mask_factory(total=5, num_valid=2, as_torch=True)
        result = tokenizer.encode_chunk(chunk, is_pad_mask=pad_mask)
        called_data = tokenizer.action_discretizer.processor.call_args.args[0]
        np.testing.assert_array_equal(called_data, chunk[:2].numpy())
        expected_tokens = torch.tensor([10, 20, tokenizer.eos_token_id, 0, 0, 0, 0, 0])
        expected_mask = torch.tensor(
            [False, False, False, True, True, True, True, True]
        )
        assert torch.equal(result[SampleKey.TOKENIZED_ACTIONS.value], expected_tokens)
        assert torch.equal(result[SampleKey.IS_PAD_ACTION.value], expected_mask)

    def test_encode_chunk_truncates_when_exceeding_max_len(
        self, action_tokenizer_factory, action_chunk_factory
    ):
        long_tokens = list(range(20))
        tokenizer = action_tokenizer_factory(max_token_len=8)
        tokenizer.action_discretizer.processor.side_effect = lambda x: [long_tokens]
        chunk = action_chunk_factory()
        result = tokenizer.encode_chunk(chunk)
        tokens = result[SampleKey.TOKENIZED_ACTIONS.value]
        assert tokens.shape == (8,)
        is_pad = result[SampleKey.IS_PAD_ACTION.value]
        assert not is_pad.any()
        # Truncated to max_token_len - 1 action tokens, then EOS appended
        assert tokens[-1].item() == tokenizer.eos_token_id
        assert tokens[0].item() == 0  # first token from range(20)

    def test_encode_chunk_truncation_logs_warning(
        self, action_tokenizer_factory, action_chunk_factory
    ):
        long_tokens = list(range(20))
        tokenizer = action_tokenizer_factory(max_token_len=8)
        tokenizer.action_discretizer.processor.side_effect = lambda x: [long_tokens]
        chunk = action_chunk_factory()
        with patch(
            "versatil.data.tokenization.action_tokenizer.logging"
        ) as mock_logging:
            tokenizer.encode_chunk(chunk)
            mock_logging.warning.assert_called_once()
            assert "truncating" in str(mock_logging.warning.call_args).lower()

    def test_encode_chunk_fits_exactly_with_eos_no_warning(
        self, action_tokenizer_factory, action_chunk_factory
    ):
        # 4 action tokens + 1 EOS = 5 = max_token_len, no truncation needed
        tokenizer = action_tokenizer_factory(max_token_len=5)
        tokenizer.action_discretizer.processor.side_effect = lambda x: [
            [10, 20, 30, 40]
        ]
        chunk = action_chunk_factory()
        with patch(
            "versatil.data.tokenization.action_tokenizer.logging"
        ) as mock_logging:
            result = tokenizer.encode_chunk(chunk)
            mock_logging.warning.assert_not_called()
        tokens = result[SampleKey.TOKENIZED_ACTIONS.value]
        is_pad = result[SampleKey.IS_PAD_ACTION.value]
        assert tokens.shape == (5,)
        assert not is_pad.any()
        assert tokens[-1].item() == tokenizer.eos_token_id

    def test_encode_chunk_raises_without_fast_processor(self, action_tokenizer_factory):
        tokenizer = action_tokenizer_factory()
        tokenizer.action_discretizer.processor = None
        with pytest.raises(
            RuntimeError,
            match=re.escape("FAST processor not initialized"),
        ):
            tokenizer.encode_chunk(np.zeros((5, 7), dtype=np.float32))


class TestActionTokenizerEncodeBatch:
    def test_encode_batch_raises_when_not_fitted(
        self, action_tokenizer_factory, action_chunk_factory
    ):
        tokenizer = action_tokenizer_factory(use_pretrained=False)
        batch = action_chunk_factory(batch_size=3)
        with pytest.raises(
            RuntimeError,
            match=re.escape("Tokenizer must be fitted or loaded before encoding"),
        ):
            tokenizer.encode_batch(batch)

    @pytest.mark.parametrize(
        "batch_size, max_token_len",
        [(2, 8), (3, 16), (5, 32)],
    )
    def test_encode_batch_returns_stacked_results(
        self, action_tokenizer_factory, action_chunk_factory, batch_size, max_token_len
    ):
        tokenizer = action_tokenizer_factory(max_token_len=max_token_len)
        tokenizer.action_discretizer.processor.side_effect = lambda x: [[10, 20]]
        batch = action_chunk_factory(batch_size=batch_size)
        result = tokenizer.encode_batch(batch)
        assert result[SampleKey.TOKENIZED_ACTIONS.value].shape == (
            batch_size,
            max_token_len,
        )
        assert result[SampleKey.IS_PAD_ACTION.value].shape == (
            batch_size,
            max_token_len,
        )

    def test_encode_batch_passes_per_sample_pad_mask(
        self, action_tokenizer_factory, action_chunk_factory
    ):
        tokenizer = action_tokenizer_factory(max_token_len=8)
        tokenizer.action_discretizer.processor.side_effect = lambda x: [[10, 20]]
        batch = action_chunk_factory(batch_size=2)
        pad_mask = np.array(
            [
                [False, False, True, True, True],
                [False, False, False, True, True],
            ]
        )
        result = tokenizer.encode_batch(batch, is_pad_mask=pad_mask)
        first_call = tokenizer.action_discretizer.processor.call_args_list[0].args[0]
        second_call = tokenizer.action_discretizer.processor.call_args_list[1].args[0]
        np.testing.assert_array_equal(first_call, batch[0, :2])
        np.testing.assert_array_equal(second_call, batch[1, :3])
        assert result[SampleKey.TOKENIZED_ACTIONS.value].shape == (2, 8)


class TestActionTokenizerEncode:
    def test_encode_2d_dispatches_to_encode_chunk(
        self, action_tokenizer_factory, action_chunk_factory
    ):
        tokenizer = action_tokenizer_factory(max_token_len=8)
        tokenizer.action_discretizer.processor.side_effect = lambda x: [[10, 20]]
        chunk = action_chunk_factory()
        result = tokenizer.encode(chunk)
        assert result[SampleKey.TOKENIZED_ACTIONS.value].ndim == 1

    def test_encode_3d_dispatches_to_encode_batch(
        self, action_tokenizer_factory, action_chunk_factory
    ):
        tokenizer = action_tokenizer_factory(max_token_len=8)
        tokenizer.action_discretizer.processor.side_effect = lambda x: [[10, 20]]
        batch = action_chunk_factory(batch_size=3)
        result = tokenizer.encode(batch)
        assert result[SampleKey.TOKENIZED_ACTIONS.value].shape[0] == 3

    def test_encode_3d_passes_pad_mask_to_encode_batch(
        self, action_tokenizer_factory, action_chunk_factory
    ):
        tokenizer = action_tokenizer_factory(max_token_len=8)
        tokenizer.action_discretizer.processor.side_effect = lambda x: [[10, 20]]
        batch = action_chunk_factory(batch_size=2)
        pad_mask = np.array(
            [
                [False, False, True, True, True],
                [False, False, False, True, True],
            ]
        )
        result = tokenizer.encode(batch, is_pad_mask=pad_mask)
        assert result[SampleKey.TOKENIZED_ACTIONS.value].shape[0] == 2

    def test_encode_invalid_ndim_raises(self, action_tokenizer_factory, rng):
        tokenizer = action_tokenizer_factory()
        data_1d = rng.standard_normal((7,)).astype(np.float32)
        expected_message = f"Expected 2D or 3D input, got shape {data_1d.shape}"
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            tokenizer.encode(data_1d)


class TestActionTokenizerDecodeChunk:
    def test_decode_chunk_raises_when_not_fitted(self, action_tokenizer_factory):
        tokenizer = action_tokenizer_factory(use_pretrained=False)
        with pytest.raises(
            RuntimeError,
            match=re.escape("Tokenizer must be fitted or loaded before decoding"),
        ):
            tokenizer.decode_chunk(torch.tensor([1, 2, 3]))

    def test_decode_chunk_strips_pad_tokens(self, action_tokenizer_factory):
        tokenizer = action_tokenizer_factory(pad_token_id=0)
        decoded_array = np.zeros((1, 5, 7), dtype=np.float32)
        tokenizer.action_discretizer.processor.decode.return_value = decoded_array
        tokens = torch.tensor([10, 20, 30, 0, 0, 0])
        tokenizer.decode_chunk(tokens)
        call_args = tokenizer.action_discretizer.processor.decode.call_args[0][0]
        assert call_args == [[10, 20, 30]]

    def test_decode_chunk_strips_eos_token(self, action_tokenizer_factory):
        tokenizer = action_tokenizer_factory(pad_token_id=0)
        eos_id = tokenizer.eos_token_id
        decoded_array = np.zeros((1, 5, 7), dtype=np.float32)
        tokenizer.action_discretizer.processor.decode.return_value = decoded_array
        tokens = torch.tensor([10, 20, 30, eos_id, 0, 0])
        tokenizer.decode_chunk(tokens)
        call_args = tokenizer.action_discretizer.processor.decode.call_args[0][0]
        assert call_args == [[10, 20, 30]]

    def test_decode_chunk_preserves_valid_zero_tokens_before_eos(
        self, action_tokenizer_factory
    ):
        tokenizer = action_tokenizer_factory(pad_token_id=0)
        eos_id = tokenizer.eos_token_id
        decoded_array = np.zeros((1, 5, 7), dtype=np.float32)
        tokenizer.action_discretizer.processor.decode.return_value = decoded_array
        tokens = torch.tensor([0, 10, 0, 20, eos_id, 0])
        tokenizer.decode_chunk(tokens)
        call_args = tokenizer.action_discretizer.processor.decode.call_args[0][0]
        assert call_args == [[0, 10, 0, 20]]

    def test_decode_chunk_only_strips_trailing_pad_tokens_without_eos(
        self, action_tokenizer_factory
    ):
        tokenizer = action_tokenizer_factory(pad_token_id=0)
        decoded_array = np.zeros((1, 5, 7), dtype=np.float32)
        tokenizer.action_discretizer.processor.decode.return_value = decoded_array
        tokens = torch.tensor([0, 10, 0, 20, 0, 0])
        tokenizer.decode_chunk(tokens)
        call_args = tokenizer.action_discretizer.processor.decode.call_args[0][0]
        assert call_args == [[0, 10, 0, 20]]

    def test_decode_chunk_raises_without_fast_processor(self, action_tokenizer_factory):
        tokenizer = action_tokenizer_factory()
        tokenizer.action_discretizer.processor = None
        with pytest.raises(
            RuntimeError,
            match=re.escape("FAST processor not initialized"),
        ):
            tokenizer.decode_chunk(torch.tensor([1, 2, 3]))

    def test_decode_chunk_accepts_list_input(self, action_tokenizer_factory):
        tokenizer = action_tokenizer_factory(pad_token_id=0)
        tokenizer.action_discretizer.processor.decode.return_value = np.zeros(
            (1, 5, 7), dtype=np.float32
        )
        result = tokenizer.decode_chunk([10, 20, 30])
        np.testing.assert_array_equal(result, np.zeros((5, 7), dtype=np.float32))

    def test_decode_chunk_accepts_numpy_input(self, action_tokenizer_factory):
        tokenizer = action_tokenizer_factory(pad_token_id=0)
        tokenizer.action_discretizer.processor.decode.return_value = np.zeros(
            (1, 5, 7), dtype=np.float32
        )
        result = tokenizer.decode_chunk(np.array([10, 20, 30]))
        np.testing.assert_array_equal(result, np.zeros((5, 7), dtype=np.float32))

    def test_decode_chunk_raises_type_error_when_processor_returns_non_ndarray(
        self, action_tokenizer_factory
    ):
        tokenizer = action_tokenizer_factory(pad_token_id=0)
        tokenizer.action_discretizer.processor.decode.return_value = [[0.1, 0.2]]
        expected_message = (
            "Expected np.ndarray from FAST processor decode, got <class 'list'>"
        )
        with pytest.raises(TypeError, match=re.escape(expected_message)):
            tokenizer.decode_chunk(torch.tensor([10, 20, 30]))


class TestActionTokenizerDecodeBatch:
    def test_decode_batch_raises_when_not_fitted(self, action_tokenizer_factory):
        tokenizer = action_tokenizer_factory(use_pretrained=False)
        tokens = torch.tensor([[10, 20, 30], [40, 50, 60]])
        with pytest.raises(
            RuntimeError,
            match=re.escape("Tokenizer must be fitted or loaded before decoding"),
        ):
            tokenizer.decode_batch(tokens)

    def test_decode_batch_strips_pad_per_sample(self, action_tokenizer_factory):
        tokenizer = action_tokenizer_factory(pad_token_id=0)
        tokenizer.action_discretizer.processor.decode.return_value = np.zeros(
            (2, 5, 7), dtype=np.float32
        )
        tokens = torch.tensor([[10, 20, 0], [30, 40, 50]])
        tokenizer.decode_batch(tokens)
        call_args = tokenizer.action_discretizer.processor.decode.call_args[0][0]
        assert call_args[0] == [10, 20]
        assert call_args[1] == [30, 40, 50]

    def test_decode_batch_strips_eos_per_sample(self, action_tokenizer_factory):
        tokenizer = action_tokenizer_factory(pad_token_id=0)
        eos_id = tokenizer.eos_token_id
        tokenizer.action_discretizer.processor.decode.return_value = np.zeros(
            (2, 5, 7), dtype=np.float32
        )
        tokens = torch.tensor([[10, 20, eos_id, 0], [30, eos_id, 0, 0]])
        tokenizer.decode_batch(tokens)
        call_args = tokenizer.action_discretizer.processor.decode.call_args[0][0]
        assert call_args[0] == [10, 20]
        assert call_args[1] == [30]

    def test_decode_batch_preserves_valid_zero_tokens(self, action_tokenizer_factory):
        tokenizer = action_tokenizer_factory(pad_token_id=0)
        eos_id = tokenizer.eos_token_id
        tokenizer.action_discretizer.processor.decode.return_value = np.zeros(
            (2, 5, 7), dtype=np.float32
        )
        tokens = torch.tensor([[0, 20, eos_id, 0], [30, 0, 40, 0]])
        tokenizer.decode_batch(tokens)
        call_args = tokenizer.action_discretizer.processor.decode.call_args[0][0]
        assert call_args[0] == [0, 20]
        assert call_args[1] == [30, 0, 40]

    def test_decode_batch_raises_without_fast_processor(self, action_tokenizer_factory):
        tokenizer = action_tokenizer_factory()
        tokenizer.action_discretizer.processor = None
        tokens = torch.tensor([[10, 20, 30]])
        with pytest.raises(
            RuntimeError,
            match=re.escape("FAST processor not initialized"),
        ):
            tokenizer.decode_batch(tokens)

    def test_decode_batch_accepts_numpy_input(self, action_tokenizer_factory):
        tokenizer = action_tokenizer_factory(pad_token_id=0)
        tokenizer.action_discretizer.processor.decode.return_value = np.zeros(
            (2, 5, 7), dtype=np.float32
        )
        tokens = np.array([[10, 20, 30], [40, 50, 60]])
        result = tokenizer.decode_batch(tokens)
        assert result.shape == (2, 5, 7)

    def test_decode_batch_raises_type_error_when_processor_returns_non_ndarray(
        self, action_tokenizer_factory
    ):
        tokenizer = action_tokenizer_factory(pad_token_id=0)
        tokenizer.action_discretizer.processor.decode.return_value = [[0.1, 0.2]]
        tokens = torch.tensor([[10, 20, 30], [40, 50, 60]])
        expected_message = (
            "Expected np.ndarray from FAST processor decode, got <class 'list'>"
        )
        with pytest.raises(TypeError, match=re.escape(expected_message)):
            tokenizer.decode_batch(tokens)


class TestActionTokenizerDecode:
    def test_decode_1d_dispatches_to_decode_chunk(self, action_tokenizer_factory):
        tokenizer = action_tokenizer_factory(pad_token_id=0)
        tokenizer.action_discretizer.processor.decode.return_value = np.zeros(
            (1, 5, 7), dtype=np.float32
        )
        result = tokenizer.decode(torch.tensor([10, 20, 30]))
        assert result.shape == (5, 7)

    def test_decode_2d_dispatches_to_decode_batch(self, action_tokenizer_factory):
        tokenizer = action_tokenizer_factory(pad_token_id=0)
        tokenizer.action_discretizer.processor.decode.return_value = np.zeros(
            (2, 5, 7), dtype=np.float32
        )
        tokens = torch.tensor([[10, 20, 30], [40, 50, 60]])
        result = tokenizer.decode(tokens)
        assert result.shape == (2, 5, 7)

    def test_decode_list_input_dispatches_to_decode_chunk(
        self, action_tokenizer_factory
    ):
        tokenizer = action_tokenizer_factory(pad_token_id=0)
        tokenizer.action_discretizer.processor.decode.return_value = np.zeros(
            (1, 5, 7), dtype=np.float32
        )
        result = tokenizer.decode([10, 20, 30])
        assert result.shape == (5, 7)

    def test_decode_invalid_ndim_raises(self, action_tokenizer_factory):
        tokenizer = action_tokenizer_factory()
        tokens_3d = torch.zeros((2, 3, 4), dtype=torch.long)
        expected_message = (
            f"Expected 1D or 2D input, got shape {tuple(tokens_3d.shape)}"
        )
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            tokenizer.decode(tokens_3d)


class TestActionTokenizerBinnedDiscretizer:
    def test_fit_encode_decode_binned_actions(self, action_chunk_factory):
        tokenizer = ActionTokenizer(
            action_discretizer=BinnedActionDiscretizer(num_bins=16),
            max_token_len=64,
        )
        training_data = action_chunk_factory(batch_size=20, scale=0.5)
        tokenizer.fit(training_data)

        result = tokenizer.encode_chunk(training_data[0])
        tokens = result[SampleKey.TOKENIZED_ACTIONS.value]
        decoded = tokenizer.decode_chunk(tokens)

        assert tokenizer.vocab_size == 17
        assert tokens.shape == (64,)
        assert decoded.shape == training_data[0].shape
        assert np.isfinite(decoded).all()

    def test_binned_decode_pads_short_generated_sequences(self, action_chunk_factory):
        tokenizer = ActionTokenizer(
            action_discretizer=BinnedActionDiscretizer(num_bins=8),
            max_token_len=64,
        )
        training_data = action_chunk_factory(
            batch_size=20, time_horizon=4, action_dimension=3, scale=0.5
        )
        tokenizer.fit(training_data)

        decoded = tokenizer.decode_chunk(torch.tensor([1, 2, tokenizer.eos_token_id]))

        assert decoded.shape == (4, 3)
        assert np.isfinite(decoded).all()

    @patch("versatil.data.tokenization.action_token_id_mapping.AutoTokenizer")
    def test_binned_actions_can_use_language_vocabulary_mapping(
        self, mock_auto_tokenizer, action_chunk_factory
    ):
        mock_language_tokenizer = MagicMock()
        mock_language_tokenizer.vocab_size = 32000
        mock_language_tokenizer.pad_token = "[PAD]"
        mock_auto_tokenizer.from_pretrained.return_value = mock_language_tokenizer
        tokenizer = ActionTokenizer(
            action_discretizer=BinnedActionDiscretizer(num_bins=8),
            token_id_mapping=LanguageVocabularyActionTokenIdMapping(
                language_tokenizer_model="test-language-model",
                num_special_tokens_to_skip=128,
            ),
            max_token_len=64,
        )
        training_data = action_chunk_factory(
            batch_size=20,
            time_horizon=4,
            action_dimension=3,
            scale=0.5,
        )
        tokenizer.fit(training_data)

        result = tokenizer.encode_chunk(training_data[0])
        tokens = result[SampleKey.TOKENIZED_ACTIONS.value]
        non_pad = tokens[~result[SampleKey.IS_PAD_ACTION.value]]
        non_eos = non_pad[non_pad != tokenizer.eos_token_id]
        decoded = tokenizer.decode_chunk(tokens)

        assert tokenizer.vocab_size == 32001
        assert tokenizer.eos_token_id == 32000
        assert non_eos.min() >= 31864
        assert non_eos.max() <= 31871
        assert decoded.shape == training_data[0].shape
        assert np.isfinite(decoded).all()


class TestActionTokenizerTo:
    def test_to_updates_device(self, action_tokenizer_factory, device):
        tokenizer = action_tokenizer_factory()
        tokenizer.to(device)
        assert tokenizer.device == device

    def test_to_returns_self(self, action_tokenizer_factory, device):
        tokenizer = action_tokenizer_factory()
        result = tokenizer.to(device)
        assert result is tokenizer


class TestActionTokenizerStateDict:
    def test_state_dict_keys(self, action_tokenizer_factory):
        tokenizer = action_tokenizer_factory()
        state = tokenizer.state_dict()
        expected_keys = {
            "action_discretizer",
            "token_id_mapping",
            "max_token_len",
            "pad_token_id",
            "vocab_size",
            "eos_token_id",
            "is_fitted",
        }
        assert set(state.keys()) == expected_keys

    @pytest.mark.parametrize(
        "use_pretrained, expected_fast_token_count",
        [
            (True, 2048),
        ],
    )
    def test_state_dict_values(
        self,
        action_tokenizer_factory,
        use_pretrained,
        expected_fast_token_count,
    ):
        tokenizer = action_tokenizer_factory(
            use_pretrained=use_pretrained,
        )
        state = tokenizer.state_dict()
        assert state["action_discretizer"]["type"] == "fast"
        assert state["action_discretizer"]["use_pretrained"] is use_pretrained
        assert state["action_discretizer"]["token_count"] == expected_fast_token_count
        assert state["token_id_mapping"]["type"] == "identity"
        assert state["is_fitted"] is True
        assert state["eos_token_id"] == expected_fast_token_count
        assert state["vocab_size"] == expected_fast_token_count + 1

    def test_state_dict_values_for_binned_discretizer(self, action_chunk_factory):
        tokenizer = ActionTokenizer(
            action_discretizer=BinnedActionDiscretizer(num_bins=16),
            max_token_len=64,
        )
        training_data = action_chunk_factory(
            batch_size=20,
            time_horizon=4,
            action_dimension=3,
        )
        tokenizer.fit(training_data)
        state = tokenizer.state_dict()
        assert state["action_discretizer"]["type"] == "binned"
        assert state["action_discretizer"]["num_bins"] == 16
        assert state["action_discretizer"]["time_horizon"] == 4
        assert state["action_discretizer"]["action_dim"] == 3
        assert state["token_id_mapping"]["type"] == "identity"
        assert state["eos_token_id"] == 16
        assert state["vocab_size"] == 17


class TestActionTokenizerLoadStateDict:
    def test_load_state_dict_restores_fields(self, action_tokenizer_factory):
        tokenizer = action_tokenizer_factory()
        state = {
            "action_discretizer": {
                "type": "fast",
                "use_pretrained": True,
                "tokenizer_model": "physical-intelligence/fast",
                "token_count": 2048,
                "is_fitted": True,
                "time_horizon": 5,
                "action_dim": 7,
            },
            "token_id_mapping": {"type": "identity"},
            "max_token_len": 64,
            "pad_token_id": 0,
            "vocab_size": 2049,
            "eos_token_id": 2048,
            "is_fitted": True,
        }
        tokenizer.load_state_dict(state)
        assert tokenizer.action_discretizer.use_pretrained is True
        assert tokenizer.vocab_size == 2049
        assert tokenizer.eos_token_id == 2048
        assert tokenizer._is_fitted is True

    def test_load_state_dict_without_eos_token_id_defaults_to_none(
        self, action_tokenizer_factory
    ):
        tokenizer = action_tokenizer_factory()
        state = {
            "action_discretizer": {
                "type": "fast",
                "use_pretrained": True,
                "tokenizer_model": "physical-intelligence/fast",
                "token_count": 2048,
                "is_fitted": True,
            },
            "token_id_mapping": {"type": "identity"},
            "vocab_size": 2048,
            "is_fitted": True,
        }
        tokenizer.load_state_dict(state)
        assert tokenizer.eos_token_id is None

    def test_load_state_dict_restores_binned_discretizer(self, action_chunk_factory):
        original = ActionTokenizer(
            action_discretizer=BinnedActionDiscretizer(num_bins=8),
            max_token_len=64,
        )
        training_data = action_chunk_factory(
            batch_size=20,
            time_horizon=4,
            action_dimension=3,
        )
        original.fit(training_data)
        state = original.state_dict()
        restored = ActionTokenizer(
            action_discretizer=BinnedActionDiscretizer(num_bins=2),
            max_token_len=8,
        )
        restored.load_state_dict(state)
        decoded = restored.decode_chunk(
            original.encode_chunk(training_data[0])[SampleKey.TOKENIZED_ACTIONS.value]
        )
        assert restored.action_discretizer.token_count == 8
        assert restored.action_discretizer.time_horizon == 4
        assert restored.action_discretizer.action_dim == 3
        assert restored.vocab_size == 9
        assert decoded.shape == training_data[0].shape

    @pytest.mark.parametrize(
        ("state_key", "state_value", "expected_message"),
        [
            (
                "action_discretizer",
                {"type": "unknown"},
                "Unsupported action discretizer type: unknown",
            ),
            (
                "token_id_mapping",
                {"type": "unknown"},
                "Unsupported action token-id mapping type: unknown",
            ),
        ],
    )
    def test_from_state_dict_rejects_unsupported_component_types(
        self,
        mock_auto_processor,
        state_key,
        state_value,
        expected_message,
    ):
        del mock_auto_processor
        state = {
            "action_discretizer": {
                "type": "fast",
                "use_pretrained": True,
                "tokenizer_model": "physical-intelligence/fast",
            },
            "token_id_mapping": {"type": "identity"},
            "max_token_len": 64,
            "pad_token_id": 0,
        }
        state[state_key] = state_value
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            ActionTokenizer._from_state_dict(
                state_dict=state,
                device=torch.device("cpu"),
            )


class TestActionTokenizerSavePretrained:
    def test_save_raises_when_not_fitted(self, action_tokenizer_factory, tmp_path):
        tokenizer = action_tokenizer_factory(use_pretrained=False)
        with pytest.raises(
            RuntimeError,
            match=re.escape("Cannot save unfitted tokenizer"),
        ):
            tokenizer.save_pretrained(tmp_path / "tokenizer")

    @patch("versatil.data.tokenization.action_tokenizer.torch.save")
    def test_save_writes_state_dict(
        self, mock_torch_save, action_tokenizer_factory, tmp_path
    ):
        tokenizer = action_tokenizer_factory()
        save_path = tmp_path / "tokenizer"
        tokenizer.save_pretrained(save_path)
        mock_torch_save.assert_called_once()

    def test_save_does_not_save_fast_processor_when_pretrained(
        self, action_tokenizer_factory, tmp_path
    ):
        tokenizer = action_tokenizer_factory(use_pretrained=True)
        save_path = tmp_path / "tokenizer"
        tokenizer.save_pretrained(save_path)
        tokenizer.action_discretizer.processor.save_pretrained.assert_not_called()

    def test_save_saves_fast_processor_when_custom(
        self, action_tokenizer_factory, tmp_path, action_chunk_factory
    ):
        tokenizer = action_tokenizer_factory(use_pretrained=False)
        tokenizer.action_discretizer.processor.fit.return_value = (
            tokenizer.action_discretizer.processor
        )
        data = action_chunk_factory(batch_size=10)
        tokenizer.fit(data)
        save_path = tmp_path / "tokenizer"
        tokenizer.save_pretrained(save_path)
        tokenizer.action_discretizer.processor.save_pretrained.assert_called_once()

    @patch("versatil.data.tokenization.action_token_id_mapping.AutoTokenizer")
    def test_save_saves_language_tokenizer(
        self, mock_auto_tokenizer, action_tokenizer_factory, tmp_path
    ):
        mock_lang_tok = MagicMock()
        mock_lang_tok.vocab_size = 32000
        mock_lang_tok.pad_token = "[PAD]"
        mock_auto_tokenizer.from_pretrained.return_value = mock_lang_tok
        tokenizer = action_tokenizer_factory(
            language_tokenizer_model="model",
        )
        save_path = tmp_path / "tokenizer"
        tokenizer.save_pretrained(save_path)
        mock_lang_tok.save_pretrained.assert_called_once_with(
            save_path / "language_tokenizer"
        )

    def test_save_pretrained_logs_info(self, action_tokenizer_factory, tmp_path):
        tokenizer = action_tokenizer_factory()
        tokenizer.action_discretizer.processor = None
        save_path = tmp_path / "tokenizer"
        with patch(
            "versatil.data.tokenization.action_tokenizer.logging"
        ) as mock_logging:
            tokenizer.save_pretrained(save_path)
            mock_logging.info.assert_called_once()
            assert str(save_path) in str(mock_logging.info.call_args)


class TestActionTokenizerFromPretrained:
    def test_raises_file_not_found(self, mock_auto_processor, tmp_path):
        missing = tmp_path / "missing"
        expected_message = f"Tokenizer path not found: {missing}"
        with pytest.raises(FileNotFoundError, match=re.escape(expected_message)):
            ActionTokenizer.from_pretrained(str(missing))

    @patch("versatil.data.tokenization.action_tokenizer.torch.load")
    def test_loads_state_and_restores_tokenizer(
        self, mock_torch_load, mock_auto_processor, tmp_path
    ):
        save_path = tmp_path / "tokenizer"
        save_path.mkdir(parents=True)
        (save_path / "action_tokenizer_state.pt").touch()
        mock_torch_load.return_value = {
            "action_discretizer": {
                "type": "fast",
                "use_pretrained": True,
                "tokenizer_model": "physical-intelligence/fast",
                "token_count": 2048,
                "is_fitted": True,
            },
            "token_id_mapping": {"type": "identity"},
            "vocab_size": 2049,
            "eos_token_id": 2048,
            "is_fitted": True,
        }
        loaded = ActionTokenizer.from_pretrained(save_path)
        assert loaded.action_discretizer.token_count == 2048
        assert loaded.vocab_size == 2049
        assert loaded._is_fitted is True

    @patch("versatil.data.tokenization.action_tokenizer.torch.load")
    def test_loads_custom_fast_processor_from_disk(
        self, mock_torch_load, mock_auto_processor, tmp_path
    ):
        save_path = tmp_path / "tokenizer"
        save_path.mkdir(parents=True)
        (save_path / "action_tokenizer_state.pt").touch()
        (save_path / "fast_processor").mkdir()
        mock_torch_load.return_value = {
            "action_discretizer": {
                "type": "fast",
                "use_pretrained": False,
                "tokenizer_model": "physical-intelligence/fast",
                "token_count": 1024,
                "is_fitted": True,
            },
            "token_id_mapping": {"type": "identity"},
            "vocab_size": 1025,
            "eos_token_id": 1024,
            "is_fitted": True,
        }
        loaded = ActionTokenizer.from_pretrained(save_path)
        assert mock_auto_processor.call_count == 2
        second_call = mock_auto_processor.call_args_list[1]
        assert second_call[0][0] == str(save_path / "fast_processor")
        assert loaded._is_fitted is True

    @patch("versatil.data.tokenization.action_tokenizer.torch.load")
    def test_loads_binned_discretizer_from_disk(
        self, mock_torch_load, action_chunk_factory, tmp_path
    ):
        original = ActionTokenizer(
            action_discretizer=BinnedActionDiscretizer(num_bins=8),
            max_token_len=64,
        )
        training_data = action_chunk_factory(
            batch_size=20,
            time_horizon=4,
            action_dimension=3,
        )
        original.fit(training_data)
        save_path = tmp_path / "tokenizer"
        save_path.mkdir(parents=True)
        (save_path / "action_tokenizer_state.pt").touch()
        mock_torch_load.return_value = original.state_dict()
        loaded = ActionTokenizer.from_pretrained(save_path)
        original_tokens = original.encode_chunk(training_data[0])[
            SampleKey.TOKENIZED_ACTIONS.value
        ]
        loaded_decoded = loaded.decode_chunk(original_tokens)
        original_decoded = original.decode_chunk(original_tokens)
        assert isinstance(loaded.action_discretizer, BinnedActionDiscretizer)
        assert loaded.action_discretizer.token_count == 8
        np.testing.assert_array_equal(loaded_decoded, original_decoded)

    @patch("versatil.data.tokenization.action_token_id_mapping.AutoTokenizer")
    @patch("versatil.data.tokenization.action_tokenizer.torch.load")
    def test_loads_language_tokenizer_from_disk(
        self, mock_torch_load, mock_auto_tokenizer, mock_auto_processor, tmp_path
    ):
        save_path = tmp_path / "tokenizer"
        save_path.mkdir(parents=True)
        (save_path / "action_tokenizer_state.pt").touch()
        (save_path / "language_tokenizer").mkdir()
        mock_lang_tok = MagicMock()
        mock_lang_tok.vocab_size = 32000
        mock_lang_tok.pad_token = "[PAD]"
        mock_auto_tokenizer.from_pretrained.return_value = mock_lang_tok
        mock_torch_load.return_value = {
            "action_discretizer": {
                "type": "fast",
                "use_pretrained": True,
                "tokenizer_model": "physical-intelligence/fast",
                "token_count": 2048,
                "is_fitted": True,
            },
            "token_id_mapping": {
                "type": "language_vocabulary",
                "language_tokenizer_model": "some-model",
                "num_special_tokens_to_skip": 128,
            },
            "vocab_size": 32001,
            "eos_token_id": 32000,
            "is_fitted": True,
        }
        loaded = ActionTokenizer.from_pretrained(save_path)
        mock_auto_tokenizer.from_pretrained.assert_any_call(
            save_path / "language_tokenizer"
        )
        assert loaded.token_id_mapping.language_tokenizer is not None

    @patch("versatil.data.tokenization.action_tokenizer.torch.load")
    def test_from_pretrained_logs_info(
        self, mock_torch_load, mock_auto_processor, tmp_path
    ):
        save_path = tmp_path / "tokenizer"
        save_path.mkdir(parents=True)
        (save_path / "action_tokenizer_state.pt").touch()
        mock_torch_load.return_value = {
            "action_discretizer": {
                "type": "fast",
                "use_pretrained": True,
                "tokenizer_model": "physical-intelligence/fast",
                "token_count": 2048,
                "is_fitted": True,
            },
            "token_id_mapping": {"type": "identity"},
            "vocab_size": 2049,
            "eos_token_id": 2048,
            "is_fitted": True,
        }
        with patch(
            "versatil.data.tokenization.action_tokenizer.logging"
        ) as mock_logging:
            ActionTokenizer.from_pretrained(save_path)
            mock_logging.info.assert_called()
            assert str(save_path) in str(mock_logging.info.call_args)


@pytest.mark.integration
class TestActionTokenizerIntegrationPretrainedFast:
    def test_encode_decode_roundtrip(self, action_chunk_factory, device):
        tokenizer = ActionTokenizer(
            action_discretizer=FastActionDiscretizer(use_pretrained=True),
            device=device,
        )
        action_chunks = action_chunk_factory(batch_size=3, scale=0.5)
        result = tokenizer.encode(action_chunks)
        tokens = result[SampleKey.TOKENIZED_ACTIONS.value]
        assert tokens.dtype == torch.long
        assert tokens.device.type == device.type
        decoded = tokenizer.decode(tokens)
        assert decoded.shape == action_chunks.shape
        np.testing.assert_allclose(decoded, action_chunks, atol=0.2)

    def test_encode_single_chunk(self, action_chunk_factory, device):
        tokenizer = ActionTokenizer(
            action_discretizer=FastActionDiscretizer(use_pretrained=True),
            max_token_len=128,
            device=device,
        )
        chunk = action_chunk_factory(scale=0.5)
        result = tokenizer.encode_chunk(chunk)
        assert result[SampleKey.TOKENIZED_ACTIONS.value].shape == (128,)
        assert result[SampleKey.IS_PAD_ACTION.value].shape == (128,)

    def test_encode_with_pad_mask(self, action_chunk_factory, pad_mask_factory, device):
        tokenizer = ActionTokenizer(
            action_discretizer=FastActionDiscretizer(use_pretrained=True),
            device=device,
        )
        chunk = action_chunk_factory(scale=0.5)
        pad_mask = pad_mask_factory(total=5, num_valid=3, as_torch=True)
        result = tokenizer.encode_chunk(chunk, is_pad_mask=pad_mask)
        assert SampleKey.TOKENIZED_ACTIONS.value in result

    def test_encode_torch_tensor_input(self, action_chunk_factory, device):
        tokenizer = ActionTokenizer(
            action_discretizer=FastActionDiscretizer(use_pretrained=True),
            device=device,
        )
        chunk = action_chunk_factory(as_torch=True).to(device)
        result = tokenizer.encode(chunk)
        assert result[SampleKey.TOKENIZED_ACTIONS.value].device.type == device.type


@pytest.mark.integration
class TestActionTokenizerIntegrationCustomFast:
    def test_fit_and_encode_decode(self, action_chunk_factory, device):
        tokenizer = ActionTokenizer(
            action_discretizer=FastActionDiscretizer(use_pretrained=False),
            device=device,
        )
        training_data = action_chunk_factory(batch_size=20, scale=0.5)
        tokenizer.fit(training_data)
        assert tokenizer._is_fitted is True
        # 1024 FAST tokens + 1 EOS token
        assert tokenizer.vocab_size == 1025
        chunk = training_data[0]
        result = tokenizer.encode_chunk(chunk)
        decoded = tokenizer.decode_chunk(result[SampleKey.TOKENIZED_ACTIONS.value])
        assert decoded.shape == chunk.shape


@pytest.mark.integration
class TestActionTokenizerIntegrationSaveLoad:
    def test_fit_save_load_decode_roundtrip(self, action_chunk_factory, tmp_path):
        tokenizer = ActionTokenizer(
            action_discretizer=FastActionDiscretizer(use_pretrained=False),
        )
        training_data = action_chunk_factory(batch_size=20, scale=0.5)
        tokenizer.fit(training_data)

        save_path = tmp_path / "action_tokenizer"
        tokenizer.save_pretrained(save_path)
        loaded = ActionTokenizer.from_pretrained(save_path)

        chunk = training_data[0]
        original_result = tokenizer.encode_chunk(chunk)
        original_tokens = original_result[SampleKey.TOKENIZED_ACTIONS.value]

        loaded_decoded = loaded.decode_chunk(original_tokens)
        assert loaded_decoded.shape == chunk.shape

        original_decoded = tokenizer.decode_chunk(original_tokens)
        np.testing.assert_array_equal(loaded_decoded, original_decoded)

    def test_loaded_tokenizer_preserves_vocab_size(
        self, action_chunk_factory, tmp_path
    ):
        tokenizer = ActionTokenizer(
            action_discretizer=FastActionDiscretizer(use_pretrained=False),
        )
        training_data = action_chunk_factory(batch_size=20, scale=0.5)
        tokenizer.fit(training_data)

        save_path = tmp_path / "action_tokenizer"
        tokenizer.save_pretrained(save_path)
        loaded = ActionTokenizer.from_pretrained(save_path)

        assert loaded.vocab_size == tokenizer.vocab_size
        assert loaded._is_fitted is True

    def test_loaded_tokenizer_encode_produces_valid_tokens(
        self, action_chunk_factory, tmp_path
    ):
        tokenizer = ActionTokenizer(
            action_discretizer=FastActionDiscretizer(use_pretrained=False),
        )
        training_data = action_chunk_factory(batch_size=20, scale=0.5)
        tokenizer.fit(training_data)

        save_path = tmp_path / "action_tokenizer"
        tokenizer.save_pretrained(save_path)
        loaded = ActionTokenizer.from_pretrained(save_path)

        chunk = training_data[0]
        result = loaded.encode_chunk(chunk)
        tokens = result[SampleKey.TOKENIZED_ACTIONS.value]
        non_pad = tokens[~result[SampleKey.IS_PAD_ACTION.value]]
        assert (non_pad >= 0).all()
        assert (non_pad < loaded.vocab_size).all()

    def test_fit_save_load_with_language_mapping(self, action_chunk_factory, tmp_path):
        tokenizer = ActionTokenizer(
            action_discretizer=FastActionDiscretizer(use_pretrained=False),
            token_id_mapping=LanguageVocabularyActionTokenIdMapping(
                language_tokenizer_model="google/bert_uncased_L-2_H-128_A-2",
                num_special_tokens_to_skip=128,
            ),
        )
        training_data = action_chunk_factory(batch_size=20, scale=0.5)
        tokenizer.fit(training_data)

        save_path = tmp_path / "action_tokenizer"
        tokenizer.save_pretrained(save_path)
        loaded = ActionTokenizer.from_pretrained(save_path)

        chunk = training_data[0]
        original_tokens = tokenizer.encode_chunk(chunk)[
            SampleKey.TOKENIZED_ACTIONS.value
        ]
        loaded_decoded = loaded.decode_chunk(original_tokens)
        original_decoded = tokenizer.decode_chunk(original_tokens)
        np.testing.assert_array_equal(loaded_decoded, original_decoded)


@pytest.mark.integration
class TestActionTokenizerIntegrationLanguageMapping:
    def test_encode_decode_with_language_mapping(self, action_chunk_factory, device):
        tokenizer = ActionTokenizer(
            action_discretizer=FastActionDiscretizer(use_pretrained=True),
            token_id_mapping=LanguageVocabularyActionTokenIdMapping(
                language_tokenizer_model="google/bert_uncased_L-2_H-128_A-2",
                num_special_tokens_to_skip=128,
            ),
            device=device,
        )
        chunks = action_chunk_factory(batch_size=3, scale=0.5)
        result = tokenizer.encode(chunks)
        assert SampleKey.TOKENIZED_ACTIONS.value in result
        tokens = result[SampleKey.TOKENIZED_ACTIONS.value]
        assert tokens.max() < tokenizer.vocab_size
        decoded = tokenizer.decode(tokens)
        assert decoded.shape == chunks.shape
        np.testing.assert_allclose(decoded, chunks, atol=0.2)

    def test_mapped_tokens_in_expected_range(self, action_chunk_factory, device):
        tokenizer = ActionTokenizer(
            action_discretizer=FastActionDiscretizer(use_pretrained=True),
            token_id_mapping=LanguageVocabularyActionTokenIdMapping(
                language_tokenizer_model="google/bert_uncased_L-2_H-128_A-2",
                num_special_tokens_to_skip=128,
            ),
            device=device,
        )
        chunk = action_chunk_factory(scale=0.5)
        result = tokenizer.encode_chunk(chunk)
        tokens = result[SampleKey.TOKENIZED_ACTIONS.value]
        non_pad = tokens[~result[SampleKey.IS_PAD_ACTION.value]]
        if len(non_pad) > 0:
            language_token_count = (
                tokenizer.token_id_mapping.language_tokenizer.vocab_size
            )
            expected_max = (
                language_token_count
                - 1
                - tokenizer.token_id_mapping.num_special_tokens_to_skip
            )
            expected_min = expected_max - tokenizer.action_discretizer.token_count + 1
            # EOS token is at vocab_size - 1 (after the +1 for EOS reservation)
            eos_id = tokenizer.eos_token_id
            non_eos = non_pad[non_pad != eos_id]
            if len(non_eos) > 0:
                assert non_eos.min() >= expected_min
                assert non_eos.max() <= expected_max
