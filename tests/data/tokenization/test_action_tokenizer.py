"""Tests for versatil.data.tokenization.action_tokenizer."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from versatil.data.constants import SampleKey, TokenizerType
from versatil.data.tokenization.action_tokenizer import ActionTokenizer


@pytest.fixture
def mock_auto_processor():
    """Patches AutoProcessor in action_tokenizer module."""
    with patch(
        "versatil.data.tokenization.action_tokenizer.AutoProcessor"
    ) as mock:
        yield mock


@pytest.fixture
def action_tokenizer_factory(mock_auto_processor):
    """Factory for ActionTokenizer with AutoProcessor mocked."""

    def factory(
        tokenizer_chain: list[str] | None = None,
        use_pretrained_fast: bool = True,
        language_tokenizer_model: str | None = None,
        fast_tokenizer_model: str = "physical-intelligence/fast",
        num_special_tokens_to_skip: int = 128,
        max_token_len: int = 256,
        pad_token_id: int = 0,
        device: torch.device | None = None,
    ) -> ActionTokenizer:
        if tokenizer_chain is None:
            tokenizer_chain = [TokenizerType.FAST.value]
        return ActionTokenizer(
            tokenizer_chain=tokenizer_chain,
            use_pretrained_fast=use_pretrained_fast,
            language_tokenizer_model=language_tokenizer_model,
            fast_tokenizer_model=fast_tokenizer_model,
            num_special_tokens_to_skip=num_special_tokens_to_skip,
            max_token_len=max_token_len,
            pad_token_id=pad_token_id,
            device=device,
        )

    return factory


class TestActionTokenizerInit:

    def test_stores_tokenizer_chain(self, action_tokenizer_factory):
        chain = [TokenizerType.FAST.value]
        tokenizer = action_tokenizer_factory(tokenizer_chain=chain)
        assert tokenizer.tokenizer_chain == chain

    def test_stores_use_pretrained_fast(self, action_tokenizer_factory):
        tokenizer = action_tokenizer_factory(use_pretrained_fast=True)
        assert tokenizer.use_pretrained_fast is True

    def test_pretrained_fast_sets_vocab_size(self, action_tokenizer_factory):
        expected_fast_vocab_size = 2048
        expected_eos_token_id = expected_fast_vocab_size
        expected_tokenizer_vocab_size = expected_fast_vocab_size + 1
        tokenizer = action_tokenizer_factory(use_pretrained_fast=True)
        assert tokenizer.fast_vocab_size == expected_fast_vocab_size
        assert tokenizer.eos_token_id == expected_eos_token_id
        assert tokenizer.vocab_size == expected_tokenizer_vocab_size

    def test_custom_fast_sets_vocab_size_1024(self, action_tokenizer_factory):
        expected_fast_vocab_size = 1024
        tokenizer = action_tokenizer_factory(use_pretrained_fast=False)
        assert tokenizer.fast_vocab_size == expected_fast_vocab_size

    def test_pretrained_fast_is_fitted_on_init(self, action_tokenizer_factory):
        tokenizer = action_tokenizer_factory(use_pretrained_fast=True)
        assert tokenizer._is_fitted is True

    def test_custom_fast_not_fitted_on_init(self, action_tokenizer_factory):
        tokenizer = action_tokenizer_factory(use_pretrained_fast=False)
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
        ActionTokenizer(
            tokenizer_chain=[TokenizerType.FAST.value], use_pretrained_fast=True
        )
        mock_auto_processor.from_pretrained.assert_called_once_with(
            "physical-intelligence/fast", trust_remote_code=True
        )

    def test_custom_fast_model_name(self, mock_auto_processor):
        ActionTokenizer(
            tokenizer_chain=[TokenizerType.FAST.value],
            use_pretrained_fast=True,
            fast_tokenizer_model="custom/model",
        )
        mock_auto_processor.from_pretrained.assert_called_once_with(
            "custom/model", trust_remote_code=True
        )

    @patch("versatil.data.tokenization.action_tokenizer.AutoTokenizer")
    def test_language_in_chain_loads_language_tokenizer(
        self, mock_auto_tokenizer, action_tokenizer_factory
    ):
        mock_lang_tok = MagicMock()
        mock_lang_tok.vocab_size = 32000
        mock_lang_tok.pad_token = "[PAD]"
        mock_auto_tokenizer.from_pretrained.return_value = mock_lang_tok
        tokenizer = action_tokenizer_factory(
            tokenizer_chain=[TokenizerType.FAST.value, TokenizerType.LANGUAGE.value],
            language_tokenizer_model="some-model",
        )
        mock_auto_tokenizer.from_pretrained.assert_called_once_with("some-model")
        assert tokenizer.vocab_size == 32001

    def test_language_without_model_raises(self, mock_auto_processor):
        with pytest.raises(
            ValueError, match="language_tokenizer_model must be provided"
        ):
            ActionTokenizer(
                tokenizer_chain=[
                    TokenizerType.FAST.value,
                    TokenizerType.LANGUAGE.value,
                ],
                use_pretrained_fast=True,
                language_tokenizer_model=None,
            )

    @patch("versatil.data.tokenization.action_tokenizer.AutoTokenizer")
    def test_small_language_vocab_raises(
        self, mock_auto_tokenizer, mock_auto_processor
    ):
        mock_lang_tok = MagicMock()
        mock_lang_tok.vocab_size = 100
        mock_lang_tok.pad_token = "[PAD]"
        mock_auto_tokenizer.from_pretrained.return_value = mock_lang_tok
        with pytest.raises(ValueError, match="vocab size .* is too small"):
            ActionTokenizer(
                tokenizer_chain=[
                    TokenizerType.FAST.value,
                    TokenizerType.LANGUAGE.value,
                ],
                use_pretrained_fast=True,
                language_tokenizer_model="tiny-model",
            )

    @patch("versatil.data.tokenization.action_tokenizer.AutoTokenizer")
    def test_sets_pad_token_from_eos_when_none(
        self, mock_auto_tokenizer, mock_auto_processor
    ):
        mock_lang_tok = MagicMock()
        mock_lang_tok.vocab_size = 32000
        mock_lang_tok.pad_token = None
        mock_lang_tok.eos_token = "<eos>"
        mock_auto_tokenizer.from_pretrained.return_value = mock_lang_tok
        ActionTokenizer(
            tokenizer_chain=[TokenizerType.FAST.value, TokenizerType.LANGUAGE.value],
            use_pretrained_fast=True,
            language_tokenizer_model="some-model",
        )
        assert mock_lang_tok.pad_token == "<eos>"


class TestActionTokenizerFit:

    def test_fit_raises_when_pretrained(
        self, action_tokenizer_factory, action_chunk_factory
    ):
        tokenizer = action_tokenizer_factory(use_pretrained_fast=True)
        data = action_chunk_factory(batch_size=10)
        with pytest.raises(ValueError, match="Cannot fit when use_pretrained_fast"):
            tokenizer.fit(data)

    def test_fit_calls_processor_fit(
        self, action_tokenizer_factory, action_chunk_factory
    ):
        tokenizer = action_tokenizer_factory(use_pretrained_fast=False)
        tokenizer.fast_processor.fit.return_value = tokenizer.fast_processor
        data = action_chunk_factory(batch_size=10)
        tokenizer.fit(data)
        tokenizer.fast_processor.fit.assert_called_once()
        assert tokenizer._is_fitted is True

    def test_fit_sets_vocab_size_without_language_tokenizer(
        self, action_tokenizer_factory, action_chunk_factory
    ):
        tokenizer = action_tokenizer_factory(use_pretrained_fast=False)
        tokenizer.fast_processor.fit.return_value = tokenizer.fast_processor
        data = action_chunk_factory(batch_size=10)
        tokenizer.fit(data)
        assert tokenizer.eos_token_id == 1024
        assert tokenizer.vocab_size == 1025

    def test_fit_raises_when_processor_none(
        self, action_tokenizer_factory, action_chunk_factory
    ):
        tokenizer = action_tokenizer_factory(use_pretrained_fast=False)
        tokenizer.fast_processor = None
        data = action_chunk_factory(batch_size=10)
        with pytest.raises(RuntimeError, match="FAST processor not initialized"):
            tokenizer.fit(data)

    def test_fit_logs_info(self, action_tokenizer_factory, action_chunk_factory):
        tokenizer = action_tokenizer_factory(use_pretrained_fast=False)
        tokenizer.fast_processor.fit.return_value = tokenizer.fast_processor
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


class TestActionTokenizerMapFastToLanguageVocab:

    @patch("versatil.data.tokenization.action_tokenizer.AutoTokenizer")
    def test_mapping_formula(self, mock_auto_tokenizer, action_tokenizer_factory):
        mock_lang_tok = MagicMock()
        mock_lang_tok.vocab_size = 32000
        mock_lang_tok.pad_token = "[PAD]"
        mock_auto_tokenizer.from_pretrained.return_value = mock_lang_tok
        tokenizer = action_tokenizer_factory(
            tokenizer_chain=[TokenizerType.FAST.value, TokenizerType.LANGUAGE.value],
            language_tokenizer_model="model",
            num_special_tokens_to_skip=128,
        )
        fast_tokens = np.array([0, 1, 2])
        mapped = tokenizer._map_fast_to_language_vocab(fast_tokens)
        expected = np.array([
            32000 - 1 - 128 - 0,
            32000 - 1 - 128 - 1,
            32000 - 1 - 128 - 2,
        ])
        np.testing.assert_array_equal(mapped, expected)

    def test_mapping_without_language_tokenizer_raises(
        self, action_tokenizer_factory
    ):
        tokenizer = action_tokenizer_factory()
        with pytest.raises(RuntimeError, match="Language tokenizer not initialized"):
            tokenizer._map_fast_to_language_vocab(np.array([0, 1]))

    @patch("versatil.data.tokenization.action_tokenizer.AutoTokenizer")
    def test_mapping_accepts_list_input(
        self, mock_auto_tokenizer, action_tokenizer_factory
    ):
        mock_lang_tok = MagicMock()
        mock_lang_tok.vocab_size = 32000
        mock_lang_tok.pad_token = "[PAD]"
        mock_auto_tokenizer.from_pretrained.return_value = mock_lang_tok
        tokenizer = action_tokenizer_factory(
            tokenizer_chain=[TokenizerType.FAST.value, TokenizerType.LANGUAGE.value],
            language_tokenizer_model="model",
        )
        mapped = tokenizer._map_fast_to_language_vocab([0, 1, 2])
        assert isinstance(mapped, np.ndarray)


class TestActionTokenizerUnmapLanguageToFastVocab:

    @patch("versatil.data.tokenization.action_tokenizer.AutoTokenizer")
    def test_roundtrip_map_unmap(
        self, mock_auto_tokenizer, action_tokenizer_factory
    ):
        mock_lang_tok = MagicMock()
        mock_lang_tok.vocab_size = 32000
        mock_lang_tok.pad_token = "[PAD]"
        mock_auto_tokenizer.from_pretrained.return_value = mock_lang_tok
        tokenizer = action_tokenizer_factory(
            tokenizer_chain=[TokenizerType.FAST.value, TokenizerType.LANGUAGE.value],
            language_tokenizer_model="model",
            num_special_tokens_to_skip=128,
        )
        original = np.array([0, 5, 100, 2047])
        mapped = tokenizer._map_fast_to_language_vocab(original)
        unmapped = tokenizer._unmap_language_to_fast_vocab(mapped)
        np.testing.assert_array_equal(unmapped, original)

    @patch("versatil.data.tokenization.action_tokenizer.AutoTokenizer")
    def test_unmap_accepts_torch_tensor(
        self, mock_auto_tokenizer, action_tokenizer_factory
    ):
        mock_lang_tok = MagicMock()
        mock_lang_tok.vocab_size = 32000
        mock_lang_tok.pad_token = "[PAD]"
        mock_auto_tokenizer.from_pretrained.return_value = mock_lang_tok
        tokenizer = action_tokenizer_factory(
            tokenizer_chain=[TokenizerType.FAST.value, TokenizerType.LANGUAGE.value],
            language_tokenizer_model="model",
        )
        lang_tokens = torch.tensor([31871, 31870, 31869])
        fast_tokens = tokenizer._unmap_language_to_fast_vocab(lang_tokens)
        assert isinstance(fast_tokens, np.ndarray)

    def test_unmap_without_language_tokenizer_raises(
        self, action_tokenizer_factory
    ):
        tokenizer = action_tokenizer_factory()
        with pytest.raises(RuntimeError, match="Language tokenizer not initialized"):
            tokenizer._unmap_language_to_fast_vocab(np.array([100]))


class TestActionTokenizerEncodeChunk:

    def test_encode_chunk_raises_when_not_fitted(
        self, action_tokenizer_factory, action_chunk_factory
    ):
        tokenizer = action_tokenizer_factory(use_pretrained_fast=False)
        chunk = action_chunk_factory()
        with pytest.raises(RuntimeError, match="fitted or loaded before encoding"):
            tokenizer.encode_chunk(chunk)

    def test_encode_chunk_returns_correct_keys(
        self, action_tokenizer_factory, action_chunk_factory
    ):
        tokenizer = action_tokenizer_factory(max_token_len=8)
        tokenizer.fast_processor.side_effect = lambda x: [[10, 20, 30]]
        chunk = action_chunk_factory()
        result = tokenizer.encode_chunk(chunk)
        assert SampleKey.TOKENIZED_ACTIONS.value in result
        assert SampleKey.IS_PAD_ACTION.value in result

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
        tokenizer.fast_processor.side_effect = lambda x: [mock_tokens]
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
        tokenizer.fast_processor.side_effect = lambda x: [[10, 20]]
        chunk = action_chunk_factory()
        pad_mask = pad_mask_factory(total=5, num_valid=2)
        result = tokenizer.encode_chunk(chunk, is_pad_mask=pad_mask)
        called_data = tokenizer.fast_processor.call_args[0][0]
        assert called_data.shape[0] == 2

    def test_encode_chunk_with_torch_pad_mask(
        self, action_tokenizer_factory, action_chunk_factory, pad_mask_factory
    ):
        tokenizer = action_tokenizer_factory(max_token_len=8)
        tokenizer.fast_processor.side_effect = lambda x: [[10, 20]]
        chunk = action_chunk_factory(as_torch=True)
        pad_mask = pad_mask_factory(total=5, num_valid=2, as_torch=True)
        result = tokenizer.encode_chunk(chunk, is_pad_mask=pad_mask)
        assert SampleKey.TOKENIZED_ACTIONS.value in result

    def test_encode_chunk_truncates_when_exceeding_max_len(
        self, action_tokenizer_factory, action_chunk_factory
    ):
        long_tokens = list(range(20))
        tokenizer = action_tokenizer_factory(max_token_len=8)
        tokenizer.fast_processor.side_effect = lambda x: [long_tokens]
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
        tokenizer.fast_processor.side_effect = lambda x: [long_tokens]
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
        tokenizer.fast_processor.side_effect = lambda x: [[10, 20, 30, 40]]
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

    def test_encode_chunk_raises_without_fast_processor(
        self, action_tokenizer_factory
    ):
        tokenizer = action_tokenizer_factory()
        tokenizer.fast_processor = None
        with pytest.raises(RuntimeError, match="No tokenizers in chain"):
            tokenizer.encode_chunk(np.zeros((5, 7), dtype=np.float32))


class TestActionTokenizerEncodeBatch:

    def test_encode_batch_raises_when_not_fitted(
        self, action_tokenizer_factory, action_chunk_factory
    ):
        tokenizer = action_tokenizer_factory(use_pretrained_fast=False)
        batch = action_chunk_factory(batch_size=3)
        with pytest.raises(RuntimeError, match="fitted or loaded before encoding"):
            tokenizer.encode_batch(batch)

    @pytest.mark.parametrize(
        "batch_size, max_token_len",
        [(2, 8), (3, 16), (5, 32)],
    )
    def test_encode_batch_returns_stacked_results(
        self, action_tokenizer_factory, action_chunk_factory, batch_size, max_token_len
    ):
        tokenizer = action_tokenizer_factory(max_token_len=max_token_len)
        tokenizer.fast_processor.side_effect = lambda x: [[10, 20]]
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
        tokenizer.fast_processor.side_effect = lambda x: [[10, 20]]
        batch = action_chunk_factory(batch_size=2)
        pad_mask = np.array([
            [False, False, True, True, True],
            [False, False, False, True, True],
        ])
        result = tokenizer.encode_batch(batch, is_pad_mask=pad_mask)
        assert result[SampleKey.TOKENIZED_ACTIONS.value].shape == (2, 8)


class TestActionTokenizerEncode:

    def test_encode_2d_dispatches_to_encode_chunk(
        self, action_tokenizer_factory, action_chunk_factory
    ):
        tokenizer = action_tokenizer_factory(max_token_len=8)
        tokenizer.fast_processor.side_effect = lambda x: [[10, 20]]
        chunk = action_chunk_factory()
        result = tokenizer.encode(chunk)
        assert result[SampleKey.TOKENIZED_ACTIONS.value].ndim == 1

    def test_encode_3d_dispatches_to_encode_batch(
        self, action_tokenizer_factory, action_chunk_factory
    ):
        tokenizer = action_tokenizer_factory(max_token_len=8)
        tokenizer.fast_processor.side_effect = lambda x: [[10, 20]]
        batch = action_chunk_factory(batch_size=3)
        result = tokenizer.encode(batch)
        assert result[SampleKey.TOKENIZED_ACTIONS.value].shape[0] == 3

    def test_encode_3d_passes_pad_mask_to_encode_batch(
        self, action_tokenizer_factory, action_chunk_factory
    ):
        tokenizer = action_tokenizer_factory(max_token_len=8)
        tokenizer.fast_processor.side_effect = lambda x: [[10, 20]]
        batch = action_chunk_factory(batch_size=2)
        pad_mask = np.array([
            [False, False, True, True, True],
            [False, False, False, True, True],
        ])
        result = tokenizer.encode(batch, is_pad_mask=pad_mask)
        assert result[SampleKey.TOKENIZED_ACTIONS.value].shape[0] == 2

    def test_encode_invalid_ndim_raises(self, action_tokenizer_factory, rng):
        tokenizer = action_tokenizer_factory()
        data_1d = rng.standard_normal((7,)).astype(np.float32)
        with pytest.raises(ValueError, match="Expected 2D or 3D input"):
            tokenizer.encode(data_1d)


class TestActionTokenizerDecodeChunk:

    def test_decode_chunk_raises_when_not_fitted(self, action_tokenizer_factory):
        tokenizer = action_tokenizer_factory(use_pretrained_fast=False)
        with pytest.raises(RuntimeError, match="fitted or loaded before decoding"):
            tokenizer.decode_chunk(torch.tensor([1, 2, 3]))

    def test_decode_chunk_strips_pad_tokens(self, action_tokenizer_factory):
        tokenizer = action_tokenizer_factory(pad_token_id=0)
        decoded_array = np.zeros((1, 5, 7), dtype=np.float32)
        tokenizer.fast_processor.decode.return_value = decoded_array
        tokens = torch.tensor([10, 20, 30, 0, 0, 0])
        tokenizer.decode_chunk(tokens)
        call_args = tokenizer.fast_processor.decode.call_args[0][0]
        assert call_args == [[10, 20, 30]]

    def test_decode_chunk_strips_eos_token(self, action_tokenizer_factory):
        tokenizer = action_tokenizer_factory(pad_token_id=0)
        eos_id = tokenizer.eos_token_id
        decoded_array = np.zeros((1, 5, 7), dtype=np.float32)
        tokenizer.fast_processor.decode.return_value = decoded_array
        tokens = torch.tensor([10, 20, 30, eos_id, 0, 0])
        tokenizer.decode_chunk(tokens)
        call_args = tokenizer.fast_processor.decode.call_args[0][0]
        assert call_args == [[10, 20, 30]]

    def test_decode_chunk_raises_without_fast_processor(
        self, action_tokenizer_factory
    ):
        tokenizer = action_tokenizer_factory()
        tokenizer.fast_processor = None
        with pytest.raises(RuntimeError, match="Cannot decode without FAST processor"):
            tokenizer.decode_chunk(torch.tensor([1, 2, 3]))

    def test_decode_chunk_accepts_list_input(self, action_tokenizer_factory):
        tokenizer = action_tokenizer_factory(pad_token_id=0)
        tokenizer.fast_processor.decode.return_value = np.zeros(
            (1, 5, 7), dtype=np.float32
        )
        result = tokenizer.decode_chunk([10, 20, 30])
        assert isinstance(result, np.ndarray)

    def test_decode_chunk_accepts_numpy_input(self, action_tokenizer_factory):
        tokenizer = action_tokenizer_factory(pad_token_id=0)
        tokenizer.fast_processor.decode.return_value = np.zeros(
            (1, 5, 7), dtype=np.float32
        )
        result = tokenizer.decode_chunk(np.array([10, 20, 30]))
        assert isinstance(result, np.ndarray)

    def test_decode_chunk_raises_type_error_when_processor_returns_non_ndarray(
        self, action_tokenizer_factory
    ):
        tokenizer = action_tokenizer_factory(pad_token_id=0)
        tokenizer.fast_processor.decode.return_value = [[0.1, 0.2]]
        with pytest.raises(TypeError, match="Expected np.ndarray"):
            tokenizer.decode_chunk(torch.tensor([10, 20, 30]))


class TestActionTokenizerDecodeBatch:

    def test_decode_batch_raises_when_not_fitted(self, action_tokenizer_factory):
        tokenizer = action_tokenizer_factory(use_pretrained_fast=False)
        tokens = torch.tensor([[10, 20, 30], [40, 50, 60]])
        with pytest.raises(RuntimeError, match="fitted or loaded before decoding"):
            tokenizer.decode_batch(tokens)

    def test_decode_batch_strips_pad_per_sample(self, action_tokenizer_factory):
        tokenizer = action_tokenizer_factory(pad_token_id=0)
        tokenizer.fast_processor.decode.return_value = np.zeros(
            (2, 5, 7), dtype=np.float32
        )
        tokens = torch.tensor([[10, 20, 0], [30, 40, 50]])
        tokenizer.decode_batch(tokens)
        call_args = tokenizer.fast_processor.decode.call_args[0][0]
        assert call_args[0] == [10, 20]
        assert call_args[1] == [30, 40, 50]

    def test_decode_batch_strips_eos_per_sample(self, action_tokenizer_factory):
        tokenizer = action_tokenizer_factory(pad_token_id=0)
        eos_id = tokenizer.eos_token_id
        tokenizer.fast_processor.decode.return_value = np.zeros(
            (2, 5, 7), dtype=np.float32
        )
        tokens = torch.tensor([[10, 20, eos_id, 0], [30, eos_id, 0, 0]])
        tokenizer.decode_batch(tokens)
        call_args = tokenizer.fast_processor.decode.call_args[0][0]
        assert call_args[0] == [10, 20]
        assert call_args[1] == [30]

    def test_decode_batch_raises_without_fast_processor(
        self, action_tokenizer_factory
    ):
        tokenizer = action_tokenizer_factory()
        tokenizer.fast_processor = None
        tokens = torch.tensor([[10, 20, 30]])
        with pytest.raises(RuntimeError, match="Cannot decode without FAST processor"):
            tokenizer.decode_batch(tokens)

    def test_decode_batch_accepts_numpy_input(self, action_tokenizer_factory):
        tokenizer = action_tokenizer_factory(pad_token_id=0)
        tokenizer.fast_processor.decode.return_value = np.zeros(
            (2, 5, 7), dtype=np.float32
        )
        tokens = np.array([[10, 20, 30], [40, 50, 60]])
        result = tokenizer.decode_batch(tokens)
        assert result.shape == (2, 5, 7)

    def test_decode_batch_raises_type_error_when_processor_returns_non_ndarray(
        self, action_tokenizer_factory
    ):
        tokenizer = action_tokenizer_factory(pad_token_id=0)
        tokenizer.fast_processor.decode.return_value = [[0.1, 0.2]]
        tokens = torch.tensor([[10, 20, 30], [40, 50, 60]])
        with pytest.raises(TypeError, match="Expected np.ndarray"):
            tokenizer.decode_batch(tokens)


class TestActionTokenizerDecode:

    def test_decode_1d_dispatches_to_decode_chunk(self, action_tokenizer_factory):
        tokenizer = action_tokenizer_factory(pad_token_id=0)
        tokenizer.fast_processor.decode.return_value = np.zeros(
            (1, 5, 7), dtype=np.float32
        )
        result = tokenizer.decode(torch.tensor([10, 20, 30]))
        assert result.shape == (5, 7)

    def test_decode_2d_dispatches_to_decode_batch(self, action_tokenizer_factory):
        tokenizer = action_tokenizer_factory(pad_token_id=0)
        tokenizer.fast_processor.decode.return_value = np.zeros(
            (2, 5, 7), dtype=np.float32
        )
        tokens = torch.tensor([[10, 20, 30], [40, 50, 60]])
        result = tokenizer.decode(tokens)
        assert result.shape == (2, 5, 7)

    def test_decode_list_input_dispatches_to_decode_chunk(
        self, action_tokenizer_factory
    ):
        tokenizer = action_tokenizer_factory(pad_token_id=0)
        tokenizer.fast_processor.decode.return_value = np.zeros(
            (1, 5, 7), dtype=np.float32
        )
        result = tokenizer.decode([10, 20, 30])
        assert result.shape == (5, 7)

    def test_decode_invalid_ndim_raises(self, action_tokenizer_factory):
        tokenizer = action_tokenizer_factory()
        tokens_3d = torch.zeros((2, 3, 4), dtype=torch.long)
        with pytest.raises(ValueError, match="Expected 1D or 2D input"):
            tokenizer.decode(tokens_3d)


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
            "tokenizer_chain",
            "use_pretrained_fast",
            "language_tokenizer_model",
            "fast_vocab_size",
            "num_special_tokens_to_skip",
            "vocab_size",
            "eos_token_id",
            "is_fitted",
        }
        assert set(state.keys()) == expected_keys

    @pytest.mark.parametrize(
        "use_pretrained_fast, num_special_tokens_to_skip, expected_fast_vocab",
        [
            (True, 64, 2048),
            (True, 128, 2048),
            (True, 256, 2048),
        ],
    )
    def test_state_dict_values(
        self,
        action_tokenizer_factory,
        use_pretrained_fast,
        num_special_tokens_to_skip,
        expected_fast_vocab,
    ):
        tokenizer = action_tokenizer_factory(
            use_pretrained_fast=use_pretrained_fast,
            num_special_tokens_to_skip=num_special_tokens_to_skip,
        )
        state = tokenizer.state_dict()
        assert state["tokenizer_chain"] == [TokenizerType.FAST.value]
        assert state["use_pretrained_fast"] is use_pretrained_fast
        assert state["fast_vocab_size"] == expected_fast_vocab
        assert state["num_special_tokens_to_skip"] == num_special_tokens_to_skip
        assert state["is_fitted"] is True
        assert state["eos_token_id"] == expected_fast_vocab
        assert state["vocab_size"] == expected_fast_vocab + 1


class TestActionTokenizerLoadStateDict:

    @pytest.mark.parametrize(
        "tokenizer_chain, use_pretrained, language_model, fast_vocab,"
        " skip_tokens, vocab_size",
        [
            (
                [TokenizerType.FAST.value, TokenizerType.LANGUAGE.value],
                False,
                "some-model",
                1024,
                64,
                32000,
            ),
            (
                [TokenizerType.FAST.value],
                True,
                None,
                2048,
                128,
                2048,
            ),
            (
                [TokenizerType.FAST.value, TokenizerType.LANGUAGE.value],
                True,
                "bert-base",
                2048,
                256,
                50000,
            ),
        ],
    )
    def test_load_state_dict_restores_fields(
        self,
        action_tokenizer_factory,
        tokenizer_chain,
        use_pretrained,
        language_model,
        fast_vocab,
        skip_tokens,
        vocab_size,
    ):
        eos_token_id = vocab_size - 1
        tokenizer = action_tokenizer_factory()
        state = {
            "tokenizer_chain": tokenizer_chain,
            "use_pretrained_fast": use_pretrained,
            "language_tokenizer_model": language_model,
            "fast_vocab_size": fast_vocab,
            "num_special_tokens_to_skip": skip_tokens,
            "vocab_size": vocab_size,
            "eos_token_id": eos_token_id,
            "is_fitted": True,
        }
        tokenizer.load_state_dict(state)
        assert tokenizer.tokenizer_chain == tokenizer_chain
        assert tokenizer.use_pretrained_fast is use_pretrained
        assert tokenizer.language_tokenizer_model == language_model
        assert tokenizer.fast_vocab_size == fast_vocab
        assert tokenizer.num_special_tokens_to_skip == skip_tokens
        assert tokenizer.vocab_size == vocab_size
        assert tokenizer.eos_token_id == eos_token_id
        assert tokenizer._is_fitted is True

    def test_load_state_dict_without_eos_token_id_defaults_to_none(
        self, action_tokenizer_factory
    ):
        tokenizer = action_tokenizer_factory()
        state = {
            "tokenizer_chain": [TokenizerType.FAST.value],
            "use_pretrained_fast": True,
            "language_tokenizer_model": None,
            "fast_vocab_size": 2048,
            "num_special_tokens_to_skip": 128,
            "vocab_size": 2048,
            "is_fitted": True,
        }
        tokenizer.load_state_dict(state)
        assert tokenizer.eos_token_id is None


class TestActionTokenizerSavePretrained:

    def test_save_raises_when_not_fitted(self, action_tokenizer_factory, tmp_path):
        tokenizer = action_tokenizer_factory(use_pretrained_fast=False)
        with pytest.raises(RuntimeError, match="Cannot save unfitted tokenizer"):
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
        tokenizer = action_tokenizer_factory(use_pretrained_fast=True)
        save_path = tmp_path / "tokenizer"
        tokenizer.save_pretrained(save_path)
        tokenizer.fast_processor.save_pretrained.assert_not_called()

    def test_save_saves_fast_processor_when_custom(
        self, action_tokenizer_factory, tmp_path, action_chunk_factory
    ):
        tokenizer = action_tokenizer_factory(use_pretrained_fast=False)
        tokenizer.fast_processor.fit.return_value = tokenizer.fast_processor
        data = action_chunk_factory(batch_size=10)
        tokenizer.fit(data)
        save_path = tmp_path / "tokenizer"
        tokenizer.save_pretrained(save_path)
        tokenizer.fast_processor.save_pretrained.assert_called_once()

    @patch("versatil.data.tokenization.action_tokenizer.AutoTokenizer")
    def test_save_saves_language_tokenizer(
        self, mock_auto_tokenizer, action_tokenizer_factory, tmp_path
    ):
        mock_lang_tok = MagicMock()
        mock_lang_tok.vocab_size = 32000
        mock_lang_tok.pad_token = "[PAD]"
        mock_auto_tokenizer.from_pretrained.return_value = mock_lang_tok
        tokenizer = action_tokenizer_factory(
            tokenizer_chain=[TokenizerType.FAST.value, TokenizerType.LANGUAGE.value],
            language_tokenizer_model="model",
        )
        save_path = tmp_path / "tokenizer"
        tokenizer.save_pretrained(save_path)
        mock_lang_tok.save_pretrained.assert_called_once_with(
            save_path / "language_tokenizer"
        )

    def test_save_pretrained_logs_info(
        self, action_tokenizer_factory, tmp_path
    ):
        tokenizer = action_tokenizer_factory()
        save_path = tmp_path / "tokenizer"
        with patch(
            "versatil.data.tokenization.action_tokenizer.logging"
        ) as mock_logging:
            tokenizer.save_pretrained(save_path)
            mock_logging.info.assert_called_once()
            assert str(save_path) in str(mock_logging.info.call_args)


class TestActionTokenizerFromPretrained:

    def test_raises_file_not_found(self, mock_auto_processor):
        with pytest.raises(FileNotFoundError, match="Tokenizer path not found"):
            ActionTokenizer.from_pretrained("/nonexistent/path")

    @patch("versatil.data.tokenization.action_tokenizer.torch.load")
    def test_loads_state_and_restores_tokenizer(
        self, mock_torch_load, mock_auto_processor, tmp_path
    ):
        save_path = tmp_path / "tokenizer"
        save_path.mkdir(parents=True)
        (save_path / "action_tokenizer_state.pt").touch()
        mock_torch_load.return_value = {
            "tokenizer_chain": [TokenizerType.FAST.value],
            "use_pretrained_fast": True,
            "language_tokenizer_model": None,
            "fast_vocab_size": 2048,
            "num_special_tokens_to_skip": 128,
            "vocab_size": 2048,
            "is_fitted": True,
        }
        loaded = ActionTokenizer.from_pretrained(save_path)
        assert loaded.tokenizer_chain == [TokenizerType.FAST.value]
        assert loaded.vocab_size == 2048
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
            "tokenizer_chain": [TokenizerType.FAST.value],
            "use_pretrained_fast": False,
            "language_tokenizer_model": None,
            "fast_vocab_size": 1024,
            "num_special_tokens_to_skip": 128,
            "vocab_size": 1024,
            "is_fitted": True,
        }
        loaded = ActionTokenizer.from_pretrained(save_path)
        assert mock_auto_processor.from_pretrained.call_count == 2
        second_call = mock_auto_processor.from_pretrained.call_args_list[1]
        assert str(save_path / "fast_processor") in second_call[0]
        assert loaded._is_fitted is True

    @patch("versatil.data.tokenization.action_tokenizer.AutoTokenizer")
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
            "tokenizer_chain": [TokenizerType.FAST.value, TokenizerType.LANGUAGE.value],
            "use_pretrained_fast": True,
            "language_tokenizer_model": "some-model",
            "fast_vocab_size": 2048,
            "num_special_tokens_to_skip": 128,
            "vocab_size": 32000,
            "is_fitted": True,
        }
        loaded = ActionTokenizer.from_pretrained(save_path)
        mock_auto_tokenizer.from_pretrained.assert_any_call(
            save_path / "language_tokenizer"
        )
        assert loaded.language_tokenizer is not None

    @patch("versatil.data.tokenization.action_tokenizer.torch.load")
    def test_from_pretrained_logs_info(
        self, mock_torch_load, mock_auto_processor, tmp_path
    ):
        save_path = tmp_path / "tokenizer"
        save_path.mkdir(parents=True)
        (save_path / "action_tokenizer_state.pt").touch()
        mock_torch_load.return_value = {
            "tokenizer_chain": [TokenizerType.FAST.value],
            "use_pretrained_fast": True,
            "language_tokenizer_model": None,
            "fast_vocab_size": 2048,
            "num_special_tokens_to_skip": 128,
            "vocab_size": 2048,
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
            tokenizer_chain=[TokenizerType.FAST.value],
            use_pretrained_fast=True,
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
            tokenizer_chain=[TokenizerType.FAST.value],
            use_pretrained_fast=True,
            max_token_len=128,
            device=device,
        )
        chunk = action_chunk_factory(scale=0.5)
        result = tokenizer.encode_chunk(chunk)
        assert result[SampleKey.TOKENIZED_ACTIONS.value].shape == (128,)
        assert result[SampleKey.IS_PAD_ACTION.value].shape == (128,)

    def test_encode_with_pad_mask(
        self, action_chunk_factory, pad_mask_factory, device
    ):
        tokenizer = ActionTokenizer(
            tokenizer_chain=[TokenizerType.FAST.value],
            use_pretrained_fast=True,
            device=device,
        )
        chunk = action_chunk_factory(scale=0.5)
        pad_mask = pad_mask_factory(total=5, num_valid=3, as_torch=True)
        result = tokenizer.encode_chunk(chunk, is_pad_mask=pad_mask)
        assert SampleKey.TOKENIZED_ACTIONS.value in result

    def test_encode_torch_tensor_input(self, action_chunk_factory, device):
        tokenizer = ActionTokenizer(
            tokenizer_chain=[TokenizerType.FAST.value],
            use_pretrained_fast=True,
            device=device,
        )
        chunk = action_chunk_factory(as_torch=True).to(device)
        result = tokenizer.encode(chunk)
        assert result[SampleKey.TOKENIZED_ACTIONS.value].device.type == device.type


@pytest.mark.integration
class TestActionTokenizerIntegrationCustomFast:

    def test_fit_and_encode_decode(self, action_chunk_factory, device):
        tokenizer = ActionTokenizer(
            tokenizer_chain=[TokenizerType.FAST.value],
            use_pretrained_fast=False,
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
class TestActionTokenizerIntegrationLanguageMapping:

    def test_encode_decode_with_language_mapping(
        self, action_chunk_factory, device
    ):
        tokenizer = ActionTokenizer(
            tokenizer_chain=[TokenizerType.FAST.value, TokenizerType.LANGUAGE.value],
            use_pretrained_fast=True,
            language_tokenizer_model="google/bert_uncased_L-2_H-128_A-2",
            num_special_tokens_to_skip=128,
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
            tokenizer_chain=[TokenizerType.FAST.value, TokenizerType.LANGUAGE.value],
            use_pretrained_fast=True,
            language_tokenizer_model="google/bert_uncased_L-2_H-128_A-2",
            num_special_tokens_to_skip=128,
            device=device,
        )
        chunk = action_chunk_factory(scale=0.5)
        result = tokenizer.encode_chunk(chunk)
        tokens = result[SampleKey.TOKENIZED_ACTIONS.value]
        non_pad = tokens[~result[SampleKey.IS_PAD_ACTION.value]]
        if len(non_pad) > 0:
            language_vocab_size = tokenizer.language_tokenizer.vocab_size
            expected_max = (
                language_vocab_size - 1 - tokenizer.num_special_tokens_to_skip
            )
            expected_min = expected_max - tokenizer.fast_vocab_size + 1
            # EOS token is at vocab_size - 1 (after the +1 for EOS reservation)
            eos_id = tokenizer.eos_token_id
            non_eos = non_pad[non_pad != eos_id]
            if len(non_eos) > 0:
                assert non_eos.min() >= expected_min
                assert non_eos.max() <= expected_max
