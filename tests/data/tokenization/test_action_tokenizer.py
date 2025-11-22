"""Tests for ActionTokenizer with FAST and language vocab mapping."""

import numpy as np
import pytest
import torch

from refactoring.data.constants import TokenizerType, TOKENIZED_ACTIONS_KEY, IS_PAD_ACTION_KEY
from refactoring.data.tokenization.action_tokenizer import ActionTokenizer


@pytest.mark.integration
class TestActionTokenizerFASTOnly:
    """Tests for ActionTokenizer with FAST tokenization only."""

    def test_initialization_pretrained_fast_only(self, device):
        """Test initialization with pretrained FAST weights."""
        tokenizer = ActionTokenizer(
            tokenizer_chain=[TokenizerType.FAST.value],
            use_pretrained_fast=True,
            fast_vocab_size=2048,
            device=device,
        )

        assert tokenizer.tokenizer_chain == [TokenizerType.FAST.value]
        assert tokenizer.use_pretrained_fast is True
        assert tokenizer.fast_processor is not None
        assert tokenizer.language_tokenizer is None
        assert tokenizer._is_fitted is True
        assert tokenizer.vocab_size == 2048

    def test_encode_decode_pretrained_fast(self, device, normalized_action_chunks):
        """Test encode/decode roundtrip with pretrained FAST."""
        tokenizer = ActionTokenizer(
            tokenizer_chain=[TokenizerType.FAST.value],
            use_pretrained_fast=True,
            device=device,
        )

        # Encode batch of chunks (N, T, D)
        result = tokenizer.encode(normalized_action_chunks)

        assert TOKENIZED_ACTIONS_KEY in result
        assert IS_PAD_ACTION_KEY in result
        tokens = result[TOKENIZED_ACTIONS_KEY]
        assert tokens.device == device
        assert tokens.dtype == torch.long

        # Decode
        decoded = tokenizer.decode(tokens)
        assert isinstance(decoded, np.ndarray)
        assert decoded.shape == normalized_action_chunks.shape
        # Assert decoded actions are close to original
        np.testing.assert_allclose(decoded, normalized_action_chunks, atol=1e-1)

    def test_encode_with_padding_mask(self, device, normalized_action_chunks):
        """Test encoding filters out padded actions."""
        tokenizer = ActionTokenizer(
            tokenizer_chain=[TokenizerType.FAST.value],
            use_pretrained_fast=True,
            device=device,
        )

        # Create padding mask (T,)
        single_chunk = normalized_action_chunks[0]  # (T, D)
        is_pad = torch.tensor([False, False, True, True, True], dtype=torch.bool)

        result = tokenizer.encode(single_chunk, is_pad_mask=is_pad)

        # Should only tokenize first 2 timesteps
        assert TOKENIZED_ACTIONS_KEY in result
        assert IS_PAD_ACTION_KEY in result

    def test_cannot_fit_pretrained(self, device, normalized_action_chunks):
        """Test that fitting raises error when using pretrained weights."""
        tokenizer = ActionTokenizer(
            tokenizer_chain=[TokenizerType.FAST.value],
            use_pretrained_fast=True,
            device=device,
        )

        with pytest.raises(ValueError, match="Cannot fit when use_pretrained_fast=True"):
            tokenizer.fit(normalized_action_chunks)


@pytest.mark.integration
class TestActionTokenizerCustomFAST:
    """Tests for ActionTokenizer with custom FAST fitting."""

    def test_initialization_custom_fast(self, device):
        """Test initialization for custom FAST fitting."""
        tokenizer = ActionTokenizer(
            tokenizer_chain=[TokenizerType.FAST.value],
            use_pretrained_fast=False,
            fast_vocab_size=1024,
            device=device,
        )

        assert tokenizer.use_pretrained_fast is False
        assert tokenizer.fast_processor is not None
        assert tokenizer._is_fitted is False
        assert tokenizer.vocab_size is None  # Not set until fitted

    def test_fit_and_encode_decode(self, device, normalized_action_chunks):
        """Test fitting and encode/decode roundtrip."""
        tokenizer = ActionTokenizer(
            tokenizer_chain=[TokenizerType.FAST.value],
            use_pretrained_fast=False,
            fast_vocab_size=1024,
            device=device,
        )

        # Fit on action chunks (N, T, D)
        tokenizer.fit(normalized_action_chunks)
        assert tokenizer._is_fitted is True
        assert tokenizer.vocab_size == 1024
        # Encode entire batch of chunks
        result = tokenizer.encode(normalized_action_chunks[0]) # Single chunk (T, D)
        tokens = result[TOKENIZED_ACTIONS_KEY]
        decoded = tokenizer.decode(tokens)
        assert decoded.shape == normalized_action_chunks[0].shape
        # Assert decoded actions are close to original
        np.testing.assert_allclose(decoded, normalized_action_chunks[0], atol=1e-1)

    def test_encode_before_fit_raises_error(self, device, normalized_action_chunks):
        """Test that encoding before fitting raises error."""
        tokenizer = ActionTokenizer(
            tokenizer_chain=[TokenizerType.FAST.value],
            use_pretrained_fast=False,
            device=device,
        )

        single_chunk = normalized_action_chunks[0]
        with pytest.raises(RuntimeError, match="Tokenizer must be fitted"):
            tokenizer.encode(single_chunk)

    def test_decode_before_fit_raises_error(self, device):
        """Test that decoding before fitting raises error."""
        tokenizer = ActionTokenizer(
            tokenizer_chain=[TokenizerType.FAST.value],
            use_pretrained_fast=False,
            device=device,
        )

        with pytest.raises(RuntimeError, match="Tokenizer must be fitted"):
            tokenizer.decode([1, 2, 3])


@pytest.mark.integration
class TestActionTokenizerWithLanguageMapping:
    """Tests for ActionTokenizer with FAST + language vocab mapping."""

    def test_initialization_with_language_mapping(
        self, device, simple_language_tokenizer_model
    ):
        """Test initialization with language tokenizer mapping."""
        tokenizer = ActionTokenizer(
            tokenizer_chain=[TokenizerType.FAST.value, TokenizerType.LANGUAGE.value],
            use_pretrained_fast=True,
            language_tokenizer_model=simple_language_tokenizer_model,
            fast_vocab_size=2048,
            device=device,
        )

        assert tokenizer.fast_processor is not None
        assert tokenizer.language_tokenizer is not None
        assert tokenizer.vocab_size == tokenizer.language_tokenizer.vocab_size
        assert tokenizer._is_fitted is True

    def test_initialization_without_language_model_raises_error(self, device):
        """Test that language tokenizer without model raises error."""
        with pytest.raises(
            ValueError, match="language_tokenizer_model must be provided"
        ):
            ActionTokenizer(
                tokenizer_chain=[TokenizerType.FAST.value, TokenizerType.LANGUAGE.value],
                use_pretrained_fast=True,
                language_tokenizer_model=None,
                device=device,
            )

    def test_initialization_with_small_language_vocab_raises_error(self, device):
        """Test that language tokenizer with insufficient vocab size raises error."""
        # Use tiny BERT which has ~30k vocab, try to use more FAST tokens than available
        with pytest.raises(ValueError, match="vocab size .* is too small"):
            ActionTokenizer(
                tokenizer_chain=[TokenizerType.FAST.value, TokenizerType.LANGUAGE.value],
                use_pretrained_fast=True,
                language_tokenizer_model="google/bert_uncased_L-2_H-128_A-2",
                fast_vocab_size=35000,  # Larger than vocab size (~30k)
                num_special_tokens_to_skip=128,
                device=device,
            )

    def test_encode_decode_with_language_mapping(
        self, device, simple_language_tokenizer_model, normalized_action_chunks
    ):
        """Test encode/decode roundtrip with language vocab mapping."""
        tokenizer = ActionTokenizer(
            tokenizer_chain=[TokenizerType.FAST.value, TokenizerType.LANGUAGE.value],
            use_pretrained_fast=True,
            language_tokenizer_model=simple_language_tokenizer_model,
            fast_vocab_size=2048,
            device=device,
        )

        result = tokenizer.encode(normalized_action_chunks)
        tokens = result[TOKENIZED_ACTIONS_KEY]

        # Tokens should be in language vocab range
        assert tokens.max() < tokenizer.vocab_size
        assert tokens.min() >= 0

        # Decode should work
        decoded = tokenizer.decode(tokens)
        assert decoded.shape == normalized_action_chunks.shape
        # Assert decoded actions are close to original
        np.testing.assert_allclose(decoded, normalized_action_chunks, atol=1e-1)

    def test_mapping_to_end_of_language_vocab(
        self, device, simple_language_tokenizer_model, normalized_action_chunks
    ):
        """Test that FAST tokens are mapped to END of language vocab."""
        tokenizer = ActionTokenizer(
            tokenizer_chain=[TokenizerType.FAST.value, TokenizerType.LANGUAGE.value],
            use_pretrained_fast=True,
            language_tokenizer_model=simple_language_tokenizer_model,
            fast_vocab_size=2048,
            num_special_tokens_to_skip=128,
            device=device,
        )

        single_chunk = normalized_action_chunks[0]
        result = tokenizer.encode(single_chunk)
        tokens = result[TOKENIZED_ACTIONS_KEY]

        # Tokens should be in the high end of vocab (excluding padding)
        non_pad_tokens = tokens[~result[IS_PAD_ACTION_KEY]]
        lang_vocab_size = tokenizer.language_tokenizer.vocab_size

        # FAST tokens mapped to: lang_vocab_size - 1 - num_special - fast_token_id
        # So they should be in range [lang_vocab_size - 1 - num_special - (fast_vocab-1), lang_vocab_size - 1 - num_special]
        expected_max = lang_vocab_size - 1 - tokenizer.num_special_tokens_to_skip
        expected_min = (
            expected_max - tokenizer.fast_vocab_size + 1
        )

        if len(non_pad_tokens) > 0:
            assert non_pad_tokens.min() >= expected_min
            assert non_pad_tokens.max() <= expected_max


@pytest.mark.unit
class TestActionTokenizerSerialization:
    """Tests for ActionTokenizer save/load."""

    @pytest.mark.integration
    def test_save_pretrained_custom_fast(
        self, device, tmp_path, normalized_action_chunks
    ):
        """Test saving custom FAST tokenizer."""
        tokenizer = ActionTokenizer(
            tokenizer_chain=[TokenizerType.FAST.value],
            use_pretrained_fast=False,
            device=device,
        )
        tokenizer.fit(normalized_action_chunks)

        save_path = tmp_path / "action_tokenizer"
        tokenizer.save_pretrained(save_path)

        assert save_path.exists()
        assert (save_path / "action_tokenizer_state.pt").exists()
        assert (save_path / "fast_processor").exists()

    def test_save_before_fit_raises_error(self, device, tmp_path):
        """Test that saving before fitting raises error."""
        tokenizer = ActionTokenizer(
            tokenizer_chain=[TokenizerType.FAST.value],
            use_pretrained_fast=False,
            device=device,
        )

        save_path = tmp_path / "action_tokenizer"
        with pytest.raises(RuntimeError, match="Cannot save unfitted tokenizer"):
            tokenizer.save_pretrained(save_path)

    @pytest.mark.integration
    def test_state_dict(self, device):
        """Test state_dict returns expected keys."""
        tokenizer = ActionTokenizer(
            tokenizer_chain=[TokenizerType.FAST.value],
            use_pretrained_fast=True,
            fast_vocab_size=2048,
            device=device,
        )

        state = tokenizer.state_dict()

        assert "tokenizer_chain" in state
        assert "use_pretrained_fast" in state
        assert "fast_vocab_size" in state
        assert "is_fitted" in state
        assert state["tokenizer_chain"] == [TokenizerType.FAST.value]
        assert state["use_pretrained_fast"] is True
        assert state["is_fitted"] is True


@pytest.mark.unit
class TestActionTokenizerDeviceHandling:
    """Tests for device handling."""

    def test_to_device(self, device):
        """Test moving tokenizer to device."""
        tokenizer = ActionTokenizer(
            tokenizer_chain=[TokenizerType.FAST.value],
            use_pretrained_fast=True,
            device=torch.device("cpu"),
        )

        result = tokenizer.to(device)

        assert result is tokenizer  # Should return self for chaining
        assert tokenizer.device == device

    @pytest.mark.integration
    def test_encode_respects_device(self, device, normalized_action_chunks):
        """Test that encoded tokens are on correct device."""
        tokenizer = ActionTokenizer(
            tokenizer_chain=[TokenizerType.FAST.value],
            use_pretrained_fast=True,
            device=device,
        )

        single_chunk = normalized_action_chunks[0]
        result = tokenizer.encode(single_chunk)

        assert result[TOKENIZED_ACTIONS_KEY].device == device
        assert result[IS_PAD_ACTION_KEY].device == device


@pytest.mark.integration
class TestActionTokenizerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_encode_torch_tensor_input(self, device, normalized_action_chunks):
        """Test encoding with torch.Tensor input."""
        tokenizer = ActionTokenizer(
            tokenizer_chain=[TokenizerType.FAST.value],
            use_pretrained_fast=True,
            device=device,
        )

        action_tensor = torch.from_numpy(normalized_action_chunks[0]).to(device)
        result = tokenizer.encode(action_tensor)

        assert TOKENIZED_ACTIONS_KEY in result
        assert result[TOKENIZED_ACTIONS_KEY].device == device

    def test_decode_batch(self, device, normalized_action_chunks):
        """Test decoding a batch of token sequences."""
        tokenizer = ActionTokenizer(
            tokenizer_chain=[TokenizerType.FAST.value],
            use_pretrained_fast=True,
            device=device,
        )

        # Encode to get valid tokens
        result = tokenizer.encode(normalized_action_chunks)
        tokens = result[TOKENIZED_ACTIONS_KEY]

        # Decode batch
        reconstructed = tokenizer.decode(tokens)

        assert isinstance(reconstructed, np.ndarray)
        assert reconstructed.shape == normalized_action_chunks.shape
        # Assert decoded actions are close to original
        np.testing.assert_allclose(reconstructed, normalized_action_chunks, atol=1e-1)

    def test_encode_with_padding(self, device, normalized_action_chunks):
        """Test that encoding pads to max_token_len."""
        max_len = 512
        tokenizer = ActionTokenizer(
            tokenizer_chain=[TokenizerType.FAST.value],
            use_pretrained_fast=True,
            max_token_len=max_len,
            device=device,
        )

        result = tokenizer.encode(normalized_action_chunks)

        # Shape should be (N, max_token_len)
        assert result[TOKENIZED_ACTIONS_KEY].shape == (len(normalized_action_chunks), max_len)
        assert result[IS_PAD_ACTION_KEY].shape == (len(normalized_action_chunks), max_len)