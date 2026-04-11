"""Tests for versatil.data.tokenization.tokenizer."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from versatil.configs.data.tokenizer import (
    ActionTokenizationConfig,
    TokenizationConfig,
)
from versatil.data.constants import TokenizerType
from versatil.data.tokenization.tokenizer import Tokenizer, validate_tokenizer_config


@pytest.fixture
def mock_observation_tokenizer():
    """Factory for a mock ObservationTokenizer."""

    def factory(vocab_size: int = 30000) -> MagicMock:
        mock = MagicMock()
        mock.vocab_size = vocab_size
        mock.device = torch.device("cpu")
        return mock

    return factory


@pytest.fixture
def mock_action_tokenizer():
    """Factory for a mock ActionTokenizer."""

    def factory(vocab_size: int = 2048) -> MagicMock:
        mock = MagicMock()
        mock.vocab_size = vocab_size
        mock.device = torch.device("cpu")
        return mock

    return factory


class TestTokenizerInit:
    def test_stores_observation_tokenizer(self, mock_observation_tokenizer):
        observation_tokenizer = mock_observation_tokenizer(vocab_size=30000)
        tokenizer = Tokenizer(observation_tokenizer=observation_tokenizer)
        assert tokenizer.observation_tokenizer is observation_tokenizer

    def test_stores_action_tokenizer(self, mock_action_tokenizer):
        action_tokenizer = mock_action_tokenizer(vocab_size=2048)
        tokenizer = Tokenizer(action_tokenizer=action_tokenizer)
        assert tokenizer.action_tokenizer is action_tokenizer

    def test_both_none_by_default(self):
        tokenizer = Tokenizer()
        assert tokenizer.observation_tokenizer is None
        assert tokenizer.action_tokenizer is None

    def test_stores_both_tokenizers(
        self, mock_observation_tokenizer, mock_action_tokenizer
    ):
        observation_tokenizer = mock_observation_tokenizer(vocab_size=30000)
        action_tokenizer = mock_action_tokenizer(vocab_size=2048)
        tokenizer = Tokenizer(
            observation_tokenizer=observation_tokenizer,
            action_tokenizer=action_tokenizer,
        )
        assert tokenizer.observation_tokenizer is observation_tokenizer
        assert tokenizer.action_tokenizer is action_tokenizer


class TestTokenizerVocabSize:
    def test_observation_vocab_size_delegates_to_tokenizer(
        self, mock_observation_tokenizer
    ):
        observation_tokenizer = mock_observation_tokenizer(vocab_size=32000)
        tokenizer = Tokenizer(observation_tokenizer=observation_tokenizer)
        assert tokenizer.observation_vocab_size == 32000

    def test_observation_vocab_size_none_when_no_tokenizer(self):
        tokenizer = Tokenizer()
        assert tokenizer.observation_vocab_size is None

    def test_action_vocab_size_delegates_to_tokenizer(self, mock_action_tokenizer):
        action_tokenizer = mock_action_tokenizer(vocab_size=1024)
        tokenizer = Tokenizer(action_tokenizer=action_tokenizer)
        assert tokenizer.action_vocab_size == 1024

    def test_action_vocab_size_none_when_no_tokenizer(self):
        tokenizer = Tokenizer()
        assert tokenizer.action_vocab_size is None


class TestTokenizerTo:
    def test_to_calls_observation_tokenizer_to(
        self, mock_observation_tokenizer, device
    ):
        observation_tokenizer = mock_observation_tokenizer()
        tokenizer = Tokenizer(observation_tokenizer=observation_tokenizer)
        tokenizer.to(device)
        observation_tokenizer.to.assert_called_once_with(device)

    def test_to_calls_action_tokenizer_to(self, mock_action_tokenizer, device):
        action_tokenizer = mock_action_tokenizer()
        tokenizer = Tokenizer(action_tokenizer=action_tokenizer)
        tokenizer.to(device)
        action_tokenizer.to.assert_called_once_with(device)

    def test_to_calls_both_tokenizers(
        self, mock_observation_tokenizer, mock_action_tokenizer, device
    ):
        observation_tokenizer = mock_observation_tokenizer()
        action_tokenizer = mock_action_tokenizer()
        tokenizer = Tokenizer(
            observation_tokenizer=observation_tokenizer,
            action_tokenizer=action_tokenizer,
        )
        tokenizer.to(device)
        observation_tokenizer.to.assert_called_once_with(device)
        action_tokenizer.to.assert_called_once_with(device)

    def test_to_returns_self(self, device):
        tokenizer = Tokenizer()
        result = tokenizer.to(device)
        assert result is tokenizer

    def test_to_skips_none_tokenizers(self, device):
        tokenizer = Tokenizer()
        tokenizer.to(device)


class TestTokenizerSavePretrained:
    def test_creates_directory(self, tmp_path):
        tokenizer = Tokenizer()
        save_path = tmp_path / "tokenizer"
        tokenizer.save_pretrained(save_path)
        assert save_path.exists()
        assert save_path.is_dir()

    def test_saves_observation_tokenizer_to_subdirectory(
        self, mock_observation_tokenizer, tmp_path
    ):
        observation_tokenizer = mock_observation_tokenizer()
        tokenizer = Tokenizer(observation_tokenizer=observation_tokenizer)
        save_path = tmp_path / "tokenizer"
        tokenizer.save_pretrained(save_path)
        observation_tokenizer.save_pretrained.assert_called_once_with(
            save_path / "observation_tokenizer"
        )

    def test_saves_action_tokenizer_to_subdirectory(
        self, mock_action_tokenizer, tmp_path
    ):
        action_tokenizer = mock_action_tokenizer()
        tokenizer = Tokenizer(action_tokenizer=action_tokenizer)
        save_path = tmp_path / "tokenizer"
        tokenizer.save_pretrained(save_path)
        action_tokenizer.save_pretrained.assert_called_once_with(
            save_path / "action_tokenizer"
        )

    def test_skips_none_observation_tokenizer(self, tmp_path):
        tokenizer = Tokenizer()
        save_path = tmp_path / "tokenizer"
        tokenizer.save_pretrained(save_path)
        assert not (save_path / "observation_tokenizer").exists()

    def test_skips_none_action_tokenizer(self, tmp_path):
        tokenizer = Tokenizer()
        save_path = tmp_path / "tokenizer"
        tokenizer.save_pretrained(save_path)
        assert not (save_path / "action_tokenizer").exists()

    def test_saves_both_tokenizers(
        self, mock_observation_tokenizer, mock_action_tokenizer, tmp_path
    ):
        observation_tokenizer = mock_observation_tokenizer()
        action_tokenizer = mock_action_tokenizer()
        tokenizer = Tokenizer(
            observation_tokenizer=observation_tokenizer,
            action_tokenizer=action_tokenizer,
        )
        save_path = tmp_path / "tokenizer"
        tokenizer.save_pretrained(save_path)
        observation_tokenizer.save_pretrained.assert_called_once()
        action_tokenizer.save_pretrained.assert_called_once()

    def test_save_logs_info_for_observation_tokenizer(
        self, mock_observation_tokenizer, tmp_path
    ):
        observation_tokenizer = mock_observation_tokenizer()
        tokenizer = Tokenizer(observation_tokenizer=observation_tokenizer)
        save_path = tmp_path / "tokenizer"
        with patch("versatil.data.tokenization.tokenizer.logging") as mock_logging:
            tokenizer.save_pretrained(save_path)
            mock_logging.info.assert_any_call(
                f"Saved observation tokenizer to {save_path / 'observation_tokenizer'}"
            )

    def test_save_logs_info_for_action_tokenizer(self, mock_action_tokenizer, tmp_path):
        action_tokenizer = mock_action_tokenizer()
        tokenizer = Tokenizer(action_tokenizer=action_tokenizer)
        save_path = tmp_path / "tokenizer"
        with patch("versatil.data.tokenization.tokenizer.logging") as mock_logging:
            tokenizer.save_pretrained(save_path)
            mock_logging.info.assert_any_call(
                f"Saved action tokenizer to {save_path / 'action_tokenizer'}"
            )


class TestTokenizerFromPretrained:
    def test_raises_file_not_found_for_nonexistent_path(self, tmp_path):
        missing = tmp_path / "missing"
        with pytest.raises(FileNotFoundError, match="Tokenizer path not found"):
            Tokenizer.from_pretrained(str(missing))

    @patch("versatil.data.tokenization.tokenizer.ActionTokenizer")
    @patch("versatil.data.tokenization.tokenizer.ObservationTokenizer")
    def test_loads_observation_tokenizer_when_directory_exists(
        self, mock_observation_class, mock_action_class, tmp_path, device
    ):
        save_path = tmp_path / "tokenizer"
        save_path.mkdir()
        observation_path = save_path / "observation_tokenizer"
        observation_path.mkdir()

        mock_observation_class.from_pretrained.return_value = MagicMock()
        loaded = Tokenizer.from_pretrained(save_path, device=device)

        mock_observation_class.from_pretrained.assert_called_once_with(
            observation_path, device=device
        )
        assert loaded.observation_tokenizer is not None

    @patch("versatil.data.tokenization.tokenizer.ActionTokenizer")
    @patch("versatil.data.tokenization.tokenizer.ObservationTokenizer")
    def test_loads_action_tokenizer_when_directory_exists(
        self, mock_observation_class, mock_action_class, tmp_path, device
    ):
        save_path = tmp_path / "tokenizer"
        save_path.mkdir()
        action_path = save_path / "action_tokenizer"
        action_path.mkdir()

        mock_action_class.from_pretrained.return_value = MagicMock()
        loaded = Tokenizer.from_pretrained(save_path, device=device)

        mock_action_class.from_pretrained.assert_called_once_with(
            action_path, device=device
        )
        assert loaded.action_tokenizer is not None

    @patch("versatil.data.tokenization.tokenizer.ActionTokenizer")
    @patch("versatil.data.tokenization.tokenizer.ObservationTokenizer")
    def test_returns_none_when_subdirectory_missing(
        self, mock_observation_class, mock_action_class, tmp_path
    ):
        save_path = tmp_path / "tokenizer"
        save_path.mkdir()
        loaded = Tokenizer.from_pretrained(save_path)
        assert loaded.observation_tokenizer is None
        assert loaded.action_tokenizer is None

    @patch("versatil.data.tokenization.tokenizer.ActionTokenizer")
    @patch("versatil.data.tokenization.tokenizer.ObservationTokenizer")
    def test_from_pretrained_logs_info(
        self, mock_observation_class, mock_action_class, tmp_path, device
    ):
        save_path = tmp_path / "tokenizer"
        save_path.mkdir()
        observation_path = save_path / "observation_tokenizer"
        observation_path.mkdir()
        action_path = save_path / "action_tokenizer"
        action_path.mkdir()

        mock_observation_class.from_pretrained.return_value = MagicMock()
        mock_action_class.from_pretrained.return_value = MagicMock()

        with patch("versatil.data.tokenization.tokenizer.logging") as mock_logging:
            Tokenizer.from_pretrained(save_path, device=device)
            assert mock_logging.info.call_count == 1
            mock_logging.info.assert_called_once_with(
                f"Loaded action tokenizer from {action_path}"
            )


class TestValidateTokenizerConfig:
    def test_tokenize_observations_without_config_raises(self):
        config = TokenizationConfig(
            tokenize_observations=True,
            observation_tokenizer=None,
        )
        with pytest.raises(ValueError, match="observation_tokenizer must be provided"):
            validate_tokenizer_config(config=config)

    def test_tokenize_actions_without_config_raises(self):
        config = TokenizationConfig(
            tokenize_actions=True,
            action_tokenizer=None,
        )
        with pytest.raises(ValueError, match="action_tokenizer must be provided"):
            validate_tokenizer_config(config=config)

    def test_invalid_tokenizer_in_chain_raises(self):
        config = TokenizationConfig(
            tokenize_actions=True,
            action_tokenizer=ActionTokenizationConfig(
                tokenizer_chain=["invalid_tokenizer"],
            ),
        )
        with pytest.raises(ValueError, match="Invalid tokenizer 'invalid_tokenizer'"):
            validate_tokenizer_config(config=config)

    def test_language_in_chain_without_model_raises(self):
        config = TokenizationConfig(
            tokenize_actions=True,
            action_tokenizer=ActionTokenizationConfig(
                tokenizer_chain=[
                    TokenizerType.FAST.value,
                    TokenizerType.LANGUAGE.value,
                ],
                language_tokenizer_model=None,
            ),
        )
        with pytest.raises(
            ValueError, match="language_tokenizer_model must be provided"
        ):
            validate_tokenizer_config(config=config)

    def test_valid_fast_only_chain_passes(self):
        config = TokenizationConfig(
            tokenize_actions=True,
            action_tokenizer=ActionTokenizationConfig(
                tokenizer_chain=[TokenizerType.FAST.value],
            ),
        )
        validate_tokenizer_config(config=config)

    def test_valid_fast_and_language_chain_passes(self):
        config = TokenizationConfig(
            tokenize_actions=True,
            action_tokenizer=ActionTokenizationConfig(
                tokenizer_chain=[
                    TokenizerType.FAST.value,
                    TokenizerType.LANGUAGE.value,
                ],
                language_tokenizer_model="some-model",
            ),
        )
        validate_tokenizer_config(config=config)

    def test_disabled_tokenization_skips_validation(self):
        config = TokenizationConfig(
            tokenize_observations=False,
            tokenize_actions=False,
            observation_tokenizer=None,
            action_tokenizer=None,
        )
        validate_tokenizer_config(config=config)

    def test_action_tokenizer_none_skips_chain_validation(self):
        config = TokenizationConfig(
            tokenize_observations=False,
            tokenize_actions=False,
            action_tokenizer=None,
        )
        validate_tokenizer_config(config=config)
