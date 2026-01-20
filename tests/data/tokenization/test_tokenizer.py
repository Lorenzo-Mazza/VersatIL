"""Tests for Tokenizer class (wrapper for ObservationTokenizer and ActionTokenizer)."""

import pytest
import torch
from pathlib import Path

from versatil.configs.data.tokenizer import TokenizationConfig
from versatil.data.tokenization.tokenizer import Tokenizer, validate_tokenizer_config
from versatil.data.tokenization.observation_tokenizer import ObservationTokenizer
from versatil.data.tokenization.action_tokenizer import ActionTokenizer


@pytest.mark.unit
class TestTokenizerBasic:
    """Basic tests for Tokenizer."""

    def test_initialization_empty(self):
        """Test initialization with no tokenizers."""
        tokenizer = Tokenizer()

        assert tokenizer.observation_tokenizer is None
        assert tokenizer.action_tokenizer is None
        assert tokenizer.observation_vocab_size is None
        assert tokenizer.action_vocab_size is None

    def test_initialization_with_observation_tokenizer(
        self, simple_language_tokenizer_model, device
    ):
        """Test initialization with observation tokenizer."""
        obs_tokenizer = ObservationTokenizer(
            tokenizer_model=simple_language_tokenizer_model,
            observation_keys=["language_instruction"],
            bin_continuous_data=False,
            device=device,
        )

        tokenizer = Tokenizer(observation_tokenizer=obs_tokenizer)

        assert tokenizer.observation_tokenizer is not None
        assert tokenizer.action_tokenizer is None
        assert tokenizer.observation_vocab_size == obs_tokenizer.vocab_size

    def test_initialization_with_action_tokenizer(self, device, normalized_action_chunks):
        """Test initialization with action tokenizer."""
        action_tokenizer = ActionTokenizer(
            tokenizer_chain=["fast"],
            use_pretrained_fast=True,
            device=device,
        )

        tokenizer = Tokenizer(action_tokenizer=action_tokenizer)

        assert tokenizer.observation_tokenizer is None
        assert tokenizer.action_tokenizer is not None
        assert tokenizer.action_vocab_size == action_tokenizer.vocab_size

    def test_initialization_with_both_tokenizers(
        self, simple_language_tokenizer_model, device
    ):
        """Test initialization with both tokenizers."""
        obs_tokenizer = ObservationTokenizer(
            tokenizer_model=simple_language_tokenizer_model,
            observation_keys=["language_instruction"],
            bin_continuous_data=False,
            device=device,
        )

        action_tokenizer = ActionTokenizer(
            tokenizer_chain=["fast"],
            use_pretrained_fast=True,
            device=device,
        )

        tokenizer = Tokenizer(
            observation_tokenizer=obs_tokenizer, action_tokenizer=action_tokenizer
        )

        assert tokenizer.observation_tokenizer is not None
        assert tokenizer.action_tokenizer is not None
        assert tokenizer.observation_vocab_size is not None
        assert tokenizer.action_vocab_size is not None


@pytest.mark.unit
class TestTokenizerVocabSize:
    """Tests for vocab size properties."""

    def test_observation_vocab_size_none_when_not_set(self):
        """Test observation vocab size is None when tokenizer not set."""
        tokenizer = Tokenizer()

        assert tokenizer.observation_vocab_size is None

    def test_action_vocab_size_none_when_not_set(self):
        """Test action vocab size is None when tokenizer not set."""
        tokenizer = Tokenizer()

        assert tokenizer.action_vocab_size is None

    @pytest.mark.integration
    def test_observation_vocab_size(self, simple_language_tokenizer_model, device):
        """Test observation vocab size returns correct value."""
        obs_tokenizer = ObservationTokenizer(
            tokenizer_model=simple_language_tokenizer_model,
            observation_keys=["language_instruction"],
            bin_continuous_data=False,
            device=device,
        )

        tokenizer = Tokenizer(observation_tokenizer=obs_tokenizer)

        assert tokenizer.observation_vocab_size == obs_tokenizer.vocab_size
        assert tokenizer.observation_vocab_size > 0

    def test_action_vocab_size(self, device):
        """Test action vocab size returns correct value."""
        action_tokenizer = ActionTokenizer(
            tokenizer_chain=["fast"],
            use_pretrained_fast=True,
            fast_vocab_size=2048,
            device=device,
        )

        tokenizer = Tokenizer(action_tokenizer=action_tokenizer)

        assert tokenizer.action_vocab_size == 2048


@pytest.mark.unit
class TestTokenizerDeviceHandling:
    """Tests for device handling."""

    @pytest.mark.integration
    def test_to_device_with_observation_tokenizer(
        self, simple_language_tokenizer_model
    ):
        """Test moving observation tokenizer to device."""
        obs_tokenizer = ObservationTokenizer(
            tokenizer_model=simple_language_tokenizer_model,
            observation_keys=["language_instruction"],
            bin_continuous_data=False,
            device=torch.device("cpu"),
        )

        tokenizer = Tokenizer(observation_tokenizer=obs_tokenizer)

        result = tokenizer.to(torch.device("cpu"))

        assert result is tokenizer  # Should return self for chaining
        assert tokenizer.observation_tokenizer.device == torch.device("cpu")

    def test_to_device_with_action_tokenizer(self, device):
        """Test moving action tokenizer to device."""
        action_tokenizer = ActionTokenizer(
            tokenizer_chain=["fast"],
            use_pretrained_fast=True,
            device=torch.device("cpu"),
        )

        tokenizer = Tokenizer(action_tokenizer=action_tokenizer)

        result = tokenizer.to(device)

        assert result is tokenizer
        assert tokenizer.action_tokenizer.device == device

    @pytest.mark.integration
    def test_to_device_with_both_tokenizers(self, simple_language_tokenizer_model):
        """Test moving both tokenizers to device."""
        obs_tokenizer = ObservationTokenizer(
            tokenizer_model=simple_language_tokenizer_model,
            observation_keys=["language_instruction"],
            bin_continuous_data=False,
            device=torch.device("cpu"),
        )

        action_tokenizer = ActionTokenizer(
            tokenizer_chain=["fast"],
            use_pretrained_fast=True,
            device=torch.device("cpu"),
        )

        tokenizer = Tokenizer(
            observation_tokenizer=obs_tokenizer, action_tokenizer=action_tokenizer
        )

        target_device = torch.device("cpu")
        result = tokenizer.to(target_device)

        assert result is tokenizer
        assert tokenizer.observation_tokenizer.device == target_device
        assert tokenizer.action_tokenizer.device == target_device

    def test_to_device_empty_tokenizer(self, device):
        """Test to() with empty tokenizer (no sub-tokenizers)."""
        tokenizer = Tokenizer()

        result = tokenizer.to(device)

        assert result is tokenizer  # Should not crash


@pytest.mark.integration
class TestTokenizerSavePretrained:
    """Tests for save_pretrained functionality."""

    def test_save_empty_tokenizer(self, tmp_path):
        """Test saving empty tokenizer creates directory."""
        tokenizer = Tokenizer()

        save_path = tmp_path / "tokenizer"
        tokenizer.save_pretrained(save_path)

        assert save_path.exists()
        assert save_path.is_dir()

    def test_save_observation_tokenizer_only(
        self, simple_language_tokenizer_model, device, tmp_path
    ):
        """Test saving tokenizer with only observation tokenizer."""
        obs_tokenizer = ObservationTokenizer(
            tokenizer_model=simple_language_tokenizer_model,
            observation_keys=["language_instruction"],
            bin_continuous_data=False,
            device=device,
        )

        tokenizer = Tokenizer(observation_tokenizer=obs_tokenizer)

        save_path = tmp_path / "tokenizer"
        tokenizer.save_pretrained(save_path)

        assert save_path.exists()
        assert (save_path / "observation_tokenizer").exists()
        assert not (save_path / "action_tokenizer").exists()

    def test_save_action_tokenizer_only(self, device, tmp_path, normalized_action_chunks):
        """Test saving tokenizer with only action tokenizer."""
        action_tokenizer = ActionTokenizer(
            tokenizer_chain=["fast"],
            use_pretrained_fast=False,
            device=device,
        )
        action_tokenizer.fit(normalized_action_chunks)

        tokenizer = Tokenizer(action_tokenizer=action_tokenizer)

        save_path = tmp_path / "tokenizer"
        tokenizer.save_pretrained(save_path)

        assert save_path.exists()
        assert (save_path / "action_tokenizer").exists()
        assert not (save_path / "observation_tokenizer").exists()

    def test_save_both_tokenizers(
        self,
        simple_language_tokenizer_model,
        device,
        tmp_path,
        normalized_action_chunks,
        normalized_proprio_data,
    ):
        """Test saving tokenizer with both tokenizers."""
        obs_tokenizer = ObservationTokenizer(
            tokenizer_model=simple_language_tokenizer_model,
            observation_keys=["language_instruction", "proprio_robot_frame"],
            bin_continuous_data=True,
            num_bins=256,
            device=device,
        )
        obs_tokenizer.fit(normalized_proprio_data)

        action_tokenizer = ActionTokenizer(
            tokenizer_chain=["fast"],
            use_pretrained_fast=False,
            device=device,
        )
        action_tokenizer.fit(normalized_action_chunks)

        tokenizer = Tokenizer(
            observation_tokenizer=obs_tokenizer, action_tokenizer=action_tokenizer
        )

        save_path = tmp_path / "tokenizer"
        tokenizer.save_pretrained(save_path)

        assert save_path.exists()
        assert (save_path / "observation_tokenizer").exists()
        assert (save_path / "action_tokenizer").exists()


@pytest.mark.integration
class TestTokenizerFromPretrained:
    """Tests for from_pretrained functionality."""

    def test_from_pretrained_empty(self, tmp_path):
        """Test loading empty tokenizer."""
        # Save empty tokenizer
        tokenizer = Tokenizer()
        save_path = tmp_path / "tokenizer"
        tokenizer.save_pretrained(save_path)

        # Load it back
        loaded_tokenizer = Tokenizer.from_pretrained(save_path)

        assert loaded_tokenizer.observation_tokenizer is None
        assert loaded_tokenizer.action_tokenizer is None

    def test_from_pretrained_observation_only(
        self, simple_language_tokenizer_model, device, tmp_path
    ):
        """Test loading tokenizer with only observation tokenizer."""
        # Save tokenizer
        obs_tokenizer = ObservationTokenizer(
            tokenizer_model=simple_language_tokenizer_model,
            observation_keys=["language_instruction"],
            bin_continuous_data=False,
            device=device,
        )
        tokenizer = Tokenizer(observation_tokenizer=obs_tokenizer)
        save_path = tmp_path / "tokenizer"
        tokenizer.save_pretrained(save_path)

        # Load it back
        loaded_tokenizer = Tokenizer.from_pretrained(save_path, device=device)

        assert loaded_tokenizer.observation_tokenizer is not None
        assert loaded_tokenizer.action_tokenizer is None
        assert loaded_tokenizer.observation_vocab_size == obs_tokenizer.vocab_size

    def test_from_pretrained_action_only(
        self, device, tmp_path, normalized_action_chunks
    ):
        """Test loading tokenizer with only action tokenizer."""
        # Save tokenizer
        action_tokenizer = ActionTokenizer(
            tokenizer_chain=["fast"],
            use_pretrained_fast=False,
            device=device,
        )
        action_tokenizer.fit(normalized_action_chunks)

        tokenizer = Tokenizer(action_tokenizer=action_tokenizer)
        save_path = tmp_path / "tokenizer"
        tokenizer.save_pretrained(save_path)

        # Load it back
        loaded_tokenizer = Tokenizer.from_pretrained(save_path, device=device)

        assert loaded_tokenizer.observation_tokenizer is None
        assert loaded_tokenizer.action_tokenizer is not None
        assert loaded_tokenizer.action_vocab_size == action_tokenizer.vocab_size

    def test_from_pretrained_both_tokenizers(
        self,
        simple_language_tokenizer_model,
        device,
        tmp_path,
        normalized_action_chunks,
        normalized_proprio_data,
    ):
        """Test loading tokenizer with both tokenizers."""
        # Save tokenizer
        obs_tokenizer = ObservationTokenizer(
            tokenizer_model=simple_language_tokenizer_model,
            observation_keys=["language_instruction", "proprio_robot_frame"],
            bin_continuous_data=True,
            num_bins=256,
            device=device,
        )
        obs_tokenizer.fit(normalized_proprio_data)

        action_tokenizer = ActionTokenizer(
            tokenizer_chain=["fast"],
            use_pretrained_fast=False,
            device=device,
        )
        action_tokenizer.fit(normalized_action_chunks)

        tokenizer = Tokenizer(
            observation_tokenizer=obs_tokenizer, action_tokenizer=action_tokenizer
        )

        save_path = tmp_path / "tokenizer"
        tokenizer.save_pretrained(save_path)

        # Load it back
        loaded_tokenizer = Tokenizer.from_pretrained(save_path, device=device)

        assert loaded_tokenizer.observation_tokenizer is not None
        assert loaded_tokenizer.action_tokenizer is not None
        assert loaded_tokenizer.observation_vocab_size == obs_tokenizer.vocab_size
        assert loaded_tokenizer.action_vocab_size == action_tokenizer.vocab_size

    def test_from_pretrained_nonexistent_path(self):
        """Test from_pretrained with nonexistent path."""
        with pytest.raises(FileNotFoundError, match="Tokenizer path not found"):
            Tokenizer.from_pretrained("/nonexistent/path")


@pytest.mark.integration
class TestTokenizerRoundtrip:
    """Tests for save/load roundtrip preserves functionality."""

    def test_roundtrip_preserves_vocab_sizes(
        self,
        simple_language_tokenizer_model,
        device,
        tmp_path,
        normalized_action_chunks,
        normalized_proprio_data,
    ):
        """Test save/load roundtrip preserves vocab sizes."""
        # Create and save tokenizer
        obs_tokenizer = ObservationTokenizer(
            tokenizer_model=simple_language_tokenizer_model,
            observation_keys=["language_instruction", "proprio_robot_frame"],
            bin_continuous_data=True,
            num_bins=256,
            device=device,
        )
        obs_tokenizer.fit(normalized_proprio_data)

        action_tokenizer = ActionTokenizer(
            tokenizer_chain=["fast"],
            use_pretrained_fast=False,
            fast_vocab_size=2048,
            device=device,
        )
        action_tokenizer.fit(normalized_action_chunks)

        tokenizer = Tokenizer(
            observation_tokenizer=obs_tokenizer, action_tokenizer=action_tokenizer
        )

        save_path = tmp_path / "tokenizer"
        tokenizer.save_pretrained(save_path)

        # Load and compare
        loaded_tokenizer = Tokenizer.from_pretrained(save_path, device=device)

        assert (
            loaded_tokenizer.observation_vocab_size == tokenizer.observation_vocab_size
        )
        assert loaded_tokenizer.action_vocab_size == tokenizer.action_vocab_size



class TestTokenizerValidation:

    def test_observation_tokenizer_missing_config_raises_error(self):
        """Test that missing observation tokenizer config raises error."""
        with pytest.raises(ValueError, match="observation_tokenizer must be provided"):
            config = TokenizationConfig(
                tokenize_observations=True,
                tokenize_actions=False,
                observation_tokenizer=None,
            )
            validate_tokenizer_config(config=config)


    def test_action_tokenizer_missing_config_raises_error(self):
        """Test that missing action tokenizer config raises error."""
        with pytest.raises(ValueError, match="action_tokenizer must be provided"):
            config = TokenizationConfig(
                tokenize_observations=False,
                tokenize_actions=True,
                action_tokenizer=None,
            )
            validate_tokenizer_config(config=config)
