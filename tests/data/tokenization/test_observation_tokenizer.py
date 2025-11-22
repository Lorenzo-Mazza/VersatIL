"""Tests for ObservationTokenizer with binning and language prompts."""

import numpy as np
import pytest
import torch

from refactoring.data.constants import (
    LANGUAGE_KEY,
    PROPRIO_OBS_ROBOT_FRAME_KEY,
    PROPRIO_OBS_CAMERA_FRAME_KEY,
    GRIPPER_STATE_OBS_KEY,
    TOKENIZED_OBSERVATIONS_KEY,
    IS_PAD_OBSERVATION_KEY,
)
from refactoring.data.tokenization.observation_tokenizer import ObservationTokenizer


@pytest.mark.integration
class TestObservationTokenizerBasic:
    """Tests for basic observation tokenizer functionality."""

    def test_initialization(self, device, simple_language_tokenizer_model):
        """Test initialization with default parameters."""
        tokenizer = ObservationTokenizer(
            tokenizer_model=simple_language_tokenizer_model,
            observation_keys=[LANGUAGE_KEY, PROPRIO_OBS_ROBOT_FRAME_KEY],
            device=device,
        )

        assert tokenizer.tokenizer_model == simple_language_tokenizer_model
        assert tokenizer.observation_keys == [LANGUAGE_KEY, PROPRIO_OBS_ROBOT_FRAME_KEY]
        assert tokenizer.bin_continuous_data is True
        assert tokenizer.num_bins == 256
        assert tokenizer.max_token_len == 256
        assert tokenizer.device == device
        assert tokenizer.vocab_size > 0
        assert tokenizer._is_fitted is False

    def test_initialization_no_binning(self, device, simple_language_tokenizer_model):
        """Test initialization without binning."""
        tokenizer = ObservationTokenizer(
            tokenizer_model=simple_language_tokenizer_model,
            observation_keys=[LANGUAGE_KEY],
            bin_continuous_data=False,
            device=device,
        )

        assert tokenizer.bin_continuous_data is False
        assert tokenizer._is_fitted is False


@pytest.mark.integration
class TestObservationTokenizerFitting:
    """Tests for fitting observation tokenizer."""

    def test_fit_with_binning(
        self, device, simple_language_tokenizer_model, normalized_proprio_data
    ):
        """Test fitting with binning enabled."""
        tokenizer = ObservationTokenizer(
            tokenizer_model=simple_language_tokenizer_model,
            observation_keys=[LANGUAGE_KEY, PROPRIO_OBS_ROBOT_FRAME_KEY],
            bin_continuous_data=True,
            num_bins=128,
            device=device,
        )

        tokenizer.fit(normalized_proprio_data)

        assert tokenizer._is_fitted is True
        assert PROPRIO_OBS_ROBOT_FRAME_KEY in tokenizer.binning_tokenizers
        assert LANGUAGE_KEY not in tokenizer.binning_tokenizers  # Language not binned

    def test_fit_without_binning(
        self, device, simple_language_tokenizer_model, normalized_proprio_data
    ):
        """Test fitting without binning."""
        tokenizer = ObservationTokenizer(
            tokenizer_model=simple_language_tokenizer_model,
            observation_keys=[LANGUAGE_KEY],
            bin_continuous_data=False,
            device=device,
        )

        tokenizer.fit(normalized_proprio_data)

        assert tokenizer._is_fitted is True
        assert len(tokenizer.binning_tokenizers) == 0

    def test_fit_missing_keys_warns(
        self, device, simple_language_tokenizer_model, normalized_proprio_data
    ):
        """Test fitting with missing observation keys."""
        tokenizer = ObservationTokenizer(
            tokenizer_model=simple_language_tokenizer_model,
            observation_keys=[LANGUAGE_KEY, PROPRIO_OBS_ROBOT_FRAME_KEY, GRIPPER_STATE_OBS_KEY],
            bin_continuous_data=True,
            device=device,
        )

        # normalized_proprio_data doesn't have GRIPPER_STATE_OBS_KEY
        tokenizer.fit(normalized_proprio_data)

        assert tokenizer._is_fitted is True
        assert PROPRIO_OBS_ROBOT_FRAME_KEY in tokenizer.binning_tokenizers
        assert GRIPPER_STATE_OBS_KEY not in tokenizer.binning_tokenizers


@pytest.mark.integration
class TestObservationTokenizerTokenization:
    """Tests for tokenization functionality."""

    def test_tokenize_language_only(
        self, device, simple_language_tokenizer_model, language_instructions
    ):
        """Test tokenizing language observations only."""
        tokenizer = ObservationTokenizer(
            tokenizer_model=simple_language_tokenizer_model,
            observation_keys=[LANGUAGE_KEY],
            bin_continuous_data=False,
            device=device,
        )
        tokenizer.fit({})

        observations = {LANGUAGE_KEY: language_instructions}
        result = tokenizer.tokenize(observations)

        assert TOKENIZED_OBSERVATIONS_KEY in result
        assert IS_PAD_OBSERVATION_KEY in result
        assert result[TOKENIZED_OBSERVATIONS_KEY].device == device
        assert result[IS_PAD_OBSERVATION_KEY].device == device
        assert result[TOKENIZED_OBSERVATIONS_KEY].shape[0] == len(language_instructions)
        assert result[TOKENIZED_OBSERVATIONS_KEY].shape[1] == tokenizer.max_token_len

    def test_tokenize_with_proprio(
        self,
        device,
        simple_language_tokenizer_model,
        language_instructions,
        normalized_proprio_data,
    ):
        """Test tokenizing with language and proprioceptive data."""
        tokenizer = ObservationTokenizer(
            tokenizer_model=simple_language_tokenizer_model,
            observation_keys=[LANGUAGE_KEY, PROPRIO_OBS_ROBOT_FRAME_KEY],
            bin_continuous_data=True,
            num_bins=128,
            device=device,
        )
        tokenizer.fit(normalized_proprio_data)

        batch_size = 5
        observations = {
            LANGUAGE_KEY: language_instructions[:batch_size],
            PROPRIO_OBS_ROBOT_FRAME_KEY: torch.from_numpy(
                normalized_proprio_data[PROPRIO_OBS_ROBOT_FRAME_KEY][:batch_size]
            ),
        }
        result = tokenizer.tokenize(observations)

        assert TOKENIZED_OBSERVATIONS_KEY in result
        assert IS_PAD_OBSERVATION_KEY in result
        assert result[TOKENIZED_OBSERVATIONS_KEY].shape == (batch_size, tokenizer.max_token_len)
        assert result[IS_PAD_OBSERVATION_KEY].shape == (batch_size, tokenizer.max_token_len)

    def test_tokenize_without_binning(
        self,
        device,
        simple_language_tokenizer_model,
        language_instructions,
        normalized_proprio_data,
    ):
        """Test tokenizing without binning (raw float values)."""
        tokenizer = ObservationTokenizer(
            tokenizer_model=simple_language_tokenizer_model,
            observation_keys=[LANGUAGE_KEY, PROPRIO_OBS_ROBOT_FRAME_KEY],
            bin_continuous_data=False,
            device=device,
        )
        tokenizer.fit({})

        batch_size = 3
        observations = {
            LANGUAGE_KEY: language_instructions[:batch_size],
            PROPRIO_OBS_ROBOT_FRAME_KEY: torch.from_numpy(
                normalized_proprio_data[PROPRIO_OBS_ROBOT_FRAME_KEY][:batch_size]
            ),
        }
        result = tokenizer.tokenize(observations)

        assert TOKENIZED_OBSERVATIONS_KEY in result
        assert IS_PAD_OBSERVATION_KEY in result
        assert result[TOKENIZED_OBSERVATIONS_KEY].shape == (batch_size, tokenizer.max_token_len)

    def test_tokenize_before_fit_raises_error(
        self, device, simple_language_tokenizer_model, language_instructions
    ):
        """Test that tokenizing before fitting raises error."""
        tokenizer = ObservationTokenizer(
            tokenizer_model=simple_language_tokenizer_model,
            observation_keys=[LANGUAGE_KEY],
            device=device,
        )

        observations = {LANGUAGE_KEY: language_instructions}
        with pytest.raises(RuntimeError, match="Tokenizer must be fitted"):
            tokenizer.tokenize(observations)


@pytest.mark.integration
class TestObservationTokenizerPromptBuilding:
    """Tests for prompt building logic."""

    def test_build_prompts_language_only(
        self, device, simple_language_tokenizer_model, language_instructions
    ):
        """Test prompt building with language only."""
        tokenizer = ObservationTokenizer(
            tokenizer_model=simple_language_tokenizer_model,
            observation_keys=[LANGUAGE_KEY],
            bin_continuous_data=False,
            device=device,
        )
        tokenizer.fit({})

        observations = {LANGUAGE_KEY: language_instructions[:2]}
        prompts = tokenizer._build_prompts(observations)

        assert len(prompts) == 2
        assert all("TaskSpace:" in p for p in prompts)
        assert all(p.endswith(";\n") for p in prompts)

    def test_build_prompts_with_proprio(
        self,
        device,
        simple_language_tokenizer_model,
        language_instructions,
        normalized_proprio_data,
    ):
        """Test prompt building with language and proprio."""
        tokenizer = ObservationTokenizer(
            tokenizer_model=simple_language_tokenizer_model,
            observation_keys=[LANGUAGE_KEY, PROPRIO_OBS_ROBOT_FRAME_KEY],
            bin_continuous_data=True,
            num_bins=64,
            device=device,
        )
        tokenizer.fit(normalized_proprio_data)

        batch_size = 2
        observations = {
            LANGUAGE_KEY: language_instructions[:batch_size],
            PROPRIO_OBS_ROBOT_FRAME_KEY: normalized_proprio_data[
                PROPRIO_OBS_ROBOT_FRAME_KEY
            ][:batch_size],
        }
        prompts = tokenizer._build_prompts(observations)

        assert len(prompts) == batch_size
        assert all("TaskSpace:" in p for p in prompts)
        assert all("proprio robot frame:" in p for p in prompts)
        assert all(p.endswith(";\n") for p in prompts)

    def test_build_prompts_multiple_keys(
        self,
        device,
        simple_language_tokenizer_model,
        language_instructions,
        normalized_proprio_data,
    ):
        """Test prompt building with multiple observation keys."""
        tokenizer = ObservationTokenizer(
            tokenizer_model=simple_language_tokenizer_model,
            observation_keys=[
                LANGUAGE_KEY,
                PROPRIO_OBS_ROBOT_FRAME_KEY,
                PROPRIO_OBS_CAMERA_FRAME_KEY,
            ],
            bin_continuous_data=True,
            device=device,
        )
        tokenizer.fit(normalized_proprio_data)

        observations = {
            LANGUAGE_KEY: [language_instructions[0]],
            PROPRIO_OBS_ROBOT_FRAME_KEY: normalized_proprio_data[
                PROPRIO_OBS_ROBOT_FRAME_KEY
            ][:1],
            PROPRIO_OBS_CAMERA_FRAME_KEY: normalized_proprio_data[
                PROPRIO_OBS_CAMERA_FRAME_KEY
            ][:1],
        }
        prompts = tokenizer._build_prompts(observations)

        assert len(prompts) == 1
        assert "TaskSpace:" in prompts[0]
        assert "proprio robot frame:" in prompts[0]
        assert "proprio camera frame:" in prompts[0]


@pytest.mark.unit
class TestObservationTokenizerSerialization:
    """Tests for save/load functionality."""

    @pytest.mark.integration
    def test_save_and_load(
        self,
        device,
        simple_language_tokenizer_model,
        normalized_proprio_data,
        tmp_path,
    ):
        """Test saving and loading tokenizer."""
        tokenizer = ObservationTokenizer(
            tokenizer_model=simple_language_tokenizer_model,
            observation_keys=[LANGUAGE_KEY, PROPRIO_OBS_ROBOT_FRAME_KEY],
            bin_continuous_data=True,
            num_bins=128,
            max_token_len=512,
            device=device,
        )
        tokenizer.fit(normalized_proprio_data)

        save_path = tmp_path / "observation_tokenizer"
        tokenizer.save_pretrained(save_path)

        assert save_path.exists()
        assert (save_path / "observation_tokenizer_state.pt").exists()
        assert (save_path / "language_tokenizer").exists()

        # Load and verify
        loaded = ObservationTokenizer.from_pretrained(save_path, device=device)
        assert loaded.tokenizer_model == tokenizer.tokenizer_model
        assert loaded.observation_keys == tokenizer.observation_keys
        assert loaded.bin_continuous_data == tokenizer.bin_continuous_data
        assert loaded.num_bins == tokenizer.num_bins
        assert loaded.max_token_len == tokenizer.max_token_len
        assert loaded.vocab_size == tokenizer.vocab_size
        assert loaded._is_fitted is True
        assert len(loaded.binning_tokenizers) == len(tokenizer.binning_tokenizers)

    @pytest.mark.integration
    def test_state_dict(
        self, device, simple_language_tokenizer_model, normalized_proprio_data
    ):
        """Test state_dict returns expected keys."""
        tokenizer = ObservationTokenizer(
            tokenizer_model=simple_language_tokenizer_model,
            observation_keys=[LANGUAGE_KEY, PROPRIO_OBS_ROBOT_FRAME_KEY],
            bin_continuous_data=True,
            device=device,
        )
        tokenizer.fit(normalized_proprio_data)

        state = tokenizer.state_dict()

        assert "tokenizer_model" in state
        assert "observation_keys" in state
        assert "bin_continuous_data" in state
        assert "num_bins" in state
        assert "max_token_len" in state
        assert "vocab_size" in state
        assert "binning_tokenizers" in state
        assert "is_fitted" in state
        assert state["is_fitted"] is True


@pytest.mark.unit
class TestObservationTokenizerDeviceHandling:
    """Tests for device handling."""

    def test_to_device(self, device, simple_language_tokenizer_model):
        """Test moving tokenizer to device."""
        tokenizer = ObservationTokenizer(
            tokenizer_model=simple_language_tokenizer_model,
            observation_keys=[LANGUAGE_KEY],
            device=torch.device("cpu"),
        )

        result = tokenizer.to(device)

        assert result is tokenizer  # Should return self for chaining
        assert tokenizer.device == device

    @pytest.mark.integration
    def test_tokenize_respects_device(
        self,
        device,
        simple_language_tokenizer_model,
        language_instructions,
        normalized_proprio_data,
    ):
        """Test that tokenized outputs are on correct device."""
        tokenizer = ObservationTokenizer(
            tokenizer_model=simple_language_tokenizer_model,
            observation_keys=[LANGUAGE_KEY, PROPRIO_OBS_ROBOT_FRAME_KEY],
            bin_continuous_data=True,
            device=device,
        )
        tokenizer.fit(normalized_proprio_data)

        observations = {
            LANGUAGE_KEY: [language_instructions[0]],
            PROPRIO_OBS_ROBOT_FRAME_KEY: normalized_proprio_data[
                PROPRIO_OBS_ROBOT_FRAME_KEY
            ][:1],
        }
        result = tokenizer.tokenize(observations)

        assert result[TOKENIZED_OBSERVATIONS_KEY].device == device
        assert result[IS_PAD_OBSERVATION_KEY].device == device


@pytest.mark.integration
class TestObservationTokenizerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_tokenize_single_sample(
        self,
        device,
        simple_language_tokenizer_model,
        language_instructions,
        normalized_proprio_data,
    ):
        """Test tokenizing a single sample (not batched)."""
        tokenizer = ObservationTokenizer(
            tokenizer_model=simple_language_tokenizer_model,
            observation_keys=[LANGUAGE_KEY, PROPRIO_OBS_ROBOT_FRAME_KEY],
            bin_continuous_data=True,
            device=device,
        )
        tokenizer.fit(normalized_proprio_data)

        observations = {
            LANGUAGE_KEY: [language_instructions[0]],
            PROPRIO_OBS_ROBOT_FRAME_KEY: normalized_proprio_data[
                PROPRIO_OBS_ROBOT_FRAME_KEY
            ][0:1],
        }
        result = tokenizer.tokenize(observations)

        assert result[TOKENIZED_OBSERVATIONS_KEY].shape[0] == 1
        assert result[TOKENIZED_OBSERVATIONS_KEY].shape[1] == tokenizer.max_token_len

    def test_tokenize_missing_observation_key(
        self, device, simple_language_tokenizer_model, language_instructions
    ):
        """Test tokenizing when some observation keys are missing."""
        tokenizer = ObservationTokenizer(
            tokenizer_model=simple_language_tokenizer_model,
            observation_keys=[LANGUAGE_KEY, PROPRIO_OBS_ROBOT_FRAME_KEY],
            bin_continuous_data=False,
            device=device,
        )
        tokenizer.fit({})

        # Only provide language, not proprio
        observations = {LANGUAGE_KEY: language_instructions[:2]}
        result = tokenizer.tokenize(observations)

        assert TOKENIZED_OBSERVATIONS_KEY in result
        assert result[TOKENIZED_OBSERVATIONS_KEY].shape[0] == 2

    def test_tokenize_numpy_input(
        self,
        device,
        simple_language_tokenizer_model,
        language_instructions,
        normalized_proprio_data,
    ):
        """Test tokenizing with numpy array input."""
        tokenizer = ObservationTokenizer(
            tokenizer_model=simple_language_tokenizer_model,
            observation_keys=[LANGUAGE_KEY, PROPRIO_OBS_ROBOT_FRAME_KEY],
            bin_continuous_data=True,
            device=device,
        )
        tokenizer.fit(normalized_proprio_data)

        observations = {
            LANGUAGE_KEY: [language_instructions[0]],
            PROPRIO_OBS_ROBOT_FRAME_KEY: normalized_proprio_data[
                PROPRIO_OBS_ROBOT_FRAME_KEY
            ][:1],  # NumPy array
        }
        result = tokenizer.tokenize(observations)

        assert TOKENIZED_OBSERVATIONS_KEY in result
        assert result[TOKENIZED_OBSERVATIONS_KEY].device == device

    def test_load_nonexistent_path_raises_error(self, device):
        """Test loading from nonexistent path raises error."""
        with pytest.raises(FileNotFoundError, match="Tokenizer path not found"):
            ObservationTokenizer.from_pretrained("/nonexistent/path", device=device)