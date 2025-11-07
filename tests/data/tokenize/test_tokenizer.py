"""Tests for Tokenizer (main class managing multiple tokenizers)."""

import json
import numpy as np
import pytest
import torch
from pathlib import Path

from refactoring.data.constants import ACTION_KEY
from refactoring.data.tokenize.tokenizer import Tokenizer
from refactoring.data.tokenize.action_tokenizer import ActionTokenizer
from refactoring.data.tokenize.binning_tokenizer import BinningTokenizer


@pytest.fixture
def action_chunks():
    """Generate synthetic action chunks."""
    np.random.seed(42)
    return np.random.randn(10, 5, 7).astype(np.float32) * 0.5


@pytest.fixture
def normalized_proprio():
    """Generate synthetic normalized proprioceptive data."""
    np.random.seed(42)
    return {
        "proprio_robot_frame": np.random.randn(100, 7).astype(np.float32) * 0.5,
        "proprio_camera_frame": np.random.randn(100, 7).astype(np.float32) * 0.5,
    }


@pytest.fixture
def device():
    """Device for testing."""
    return torch.device("cpu")


@pytest.mark.unit
class TestTokenizerBasic:
    """Basic tests for Tokenizer."""

    def test_initialization(self, device):
        """Test initialization."""
        tokenizer = Tokenizer(device=device)

        assert tokenizer.device == device
        assert tokenizer.tokenizers == {}

    def test_initialization_default_device(self):
        """Test initialization with default device."""
        tokenizer = Tokenizer()

        assert tokenizer.device == torch.device("cpu")


@pytest.mark.integration
class TestTokenizerActionFitting:
    """Tests for action tokenizer fitting."""

    def test_fit_action_tokenizer_pretrained(self, device, action_chunks):
        """Test fitting action tokenizer with pretrained weights."""
        tokenizer = Tokenizer(device=device)
        tokenizer.fit_action_tokenizer(action_chunks, use_pretrained_weights=True)

        assert ACTION_KEY in tokenizer.tokenizers
        assert isinstance(tokenizer.tokenizers[ACTION_KEY], ActionTokenizer)
        assert tokenizer.tokenizers[ACTION_KEY].use_pretrained_weights is True

    def test_fit_action_tokenizer_custom(self, device, action_chunks):
        """Test fitting action tokenizer with custom weights."""
        tokenizer = Tokenizer(device=device)
        tokenizer.fit_action_tokenizer(action_chunks, use_pretrained_weights=False)

        assert ACTION_KEY in tokenizer.tokenizers
        assert isinstance(tokenizer.tokenizers[ACTION_KEY], ActionTokenizer)
        assert tokenizer.tokenizers[ACTION_KEY].use_pretrained_weights is False
        assert tokenizer.tokenizers[ACTION_KEY]._is_fitted is True


@pytest.mark.unit
class TestTokenizerProprioFitting:
    """Tests for proprioceptive tokenizer fitting."""

    def test_fit_proprio_tokenizer(self, device, normalized_proprio):
        """Test fitting proprioceptive tokenizer."""
        tokenizer = Tokenizer(device=device)
        tokenizer.fit_proprio_tokenizer(normalized_proprio, num_bins=256)

        assert "proprio_robot_frame" in tokenizer.tokenizers
        assert "proprio_camera_frame" in tokenizer.tokenizers
        assert isinstance(tokenizer.tokenizers["proprio_robot_frame"], BinningTokenizer)
        assert isinstance(tokenizer.tokenizers["proprio_camera_frame"], BinningTokenizer)
        assert tokenizer.tokenizers["proprio_robot_frame"]._is_fitted is True

    def test_fit_proprio_tokenizer_custom_bins(self, device, normalized_proprio):
        """Test fitting proprioceptive tokenizer with custom bins."""
        tokenizer = Tokenizer(device=device)
        tokenizer.fit_proprio_tokenizer(normalized_proprio, num_bins=128)

        assert tokenizer.tokenizers["proprio_robot_frame"].num_bins == 128


@pytest.mark.integration
class TestTokenizerEncodeDecodeAction:
    """Tests for tokenization of action data."""

    def test_tokenize_action_data(self, device, action_chunks):
        """Test tokenizing action data."""
        tokenizer = Tokenizer(device=device)
        tokenizer.fit_action_tokenizer(action_chunks, use_pretrained_weights=True)

        data = {ACTION_KEY: action_chunks}
        tokenized = tokenizer.tokenize(data)

        assert ACTION_KEY in tokenized
        assert isinstance(tokenized[ACTION_KEY], list)

    def test_detokenize_action_data(self, device, action_chunks):
        """Test detokenizing action data."""
        tokenizer = Tokenizer(device=device)
        tokenizer.fit_action_tokenizer(action_chunks, use_pretrained_weights=True)

        data = {ACTION_KEY: action_chunks}
        tokenized = tokenizer.tokenize(data)
        detokenized = tokenizer.detokenize(tokenized)

        assert ACTION_KEY in detokenized
        assert isinstance(detokenized[ACTION_KEY], np.ndarray)

    def test_tokenize_passthrough_unknown_keys(self, device, action_chunks):
        """Test that unknown keys are passed through."""
        tokenizer = Tokenizer(device=device)
        tokenizer.fit_action_tokenizer(action_chunks, use_pretrained_weights=True)

        data = {
            ACTION_KEY: action_chunks,
            "unknown_key": np.array([1, 2, 3]),
        }
        tokenized = tokenizer.tokenize(data)

        assert "unknown_key" in tokenized
        assert np.array_equal(tokenized["unknown_key"], data["unknown_key"])


@pytest.mark.unit
class TestTokenizerEncodeDecodeProprio:
    """Tests for tokenization of proprioceptive data."""

    def test_tokenize_proprio_data(self, device, normalized_proprio):
        """Test tokenizing proprio data."""
        tokenizer = Tokenizer(device=device)
        tokenizer.fit_proprio_tokenizer(normalized_proprio, num_bins=256)

        tokenized = tokenizer.tokenize(normalized_proprio)

        assert "proprio_robot_frame" in tokenized
        assert "proprio_camera_frame" in tokenized
        assert isinstance(tokenized["proprio_robot_frame"], torch.Tensor)

    def test_detokenize_proprio_data(self, device, normalized_proprio):
        """Test detokenizing proprio data."""
        tokenizer = Tokenizer(device=device)
        tokenizer.fit_proprio_tokenizer(normalized_proprio, num_bins=256)

        tokenized = tokenizer.tokenize(normalized_proprio)
        detokenized = tokenizer.detokenize(tokenized)

        assert "proprio_robot_frame" in detokenized
        assert "proprio_camera_frame" in detokenized
        assert isinstance(detokenized["proprio_robot_frame"], torch.Tensor)


@pytest.mark.unit
class TestTokenizerDeviceHandling:
    """Tests for device handling."""

    def test_to_device(self, device):
        """Test moving tokenizer to device."""
        tokenizer = Tokenizer(device=torch.device("cpu"))

        result = tokenizer.to(device)

        assert result is tokenizer  # Should return self
        assert tokenizer.device == device

    def test_to_device_moves_sub_tokenizers(self, device, normalized_proprio):
        """Test that to() moves all sub-tokenizers."""
        tokenizer = Tokenizer(device=torch.device("cpu"))
        tokenizer.fit_proprio_tokenizer(normalized_proprio, num_bins=256)

        tokenizer.to(device)

        for sub_tok in tokenizer.tokenizers.values():
            assert sub_tok.device == device


@pytest.mark.unit
class TestTokenizerStateDictBasic:
    """Tests for state_dict (basic functionality, no HuggingFace)."""

    def test_state_dict_empty(self, device):
        """Test state_dict with no fitted tokenizers."""
        tokenizer = Tokenizer(device=device)

        state = tokenizer.state_dict()

        assert "device" in state
        assert "tokenizers" in state
        assert state["tokenizers"] == {}

    def test_state_dict_with_proprio(self, device, normalized_proprio):
        """Test state_dict with fitted proprio tokenizers."""
        tokenizer = Tokenizer(device=device)
        tokenizer.fit_proprio_tokenizer(normalized_proprio, num_bins=256)

        state = tokenizer.state_dict()

        assert "tokenizers" in state
        assert "proprio_robot_frame" in state["tokenizers"]
        assert "proprio_camera_frame" in state["tokenizers"]

    def test_load_state_dict_with_proprio(self, device, normalized_proprio):
        """Test load_state_dict restores proprio tokenizers."""
        tokenizer = Tokenizer(device=device)
        tokenizer.fit_proprio_tokenizer(normalized_proprio, num_bins=256)

        state = tokenizer.state_dict()

        # Create new tokenizer and load state
        new_tokenizer = Tokenizer(device=device)
        # Need to create empty tokenizers first for load_state_dict to work
        new_tokenizer.tokenizers["proprio_robot_frame"] = BinningTokenizer(num_bins=256, device=device)
        new_tokenizer.tokenizers["proprio_camera_frame"] = BinningTokenizer(num_bins=256, device=device)
        new_tokenizer.load_state_dict(state)

        assert new_tokenizer.device == device
        assert "proprio_robot_frame" in new_tokenizer.tokenizers


@pytest.mark.integration
class TestTokenizerSavePretrained:
    """Tests for save_pretrained functionality."""

    def test_save_pretrained_action_only(self, device, action_chunks, tmp_path):
        """Test save_pretrained with action tokenizer only."""
        tokenizer = Tokenizer(device=device)
        tokenizer.fit_action_tokenizer(action_chunks, use_pretrained_weights=False)

        save_path = tmp_path / "tokenizer"
        tokenizer.save_pretrained(save_path)

        # Check directory structure
        assert save_path.exists()
        assert (save_path / "config.json").exists()
        assert (save_path / ACTION_KEY).exists()

        # Check config.json
        with open(save_path / "config.json") as f:
            config = json.load(f)
        assert "device" in config
        assert "tokenizer_keys" in config
        assert ACTION_KEY in config["tokenizer_keys"]
        assert config["tokenizer_keys"][ACTION_KEY]["type"] == "action"

    def test_save_pretrained_proprio_only(self, device, normalized_proprio, tmp_path):
        """Test save_pretrained with proprio tokenizers only."""
        tokenizer = Tokenizer(device=device)
        tokenizer.fit_proprio_tokenizer(normalized_proprio, num_bins=256)

        save_path = tmp_path / "tokenizer"
        tokenizer.save_pretrained(save_path)

        # Check directory structure
        assert save_path.exists()
        assert (save_path / "config.json").exists()
        assert (save_path / "proprio_robot_frame").exists()
        assert (save_path / "proprio_camera_frame").exists()

        # Check binning state files
        assert (save_path / "proprio_robot_frame" / "binning_state.pt").exists()
        assert (save_path / "proprio_camera_frame" / "binning_state.pt").exists()

    def test_save_pretrained_mixed(self, device, action_chunks, normalized_proprio, tmp_path):
        """Test save_pretrained with both action and proprio tokenizers."""
        tokenizer = Tokenizer(device=device)
        tokenizer.fit_action_tokenizer(action_chunks, use_pretrained_weights=False)
        tokenizer.fit_proprio_tokenizer(normalized_proprio, num_bins=256)

        save_path = tmp_path / "tokenizer"
        tokenizer.save_pretrained(save_path)

        # Check all subdirectories exist
        assert (save_path / ACTION_KEY).exists()
        assert (save_path / "proprio_robot_frame").exists()
        assert (save_path / "proprio_camera_frame").exists()

        # Check config has all keys
        with open(save_path / "config.json") as f:
            config = json.load(f)
        assert len(config["tokenizer_keys"]) == 3


@pytest.mark.integration
class TestTokenizerFromPretrained:
    """Tests for from_pretrained functionality."""

    def test_from_pretrained_action_only(self, device, action_chunks, tmp_path):
        """Test from_pretrained with action tokenizer."""
        # Save tokenizer
        tokenizer = Tokenizer(device=device)
        tokenizer.fit_action_tokenizer(action_chunks, use_pretrained_weights=False)

        save_path = tmp_path / "tokenizer"
        tokenizer.save_pretrained(save_path)

        # Load tokenizer
        loaded_tokenizer = Tokenizer.from_pretrained(save_path, device=device)

        assert loaded_tokenizer.device == device
        assert ACTION_KEY in loaded_tokenizer.tokenizers
        assert isinstance(loaded_tokenizer.tokenizers[ACTION_KEY], ActionTokenizer)
        assert loaded_tokenizer.tokenizers[ACTION_KEY]._is_fitted is True

    def test_from_pretrained_proprio_only(self, device, normalized_proprio, tmp_path):
        """Test from_pretrained with proprio tokenizers."""
        # Save tokenizer
        tokenizer = Tokenizer(device=device)
        tokenizer.fit_proprio_tokenizer(normalized_proprio, num_bins=256)

        save_path = tmp_path / "tokenizer"
        tokenizer.save_pretrained(save_path)

        # Load tokenizer
        loaded_tokenizer = Tokenizer.from_pretrained(save_path, device=device)

        assert loaded_tokenizer.device == device
        assert "proprio_robot_frame" in loaded_tokenizer.tokenizers
        assert "proprio_camera_frame" in loaded_tokenizer.tokenizers
        assert isinstance(loaded_tokenizer.tokenizers["proprio_robot_frame"], BinningTokenizer)
        assert loaded_tokenizer.tokenizers["proprio_robot_frame"]._is_fitted is True

    def test_from_pretrained_mixed(self, device, action_chunks, normalized_proprio, tmp_path):
        """Test from_pretrained with mixed tokenizers."""
        # Save tokenizer
        tokenizer = Tokenizer(device=device)
        tokenizer.fit_action_tokenizer(action_chunks, use_pretrained_weights=False)
        tokenizer.fit_proprio_tokenizer(normalized_proprio, num_bins=256)

        save_path = tmp_path / "tokenizer"
        tokenizer.save_pretrained(save_path)

        # Load tokenizer
        loaded_tokenizer = Tokenizer.from_pretrained(save_path, device=device)

        assert len(loaded_tokenizer.tokenizers) == 3
        assert ACTION_KEY in loaded_tokenizer.tokenizers
        assert "proprio_robot_frame" in loaded_tokenizer.tokenizers
        assert "proprio_camera_frame" in loaded_tokenizer.tokenizers

    def test_from_pretrained_nonexistent_path(self, device):
        """Test from_pretrained with nonexistent path."""
        with pytest.raises(FileNotFoundError, match="Tokenizer path not found"):
            Tokenizer.from_pretrained("/nonexistent/path", device=device)

    def test_from_pretrained_missing_config(self, device, tmp_path):
        """Test from_pretrained with missing config.json."""
        save_path = tmp_path / "tokenizer"
        save_path.mkdir()

        with pytest.raises(FileNotFoundError, match="config.json not found"):
            Tokenizer.from_pretrained(save_path, device=device)


@pytest.mark.integration
class TestTokenizerRoundtrip:
    """Tests for save/load roundtrip."""

    def test_roundtrip_proprio(self, device, normalized_proprio, tmp_path):
        """Test save/load roundtrip preserves functionality."""
        # Original tokenizer
        tokenizer = Tokenizer(device=device)
        tokenizer.fit_proprio_tokenizer(normalized_proprio, num_bins=256)

        # Tokenize data
        tokenized_original = tokenizer.tokenize(normalized_proprio)

        # Save and load
        save_path = tmp_path / "tokenizer"
        tokenizer.save_pretrained(save_path)
        loaded_tokenizer = Tokenizer.from_pretrained(save_path, device=device)

        # Tokenize with loaded tokenizer
        tokenized_loaded = loaded_tokenizer.tokenize(normalized_proprio)

        # Check that tokenization is the same
        for key in tokenized_original:
            assert torch.equal(tokenized_original[key], tokenized_loaded[key])


@pytest.mark.unit
class TestTokenizerEdgeCases:
    """Tests for edge cases."""

    def test_empty_tokenizer_save_load(self, device, tmp_path):
        """Test saving and loading empty tokenizer."""
        tokenizer = Tokenizer(device=device)

        save_path = tmp_path / "tokenizer"
        tokenizer.save_pretrained(save_path)

        loaded_tokenizer = Tokenizer.from_pretrained(save_path, device=device)

        assert loaded_tokenizer.tokenizers == {}
        assert loaded_tokenizer.device == device

    def test_tokenize_empty_dict(self, device):
        """Test tokenizing empty dictionary."""
        tokenizer = Tokenizer(device=device)

        result = tokenizer.tokenize({})

        assert result == {}

    def test_detokenize_empty_dict(self, device):
        """Test detokenizing empty dictionary."""
        tokenizer = Tokenizer(device=device)

        result = tokenizer.detokenize({})

        assert result == {}