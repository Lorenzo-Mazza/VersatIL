"""Tests for ActionTokenizer."""

import numpy as np
import pytest
import torch

from refactoring.data.tokenize.action_tokenizer import ActionTokenizer


@pytest.fixture
def action_chunks():
    """Generate synthetic action chunks for testing."""
    # (N_chunks, T, D) = (10, 5, 7)
    return np.random.randn(10, 5, 7).astype(np.float32) * 0.5  # Normalized to ~[-1, 1]


@pytest.fixture
def device():
    """Device for testing."""
    return torch.device("cpu")


@pytest.mark.unit
class TestActionTokenizerPretrained:
    """Tests for ActionTokenizer with pretrained weights."""

    def test_initialization_pretrained(self, device):
        """Test initialization with pretrained weights."""
        tokenizer = ActionTokenizer(use_pretrained_weights=True, device=device)

        assert tokenizer.use_pretrained_weights is True
        assert tokenizer.device == device
        assert tokenizer._is_fitted is True
        assert tokenizer.processor is not None

    def test_encode_decode_pretrained(self, device, action_chunks):
        """Test encode/decode roundtrip with pretrained weights."""
        tokenizer = ActionTokenizer(use_pretrained_weights=True, device=device)

        # Encode
        tokens = tokenizer.encode(action_chunks)
        assert isinstance(tokens, list)
        assert all(all(isinstance(t, int) for t in token_chunk) for token_chunk in tokens)
        # Decode
        reconstructed = tokenizer.decode(tokens)
        assert isinstance(reconstructed, np.ndarray)
        assert reconstructed.shape == action_chunks.shape

    def test_cannot_fit_pretrained(self, device, action_chunks):
        """Test that fitting raises error when using pretrained weights."""
        tokenizer = ActionTokenizer(use_pretrained_weights=True, device=device)

        with pytest.raises(ValueError, match="Cannot fit when use_pretrained_weights=True"):
            tokenizer.fit(action_chunks)


@pytest.mark.integration
class TestActionTokenizerCustom:
    """Tests for ActionTokenizer with custom fitting."""

    def test_initialization_custom(self, device):
        """Test initialization for custom fitting."""
        tokenizer = ActionTokenizer(use_pretrained_weights=False, device=device)

        assert tokenizer.use_pretrained_weights is False
        assert tokenizer.device == device
        assert tokenizer._is_fitted is False

    def test_fit_and_encode_decode(self, device, action_chunks):
        """Test fitting and encode/decode roundtrip."""
        tokenizer = ActionTokenizer(use_pretrained_weights=False, device=device)

        # Fit
        tokenizer.fit(action_chunks)
        assert tokenizer._is_fitted is True

        # Encode
        tokens = tokenizer.encode(action_chunks)
        assert isinstance(tokens, list)

        # Decode
        reconstructed = tokenizer.decode(tokens)
        assert isinstance(reconstructed, np.ndarray)

    def test_encode_before_fit_raises_error(self, device, action_chunks):
        """Test that encoding before fitting raises error."""
        tokenizer = ActionTokenizer(use_pretrained_weights=False, device=device)

        with pytest.raises(RuntimeError, match="Tokenizer must be fitted"):
            tokenizer.encode(action_chunks)

    def test_decode_before_fit_raises_error(self, device):
        """Test that decoding before fitting raises error."""
        tokenizer = ActionTokenizer(use_pretrained_weights=False, device=device)

        with pytest.raises(RuntimeError, match="Tokenizer must be fitted"):
            tokenizer.decode([1, 2, 3])


@pytest.mark.unit
class TestActionTokenizerSerialization:
    """Tests for ActionTokenizer save/load."""

    def test_save_pretrained(self, device, action_chunks, tmp_path):
        """Test saving pretrained tokenizer."""
        tokenizer = ActionTokenizer(use_pretrained_weights=False, device=device)
        tokenizer.fit(action_chunks)

        save_path = tmp_path / "tokenizer"
        tokenizer.save_pretrained(str(save_path))

        # Check that files were created
        assert save_path.exists()
        assert len(list(save_path.iterdir())) > 0  # Has files

    def test_save_before_fit_raises_error(self, device, tmp_path):
        """Test that saving before fitting raises error."""
        tokenizer = ActionTokenizer(use_pretrained_weights=False, device=device)

        save_path = tmp_path / "tokenizer"
        with pytest.raises(RuntimeError, match="Cannot save unfitted tokenizer"):
            tokenizer.save_pretrained(str(save_path))

    def test_state_dict(self, device):
        """Test state_dict returns expected keys."""
        tokenizer = ActionTokenizer(use_pretrained_weights=True, device=device)

        state = tokenizer.state_dict()
        assert "use_pretrained_weights" in state
        assert "device" in state
        assert "is_fitted" in state
        assert state["use_pretrained_weights"] is True
        assert state["is_fitted"] is True

    def test_load_state_dict(self, device):
        """Test load_state_dict restores state."""
        tokenizer = ActionTokenizer(use_pretrained_weights=True, device=device)

        state = tokenizer.state_dict()

        # Create new tokenizer and load state
        new_tokenizer = ActionTokenizer(use_pretrained_weights=False, device=device)
        new_tokenizer.load_state_dict(state)

        assert new_tokenizer.use_pretrained_weights is True
        assert new_tokenizer._is_fitted is True
        assert new_tokenizer.device == device


@pytest.mark.unit
class TestActionTokenizerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_encode_torch_tensor(self, device, action_chunks):
        """Test encoding with torch.Tensor input."""
        tokenizer = ActionTokenizer(use_pretrained_weights=True, device=device)

        action_tensor = torch.from_numpy(action_chunks).to(device)
        tokens = tokenizer.encode(action_tensor)

        assert isinstance(tokens, list)

    def test_decode_single_token(self, device, action_chunks):
        """Test decoding a single token."""
        tokenizer = ActionTokenizer(use_pretrained_weights=True, device=device)

        # First encode to initialize processor dimensions
        tokens = tokenizer.encode(action_chunks[:1])

        # Decode single token (should be converted to list internally)
        reconstructed = tokenizer.decode(tokens[0])
        assert isinstance(reconstructed, np.ndarray)

    def test_to_device(self, device):
        """Test moving tokenizer to device."""
        tokenizer = ActionTokenizer(use_pretrained_weights=True, device=torch.device("cpu"))
        new_device = device
        result = tokenizer.to(new_device)
        assert result is tokenizer  # Should return self
        assert tokenizer.device == new_device