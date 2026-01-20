"""Tests for BinningTokenizer."""

import numpy as np
import pytest
import torch

from versatil.data.tokenization.binning_tokenizer import BinningTokenizer


@pytest.fixture
def normalized_data():
    """Generate synthetic normalized proprioceptive data."""
    # (N, D) = (100, 7) - 100 samples, 7 dimensions
    np.random.seed(42)
    return np.random.randn(100, 7).astype(np.float32) * 0.5  # Normalized to ~[-1, 1]


@pytest.fixture
def normalized_data_3d():
    """Generate synthetic normalized data with temporal dimension."""
    # (B, T, D) = (10, 5, 7)
    np.random.seed(42)
    return np.random.randn(10, 5, 7).astype(np.float32) * 0.5


@pytest.fixture
def device():
    """Device for testing."""
    return torch.device("cpu")


@pytest.mark.unit
class TestBinningTokenizerBasic:
    """Basic tests for BinningTokenizer."""

    def test_initialization(self, device):
        """Test initialization with default parameters."""
        tokenizer = BinningTokenizer(num_bins=256, device=device)

        assert tokenizer.num_bins == 256
        assert tokenizer.device == device
        assert tokenizer.bin_edges is None
        assert tokenizer._is_fitted is False

    def test_initialization_custom_bins(self, device):
        """Test initialization with custom number of bins."""
        tokenizer = BinningTokenizer(num_bins=128, device=device)

        assert tokenizer.num_bins == 128

    def test_fit(self, device, normalized_data):
        """Test fitting tokenizer on data."""
        tokenizer = BinningTokenizer(num_bins=256, device=device)
        tokenizer.fit(normalized_data)

        assert tokenizer._is_fitted is True
        assert tokenizer.bin_edges is not None
        assert tokenizer.bin_edges.shape == (7, 255)  # (D, num_bins-1)
        assert tokenizer.bin_edges.device == device

    def test_fit_3d_data(self, device, normalized_data_3d):
        """Test fitting tokenizer on 3D data (flattens to 2D)."""
        tokenizer = BinningTokenizer(num_bins=256, device=device)
        tokenizer.fit(normalized_data_3d)

        assert tokenizer._is_fitted is True
        assert tokenizer.bin_edges is not None
        assert tokenizer.bin_edges.shape == (7, 255)  # (D, num_bins-1)


@pytest.mark.unit
class TestBinningTokenizerEncoding:
    """Tests for encoding functionality."""

    def test_encode_numpy(self, device, normalized_data):
        """Test encoding numpy array."""
        tokenizer = BinningTokenizer(num_bins=256, device=device)
        tokenizer.fit(normalized_data)

        tokens = tokenizer.encode(normalized_data)

        assert isinstance(tokens, torch.Tensor)
        assert tokens.shape == normalized_data.shape
        assert tokens.dtype == torch.long
        assert tokens.min() >= 0
        assert tokens.max() < 256

    def test_encode_torch(self, device, normalized_data):
        """Test encoding torch tensor."""
        tokenizer = BinningTokenizer(num_bins=256, device=device)
        tokenizer.fit(normalized_data)

        data_tensor = torch.from_numpy(normalized_data).to(device)
        tokens = tokenizer.encode(data_tensor)

        assert isinstance(tokens, torch.Tensor)
        assert tokens.shape == data_tensor.shape
        assert tokens.dtype == torch.long

    def test_encode_before_fit_raises_error(self, device, normalized_data):
        """Test that encoding before fitting raises error."""
        tokenizer = BinningTokenizer(num_bins=256, device=device)

        with pytest.raises(RuntimeError, match="Tokenizer must be fitted"):
            tokenizer.encode(normalized_data)

    def test_encode_3d_preserves_shape(self, device, normalized_data_3d):
        """Test encoding preserves 3D shape."""
        tokenizer = BinningTokenizer(num_bins=256, device=device)
        tokenizer.fit(normalized_data_3d)

        tokens = tokenizer.encode(normalized_data_3d)

        assert tokens.shape == normalized_data_3d.shape


@pytest.mark.unit
class TestBinningTokenizerDecoding:
    """Tests for decoding functionality."""

    def test_decode(self, device, normalized_data):
        """Test decoding tokens back to continuous values."""
        tokenizer = BinningTokenizer(num_bins=256, device=device)
        tokenizer.fit(normalized_data)

        tokens = tokenizer.encode(normalized_data)
        decoded = tokenizer.decode(tokens)

        assert isinstance(decoded, torch.Tensor)
        assert decoded.shape == normalized_data.shape
        assert decoded.dtype == torch.float32

    def test_decode_numpy_input(self, device, normalized_data):
        """Test decoding with numpy input."""
        tokenizer = BinningTokenizer(num_bins=256, device=device)
        tokenizer.fit(normalized_data)

        tokens = tokenizer.encode(normalized_data)
        tokens_np = tokens.cpu().numpy()
        decoded = tokenizer.decode(tokens_np)

        assert isinstance(decoded, torch.Tensor)
        assert decoded.shape == normalized_data.shape

    def test_decode_before_fit_raises_error(self, device):
        """Test that decoding before fitting raises error."""
        tokenizer = BinningTokenizer(num_bins=256, device=device)
        fake_tokens = torch.randint(0, 256, (10, 7))

        with pytest.raises(RuntimeError, match="Tokenizer must be fitted"):
            tokenizer.decode(fake_tokens)

    def test_encode_decode_roundtrip(self, device, normalized_data):
        """Test encode/decode roundtrip maintains approximate values."""
        tokenizer = BinningTokenizer(num_bins=256, device=device)
        tokenizer.fit(normalized_data)

        tokens = tokenizer.encode(normalized_data)
        decoded = tokenizer.decode(tokens)

        # Check that decoded values are reasonably close (within bin width)
        # With 256 bins, bin width is roughly 2.0 / 256 ~ 0.008
        diff = torch.abs(torch.from_numpy(normalized_data).to(device) - decoded)
        assert diff.mean() < 0.05  # Mean error should be small


@pytest.mark.unit
class TestBinningTokenizerBinCenters:
    """Tests for bin center computation."""

    def test_get_bin_centers(self, device, normalized_data):
        """Test bin center computation."""
        tokenizer = BinningTokenizer(num_bins=8, device=device)
        tokenizer.fit(normalized_data)

        bin_centers = tokenizer._get_bin_centers(dim=0)

        assert bin_centers.shape == (8,)
        assert bin_centers.device == device
        # Centers should be monotonically increasing
        assert torch.all(bin_centers[1:] > bin_centers[:-1])


@pytest.mark.unit
class TestBinningTokenizerSerialization:
    """Tests for state_dict save/load."""

    def test_state_dict(self, device, normalized_data):
        """Test state_dict returns expected keys."""
        tokenizer = BinningTokenizer(num_bins=256, device=device)
        tokenizer.fit(normalized_data)

        state = tokenizer.state_dict()

        assert "num_bins" in state
        assert "device" in state
        assert "bin_edges" in state
        assert "is_fitted" in state
        assert state["num_bins"] == 256
        assert state["is_fitted"] is True
        assert state["bin_edges"] is not None

    def test_load_state_dict(self, device, normalized_data):
        """Test load_state_dict restores state."""
        tokenizer = BinningTokenizer(num_bins=256, device=device)
        tokenizer.fit(normalized_data)

        state = tokenizer.state_dict()

        # Create new tokenizer and load state
        new_tokenizer = BinningTokenizer(num_bins=128, device=device)  # Different initial bins
        new_tokenizer.load_state_dict(state)

        assert new_tokenizer.num_bins == 256  # Should be restored
        assert new_tokenizer._is_fitted is True
        assert new_tokenizer.bin_edges is not None
        assert torch.equal(new_tokenizer.bin_edges, tokenizer.bin_edges)

    def test_state_dict_before_fit(self, device):
        """Test state_dict before fitting."""
        tokenizer = BinningTokenizer(num_bins=256, device=device)

        state = tokenizer.state_dict()

        assert state["is_fitted"] is False
        assert state["bin_edges"] is None


@pytest.mark.unit
class TestBinningTokenizerDeviceHandling:
    """Tests for device handling."""

    def test_to_device(self, device, normalized_data):
        """Test moving tokenizer to device."""
        tokenizer = BinningTokenizer(num_bins=256, device=torch.device("cpu"))
        tokenizer.fit(normalized_data)

        initial_device = tokenizer.device
        result = tokenizer.to(device)

        assert result is tokenizer  # Should return self
        assert tokenizer.device == device
        assert tokenizer.bin_edges.device == device

    def test_to_device_before_fit(self, device):
        """Test moving unfitted tokenizer to device."""
        tokenizer = BinningTokenizer(num_bins=256, device=torch.device("cpu"))

        result = tokenizer.to(device)

        assert result is tokenizer
        assert tokenizer.device == device
        assert tokenizer.bin_edges is None  # Still unfitted


@pytest.mark.unit
class TestBinningTokenizerEdgeCases:
    """Tests for edge cases."""

    def test_small_number_of_bins(self, device, normalized_data):
        """Test with very small number of bins."""
        tokenizer = BinningTokenizer(num_bins=4, device=device)
        tokenizer.fit(normalized_data)

        tokens = tokenizer.encode(normalized_data)

        assert tokens.min() >= 0
        assert tokens.max() < 4

    def test_large_number_of_bins(self, device, normalized_data):
        """Test with large number of bins."""
        tokenizer = BinningTokenizer(num_bins=1024, device=device)
        tokenizer.fit(normalized_data)

        tokens = tokenizer.encode(normalized_data)

        assert tokens.min() >= 0
        assert tokens.max() < 1024

    def test_single_sample(self, device):
        """Test with single sample."""
        tokenizer = BinningTokenizer(num_bins=256, device=device)
        data = np.array([[0.1, 0.2, 0.3]])

        tokenizer.fit(data)
        tokens = tokenizer.encode(data)

        assert tokens.shape == (1, 3)

    def test_single_dimension(self, device):
        """Test with single dimension."""
        tokenizer = BinningTokenizer(num_bins=256, device=device)
        data = np.random.randn(100, 1).astype(np.float32)

        tokenizer.fit(data)
        tokens = tokenizer.encode(data)

        assert tokens.shape == (100, 1)
        assert tokens.min() >= 0
        assert tokens.max() < 256