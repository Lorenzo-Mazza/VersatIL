"""Tests for Binary Mapper from Free Transformer."""

import pytest
import torch

from versatil.models.layers.free_transformer.binary_mapper import BinaryMapper


class TestBinaryMapper:
    """Test suite for BinaryMapper implementation."""

    @pytest.fixture
    def binary_mapper(self):
        """Create a standard binary mapper for testing."""
        return BinaryMapper(latent_bits=8, embedding_dimension=64)

    @pytest.fixture
    def sample_features_2d(self):
        """Create 2D sample features (B, D)."""
        return torch.randn(4, 64)

    @pytest.fixture
    def sample_features_3d(self):
        """Create 3D sample features (B, T, D)."""
        return torch.randn(4, 10, 64)

    def test_initialization(self):
        """Test Binary Mapper initializes with correct parameters."""
        latent_bits = 8
        embedding_dim = 64
        mapper = BinaryMapper(latent_bits=latent_bits, embedding_dimension=embedding_dim)

        assert mapper.latent_bits == latent_bits
        assert mapper.latent_dim == 2**latent_bits  # 256 for 8 bits
        assert mapper.embedding_dimension == embedding_dim

    def test_output_shapes_2d(self, binary_mapper, sample_features_2d):
        """Test output shapes for 2D input."""
        one_hot, logits = binary_mapper(sample_features_2d, deterministic=False)

        batch_size = sample_features_2d.shape[0]
        assert one_hot.shape == (batch_size, binary_mapper.latent_dim)
        assert logits.shape == (batch_size, binary_mapper.latent_bits)

    def test_output_shapes_3d(self, binary_mapper, sample_features_3d):
        """Test output shapes for 3D input."""
        one_hot, logits = binary_mapper(sample_features_3d, deterministic=False)

        batch_size, seq_len = sample_features_3d.shape[:2]
        assert one_hot.shape == (batch_size, seq_len, binary_mapper.latent_dim)
        assert logits.shape == (batch_size, seq_len, binary_mapper.latent_bits)

    def test_one_hot_property(self, binary_mapper, sample_features_2d):
        """Test that output is valid one-hot encoding."""
        one_hot, _ = binary_mapper(sample_features_2d, deterministic=True)

        # Each row should sum to 1 (one-hot property)
        row_sums = one_hot.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

        # Each element should be 0 or 1
        assert torch.all((one_hot == 0) | (one_hot == 1))

    def test_deterministic_vs_stochastic(self, binary_mapper, sample_features_2d):
        """Test deterministic and stochastic sampling produce valid outputs."""
        # Deterministic sampling
        one_hot_det, logits_det = binary_mapper(sample_features_2d, deterministic=True)

        # Stochastic sampling
        torch.manual_seed(42)
        one_hot_stoch, logits_stoch = binary_mapper(sample_features_2d, deterministic=False)

        # Logits should be the same
        assert torch.allclose(logits_det, logits_stoch)

        # Both should be valid one-hot
        assert torch.allclose(one_hot_det.sum(dim=-1), torch.ones(sample_features_2d.shape[0]))
        assert torch.allclose(one_hot_stoch.sum(dim=-1), torch.ones(sample_features_2d.shape[0]))

    def test_gradient_flow(self, binary_mapper, sample_features_2d):
        """Test gradients flow through straight-through estimator."""
        features = sample_features_2d.clone().requires_grad_(True)

        one_hot, logits = binary_mapper(features, deterministic=False)

        # Compute loss from logits (logits have grad_fn)
        loss = logits.sum()
        loss.backward()

        # Gradients should flow through features
        assert features.grad is not None
        assert not torch.allclose(features.grad, torch.zeros_like(features.grad))

    @pytest.mark.parametrize("latent_bits", [4, 8, 12, 16])
    def test_different_latent_bits(self, latent_bits):
        """Test mapper works with different latent_bits values."""
        mapper = BinaryMapper(latent_bits=latent_bits, embedding_dimension=64)
        features = torch.randn(2, 64)

        one_hot, logits = mapper(features)

        assert mapper.latent_dim == 2**latent_bits
        assert one_hot.shape == (2, 2**latent_bits)
        assert logits.shape == (2, latent_bits)

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_batch_size_independence(self, batch_size):
        """Test mapper works with different batch sizes."""
        mapper = BinaryMapper(latent_bits=8, embedding_dimension=64)
        features = torch.randn(batch_size, 64)

        one_hot, logits = mapper(features)

        assert one_hot.shape[0] == batch_size
        assert logits.shape[0] == batch_size

    def test_logits_range(self, binary_mapper, sample_features_2d):
        """Test logits can span reasonable range for sigmoid."""
        _, logits = binary_mapper(sample_features_2d)

        # Logits should not be all zeros or extremely large
        assert not torch.allclose(logits, torch.zeros_like(logits))
        # Check that sigmoid produces reasonable probabilities
        probs = torch.sigmoid(logits)
        assert torch.all((probs >= 0) & (probs <= 1))

    def test_device_compatibility(self, binary_mapper):
        """Test mapper works on different devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        mapper = binary_mapper.cuda()
        features = torch.randn(2, 64).cuda()

        one_hot, logits = mapper(features)

        assert one_hot.device.type == 'cuda'
        assert logits.device.type == 'cuda'

    def test_stochastic_sampling_varies(self, binary_mapper, sample_features_2d):
        """Test that stochastic sampling produces different results."""
        # Run multiple times to check variation
        results = []
        for _ in range(10):
            one_hot, _ = binary_mapper(sample_features_2d, deterministic=False)
            results.append(one_hot)

        # Not all results should be identical (very low probability with 256 codes)
        all_same = all(torch.allclose(results[0], r) for r in results[1:])
        assert not all_same, "Stochastic sampling should produce variation"

    def test_deterministic_sampling_consistent(self, binary_mapper, sample_features_2d):
        """Test that deterministic sampling is consistent."""
        one_hot1, logits1 = binary_mapper(sample_features_2d, deterministic=True)
        one_hot2, logits2 = binary_mapper(sample_features_2d, deterministic=True)

        assert torch.allclose(one_hot1, one_hot2)
        assert torch.allclose(logits1, logits2)

    def test_is_nn_module(self, binary_mapper):
        """Test that BinaryMapper is a proper nn.Module."""
        assert isinstance(binary_mapper, torch.nn.Module)

    def test_trainable_parameters(self, binary_mapper):
        """Test that mapper has trainable parameters."""
        params = list(binary_mapper.parameters())
        assert len(params) > 0

        # Check that parameters require grad
        for param in params:
            assert param.requires_grad
