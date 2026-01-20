"""Tests for normalization factory."""

import pytest
import torch

from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.normalization.factory import create_normalization_layer
from versatil.models.layers.normalization.rms_norm import RMSNorm


@pytest.mark.unit
class TestCreateNormalizationLayer:
    """Tests for create_normalization_layer factory function."""

    def test_create_layer_norm(self):
        """Test creation of LayerNorm."""
        dimension = 512
        epsilon = 1e-5

        layer = create_normalization_layer(
            normalization_type=NormalizationType.LAYER_NORM.value,
            dimension=dimension,
            epsilon=epsilon,
        )

        assert isinstance(layer, torch.nn.LayerNorm)
        assert layer.normalized_shape == (dimension,)
        assert layer.eps == epsilon

    def test_create_rms_norm(self):
        """Test creation of RMSNorm."""
        dimension = 512
        epsilon = 1e-5

        layer = create_normalization_layer(
            normalization_type=NormalizationType.RMS_NORM.value,
            dimension=dimension,
            epsilon=epsilon,
        )

        assert isinstance(layer, RMSNorm)
        assert layer.weight.shape == (dimension,)
        assert layer.eps == epsilon

    @pytest.mark.parametrize("normalization_type", [
        NormalizationType.LAYER_NORM.value,
        NormalizationType.RMS_NORM.value,
    ])
    def test_forward_pass(self, normalization_type):
        """Test forward pass of created normalization layers."""
        batch_size, seq_len, dimension = 2, 10, 512

        layer = create_normalization_layer(
            normalization_type=normalization_type,
            dimension=dimension,
        )

        input_tensor = torch.randn(batch_size, seq_len, dimension)
        output = layer(input_tensor)

        assert output.shape == (batch_size, seq_len, dimension)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_raises_error_unsupported_type(self):
        """Test that error is raised for unsupported normalization type."""
        with pytest.raises(ValueError, match="Unsupported normalization type"):
            create_normalization_layer(
                normalization_type="invalid_type",
                dimension=512,
            )

    def test_default_epsilon(self):
        """Test that default epsilon is used correctly."""
        layer = create_normalization_layer(
            normalization_type=NormalizationType.LAYER_NORM.value,
            dimension=512,
        )

        assert layer.eps == 1e-6

    def test_layer_norm_properties(self):
        """Test that LayerNorm has expected statistical properties."""
        batch_size, seq_len, dimension = 4, 20, 512

        layer = create_normalization_layer(
            normalization_type=NormalizationType.LAYER_NORM.value,
            dimension=dimension,
        )

        # Input with non-zero mean and non-unit variance
        input_tensor = torch.randn(batch_size, seq_len, dimension) * 5.0 + 10.0
        output = layer(input_tensor)

        # Check that output has approximately zero mean
        output_mean = output.mean(dim=-1)
        assert torch.allclose(output_mean, torch.zeros_like(output_mean), atol=1e-5)

        # Check that output has approximately unit variance
        output_var = output.var(dim=-1, unbiased=False)
        assert torch.allclose(output_var, torch.ones_like(output_var), atol=1e-4)

    def test_rms_norm_properties(self):
        """Test that RMSNorm normalizes by RMS."""
        batch_size, seq_len, dimension = 4, 20, 512

        layer = create_normalization_layer(
            normalization_type=NormalizationType.RMS_NORM.value,
            dimension=dimension,
        )

        # Input with non-zero mean and non-unit variance
        input_tensor = torch.randn(batch_size, seq_len, dimension) * 5.0 + 10.0
        output = layer(input_tensor)

        # Check that output has approximately unit RMS (root mean square)
        # RMS = sqrt(mean(x^2))
        output_rms = torch.sqrt(torch.mean(output ** 2, dim=-1))
        assert torch.allclose(output_rms, torch.ones_like(output_rms), atol=1e-4)