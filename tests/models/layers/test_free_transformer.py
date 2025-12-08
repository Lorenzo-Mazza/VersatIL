"""Tests for Free Transformer layer components."""
import pytest
import torch

from refactoring.models.layers.free_transformer.free_transformer import (
    FreeTransformerDecoderBlock,
    LatentConditionedTransformerBlock,
    FreeTransformerEncoderBlock,
    FreeTransformerEncoder,
    FreeTransformerDecoder,
)
from refactoring.models.layers.activation import ActivationFunction


@pytest.fixture
def device():
    """Get available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def batch_size():
    """Default batch size."""
    return 2


@pytest.fixture
def sequence_length():
    """Default sequence length."""
    return 10


@pytest.fixture
def embedding_dimension():
    """Default embedding dimension."""
    return 256


@pytest.mark.unit
class TestFreeTransformerDecoderBlock:
    """Test Free Transformer Decoder Block (cross-attention)."""

    def test_init_and_forward(self, batch_size, embedding_dimension, device):
        """Test basic initialization and forward pass."""
        block = FreeTransformerDecoderBlock(
            embedding_dimension=embedding_dimension,
            number_of_heads=8,
            feedforward_dimension=1024,
            causal=False,
        ).to(device)

        target_length = 10
        memory_length = 5
        target = torch.randn(batch_size, target_length, embedding_dimension, device=device)
        memory = torch.randn(batch_size, memory_length, embedding_dimension, device=device)
        output = block(target, memory)

        assert output.shape == target.shape
        assert not torch.isnan(output).any()

    def test_causal_vs_non_causal(self, batch_size, embedding_dimension, device):
        """Test causal vs non-causal attention."""
        causal_block = FreeTransformerDecoderBlock(
            embedding_dimension=embedding_dimension,
            number_of_heads=8,
            causal=True,
        ).to(device)

        non_causal_block = FreeTransformerDecoderBlock(
            embedding_dimension=embedding_dimension,
            number_of_heads=8,
            causal=False,
        ).to(device)

        target_length = 10
        memory_length = 5
        target = torch.randn(batch_size, target_length, embedding_dimension, device=device)
        memory = torch.randn(batch_size, memory_length, embedding_dimension, device=device)

        causal_output = causal_block(target, memory)
        non_causal_output = non_causal_block(target, memory)

        assert causal_output.shape == non_causal_output.shape == target.shape
        assert not torch.allclose(causal_output, non_causal_output, atol=1e-2)

    @pytest.mark.parametrize("activation", [
        ActivationFunction.SWIGLU,
        ActivationFunction.GELU,
        ActivationFunction.RELU,
    ])
    def test_different_activations(self, batch_size, embedding_dimension, device, activation):
        """Test with different activation functions."""
        block = FreeTransformerDecoderBlock(
            embedding_dimension=embedding_dimension,
            number_of_heads=8,
            activation=activation,
        ).to(device)

        target_length = 10
        memory_length = 5
        target = torch.randn(batch_size, target_length, embedding_dimension, device=device)
        memory = torch.randn(batch_size, memory_length, embedding_dimension, device=device)
        output = block(target, memory)

        assert output.shape == target.shape
        assert not torch.isnan(output).any()

    def test_with_memory_key_padding_mask(self, batch_size, embedding_dimension, device):
        """Test with memory key padding mask."""
        block = FreeTransformerDecoderBlock(
            embedding_dimension=embedding_dimension,
            number_of_heads=8,
        ).to(device)

        target_length = 10
        memory_length = 5
        target = torch.randn(batch_size, target_length, embedding_dimension, device=device)
        memory = torch.randn(batch_size, memory_length, embedding_dimension, device=device)
        memory_key_padding_mask = torch.zeros(batch_size, memory_length, dtype=torch.bool, device=device)
        memory_key_padding_mask[:, -2:] = True

        output = block(target, memory, memory_key_padding_mask=memory_key_padding_mask)

        assert output.shape == target.shape


@pytest.mark.unit
class TestLatentConditionedTransformerBlock:
    """Test Latent-Conditioned Transformer Block."""

    def test_init_and_forward(self, batch_size, sequence_length, embedding_dimension, device):
        """Test basic initialization and forward pass."""
        latent_dim = 256
        block = LatentConditionedTransformerBlock(
            embedding_dimension=embedding_dimension,
            number_of_heads=8,
            feedforward_dimension=1024,
            latent_dim=latent_dim,
        ).to(device)

        x = torch.randn(batch_size, sequence_length, embedding_dimension, device=device)
        latent = torch.randn(batch_size, sequence_length, latent_dim, device=device)
        output = block(x, latent)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_latent_broadcasting(self, batch_size, sequence_length, embedding_dimension, device):
        """Test that single latent is broadcasted to sequence."""
        latent_dim = 256
        block = LatentConditionedTransformerBlock(
            embedding_dimension=embedding_dimension,
            number_of_heads=8,
            latent_dim=latent_dim,
        ).to(device)

        x = torch.randn(batch_size, sequence_length, embedding_dimension, device=device)
        latent_single = torch.randn(batch_size, 1, latent_dim, device=device)

        output = block(x, latent_single)

        assert output.shape == x.shape

    def test_with_rope(self, batch_size, sequence_length, embedding_dimension, device):
        """Test with Rotary Position Embeddings."""
        latent_dim = 256
        block = LatentConditionedTransformerBlock(
            embedding_dimension=embedding_dimension,
            number_of_heads=8,
            latent_dim=latent_dim,
            use_rope=True,
        ).to(device)

        x = torch.randn(batch_size, sequence_length, embedding_dimension, device=device)
        latent = torch.randn(batch_size, sequence_length, latent_dim, device=device)
        output = block(x, latent)

        assert output.shape == x.shape

    @pytest.mark.parametrize("activation", [
        ActivationFunction.SWIGLU,
        ActivationFunction.GELU,
    ])
    def test_different_activations(self, batch_size, sequence_length, embedding_dimension, device, activation):
        """Test with different activation functions."""
        latent_dim = 256
        block = LatentConditionedTransformerBlock(
            embedding_dimension=embedding_dimension,
            number_of_heads=8,
            latent_dim=latent_dim,
            activation=activation,
        ).to(device)

        x = torch.randn(batch_size, sequence_length, embedding_dimension, device=device)
        latent = torch.randn(batch_size, sequence_length, latent_dim, device=device)
        output = block(x, latent)

        assert output.shape == x.shape


@pytest.mark.unit
class TestFreeTransformerEncoderBlock:
    """Test Free Transformer Encoder Block."""

    def test_init_and_forward(self, batch_size, sequence_length, embedding_dimension, device):
        """Test basic initialization and forward pass."""
        block = FreeTransformerEncoderBlock(
            embedding_dimension=embedding_dimension,
            number_of_heads=8,
            feedforward_dimension=1024,
        ).to(device)

        x = torch.randn(batch_size, sequence_length, embedding_dimension, device=device)
        queries = torch.randn(batch_size, sequence_length, embedding_dimension, device=device)
        output = block(x, queries)

        assert output.shape == queries.shape
        assert not torch.isnan(output).any()

    def test_rejects_causal(self, embedding_dimension):
        """Test that encoder block rejects causal=True."""
        with pytest.raises(ValueError, match="must be non-causal"):
            FreeTransformerEncoderBlock(
                embedding_dimension=embedding_dimension,
                number_of_heads=8,
                causal=True,
            )

    @pytest.mark.parametrize("activation", [
        ActivationFunction.SWIGLU,
        ActivationFunction.GELU,
    ])
    def test_different_activations(self, batch_size, sequence_length, embedding_dimension, device, activation):
        """Test with different activation functions."""
        block = FreeTransformerEncoderBlock(
            embedding_dimension=embedding_dimension,
            number_of_heads=8,
            activation=activation,
        ).to(device)

        x = torch.randn(batch_size, sequence_length, embedding_dimension, device=device)
        queries = torch.randn(batch_size, sequence_length, embedding_dimension, device=device)
        output = block(x, queries)

        assert output.shape == queries.shape


@pytest.mark.unit
class TestFreeTransformerEncoder:
    """Test Free Transformer Encoder."""

    def test_init_and_forward(self, batch_size, sequence_length, embedding_dimension, device):
        """Test basic initialization and forward pass."""
        encoder = FreeTransformerEncoder(
            embedding_dimension=embedding_dimension,
            number_of_layers=2,
            number_of_heads=8,
            latent_bits=8,
        ).to(device)

        x = torch.randn(batch_size, sequence_length, embedding_dimension, device=device)
        latent_codes, binary_logits = encoder(x)

        assert latent_codes.shape == (batch_size, sequence_length, 2**8)
        assert binary_logits.shape == (batch_size, sequence_length, 8)

        assert torch.allclose(latent_codes.sum(dim=-1), torch.ones_like(latent_codes.sum(dim=-1)))

    def test_deterministic_sampling(self, batch_size, sequence_length, embedding_dimension, device):
        """Test deterministic latent sampling."""
        encoder = FreeTransformerEncoder(
            embedding_dimension=embedding_dimension,
            number_of_layers=1,
            latent_bits=8,
        ).to(device)

        x = torch.randn(batch_size, sequence_length, embedding_dimension, device=device)

        encoder.eval()
        latent_codes1, _ = encoder(x, deterministic=True)
        latent_codes2, _ = encoder(x, deterministic=True)

        assert torch.allclose(latent_codes1, latent_codes2)

    @pytest.mark.parametrize("latent_bits", [8, 12, 16])
    def test_different_latent_bits(self, batch_size, sequence_length, embedding_dimension, device, latent_bits):
        """Test with different latent bits."""
        encoder = FreeTransformerEncoder(
            embedding_dimension=embedding_dimension,
            number_of_layers=1,
            latent_bits=latent_bits,
        ).to(device)

        x = torch.randn(batch_size, sequence_length, embedding_dimension, device=device)
        latent_codes, binary_logits = encoder(x)

        assert latent_codes.shape == (batch_size, sequence_length, 2**latent_bits)
        assert binary_logits.shape == (batch_size, sequence_length, latent_bits)


@pytest.mark.unit
class TestFreeTransformerDecoder:
    """Test Free Transformer Decoder."""

    def test_init_and_forward(self, batch_size, sequence_length, embedding_dimension, device):
        """Test basic initialization and forward pass."""
        latent_dim = 256
        decoder = FreeTransformerDecoder(
            embedding_dimension=embedding_dimension,
            number_of_layers=4,
            number_of_heads=8,
            latent_dim=latent_dim,
        ).to(device)

        x = torch.randn(batch_size, sequence_length, embedding_dimension, device=device)
        latent = torch.randn(batch_size, sequence_length, latent_dim, device=device)
        output, _ = decoder(x, latent)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_forward_to_mid(self, batch_size, sequence_length, embedding_dimension, device):
        """Test forward to mid-layer."""
        latent_dim = 256
        decoder = FreeTransformerDecoder(
            embedding_dimension=embedding_dimension,
            number_of_layers=4,
            number_of_heads=8,
            latent_dim=latent_dim,
        ).to(device)

        x = torch.randn(batch_size, sequence_length, embedding_dimension, device=device)
        mid_features = decoder.forward_to_mid(x)

        assert mid_features.shape == x.shape

    def test_forward_from_mid(self, batch_size, sequence_length, embedding_dimension, device):
        """Test forward from mid-layer."""
        latent_dim = 256
        decoder = FreeTransformerDecoder(
            embedding_dimension=embedding_dimension,
            number_of_layers=4,
            number_of_heads=8,
            latent_dim=latent_dim,
        ).to(device)

        mid_features = torch.randn(batch_size, sequence_length, embedding_dimension, device=device)
        latent = torch.randn(batch_size, sequence_length, latent_dim, device=device)
        output = decoder.forward_from_mid(mid_features, latent)

        assert output.shape == mid_features.shape

    def test_return_mid_features(self, batch_size, sequence_length, embedding_dimension, device):
        """Test returning mid-layer features."""
        latent_dim = 256
        decoder = FreeTransformerDecoder(
            embedding_dimension=embedding_dimension,
            number_of_layers=4,
            number_of_heads=8,
            latent_dim=latent_dim,
        ).to(device)

        x = torch.randn(batch_size, sequence_length, embedding_dimension, device=device)
        latent = torch.randn(batch_size, sequence_length, latent_dim, device=device)
        output, mid_features = decoder(x, latent, return_mid_features=True)

        assert output.shape == x.shape
        assert mid_features.shape == x.shape

    def test_even_layers_required(self, embedding_dimension):
        """Test that decoder requires even number of layers."""
        with pytest.raises(ValueError, match="must be even"):
            FreeTransformerDecoder(
                embedding_dimension=embedding_dimension,
                number_of_layers=5,
                number_of_heads=8,
                latent_dim=256,
            )

    @pytest.mark.parametrize("number_of_layers", [2, 4, 6, 8])
    def test_different_layer_counts(self, batch_size, sequence_length, embedding_dimension, device, number_of_layers):
        """Test with different even layer counts."""
        latent_dim = 256
        decoder = FreeTransformerDecoder(
            embedding_dimension=embedding_dimension,
            number_of_layers=number_of_layers,
            number_of_heads=8,
            latent_dim=latent_dim,
        ).to(device)

        x = torch.randn(batch_size, sequence_length, embedding_dimension, device=device)
        latent = torch.randn(batch_size, sequence_length, latent_dim, device=device)
        output, _ = decoder(x, latent)

        assert output.shape == x.shape

    @pytest.mark.parametrize("activation", [
        ActivationFunction.SWIGLU,
        ActivationFunction.GELU,
    ])
    def test_different_activations(self, batch_size, sequence_length, embedding_dimension, device, activation):
        """Test with different activation functions."""
        latent_dim = 256
        decoder = FreeTransformerDecoder(
            embedding_dimension=embedding_dimension,
            number_of_layers=4,
            number_of_heads=8,
            latent_dim=latent_dim,
            activation=activation,
        ).to(device)

        x = torch.randn(batch_size, sequence_length, embedding_dimension, device=device)
        latent = torch.randn(batch_size, sequence_length, latent_dim, device=device)
        output, _ = decoder(x, latent)

        assert output.shape == x.shape

    def test_causal_vs_non_causal(self, batch_size, sequence_length, embedding_dimension, device):
        """Test causal vs non-causal decoder."""
        latent_dim = 256

        causal_decoder = FreeTransformerDecoder(
            embedding_dimension=embedding_dimension,
            number_of_layers=4,
            number_of_heads=8,
            latent_dim=latent_dim,
            causal=True,
        ).to(device)

        non_causal_decoder = FreeTransformerDecoder(
            embedding_dimension=embedding_dimension,
            number_of_layers=4,
            number_of_heads=8,
            latent_dim=latent_dim,
            causal=False,
        ).to(device)

        x = torch.randn(batch_size, sequence_length, embedding_dimension, device=device)
        latent = torch.randn(batch_size, sequence_length, latent_dim, device=device)

        causal_output, _ = causal_decoder(x, latent)
        non_causal_output, _ = non_causal_decoder(x, latent)

        assert causal_output.shape == non_causal_output.shape == x.shape


@pytest.mark.integration
class TestFreeTransformerIntegration:
    """Integration tests for Free Transformer components."""

    def test_encoder_decoder_integration(self, batch_size, sequence_length, embedding_dimension, device):
        """Test encoder and decoder integration."""
        latent_bits = 8
        latent_dim = 2**latent_bits

        encoder = FreeTransformerEncoder(
            embedding_dimension=embedding_dimension,
            number_of_layers=2,
            latent_bits=latent_bits,
        ).to(device)

        decoder = FreeTransformerDecoder(
            embedding_dimension=embedding_dimension,
            number_of_layers=4,
            latent_dim=latent_dim,
        ).to(device)

        x = torch.randn(batch_size, sequence_length, embedding_dimension, device=device)

        mid_features = decoder.forward_to_mid(x)
        latent_codes, binary_logits = encoder(mid_features)
        output = decoder.forward_from_mid(mid_features, latent_codes)

        assert output.shape == x.shape
        assert latent_codes.shape == (batch_size, sequence_length, latent_dim)
        assert binary_logits.shape == (batch_size, sequence_length, latent_bits)
