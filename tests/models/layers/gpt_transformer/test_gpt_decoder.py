"""Tests for GPT decoder and decoder layer."""

import pytest
import torch

from refactoring.models.layers.activation import ActivationFunction
from refactoring.models.layers.constants import AttentionType, NormalizationType, PositionalEncodingType
from refactoring.models.layers.gpt_transformer.gpt_decoder import GPTDecoder
from refactoring.models.layers.gpt_transformer.gpt_decoder_layer import GPTDecoderLayer


@pytest.mark.unit
class TestGPTDecoderLayer:
    """Tests for GPTDecoderLayer."""

    @pytest.mark.parametrize("use_cross_attention", [True, False])
    def test_initialization(self, use_cross_attention):
        """Test layer initialization with/without cross-attention."""
        layer = GPTDecoderLayer(
            embedding_dimension=512,
            number_of_heads=8,
            attention_type=AttentionType.MULTI_HEAD.value,
            use_cross_attention=use_cross_attention,
        )

        assert layer.use_cross_attention is use_cross_attention
        if use_cross_attention:
            assert layer.cross_attention is not None
            assert layer.cross_attention_normalization is not None
        else:
            assert layer.cross_attention is None
            assert layer.cross_attention_normalization is None

    @pytest.mark.parametrize("use_cross_attention", [True, False])
    @pytest.mark.parametrize("activation", [
        ActivationFunction.GELU.value,
        ActivationFunction.SWIGLU.value,
        ActivationFunction.SILU.value,
    ])
    @pytest.mark.parametrize("normalization", [
        NormalizationType.LAYER_NORM.value,
        NormalizationType.RMS_NORM.value,
    ])
    def test_forward_configurations(self, use_cross_attention, activation, normalization):
        """Test forward pass with different configurations."""
        batch_size, seq_len, feature_len, embedding_dim = 2, 10, 20, 512

        layer = GPTDecoderLayer(
            embedding_dimension=embedding_dim,
            number_of_heads=8,
            attention_type=AttentionType.MULTI_HEAD.value,
            activation=activation,
            normalization_type=normalization,
            use_cross_attention=use_cross_attention,
        )

        hidden_states = torch.randn(batch_size, seq_len, embedding_dim)

        if use_cross_attention:
            encoded_features = torch.randn(batch_size, feature_len, embedding_dim)
            output, cache = layer(
                hidden_states=hidden_states,
                encoded_features=encoded_features,
                use_cache=True,
            )
        else:
            output, cache = layer(
                hidden_states=hidden_states,
                use_cache=True,
            )

        assert output.shape == (batch_size, seq_len, embedding_dim)
        assert cache is not None


@pytest.mark.unit
class TestGPTDecoder:
    """Tests for GPTDecoder."""

    @pytest.mark.parametrize("use_cross_attention", [True, False])
    def test_initialization(self, use_cross_attention):
        """Test decoder initialization with/without cross-attention."""
        decoder = GPTDecoder(
            number_of_layers=4,
            embedding_dimension=512,
            number_of_heads=8,
            attention_type=AttentionType.MULTI_HEAD.value,
            use_cross_attention=use_cross_attention,
        )

        assert decoder.use_cross_attention is use_cross_attention
        assert len(decoder.layers) == 4
        assert all(layer.use_cross_attention == use_cross_attention for layer in decoder.layers)

    def test_initialization_with_gqa(self):
        """Test decoder initialization with GQA."""
        decoder = GPTDecoder(
            number_of_layers=4,
            embedding_dimension=512,
            number_of_heads=8,
            number_of_key_value_heads=2,
            attention_type=AttentionType.GROUPED_QUERY.value,
        )

        assert decoder.number_of_key_value_heads == 2
        assert decoder.head_dimension == 64

    @pytest.mark.parametrize("use_cross_attention", [True, False])
    @pytest.mark.parametrize("attention_type", [
        AttentionType.MULTI_HEAD.value,
        AttentionType.GROUPED_QUERY.value,
    ])
    def test_forward(self, use_cross_attention, attention_type):
        """Test forward pass with different attention configurations."""
        batch_size, seq_len, feature_len, embedding_dim = 2, 10, 20, 512

        decoder_kwargs = {
            "number_of_layers": 2,
            "embedding_dimension": embedding_dim,
            "number_of_heads": 8,
            "attention_type": attention_type,
            "use_cross_attention": use_cross_attention,
        }
        if attention_type == AttentionType.GROUPED_QUERY.value:
            decoder_kwargs["number_of_key_value_heads"] = 2

        decoder = GPTDecoder(**decoder_kwargs)

        hidden_states = torch.randn(batch_size, seq_len, embedding_dim)

        if use_cross_attention:
            encoded_features = torch.randn(batch_size, feature_len, embedding_dim)
            output, cache = decoder(
                hidden_states=hidden_states,
                encoded_features=encoded_features,
                use_cache=True,
            )
        else:
            output, cache = decoder(
                hidden_states=hidden_states,
                use_cache=True,
            )

        assert output.shape == (batch_size, seq_len, embedding_dim)
        assert cache is not None
        assert len(cache.layers) == 2

    @pytest.mark.parametrize("attention_type", [
        AttentionType.MULTI_HEAD.value,
        AttentionType.GROUPED_QUERY.value,
    ])
    def test_autoregressive_generation_with_cache(self, attention_type):
        """Test autoregressive generation with KV caching."""
        batch_size, initial_len, new_len, embedding_dim = 2, 10, 1, 512

        decoder_kwargs = {
            "number_of_layers": 2,
            "embedding_dimension": embedding_dim,
            "number_of_heads": 8,
            "attention_type": attention_type,
            "use_cross_attention": False,
        }
        if attention_type == AttentionType.GROUPED_QUERY.value:
            decoder_kwargs["number_of_key_value_heads"] = 2

        decoder = GPTDecoder(**decoder_kwargs)

        # Initial forward pass
        initial_input = torch.randn(batch_size, initial_len, embedding_dim)
        _, cache = decoder(
            hidden_states=initial_input,
            use_cache=True,
        )

        # Autoregressive step
        new_input = torch.randn(batch_size, new_len, embedding_dim)
        output, new_cache = decoder(
            hidden_states=new_input,
            decoder_cache=cache,
            use_cache=True,
        )

        assert output.shape == (batch_size, new_len, embedding_dim)
        assert new_cache.get_length() == initial_len + new_len

    @pytest.mark.parametrize("attention_type", [
        AttentionType.MULTI_HEAD.value,
        AttentionType.GROUPED_QUERY.value,
    ])
    def test_cross_attention_kv_precomputation(self, attention_type):
        """Test that cross-attention K/V are precomputed and cached."""
        batch_size, seq_len, feature_len, embedding_dim = 2, 10, 20, 512
        num_heads = 8

        decoder_kwargs = {
            "number_of_layers": 2,
            "embedding_dimension": embedding_dim,
            "number_of_heads": num_heads,
            "attention_type": attention_type,
            "use_cross_attention": True,
        }
        if attention_type == AttentionType.GROUPED_QUERY.value:
            decoder_kwargs["number_of_key_value_heads"] = 2
            num_kv_heads = 2
        else:
            num_kv_heads = num_heads

        head_dim = embedding_dim // num_heads

        decoder = GPTDecoder(**decoder_kwargs)

        hidden_states = torch.randn(batch_size, seq_len, embedding_dim)
        encoded_features = torch.randn(batch_size, feature_len, embedding_dim)

        # First forward: precompute cross KV
        _, cache = decoder(
            hidden_states=hidden_states,
            encoded_features=encoded_features,
            use_cache=True,
        )

        # Check cross KV are cached for all layers
        for layer_cache in cache.layers:
            assert layer_cache.cross_attention_keys is not None
            assert layer_cache.cross_attention_values is not None
            assert layer_cache.cross_attention_keys.shape == (batch_size, num_kv_heads, feature_len, head_dim)

        # Second forward: reuse cached cross KV (no encoded_features needed)
        new_input = torch.randn(batch_size, 1, embedding_dim)
        output, new_cache = decoder(
            hidden_states=new_input,
            decoder_cache=cache,
            use_cache=True,
        )

        assert output.shape == (batch_size, 1, embedding_dim)

    def test_causal_mask_with_cache(self):
        """Test that causal mask accounts for cache length."""
        batch_size, initial_len, new_len, embedding_dim = 2, 10, 1, 512

        decoder = GPTDecoder(
            number_of_layers=1,
            embedding_dimension=embedding_dim,
            number_of_heads=8,
            attention_type=AttentionType.MULTI_HEAD.value,
            use_cross_attention=False,
        )

        # Initial forward
        initial_input = torch.randn(batch_size, initial_len, embedding_dim)
        _, cache = decoder(
            hidden_states=initial_input,
            use_cache=True,
        )

        # Generate causal mask for next token (should account for cache)
        total_length = initial_len + new_len
        mask = decoder.generate_causal_mask(total_length, torch.device("cpu"))
        mask_slice = mask[:, :, -new_len:, :]

        # Mask should allow attention to all previous tokens
        assert mask_slice.shape == (1, 1, new_len, total_length)

    @pytest.mark.parametrize("attention_type", [
        AttentionType.MULTI_HEAD.value,
        AttentionType.GROUPED_QUERY.value,
    ])
    @pytest.mark.parametrize("positional_encoding_type", [
        None,
        PositionalEncodingType.ROPE.value,
        PositionalEncodingType.SINUSOIDAL.value,
    ])
    def test_with_positional_encodings(self, attention_type, positional_encoding_type):
        """Test decoder with different positional encoding types and attention mechanisms."""
        batch_size, seq_len, embedding_dim = 2, 10, 512

        decoder_kwargs = {
            "number_of_layers": 2,
            "embedding_dimension": embedding_dim,
            "number_of_heads": 8,
            "attention_type": attention_type,
            "positional_encoding_type": positional_encoding_type,
            "use_cross_attention": False,
        }
        if attention_type == AttentionType.GROUPED_QUERY.value:
            decoder_kwargs["number_of_key_value_heads"] = 2

        decoder = GPTDecoder(**decoder_kwargs)

        hidden_states = torch.randn(batch_size, seq_len, embedding_dim)

        output, _ = decoder(hidden_states=hidden_states)

        assert output.shape == (batch_size, seq_len, embedding_dim)

    def test_raises_error_cross_attention_without_features(self):
        """Test that error is raised when cross-attention enabled but no features provided."""
        decoder = GPTDecoder(
            number_of_layers=2,
            embedding_dimension=512,
            number_of_heads=8,
            attention_type=AttentionType.MULTI_HEAD.value,
            use_cross_attention=True,
        )

        hidden_states = torch.randn(2, 10, 512)

        with pytest.raises(ValueError, match="encoded_features required"):
            decoder(hidden_states=hidden_states, use_cache=True)