"""Tests for versatil.models.layers.transformer.autoregressive_decoder module."""

import re
from collections.abc import Callable

import numpy as np
import pytest
import torch

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType, PositionalEncodingType
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.transformer.autoregressive_decoder import GPTDecoder


@pytest.fixture
def gpt_decoder_factory() -> Callable[..., GPTDecoder]:
    """Factory for GPTDecoder modules."""

    def factory(
        number_of_layers: int = 2,
        embedding_dimension: int = 32,
        number_of_heads: int = 4,
        number_of_key_value_heads: int | None = None,
        feedforward_dimension: int | None = None,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation: str = ActivationFunction.GELU.value,
        normalization_type: str = NormalizationType.LAYER_NORM.value,
        attention_type: str = AttentionType.MULTI_HEAD.value,
        use_cross_attention: bool = False,
        positional_encoding_type: str | None = None,
        maximum_sequence_length: int = 128,
        bias: bool = True,
        normalization_epsilon: float = 1e-6,
        initializer_range: float = 0.02,
    ) -> GPTDecoder:
        return GPTDecoder(
            number_of_layers=number_of_layers,
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_key_value_heads=number_of_key_value_heads,
            feedforward_dimension=feedforward_dimension,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation=activation,
            normalization_type=normalization_type,
            attention_type=attention_type,
            use_cross_attention=use_cross_attention,
            positional_encoding_type=positional_encoding_type,
            maximum_sequence_length=maximum_sequence_length,
            bias=bias,
            normalization_epsilon=normalization_epsilon,
            initializer_range=initializer_range,
        )

    return factory


class TestGPTDecoderInitialization:
    @pytest.mark.parametrize("number_of_layers", [1, 4])
    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    @pytest.mark.parametrize("use_cross_attention", [True, False])
    def test_stores_configuration(
        self,
        gpt_decoder_factory: Callable[..., GPTDecoder],
        number_of_layers: int,
        embedding_dimension: int,
        use_cross_attention: bool,
    ):
        decoder = gpt_decoder_factory(
            number_of_layers=number_of_layers,
            embedding_dimension=embedding_dimension,
            use_cross_attention=use_cross_attention,
        )
        assert decoder.number_of_layers == number_of_layers
        assert decoder.embedding_dimension == embedding_dimension
        assert decoder.use_cross_attention == use_cross_attention

    def test_creates_correct_number_of_layers(
        self, gpt_decoder_factory: Callable[..., GPTDecoder]
    ):
        decoder = gpt_decoder_factory(number_of_layers=3)
        assert len(decoder.layers) == 3

    def test_layers_are_autoregressive(
        self, gpt_decoder_factory: Callable[..., GPTDecoder]
    ):
        decoder = gpt_decoder_factory(number_of_layers=2)
        for layer in decoder.layers:
            assert layer.autoregressive is True

    def test_no_positional_encoding_by_default(
        self, gpt_decoder_factory: Callable[..., GPTDecoder]
    ):
        decoder = gpt_decoder_factory(positional_encoding_type=None)
        assert decoder.positional_encoding is None

    @pytest.mark.parametrize(
        "positional_encoding_type",
        [PositionalEncodingType.SINUSOIDAL.value, PositionalEncodingType.LEARNED.value],
    )
    def test_positional_encoding_created_when_specified(
        self,
        gpt_decoder_factory: Callable[..., GPTDecoder],
        positional_encoding_type: str,
    ):
        decoder = gpt_decoder_factory(
            positional_encoding_type=positional_encoding_type,
        )
        assert decoder.positional_encoding is not None

    def test_gqa_requires_kv_heads(
        self, gpt_decoder_factory: Callable[..., GPTDecoder]
    ):
        with pytest.raises(
            ValueError,
            match=re.escape("number_of_key_value_heads required for GQA"),
        ):
            gpt_decoder_factory(
                attention_type=AttentionType.GROUPED_QUERY.value,
                number_of_key_value_heads=None,
            )

    def test_gqa_stores_kv_heads(self, gpt_decoder_factory: Callable[..., GPTDecoder]):
        decoder = gpt_decoder_factory(
            number_of_heads=8,
            number_of_key_value_heads=2,
            attention_type=AttentionType.GROUPED_QUERY.value,
        )
        assert decoder.number_of_key_value_heads == 2

    def test_mha_sets_kv_heads_to_query_heads(
        self, gpt_decoder_factory: Callable[..., GPTDecoder]
    ):
        decoder = gpt_decoder_factory(
            number_of_heads=8,
            attention_type=AttentionType.MULTI_HEAD.value,
        )
        assert decoder.number_of_key_value_heads == 8

    def test_head_dimension_computed(
        self, gpt_decoder_factory: Callable[..., GPTDecoder]
    ):
        decoder = gpt_decoder_factory(embedding_dimension=64, number_of_heads=8)
        assert decoder.head_dimension == 8


class TestGPTDecoderForward:
    def test_output_shape_decoder_only(
        self,
        gpt_decoder_factory: Callable[..., GPTDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        decoder = gpt_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=False,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=5, embedding_dimension=32
        )
        output, cache = decoder(hidden_states=hidden_states)
        assert output.shape == (2, 5, 32)
        assert cache is None

    def test_output_shape_with_cross_attention(
        self,
        gpt_decoder_factory: Callable[..., GPTDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        decoder = gpt_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=True,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=5, embedding_dimension=32
        )
        memory = sequence_tensor_factory(
            batch_size=2, sequence_length=8, embedding_dimension=32
        )
        output, cache = decoder(hidden_states=hidden_states, encoded_features=memory)
        assert output.shape == (2, 5, 32)

    def test_use_cache_returns_decoder_cache(
        self,
        gpt_decoder_factory: Callable[..., GPTDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        decoder = gpt_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=False,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=3, embedding_dimension=32
        )
        output, cache = decoder(hidden_states=hidden_states, use_cache=True)
        assert cache is not None
        assert len(cache.layers) == 2
        assert cache.get_length() == 3

    def test_causal_masking_earlier_tokens_unaffected_by_later_changes(
        self,
        gpt_decoder_factory: Callable[..., GPTDecoder],
        rng: np.random.Generator,
    ):
        decoder = gpt_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=False,
            initializer_range=0.5,
        )
        decoder.eval()
        hidden_states = torch.from_numpy(
            rng.standard_normal((1, 6, 32)).astype(np.float32)
        )
        output_original, _ = decoder(hidden_states=hidden_states)
        output_original = output_original.clone()
        # Modify token at position 3 (middle of sequence)
        modified = hidden_states.clone()
        modified[0, 3, :] *= 100.0
        output_modified, _ = decoder(hidden_states=modified)
        # Tokens before position 3 (0, 1, 2) should be unchanged due to causal masking
        assert torch.allclose(output_original[0, :3], output_modified[0, :3], atol=1e-5)
        # Tokens at or after position 3 should change
        assert not torch.allclose(
            output_original[0, 3:], output_modified[0, 3:], atol=1e-5
        )

    def test_cached_forward_matches_full_forward(
        self,
        gpt_decoder_factory: Callable[..., GPTDecoder],
        rng: np.random.Generator,
    ):
        decoder = gpt_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=False,
        )
        decoder.eval()
        sequence_length = 5
        batch_size = 2
        full_sequence = torch.from_numpy(
            rng.standard_normal((batch_size, sequence_length, 32)).astype(np.float32)
        )
        # Full forward
        full_output, _ = decoder(hidden_states=full_sequence)
        # Incremental forward with caching
        cache = None
        cached_outputs = []
        for step in range(sequence_length):
            token = full_sequence[:, step : step + 1, :]
            step_output, cache = decoder(
                hidden_states=token,
                decoder_cache=cache,
                use_cache=True,
            )
            cached_outputs.append(step_output)
        cached_full_output = torch.cat(cached_outputs, dim=1)
        assert torch.allclose(full_output, cached_full_output, atol=1e-5)

    def test_cached_forward_with_cross_attention_matches_full(
        self,
        gpt_decoder_factory: Callable[..., GPTDecoder],
        rng: np.random.Generator,
    ):
        decoder = gpt_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=True,
        )
        decoder.eval()
        sequence_length = 4
        batch_size = 2
        full_sequence = torch.from_numpy(
            rng.standard_normal((batch_size, sequence_length, 32)).astype(np.float32)
        )
        memory = torch.from_numpy(
            rng.standard_normal((batch_size, 6, 32)).astype(np.float32)
        )
        # Full forward
        full_output, _ = decoder(hidden_states=full_sequence, encoded_features=memory)
        # Incremental forward
        cache = None
        cached_outputs = []
        for step in range(sequence_length):
            token = full_sequence[:, step : step + 1, :]
            step_output, cache = decoder(
                hidden_states=token,
                encoded_features=memory,
                decoder_cache=cache,
                use_cache=True,
            )
            cached_outputs.append(step_output)
        cached_full_output = torch.cat(cached_outputs, dim=1)
        assert torch.allclose(full_output, cached_full_output, atol=1e-5)

    def test_cross_attention_without_features_or_cache_raises(
        self,
        gpt_decoder_factory: Callable[..., GPTDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        decoder = gpt_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=True,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=5, embedding_dimension=32
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                "encoded_features required when use_cross_attention=True and no cached cross KV"
            ),
        ):
            decoder(hidden_states=hidden_states, encoded_features=None)

    def test_key_padding_mask_affects_output(
        self,
        gpt_decoder_factory: Callable[..., GPTDecoder],
        rng: np.random.Generator,
        padding_mask_factory: Callable[..., torch.Tensor],
    ):
        decoder = gpt_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=False,
        )
        decoder.eval()
        hidden_states = torch.from_numpy(
            rng.standard_normal((2, 4, 32)).astype(np.float32)
        )
        mask = padding_mask_factory(
            batch_size=2, sequence_length=4, padded_positions=[[1, 2], []]
        )
        output_masked, _ = decoder(hidden_states=hidden_states, key_padding_mask=mask)
        output_unmasked, _ = decoder(hidden_states=hidden_states)
        assert not torch.allclose(output_masked[0], output_unmasked[0], atol=1e-5)

    def test_custom_self_attention_mask(
        self,
        gpt_decoder_factory: Callable[..., GPTDecoder],
        rng: np.random.Generator,
    ):
        decoder = gpt_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=False,
        )
        decoder.eval()
        hidden_states = torch.from_numpy(
            rng.standard_normal((2, 4, 32)).astype(np.float32)
        )
        # Fully open mask (no masking at all)
        open_mask = torch.zeros(2, 1, 4, 4, dtype=torch.bool)
        output_open, _ = decoder(
            hidden_states=hidden_states,
            self_attention_mask=open_mask,
        )
        output_default, _ = decoder(hidden_states=hidden_states)
        # Default is causal mask, open mask allows full attention -> different output
        assert not torch.allclose(output_open, output_default, atol=1e-5)

    @pytest.mark.parametrize(
        "positional_encoding_type",
        [PositionalEncodingType.SINUSOIDAL.value, PositionalEncodingType.LEARNED.value],
    )
    def test_with_additive_positional_encoding(
        self,
        gpt_decoder_factory: Callable[..., GPTDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        positional_encoding_type: str,
    ):
        decoder = gpt_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=False,
            positional_encoding_type=positional_encoding_type,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=5, embedding_dimension=32
        )
        output, _ = decoder(hidden_states=hidden_states)
        assert output.shape == (2, 5, 32)

    def test_with_rope_positional_encoding(
        self,
        gpt_decoder_factory: Callable[..., GPTDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        decoder = gpt_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=False,
            positional_encoding_type=PositionalEncodingType.ROPE.value,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=5, embedding_dimension=32
        )
        output, _ = decoder(hidden_states=hidden_states)
        assert output.shape == (2, 5, 32)

    def test_cached_forward_with_rope_matches_full_forward(
        self,
        gpt_decoder_factory: Callable[..., GPTDecoder],
        rng: np.random.Generator,
    ):
        decoder = gpt_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=False,
            positional_encoding_type=PositionalEncodingType.ROPE.value,
        )
        decoder.eval()
        sequence_length = 5
        batch_size = 2
        full_sequence = torch.from_numpy(
            rng.standard_normal((batch_size, sequence_length, 32)).astype(np.float32)
        )
        full_output, _ = decoder(hidden_states=full_sequence)
        cache = None
        cached_outputs = []
        for step in range(sequence_length):
            token = full_sequence[:, step : step + 1, :]
            step_output, cache = decoder(
                hidden_states=token,
                decoder_cache=cache,
                use_cache=True,
            )
            cached_outputs.append(step_output)
        cached_full_output = torch.cat(cached_outputs, dim=1)
        assert torch.allclose(full_output, cached_full_output, atol=1e-5)

    def test_cached_forward_with_sinusoidal_matches_full_forward(
        self,
        gpt_decoder_factory: Callable[..., GPTDecoder],
        rng: np.random.Generator,
    ):
        decoder = gpt_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=False,
            positional_encoding_type=PositionalEncodingType.SINUSOIDAL.value,
        )
        decoder.eval()
        sequence_length = 5
        batch_size = 2
        full_sequence = torch.from_numpy(
            rng.standard_normal((batch_size, sequence_length, 32)).astype(np.float32)
        )
        full_output, _ = decoder(hidden_states=full_sequence)
        cache = None
        cached_outputs = []
        for step in range(sequence_length):
            token = full_sequence[:, step : step + 1, :]
            step_output, cache = decoder(
                hidden_states=token,
                decoder_cache=cache,
                use_cache=True,
            )
            cached_outputs.append(step_output)
        cached_full_output = torch.cat(cached_outputs, dim=1)
        assert torch.allclose(full_output, cached_full_output, atol=1e-5)


class TestGPTDecoderPrecomputeCrossAttentionKV:
    def test_returns_one_kv_pair_per_layer(
        self,
        gpt_decoder_factory: Callable[..., GPTDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        decoder = gpt_decoder_factory(
            number_of_layers=3,
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=True,
        )
        memory = sequence_tensor_factory(
            batch_size=2, sequence_length=8, embedding_dimension=32
        )
        cross_kv = decoder.precompute_cross_attention_kv(encoded_features=memory)
        assert len(cross_kv) == 3

    def test_kv_shapes(
        self,
        gpt_decoder_factory: Callable[..., GPTDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        decoder = gpt_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=True,
        )
        memory = sequence_tensor_factory(
            batch_size=2, sequence_length=8, embedding_dimension=32
        )
        cross_kv = decoder.precompute_cross_attention_kv(encoded_features=memory)
        for keys, values in cross_kv:
            # (B, kv_heads, memory_length, head_dim)
            assert keys.shape == (2, 4, 8, 8)
            assert values.shape == (2, 4, 8, 8)

    def test_gqa_precompute_shapes(
        self,
        gpt_decoder_factory: Callable[..., GPTDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        decoder = gpt_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            number_of_heads=4,
            number_of_key_value_heads=2,
            attention_type=AttentionType.GROUPED_QUERY.value,
            use_cross_attention=True,
        )
        memory = sequence_tensor_factory(
            batch_size=2, sequence_length=8, embedding_dimension=32
        )
        cross_kv = decoder.precompute_cross_attention_kv(encoded_features=memory)
        for keys, values in cross_kv:
            # GQA: kv_heads=2, head_dim=8
            assert keys.shape == (2, 2, 8, 8)
            assert values.shape == (2, 2, 8, 8)


class TestGPTDecoderCacheManagement:
    def test_cache_accumulates_across_steps(
        self,
        gpt_decoder_factory: Callable[..., GPTDecoder],
        rng: np.random.Generator,
    ):
        decoder = gpt_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=False,
        )
        decoder.eval()
        batch_size = 2
        cache = None
        for step in range(5):
            token = torch.from_numpy(
                rng.standard_normal((batch_size, 1, 32)).astype(np.float32)
            )
            _, cache = decoder(
                hidden_states=token,
                decoder_cache=cache,
                use_cache=True,
            )
            assert cache.get_length() == step + 1

    def test_cache_key_padding_mask_propagated(
        self,
        gpt_decoder_factory: Callable[..., GPTDecoder],
        rng: np.random.Generator,
    ):
        decoder = gpt_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=False,
        )
        decoder.eval()
        batch_size = 2
        hidden_states = torch.from_numpy(
            rng.standard_normal((batch_size, 3, 32)).astype(np.float32)
        )
        key_padding = torch.tensor([[False, True, False], [False, False, True]])
        _, cache = decoder(
            hidden_states=hidden_states,
            key_padding_mask=key_padding,
            use_cache=True,
        )
        assert cache.key_padding_mask is not None
        assert cache.key_padding_mask.shape[1] == 3

    def test_cross_attention_kv_cached_across_steps(
        self,
        gpt_decoder_factory: Callable[..., GPTDecoder],
        rng: np.random.Generator,
    ):
        decoder = gpt_decoder_factory(
            number_of_layers=2,
            embedding_dimension=32,
            number_of_heads=4,
            use_cross_attention=True,
        )
        decoder.eval()
        batch_size = 2
        memory = torch.from_numpy(
            rng.standard_normal((batch_size, 6, 32)).astype(np.float32)
        )
        first_token = torch.from_numpy(
            rng.standard_normal((batch_size, 1, 32)).astype(np.float32)
        )
        _, cache = decoder(
            hidden_states=first_token,
            encoded_features=memory,
            use_cache=True,
        )
        # Cross-attention K/V should be cached
        for layer_cache in cache.layers:
            assert layer_cache.cross_attention_keys is not None
            assert layer_cache.cross_attention_values is not None
        # Second step should use cached cross K/V (no need to pass memory again
        # through the decoder, since it retrieves from cache)
        second_token = torch.from_numpy(
            rng.standard_normal((batch_size, 1, 32)).astype(np.float32)
        )
        output, cache_2 = decoder(
            hidden_states=second_token,
            encoded_features=memory,
            decoder_cache=cache,
            use_cache=True,
        )
        assert output.shape == (batch_size, 1, 32)
        assert cache_2.get_length() == 2
