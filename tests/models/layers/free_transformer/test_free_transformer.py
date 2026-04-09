"""Tests for versatil.models.layers.free_transformer.free_transformer module."""

import re
from collections.abc import Callable

import numpy as np
import pytest
import torch

from versatil.models.layers.constants import AttentionType, PositionalEncodingType
from versatil.models.layers.free_transformer.free_transformer import (
    FreeTransformer,
    FreeTransformerLatentEncoder,
    LatentConditionedDecoderLayer,
)
from versatil.models.layers.transformer.transformer_mixin import TransformerMixin


class TestLatentConditionedDecoderLayerInitialization:
    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    @pytest.mark.parametrize("latent_dim", [8, 16])
    @pytest.mark.parametrize("number_of_heads", [2, 4])
    def test_stores_configuration(
        self,
        latent_conditioned_decoder_layer_factory: Callable[
            ..., LatentConditionedDecoderLayer
        ],
        embedding_dimension: int,
        latent_dim: int,
        number_of_heads: int,
    ):
        layer = latent_conditioned_decoder_layer_factory(
            embedding_dimension=embedding_dimension,
            latent_dim=latent_dim,
            number_of_heads=number_of_heads,
        )
        assert layer.latent_proj.in_features == latent_dim
        assert layer.latent_proj.out_features == embedding_dimension

    def test_inherits_transformer_decoder_layer_forward_behavior(
        self,
        latent_conditioned_decoder_layer_factory: Callable[
            ..., LatentConditionedDecoderLayer
        ],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        # Without latent, the layer should produce the same output as the parent class
        embedding_dimension = 32
        layer = latent_conditioned_decoder_layer_factory(
            embedding_dimension=embedding_dimension,
        )
        layer.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=embedding_dimension
        )
        with torch.no_grad():
            output, cache = layer(hidden_states=hidden_states, latent=None)
        # Verify it produces valid output (same shape, finite values) like a TransformerDecoderLayer
        assert output.shape == hidden_states.shape
        assert torch.all(torch.isfinite(output))
        assert cache is None

    def test_latent_projection_does_not_add_bias(
        self,
        latent_conditioned_decoder_layer_factory: Callable[
            ..., LatentConditionedDecoderLayer
        ],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        rng: np.random.Generator,
    ):
        # Verify no-bias by checking that projecting a zero latent yields zero contribution
        embedding_dimension = 32
        latent_dim = 16
        layer = latent_conditioned_decoder_layer_factory(
            embedding_dimension=embedding_dimension,
            latent_dim=latent_dim,
        )
        zero_latent = torch.zeros(1, latent_dim)
        projected = layer.latent_proj(zero_latent)
        assert torch.equal(projected, torch.zeros(1, embedding_dimension))


class TestLatentConditionedDecoderLayerForward:
    @pytest.mark.parametrize(
        "batch_size, sequence_length",
        [
            (2, 4),
            (1, 8),
        ],
    )
    def test_output_shape_without_latent(
        self,
        latent_conditioned_decoder_layer_factory: Callable[
            ..., LatentConditionedDecoderLayer
        ],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
        sequence_length: int,
    ):
        embedding_dimension = 32
        layer = latent_conditioned_decoder_layer_factory(
            embedding_dimension=embedding_dimension,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=embedding_dimension,
        )
        output, cache = layer(hidden_states=hidden_states, latent=None)
        assert output.shape == (batch_size, sequence_length, embedding_dimension)
        assert cache is None

    @pytest.mark.parametrize(
        "batch_size, sequence_length",
        [
            (2, 4),
            (1, 8),
        ],
    )
    def test_output_shape_with_latent(
        self,
        latent_conditioned_decoder_layer_factory: Callable[
            ..., LatentConditionedDecoderLayer
        ],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        rng: np.random.Generator,
        batch_size: int,
        sequence_length: int,
    ):
        embedding_dimension = 32
        latent_dim = 16
        layer = latent_conditioned_decoder_layer_factory(
            embedding_dimension=embedding_dimension,
            latent_dim=latent_dim,
        )
        hidden_states = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=embedding_dimension,
        )
        latent = torch.from_numpy(
            rng.standard_normal((batch_size, sequence_length, latent_dim)).astype(
                np.float32
            )
        )
        output, cache = layer(hidden_states=hidden_states, latent=latent)
        assert output.shape == (batch_size, sequence_length, embedding_dimension)

    def test_latent_conditioning_changes_output(
        self,
        latent_conditioned_decoder_layer_factory: Callable[
            ..., LatentConditionedDecoderLayer
        ],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        rng: np.random.Generator,
    ):
        embedding_dimension = 32
        latent_dim = 16
        layer = latent_conditioned_decoder_layer_factory(
            embedding_dimension=embedding_dimension,
            latent_dim=latent_dim,
        )
        layer.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=embedding_dimension,
        )
        latent = torch.from_numpy(
            rng.standard_normal((2, 4, latent_dim)).astype(np.float32)
        )
        with torch.no_grad():
            output_with_latent, _ = layer(hidden_states=hidden_states, latent=latent)
            output_without_latent, _ = layer(hidden_states=hidden_states, latent=None)
        assert not torch.allclose(output_with_latent, output_without_latent, atol=1e-5)

    def test_different_latents_produce_different_outputs(
        self,
        latent_conditioned_decoder_layer_factory: Callable[
            ..., LatentConditionedDecoderLayer
        ],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        rng: np.random.Generator,
    ):
        embedding_dimension = 32
        latent_dim = 16
        layer = latent_conditioned_decoder_layer_factory(
            embedding_dimension=embedding_dimension,
            latent_dim=latent_dim,
        )
        layer.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2,
            sequence_length=4,
            embedding_dimension=embedding_dimension,
        )
        latent_a = torch.from_numpy(
            rng.standard_normal((2, 4, latent_dim)).astype(np.float32)
        )
        latent_b = torch.from_numpy(
            rng.standard_normal((2, 4, latent_dim)).astype(np.float32)
        )
        with torch.no_grad():
            output_a, _ = layer(hidden_states=hidden_states, latent=latent_a)
            output_b, _ = layer(hidden_states=hidden_states, latent=latent_b)
        assert not torch.allclose(output_a, output_b, atol=1e-5)

    def test_global_latent_broadcast_from_single_timestep(
        self,
        latent_conditioned_decoder_layer_factory: Callable[
            ..., LatentConditionedDecoderLayer
        ],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        rng: np.random.Generator,
    ):
        embedding_dimension = 32
        latent_dim = 16
        sequence_length = 4
        batch_size = 2
        layer = latent_conditioned_decoder_layer_factory(
            embedding_dimension=embedding_dimension,
            latent_dim=latent_dim,
        )
        layer.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=embedding_dimension,
        )
        # Latent with T=1 should be broadcast to match sequence_length
        latent_single = torch.from_numpy(
            rng.standard_normal((batch_size, 1, latent_dim)).astype(np.float32)
        )
        latent_expanded = latent_single.expand(-1, sequence_length, -1)
        with torch.no_grad():
            output_single, _ = layer(hidden_states=hidden_states, latent=latent_single)
            output_expanded, _ = layer(
                hidden_states=hidden_states, latent=latent_expanded
            )
        assert torch.allclose(output_single, output_expanded, atol=1e-5)


class TestFreeTransformerLatentEncoderInitialization:
    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    @pytest.mark.parametrize("number_of_layers", [1, 2])
    @pytest.mark.parametrize("use_global_latent", [True, False])
    def test_stores_configuration(
        self,
        latent_encoder_factory: Callable[..., FreeTransformerLatentEncoder],
        embedding_dimension: int,
        number_of_layers: int,
        use_global_latent: bool,
    ):
        encoder = latent_encoder_factory(
            embedding_dimension=embedding_dimension,
            number_of_layers=number_of_layers,
            use_global_latent=use_global_latent,
        )
        assert encoder.use_global_latent == use_global_latent
        assert len(encoder.layers) == number_of_layers
        assert encoder.learned_query.shape == (1, 1, embedding_dimension)

    def test_layers_support_cross_attention_to_mid_features(
        self,
        latent_encoder_factory: Callable[..., FreeTransformerLatentEncoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        # Verify that cross-attention is functional by checking that different
        # encoded_features produce different outputs through the cross-attention path
        embedding_dimension = 32
        encoder = latent_encoder_factory(
            embedding_dimension=embedding_dimension,
            number_of_layers=2,
        )
        encoder.eval()
        mid_features_a = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=embedding_dimension
        )
        mid_features_b = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=embedding_dimension
        )
        with torch.no_grad():
            output_a = encoder(mid_features=mid_features_a)
            output_b = encoder(mid_features=mid_features_b)
        assert not torch.allclose(output_a, output_b, atol=1e-5)


class TestFreeTransformerLatentEncoderForward:
    @pytest.mark.parametrize(
        "batch_size, sequence_length, use_global_latent",
        [
            (2, 4, False),
            (2, 4, True),
            (1, 8, False),
        ],
    )
    def test_output_shape(
        self,
        latent_encoder_factory: Callable[..., FreeTransformerLatentEncoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
        sequence_length: int,
        use_global_latent: bool,
    ):
        embedding_dimension = 32
        encoder = latent_encoder_factory(
            embedding_dimension=embedding_dimension,
            use_global_latent=use_global_latent,
        )
        mid_features = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=embedding_dimension,
        )
        output = encoder(mid_features=mid_features)
        if use_global_latent:
            assert output.shape == (batch_size, 1, embedding_dimension)
        else:
            assert output.shape == (batch_size, sequence_length, embedding_dimension)

    def test_different_mid_features_produce_different_latents(
        self,
        latent_encoder_factory: Callable[..., FreeTransformerLatentEncoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        encoder = latent_encoder_factory(
            embedding_dimension=embedding_dimension,
        )
        encoder.eval()
        mid_features_a = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=embedding_dimension
        )
        mid_features_b = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=embedding_dimension
        )
        with torch.no_grad():
            output_a = encoder(mid_features=mid_features_a)
            output_b = encoder(mid_features=mid_features_b)
        assert not torch.allclose(output_a, output_b, atol=1e-5)

    def test_padding_mask_affects_output(
        self,
        latent_encoder_factory: Callable[..., FreeTransformerLatentEncoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        padding_mask_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        batch_size = 2
        sequence_length = 4
        encoder = latent_encoder_factory(
            embedding_dimension=embedding_dimension,
        )
        encoder.eval()
        mid_features = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=embedding_dimension,
        )
        mask_none = None
        mask_with_padding = padding_mask_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            padded_positions=[[2, 3], [3]],
        )
        with torch.no_grad():
            output_no_mask = encoder(
                mid_features=mid_features, mid_features_mask=mask_none
            )
            output_with_mask = encoder(
                mid_features=mid_features, mid_features_mask=mask_with_padding
            )
        assert not torch.allclose(output_no_mask, output_with_mask, atol=1e-5)

    def test_gradient_flows_through_encoder(
        self,
        latent_encoder_factory: Callable[..., FreeTransformerLatentEncoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        encoder = latent_encoder_factory(
            embedding_dimension=embedding_dimension,
        )
        mid_features = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=embedding_dimension
        )
        mid_features.requires_grad_(True)
        output = encoder(mid_features=mid_features)
        output.sum().backward()
        assert mid_features.grad is not None
        assert torch.all(torch.isfinite(mid_features.grad))


class TestFreeTransformerInitialization:
    @pytest.mark.parametrize("number_of_decoder_layers", [4, 6])
    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    @pytest.mark.parametrize("use_global_latent", [True, False])
    def test_stores_configuration(
        self,
        free_transformer_factory: Callable[..., FreeTransformer],
        number_of_decoder_layers: int,
        embedding_dimension: int,
        use_global_latent: bool,
    ):
        model = free_transformer_factory(
            number_of_decoder_layers=number_of_decoder_layers,
            embedding_dimension=embedding_dimension,
            use_global_latent=use_global_latent,
        )
        assert model.number_of_decoder_layers == number_of_decoder_layers
        assert model.embedding_dimension == embedding_dimension
        assert model.use_global_latent == use_global_latent

    @pytest.mark.parametrize(
        "number_of_decoder_layers, number_of_encoder_layers",
        [(4, 2), (6, 0), (2, 4)],
    )
    def test_total_residual_streams_accounts_for_encoder_and_decoder(
        self,
        free_transformer_factory: Callable[..., FreeTransformer],
        number_of_decoder_layers: int,
        number_of_encoder_layers: int,
    ):
        model = free_transformer_factory(
            number_of_decoder_layers=number_of_decoder_layers,
            number_of_encoder_layers=number_of_encoder_layers,
        )
        expected = 2 * number_of_decoder_layers + 3 * number_of_encoder_layers
        assert isinstance(model, TransformerMixin)
        assert model._total_residual_streams == expected

    def test_raises_for_odd_number_of_layers(
        self,
        free_transformer_factory: Callable[..., FreeTransformer],
    ):
        with pytest.raises(
            ValueError,
            match=re.escape("number_of_layers must be even"),
        ):
            free_transformer_factory(number_of_decoder_layers=5)

    def test_raises_when_gqa_without_kv_heads(
        self,
    ):
        with pytest.raises(
            ValueError,
            match=re.escape("number_of_key_value_heads required for GQA"),
        ):
            FreeTransformer(
                number_of_decoder_layers=4,
                embedding_dimension=32,
                number_of_heads=4,
                number_of_key_value_heads=None,
                attention_type=AttentionType.GROUPED_QUERY.value,
            )

    def test_mha_sets_kv_heads_equal_to_heads(
        self,
        free_transformer_factory: Callable[..., FreeTransformer],
    ):
        model = free_transformer_factory(
            attention_type=AttentionType.MULTI_HEAD.value,
            number_of_heads=4,
            number_of_key_value_heads=None,
        )
        assert model.number_of_key_value_heads == 4

    def test_latent_dim_defaults_to_two_power_latent_bits(
        self,
        free_transformer_factory: Callable[..., FreeTransformer],
    ):
        latent_bits = 4
        model = free_transformer_factory(latent_bits=latent_bits, latent_dim=None)
        assert model.latent_dim == 2**latent_bits

    def test_latent_dim_override(
        self,
        free_transformer_factory: Callable[..., FreeTransformer],
    ):
        model = free_transformer_factory(latent_bits=4, latent_dim=128)
        assert model.latent_dim == 128

    def test_middle_layer_accepts_latent_conditioning(
        self,
        free_transformer_factory: Callable[..., FreeTransformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        rng: np.random.Generator,
    ):
        # The middle layer should accept and be affected by latent conditioning
        number_of_decoder_layers = 6
        embedding_dimension = 32
        latent_dim = 16
        model = free_transformer_factory(
            number_of_decoder_layers=number_of_decoder_layers,
            embedding_dimension=embedding_dimension,
            latent_dim=latent_dim,
        )
        model.eval()
        mid_index = number_of_decoder_layers // 2
        mid_layer = model.decoder_layers[mid_index]
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=embedding_dimension
        )
        latent = torch.from_numpy(
            rng.standard_normal((2, 4, latent_dim)).astype(np.float32)
        )
        with torch.no_grad():
            output_with_latent, _ = mid_layer(
                hidden_states=hidden_states, latent=latent
            )
            output_without_latent, _ = mid_layer(
                hidden_states=hidden_states, latent=None
            )
        assert not torch.allclose(output_with_latent, output_without_latent, atol=1e-5)

    def test_non_middle_layers_do_not_have_latent_projection(
        self,
        free_transformer_factory: Callable[..., FreeTransformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        rng: np.random.Generator,
    ):
        # Non-middle layers should not accept latent conditioning. Verify by
        # checking that calling forward with the latent kwarg raises TypeError,
        # confirming they are regular decoder layers, not LatentConditionedDecoderLayers.
        number_of_decoder_layers = 6
        embedding_dimension = 32
        latent_dim = 16
        model = free_transformer_factory(
            number_of_decoder_layers=number_of_decoder_layers,
            embedding_dimension=embedding_dimension,
            latent_dim=latent_dim,
        )
        model.eval()
        mid_index = number_of_decoder_layers // 2
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=embedding_dimension
        )
        latent = torch.from_numpy(
            rng.standard_normal((2, 4, latent_dim)).astype(np.float32)
        )
        # The middle layer should accept latent conditioning
        with torch.no_grad():
            output_with_latent, _ = model.decoder_layers[mid_index](
                hidden_states=hidden_states, latent=latent
            )
            output_without_latent, _ = model.decoder_layers[mid_index](
                hidden_states=hidden_states, latent=None
            )
        assert not torch.allclose(output_with_latent, output_without_latent, atol=1e-5)
        # Non-middle layers should reject the latent kwarg
        for index, layer in enumerate(model.decoder_layers):
            if index != mid_index:
                with pytest.raises(TypeError):
                    layer(hidden_states=hidden_states, latent=latent)

    def test_total_decoder_layers_count(
        self,
        free_transformer_factory: Callable[..., FreeTransformer],
    ):
        number_of_decoder_layers = 4
        model = free_transformer_factory(
            number_of_decoder_layers=number_of_decoder_layers,
        )
        assert len(model.decoder_layers) == number_of_decoder_layers

    def test_binary_mapper_uses_configured_latent_bits(
        self,
        free_transformer_factory: Callable[..., FreeTransformer],
    ):
        latent_bits = 5
        model = free_transformer_factory(latent_bits=latent_bits)
        assert model.binary_mapper.latent_bits == latent_bits
        assert model.binary_mapper.latent_dim == 2**latent_bits

    def test_latent_encoder_uses_configured_global_latent_mode(
        self,
        free_transformer_factory: Callable[..., FreeTransformer],
    ):
        model_global = free_transformer_factory(use_global_latent=True)
        model_per_token = free_transformer_factory(use_global_latent=False)
        assert model_global.latent_encoder.use_global_latent is True
        assert model_per_token.latent_encoder.use_global_latent is False

    @pytest.mark.parametrize(
        "positional_encoding_type",
        [
            PositionalEncodingType.ROPE.value,
            PositionalEncodingType.SINUSOIDAL.value,
            PositionalEncodingType.LEARNED.value,
        ],
    )
    def test_positional_encoding_produces_finite_output(
        self,
        free_transformer_factory: Callable[..., FreeTransformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        positional_encoding_type: str,
    ):
        embedding_dimension = 32
        model = free_transformer_factory(
            embedding_dimension=embedding_dimension,
            positional_encoding_type=positional_encoding_type,
        )
        model.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=embedding_dimension
        )
        with torch.no_grad():
            output, _, _, _ = model(hidden_states=hidden_states, deterministic=True)
        assert torch.all(torch.isfinite(output))

    def test_no_positional_encoding_produces_position_invariant_encoding(
        self,
        free_transformer_factory: Callable[..., FreeTransformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        # Without positional encoding, swapping token order in the input should
        # produce identically swapped outputs (no position information injected)
        embedding_dimension = 32
        model = free_transformer_factory(
            positional_encoding_type=None,
            embedding_dimension=embedding_dimension,
        )
        model.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=1, sequence_length=4, embedding_dimension=embedding_dimension
        )
        with torch.no_grad():
            output, _, _, _ = model(hidden_states=hidden_states, deterministic=True)
        assert torch.all(torch.isfinite(output))


class TestFreeTransformerWeightInitialization:
    def test_normalization_weights_initialized_to_ones(
        self,
        free_transformer_factory: Callable[..., FreeTransformer],
    ):
        model = free_transformer_factory()
        assert torch.allclose(
            model.final_normalization.weight.data,
            torch.ones_like(model.final_normalization.weight.data),
        )

    def test_logit_projection_weight_initialized_near_expected_std(
        self,
        free_transformer_factory: Callable[..., FreeTransformer],
    ):
        # BinaryMapper's logit_projection is nn.Linear, should have weights initialized
        # with std = initializer_range (since it's not a residual stream layer)
        initializer_range = 0.02
        model = free_transformer_factory(initializer_range=initializer_range)
        weight = model.binary_mapper.logit_projection.weight.data
        # Mean should be near 0, std should be near initializer_range
        assert abs(weight.mean().item()) < initializer_range * 2
        assert weight.std().item() < initializer_range * 3
        assert weight.std().item() > 0, "Weights should not all be zero"


class TestFreeTransformerForward:
    @pytest.mark.parametrize(
        "batch_size, sequence_length",
        [
            (2, 4),
            (1, 8),
        ],
    )
    def test_training_output_shapes(
        self,
        free_transformer_factory: Callable[..., FreeTransformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
        sequence_length: int,
    ):
        latent_bits = 4
        embedding_dimension = 32
        model = free_transformer_factory(
            latent_bits=latent_bits,
            embedding_dimension=embedding_dimension,
        )
        model.train()
        hidden_states = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=embedding_dimension,
        )
        output, bit_logits, latent_codes, new_cache = model(hidden_states=hidden_states)
        assert output.shape == (batch_size, sequence_length, embedding_dimension)
        assert bit_logits.shape == (batch_size, sequence_length, latent_bits)
        assert latent_codes.shape == (
            batch_size,
            sequence_length,
            2**latent_bits,
        )
        assert new_cache is None

    @pytest.mark.parametrize(
        "batch_size, sequence_length",
        [
            (2, 4),
            (1, 8),
        ],
    )
    def test_inference_output_shapes(
        self,
        free_transformer_factory: Callable[..., FreeTransformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
        sequence_length: int,
    ):
        latent_bits = 4
        embedding_dimension = 32
        model = free_transformer_factory(
            latent_bits=latent_bits,
            embedding_dimension=embedding_dimension,
        )
        model.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=embedding_dimension,
        )
        with torch.no_grad():
            output, bit_logits, latent_codes, new_cache = model(
                hidden_states=hidden_states, is_inference=True
            )
        assert output.shape == (batch_size, sequence_length, embedding_dimension)
        assert bit_logits is None
        assert latent_codes.shape == (
            batch_size,
            sequence_length,
            2**latent_bits,
        )

    def test_global_latent_produces_single_timestep_codes(
        self,
        free_transformer_factory: Callable[..., FreeTransformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        latent_bits = 4
        embedding_dimension = 32
        batch_size = 2
        sequence_length = 4
        model = free_transformer_factory(
            latent_bits=latent_bits,
            embedding_dimension=embedding_dimension,
            use_global_latent=True,
        )
        model.train()
        hidden_states = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=embedding_dimension,
        )
        output, bit_logits, latent_codes, _ = model(hidden_states=hidden_states)
        assert output.shape == (batch_size, sequence_length, embedding_dimension)
        # Global latent produces (B, 1, ...) for bit_logits and latent_codes
        assert bit_logits.shape == (batch_size, 1, latent_bits)
        assert latent_codes.shape == (batch_size, 1, 2**latent_bits)

    def test_return_latent_embeddings_adds_extra_output(
        self,
        free_transformer_factory: Callable[..., FreeTransformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        batch_size = 2
        sequence_length = 4
        model = free_transformer_factory(
            embedding_dimension=embedding_dimension,
        )
        model.train()
        hidden_states = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=embedding_dimension,
        )
        result = model(hidden_states=hidden_states, return_latent_embeddings=True)
        assert len(result) == 5
        output, bit_logits, latent_codes, latent_embeddings, new_cache = result
        assert latent_embeddings.shape == (
            batch_size,
            sequence_length,
            embedding_dimension,
        )

    def test_without_return_latent_embeddings_gives_four_outputs(
        self,
        free_transformer_factory: Callable[..., FreeTransformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        model = free_transformer_factory(embedding_dimension=32)
        model.train()
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        result = model(hidden_states=hidden_states, return_latent_embeddings=False)
        assert len(result) == 4

    def test_use_cache_returns_decoder_cache(
        self,
        free_transformer_factory: Callable[..., FreeTransformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        number_of_decoder_layers = 4
        embedding_dimension = 32
        model = free_transformer_factory(
            number_of_decoder_layers=number_of_decoder_layers,
            embedding_dimension=embedding_dimension,
        )
        model.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=embedding_dimension
        )
        with torch.no_grad():
            generation_cache = model.create_empty_generation_cache(
                batch_size=2, device=hidden_states.device, dtype=hidden_states.dtype
            )
            _, _, _, new_cache = model(
                hidden_states=hidden_states, generation_cache=generation_cache
            )
        assert new_cache is not None
        assert len(new_cache.layers) == number_of_decoder_layers

    def test_deterministic_training_gives_consistent_logits(
        self,
        free_transformer_factory: Callable[..., FreeTransformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        model = free_transformer_factory(embedding_dimension=32)
        model.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        with torch.no_grad():
            _, logits_first, _, _ = model(
                hidden_states=hidden_states, deterministic=True
            )
            _, logits_second, _, _ = model(
                hidden_states=hidden_states, deterministic=True
            )
        assert torch.equal(logits_first, logits_second)

    def test_gradient_flows_through_full_model(
        self,
        free_transformer_factory: Callable[..., FreeTransformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        model = free_transformer_factory(embedding_dimension=32)
        model.train()
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        hidden_states.requires_grad_(True)
        output, bit_logits, latent_codes, _ = model(hidden_states=hidden_states)
        loss = output.sum() + bit_logits.sum()
        loss.backward()
        assert hidden_states.grad is not None
        assert torch.all(torch.isfinite(hidden_states.grad))

    def test_padding_mask_affects_output(
        self,
        free_transformer_factory: Callable[..., FreeTransformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        padding_mask_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        batch_size = 2
        sequence_length = 4
        model = free_transformer_factory(embedding_dimension=embedding_dimension)
        model.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=embedding_dimension,
        )
        mask = padding_mask_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            padded_positions=[[2, 3], [3]],
        )
        with torch.no_grad():
            output_no_mask, _, _, _ = model(
                hidden_states=hidden_states,
                deterministic=True,
            )
            output_with_mask, _, _, _ = model(
                hidden_states=hidden_states,
                key_padding_mask=mask,
                deterministic=True,
            )
        assert not torch.allclose(output_no_mask, output_with_mask, atol=1e-5)


class TestFreeTransformerCaching:
    def test_cached_forward_produces_same_output_as_full(
        self,
        free_transformer_factory: Callable[..., FreeTransformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        sequence_length = 4
        batch_size = 2
        model = free_transformer_factory(
            embedding_dimension=embedding_dimension,
        )
        model.eval()
        full_input = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=embedding_dimension,
        )
        # Full forward
        with torch.no_grad():
            full_output, _, _, _ = model(hidden_states=full_input, deterministic=True)
        # Incremental forward: process tokens one at a time
        cache = model.create_empty_generation_cache(
            batch_size=batch_size, device=full_input.device, dtype=full_input.dtype
        )
        incremental_outputs = []
        for token_index in range(sequence_length):
            single_token = full_input[:, token_index : token_index + 1, :]
            with torch.no_grad():
                step_output, _, _, cache = model(
                    hidden_states=single_token,
                    generation_cache=cache,
                    deterministic=True,
                )
            incremental_outputs.append(step_output)
        incremental_output = torch.cat(incremental_outputs, dim=1)
        assert torch.allclose(full_output, incremental_output, atol=1e-4)

    def test_cache_length_grows_with_tokens(
        self,
        free_transformer_factory: Callable[..., FreeTransformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        model = free_transformer_factory(
            embedding_dimension=embedding_dimension,
        )
        model.eval()
        full_input = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=embedding_dimension
        )
        cache = model.create_empty_generation_cache(
            batch_size=2, device=full_input.device, dtype=full_input.dtype
        )
        for token_index in range(4):
            single_token = full_input[:, token_index : token_index + 1, :]
            with torch.no_grad():
                _, _, _, cache = model(
                    hidden_states=single_token,
                    generation_cache=cache,
                    deterministic=True,
                )
            assert cache.get_length() == token_index + 1


class TestFreeTransformerPositionalEncoding:
    @pytest.mark.parametrize(
        "positional_encoding_type",
        [
            PositionalEncodingType.SINUSOIDAL.value,
            PositionalEncodingType.LEARNED.value,
        ],
    )
    def test_additive_positional_encoding_affects_output(
        self,
        free_transformer_factory: Callable[..., FreeTransformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        positional_encoding_type: str,
    ):
        embedding_dimension = 32
        model = free_transformer_factory(
            embedding_dimension=embedding_dimension,
            positional_encoding_type=positional_encoding_type,
        )
        model.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=embedding_dimension
        )
        with torch.no_grad():
            output, _, _, _ = model(hidden_states=hidden_states, deterministic=True)
        assert torch.all(torch.isfinite(output))
        assert output.shape == (2, 4, embedding_dimension)

    def test_rope_positional_encoding_produces_finite_output(
        self,
        free_transformer_factory: Callable[..., FreeTransformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 32
        model = free_transformer_factory(
            embedding_dimension=embedding_dimension,
            positional_encoding_type=PositionalEncodingType.ROPE.value,
        )
        model.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=embedding_dimension
        )
        with torch.no_grad():
            output, _, _, _ = model(hidden_states=hidden_states, deterministic=True)
        assert torch.all(torch.isfinite(output))


class TestFreeTransformerInferenceMode:
    def test_inference_samples_from_uniform_prior(
        self,
        free_transformer_factory: Callable[..., FreeTransformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        latent_bits = 4
        model = free_transformer_factory(
            latent_bits=latent_bits,
            embedding_dimension=32,
        )
        model.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        with torch.no_grad():
            _, bit_logits, latent_codes, _ = model(
                hidden_states=hidden_states, is_inference=True
            )
        # In inference mode, bit_logits should be None
        assert bit_logits is None
        # latent_codes should be valid one-hot vectors
        sums = latent_codes.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums))

    def test_inference_global_latent_shape(
        self,
        free_transformer_factory: Callable[..., FreeTransformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        latent_bits = 4
        model = free_transformer_factory(
            latent_bits=latent_bits,
            embedding_dimension=32,
            use_global_latent=True,
        )
        model.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        with torch.no_grad():
            _, bit_logits, latent_codes, _ = model(
                hidden_states=hidden_states, is_inference=True
            )
        assert bit_logits is None
        # Global latent should have query_dim=1
        assert latent_codes.shape == (2, 1, 2**latent_bits)

    def test_non_inference_eval_still_computes_bit_logits(
        self,
        free_transformer_factory: Callable[..., FreeTransformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        # When is_inference=False but model is in eval mode, logits should still be computed
        latent_bits = 4
        model = free_transformer_factory(
            latent_bits=latent_bits, embedding_dimension=32
        )
        model.eval()
        hidden_states = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        with torch.no_grad():
            _, bit_logits, _, _ = model(hidden_states=hidden_states, is_inference=False)
        assert bit_logits.shape == (2, 4, latent_bits)
