"""Tests for versatil.models.layers.detr_transformer.transformer_encoder module."""
from collections.abc import Callable

import pytest
import torch

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.detr_transformer.transformer_encoder import (
    TransformerEncoder,
    TransformerEncoderLayer,
)

EMBEDDING_DIMENSION = 64
NUMBER_OF_HEADS = 4
FEEDFORWARD_DIMENSION = 128
SOURCE_LENGTH = 8


class TestTransformerEncoderLayerInitialization:

    @pytest.mark.parametrize("embedding_dimension", [EMBEDDING_DIMENSION, 128])
    @pytest.mark.parametrize("number_of_heads", [NUMBER_OF_HEADS, 8])
    @pytest.mark.parametrize("normalize_before", [False, True])
    def test_stores_configuration(
        self,
        encoder_layer_factory: Callable[..., TransformerEncoderLayer],
        embedding_dimension: int,
        number_of_heads: int,
        normalize_before: bool,
    ):
        layer = encoder_layer_factory(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            normalize_before=normalize_before,
        )
        assert layer.normalize_before == normalize_before
        assert layer.self_attention.embedding_dimension == embedding_dimension
        assert layer.self_attention.number_of_heads == number_of_heads

    @pytest.mark.parametrize("activation", [
        ActivationFunction.RELU.value,
        ActivationFunction.GELU.value,
    ])
    def test_standard_activation_creates_feedforward_linear1(
        self,
        encoder_layer_factory: Callable[..., TransformerEncoderLayer],
        activation: str,
    ):
        layer = encoder_layer_factory(activation=activation)
        assert layer.feedforward_linear1.in_features == EMBEDDING_DIMENSION
        assert layer.feedforward_linear1.out_features == FEEDFORWARD_DIMENSION
        assert layer.feedforward_linear2.in_features == FEEDFORWARD_DIMENSION
        assert layer.feedforward_linear2.out_features == EMBEDDING_DIMENSION

    def test_swiglu_activation_omits_feedforward_linear1(
        self,
        encoder_layer_factory: Callable[..., TransformerEncoderLayer],
    ):
        layer = encoder_layer_factory(activation=ActivationFunction.SWIGLU.value)
        assert not hasattr(layer, "feedforward_linear1")
        assert layer.feedforward_linear2.in_features == FEEDFORWARD_DIMENSION
        assert layer.feedforward_linear2.out_features == EMBEDDING_DIMENSION


class TestTransformerEncoderLayerForward:

    def test_output_shape(
        self,
        encoder_layer_factory: Callable[..., TransformerEncoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        layer = encoder_layer_factory()
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output = layer(source=source)
        assert output.shape == (batch_size, SOURCE_LENGTH, EMBEDDING_DIMENSION)

    def test_residual_connection_modifies_output(
        self,
        encoder_layer_factory: Callable[..., TransformerEncoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        layer = encoder_layer_factory()
        layer.eval()
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output = layer(source=source)
        # With random initialization, the output should differ from input
        assert not torch.allclose(output, source, atol=1e-5)

    def test_normalization_placement_produces_different_outputs(
        self,
        encoder_layer_factory: Callable[..., TransformerEncoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        torch.manual_seed(0)
        layer_pre = encoder_layer_factory(normalize_before=True)
        torch.manual_seed(0)
        layer_post = encoder_layer_factory(normalize_before=False)
        layer_pre.eval()
        layer_post.eval()
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output_pre = layer_pre(source=source)
        output_post = layer_post(source=source)
        assert not torch.allclose(output_pre, output_post)

    def test_positional_encoding_changes_output(
        self,
        encoder_layer_factory: Callable[..., TransformerEncoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        layer = encoder_layer_factory()
        layer.eval()
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        positional_encoding = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output_without = layer(source=source)
        output_with = layer(
            source=source,
            positional_encoding=positional_encoding,
        )
        assert not torch.allclose(output_without, output_with)

    def test_padding_mask_changes_output(
        self,
        encoder_layer_factory: Callable[..., TransformerEncoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        padding_mask_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        layer = encoder_layer_factory()
        layer.eval()
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        padding_mask = padding_mask_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            mask_last_n=1,
        )
        output_without = layer(source=source)
        output_with = layer(
            source=source,
            source_key_padding_mask=padding_mask,
        )
        assert not torch.allclose(output_without, output_with)

    @pytest.mark.parametrize("activation", [
        ActivationFunction.RELU.value,
        ActivationFunction.GELU.value,
        ActivationFunction.SWIGLU.value,
    ])
    def test_forward_with_different_activations(
        self,
        encoder_layer_factory: Callable[..., TransformerEncoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
        activation: str,
    ):
        layer = encoder_layer_factory(activation=activation)
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output = layer(source=source)
        assert output.shape == (batch_size, SOURCE_LENGTH, EMBEDDING_DIMENSION)

    def test_gradients_flow_through_all_parameters(
        self,
        encoder_layer_factory: Callable[..., TransformerEncoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        layer = encoder_layer_factory()
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        source.requires_grad_(True)
        output = layer(source=source)
        loss = output.sum()
        loss.backward()
        assert source.grad is not None
        for name, parameter in layer.named_parameters():
            assert parameter.grad is not None, f"No gradient for {name}"

    def test_source_mask_restricts_attention_pattern(
        self,
        encoder_layer_factory: Callable[..., TransformerEncoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        layer = encoder_layer_factory()
        layer.eval()
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        # Causal mask: prevent attending to future positions
        causal_mask = torch.triu(
            torch.ones(SOURCE_LENGTH, SOURCE_LENGTH, dtype=torch.bool),
            diagonal=1,
        )
        output_causal = layer(source=source, source_mask=causal_mask)
        # Modify the last token
        modified_source = source.clone()
        modified_source[:, -1, :] = 0.0
        output_modified = layer(source=modified_source, source_mask=causal_mask)
        # First token cannot see last token due to causal mask
        assert torch.allclose(
            output_causal[:, 0, :], output_modified[:, 0, :], atol=1e-5
        )


class TestTransformerEncoderInitialization:

    @pytest.mark.parametrize("number_of_layers", [1, 3])
    def test_stores_number_of_layers(
        self,
        transformer_encoder_factory: Callable[..., TransformerEncoder],
        number_of_layers: int,
    ):
        encoder = transformer_encoder_factory(number_of_layers=number_of_layers)
        assert encoder.number_of_layers == number_of_layers
        assert len(encoder.layers) == number_of_layers

    def test_layers_are_independent_copies(
        self,
        transformer_encoder_factory: Callable[..., TransformerEncoder],
    ):
        encoder = transformer_encoder_factory(number_of_layers=2)
        # Mutate one layer's weight and verify the other is unaffected
        original_weight = encoder.layers[1].normalization1.weight.data.clone()
        encoder.layers[0].normalization1.weight.data.fill_(999.0)
        assert torch.allclose(encoder.layers[1].normalization1.weight.data, original_weight)

    def test_normalization_applied_when_configured(
        self,
        transformer_encoder_factory: Callable[..., TransformerEncoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        # With normalization, the encoder output should be layer-normalized
        # (near-zero mean along the embedding dimension).
        encoder = transformer_encoder_factory(use_normalization=True)
        encoder.eval()
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output = encoder(source=source)
        mean = output.mean(dim=-1)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=0.1)

    def test_normalization_absent_when_not_configured(
        self,
        transformer_encoder_factory: Callable[..., TransformerEncoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        # Without final normalization, mutating the normalization layer weights
        # of the "with normalization" encoder should change its output but
        # the "without" encoder should be unaffected. Instead, we verify that
        # the two configurations produce different outputs when using pre-norm layers.
        encoder_with = transformer_encoder_factory(
            use_normalization=True,
            normalize_before=True,
        )
        encoder_without = transformer_encoder_factory(
            use_normalization=False,
            normalize_before=True,
        )
        # Load same layer weights
        layer_state = {
            key: value
            for key, value in encoder_with.state_dict().items()
            if key.startswith("layers.")
        }
        encoder_without.load_state_dict(layer_state, strict=False)
        encoder_with.eval()
        encoder_without.eval()
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output_with = encoder_with(source=source)
        output_without = encoder_without(source=source)
        # Pre-norm layers don't normalize at the end of each layer,
        # so the final normalization layer makes a real difference
        assert not torch.allclose(output_with, output_without, atol=1e-5)


class TestTransformerEncoderForward:

    def test_output_shape(
        self,
        transformer_encoder_factory: Callable[..., TransformerEncoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        encoder = transformer_encoder_factory()
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output = encoder(source=source)
        assert output.shape == (batch_size, SOURCE_LENGTH, EMBEDDING_DIMENSION)

    def test_multiple_layers_transform_input(
        self,
        transformer_encoder_factory: Callable[..., TransformerEncoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        encoder = transformer_encoder_factory(number_of_layers=2)
        encoder.eval()
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output = encoder(source=source)
        assert not torch.allclose(output, source, atol=1e-5)

    def test_more_layers_produces_different_output(
        self,
        transformer_encoder_factory: Callable[..., TransformerEncoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        torch.manual_seed(0)
        encoder_one = transformer_encoder_factory(number_of_layers=1)
        torch.manual_seed(0)
        encoder_two = transformer_encoder_factory(number_of_layers=2)
        encoder_one.eval()
        encoder_two.eval()
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output_one = encoder_one(source=source)
        output_two = encoder_two(source=source)
        assert not torch.allclose(output_one, output_two)

    def test_final_normalization_applied_when_configured(
        self,
        transformer_encoder_factory: Callable[..., TransformerEncoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        torch.manual_seed(0)
        encoder_with_norm = transformer_encoder_factory(use_normalization=True)
        torch.manual_seed(0)
        encoder_without_norm = transformer_encoder_factory(use_normalization=False)
        encoder_with_norm.eval()
        encoder_without_norm.eval()
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output_with = encoder_with_norm(source=source)
        output_without = encoder_without_norm(source=source)
        assert not torch.allclose(output_with, output_without)

    def test_positional_encoding_passed_through_layers(
        self,
        transformer_encoder_factory: Callable[..., TransformerEncoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        encoder = transformer_encoder_factory()
        encoder.eval()
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        positional_encoding = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output_without = encoder(source=source)
        output_with = encoder(
            source=source,
            positional_encoding=positional_encoding,
        )
        assert not torch.allclose(output_without, output_with)

    def test_mask_passed_through_all_layers(
        self,
        transformer_encoder_factory: Callable[..., TransformerEncoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        encoder = transformer_encoder_factory(number_of_layers=2)
        encoder.eval()
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        causal_mask = torch.triu(
            torch.ones(SOURCE_LENGTH, SOURCE_LENGTH, dtype=torch.bool),
            diagonal=1,
        )
        output_without = encoder(source=source)
        output_with = encoder(source=source, mask=causal_mask)
        assert not torch.allclose(output_without, output_with)

    def test_gradients_flow_through_all_parameters(
        self,
        transformer_encoder_factory: Callable[..., TransformerEncoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        encoder = transformer_encoder_factory(number_of_layers=2)
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        source.requires_grad_(True)
        output = encoder(source=source)
        loss = output.sum()
        loss.backward()
        assert source.grad is not None
        for name, parameter in encoder.named_parameters():
            assert parameter.grad is not None, f"No gradient for {name}"
