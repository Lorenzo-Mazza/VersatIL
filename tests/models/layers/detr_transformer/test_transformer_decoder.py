"""Tests for versatil.models.layers.detr_transformer.transformer_decoder module."""

from collections.abc import Callable

import pytest
import torch

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.detr_transformer.transformer_decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
)

EMBEDDING_DIMENSION = 64
NUMBER_OF_HEADS = 4
FEEDFORWARD_DIMENSION = 128
SOURCE_LENGTH = 8
TARGET_LENGTH = 6


class TestTransformerDecoderLayerInitialization:
    @pytest.mark.parametrize("embedding_dimension", [EMBEDDING_DIMENSION, 128])
    @pytest.mark.parametrize("number_of_heads", [NUMBER_OF_HEADS, 8])
    @pytest.mark.parametrize("normalize_before", [False, True])
    def test_stores_configuration(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
        embedding_dimension: int,
        number_of_heads: int,
        normalize_before: bool,
    ):
        layer = decoder_layer_factory(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            normalize_before=normalize_before,
        )
        assert layer.normalize_before == normalize_before
        assert layer.self_attention.embedding_dimension == embedding_dimension
        assert layer.self_attention.number_of_heads == number_of_heads
        assert layer.cross_attention.embedding_dimension == embedding_dimension
        assert layer.cross_attention.number_of_heads == number_of_heads

    def test_has_three_normalization_layers(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
    ):
        layer = decoder_layer_factory()
        assert layer.normalization1.normalized_shape == (EMBEDDING_DIMENSION,)
        assert layer.normalization2.normalized_shape == (EMBEDDING_DIMENSION,)
        assert layer.normalization3.normalized_shape == (EMBEDDING_DIMENSION,)

    @pytest.mark.parametrize(
        "activation",
        [
            ActivationFunction.RELU.value,
            ActivationFunction.GELU.value,
        ],
    )
    def test_standard_activation_creates_feedforward_linear1(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
        activation: str,
    ):
        layer = decoder_layer_factory(activation=activation)
        assert layer.feedforward_linear1.in_features == EMBEDDING_DIMENSION
        assert layer.feedforward_linear1.out_features == FEEDFORWARD_DIMENSION

    def test_swiglu_activation_omits_feedforward_linear1(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
    ):
        layer = decoder_layer_factory(activation=ActivationFunction.SWIGLU.value)
        assert not hasattr(layer, "feedforward_linear1")
        assert layer.feedforward_linear2.in_features == FEEDFORWARD_DIMENSION
        assert layer.feedforward_linear2.out_features == EMBEDDING_DIMENSION


class TestTransformerDecoderLayerForward:
    def test_output_shape(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        layer = decoder_layer_factory()
        target = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        memory = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output = layer(target=target, memory=memory)
        assert output.shape == (batch_size, TARGET_LENGTH, EMBEDDING_DIMENSION)

    def test_residual_connection_modifies_output(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        layer = decoder_layer_factory()
        layer.eval()
        target = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        memory = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output = layer(target=target, memory=memory)
        assert not torch.allclose(output, target, atol=1e-5)

    def test_different_memory_produces_different_output(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        layer = decoder_layer_factory()
        layer.eval()
        target = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        memory_a = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        memory_b = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output_a = layer(target=target, memory=memory_a)
        output_b = layer(target=target, memory=memory_b)
        # Cross-attention with different memory should produce different results
        assert not torch.allclose(output_a, output_b)

    def test_query_positional_encoding_changes_output(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        layer = decoder_layer_factory()
        layer.eval()
        target = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        memory = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        query_pe = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output_without = layer(target=target, memory=memory)
        output_with = layer(
            target=target,
            memory=memory,
            query_positional_encoding=query_pe,
        )
        assert not torch.allclose(output_without, output_with)

    def test_memory_positional_encoding_changes_output(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        layer = decoder_layer_factory()
        layer.eval()
        target = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        memory = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        memory_pe = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output_without = layer(target=target, memory=memory)
        output_with = layer(
            target=target,
            memory=memory,
            memory_positional_encoding=memory_pe,
        )
        assert not torch.allclose(output_without, output_with)

    def test_target_padding_mask_changes_output(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        layer = decoder_layer_factory()
        layer.eval()
        target = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        memory = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        target_padding_mask = torch.zeros(batch_size, TARGET_LENGTH, dtype=torch.bool)
        target_padding_mask[:, -1] = True
        output_without = layer(target=target, memory=memory)
        output_with = layer(
            target=target,
            memory=memory,
            target_key_padding_mask=target_padding_mask,
        )
        assert not torch.allclose(output_without, output_with)

    def test_memory_padding_mask_changes_output(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        layer = decoder_layer_factory()
        layer.eval()
        target = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        memory = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        memory_padding_mask = torch.zeros(batch_size, SOURCE_LENGTH, dtype=torch.bool)
        memory_padding_mask[:, -1] = True
        output_without = layer(target=target, memory=memory)
        output_with = layer(
            target=target,
            memory=memory,
            memory_key_padding_mask=memory_padding_mask,
        )
        assert not torch.allclose(output_without, output_with)

    @pytest.mark.parametrize(
        "activation",
        [
            ActivationFunction.RELU.value,
            ActivationFunction.GELU.value,
            ActivationFunction.SWIGLU.value,
        ],
    )
    def test_forward_with_different_activations(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
        activation: str,
    ):
        layer = decoder_layer_factory(activation=activation)
        target = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        memory = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output = layer(target=target, memory=memory)
        assert output.shape == (batch_size, TARGET_LENGTH, EMBEDDING_DIMENSION)

    def test_causal_mask_prevents_future_token_influence(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        layer = decoder_layer_factory()
        layer.eval()
        target = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        memory = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        # Causal mask: True = masked, upper triangular
        causal_mask = torch.triu(
            torch.ones(TARGET_LENGTH, TARGET_LENGTH, dtype=torch.bool),
            diagonal=1,
        )
        output_causal = layer(
            target=target,
            memory=memory,
            target_mask=causal_mask,
        )
        # Modify the last target token
        modified_target = target.clone()
        modified_target[:, -1, :] = 0.0
        output_modified = layer(
            target=modified_target,
            memory=memory,
            target_mask=causal_mask,
        )
        # With causal masking, the first token's output should be unchanged
        # because it cannot attend to future tokens
        assert torch.allclose(
            output_causal[:, 0, :], output_modified[:, 0, :], atol=1e-5
        )
        # But later tokens that CAN see the modified position should change
        assert not torch.allclose(output_causal[:, -1, :], output_modified[:, -1, :])

    def test_normalization_placement_produces_different_outputs(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        torch.manual_seed(0)
        layer_pre = decoder_layer_factory(normalize_before=True)
        torch.manual_seed(0)
        layer_post = decoder_layer_factory(normalize_before=False)
        layer_pre.eval()
        layer_post.eval()
        target = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        memory = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output_pre = layer_pre(target=target, memory=memory)
        output_post = layer_post(target=target, memory=memory)
        assert not torch.allclose(output_pre, output_post)

    def test_gradients_flow_through_all_parameters(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        layer = decoder_layer_factory()
        target = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        memory = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        target.requires_grad_(True)
        memory.requires_grad_(True)
        output = layer(target=target, memory=memory)
        loss = output.sum()
        loss.backward()
        assert target.grad is not None
        assert memory.grad is not None
        for name, parameter in layer.named_parameters():
            assert parameter.grad is not None, f"No gradient for {name}"

    def test_reset_parameters_applies_xavier_uniform(
        self,
        decoder_layer_factory: Callable[..., TransformerDecoderLayer],
    ):
        layer = decoder_layer_factory()
        for name, parameter in layer.named_parameters():
            if parameter.dim() > 1:
                fan_in = parameter.shape[1]
                fan_out = parameter.shape[0]
                xavier_bound = (6.0 / (fan_in + fan_out)) ** 0.5
                assert parameter.data.abs().max().item() <= xavier_bound + 1e-6, (
                    f"Parameter {name} exceeds Xavier uniform bound"
                )


class TestTransformerDecoderInitialization:
    @pytest.mark.parametrize("number_of_layers", [1, 3])
    def test_stores_number_of_layers(
        self,
        transformer_decoder_factory: Callable[..., TransformerDecoder],
        number_of_layers: int,
    ):
        decoder = transformer_decoder_factory(number_of_layers=number_of_layers)
        assert decoder.number_of_layers == number_of_layers
        assert len(decoder.layers) == number_of_layers

    @pytest.mark.parametrize("return_intermediate", [False, True])
    def test_stores_return_intermediate(
        self,
        transformer_decoder_factory: Callable[..., TransformerDecoder],
        return_intermediate: bool,
    ):
        decoder = transformer_decoder_factory(return_intermediate=return_intermediate)
        assert decoder.return_intermediate == return_intermediate

    def test_layers_are_independent_copies(
        self,
        transformer_decoder_factory: Callable[..., TransformerDecoder],
    ):
        decoder = transformer_decoder_factory(number_of_layers=2)
        original_weight = decoder.layers[1].normalization1.weight.data.clone()
        decoder.layers[0].normalization1.weight.data.fill_(999.0)
        assert torch.allclose(
            decoder.layers[1].normalization1.weight.data, original_weight
        )


class TestTransformerDecoderForward:
    def test_output_shape_without_intermediate(
        self,
        transformer_decoder_factory: Callable[..., TransformerDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        decoder = transformer_decoder_factory(
            return_intermediate=False,
            number_of_layers=2,
        )
        target = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        memory = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output = decoder(target=target, memory=memory)
        # Without intermediate: (1, B, T, C)
        assert output.shape == (1, batch_size, TARGET_LENGTH, EMBEDDING_DIMENSION)

    @pytest.mark.parametrize("number_of_layers", [2, 3])
    def test_output_shape_with_intermediate(
        self,
        transformer_decoder_factory: Callable[..., TransformerDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
        number_of_layers: int,
    ):
        decoder = transformer_decoder_factory(
            return_intermediate=True,
            number_of_layers=number_of_layers,
        )
        target = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        memory = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output = decoder(target=target, memory=memory)
        # With intermediate: (number_of_layers, B, T, C)
        assert output.shape == (
            number_of_layers,
            batch_size,
            TARGET_LENGTH,
            EMBEDDING_DIMENSION,
        )

    def test_intermediate_last_layer_matches_final_output(
        self,
        transformer_decoder_factory: Callable[..., TransformerDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        number_of_layers = 3
        torch.manual_seed(0)
        decoder_intermediate = transformer_decoder_factory(
            return_intermediate=True,
            number_of_layers=number_of_layers,
        )
        torch.manual_seed(0)
        decoder_final = transformer_decoder_factory(
            return_intermediate=False,
            number_of_layers=number_of_layers,
        )
        decoder_intermediate.eval()
        decoder_final.eval()
        # Copy weights to ensure identical layers
        decoder_final.load_state_dict(decoder_intermediate.state_dict())
        target = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        memory = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output_intermediate = decoder_intermediate(target=target, memory=memory)
        output_final = decoder_final(target=target, memory=memory)
        # Last intermediate layer output should match final output
        assert torch.allclose(output_intermediate[-1], output_final[0], atol=1e-5)

    def test_intermediate_layers_differ(
        self,
        transformer_decoder_factory: Callable[..., TransformerDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        decoder = transformer_decoder_factory(
            return_intermediate=True,
            number_of_layers=3,
        )
        decoder.eval()
        target = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        memory = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output = decoder(target=target, memory=memory)
        # Different layers should produce different intermediate outputs
        assert not torch.allclose(output[0], output[1])
        assert not torch.allclose(output[1], output[2])

    def test_positional_encodings_passed_through_layers(
        self,
        transformer_decoder_factory: Callable[..., TransformerDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        decoder = transformer_decoder_factory()
        decoder.eval()
        target = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        memory = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        query_pe = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        memory_pe = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output_without = decoder(target=target, memory=memory)
        output_with = decoder(
            target=target,
            memory=memory,
            query_positional_encoding=query_pe,
            memory_positional_encoding=memory_pe,
        )
        assert not torch.allclose(output_without, output_with)

    def test_no_normalization_with_intermediate_returns_unnormalized(
        self,
        transformer_decoder_factory: Callable[..., TransformerDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        torch.manual_seed(0)
        decoder_with_norm = transformer_decoder_factory(
            return_intermediate=True,
            use_normalization=True,
            number_of_layers=2,
        )
        torch.manual_seed(0)
        decoder_without_norm = transformer_decoder_factory(
            return_intermediate=True,
            use_normalization=False,
            number_of_layers=2,
        )
        decoder_with_norm.eval()
        decoder_without_norm.eval()
        target = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        memory = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output_with = decoder_with_norm(target=target, memory=memory)
        output_without = decoder_without_norm(target=target, memory=memory)
        # Final LayerNorm produces small but consistent differences on the last layer
        assert not torch.allclose(
            output_with[-1], output_without[-1], rtol=1e-6, atol=0
        )

    def test_gradients_flow_through_all_parameters(
        self,
        transformer_decoder_factory: Callable[..., TransformerDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        decoder = transformer_decoder_factory(number_of_layers=2)
        target = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        memory = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        target.requires_grad_(True)
        memory.requires_grad_(True)
        output = decoder(target=target, memory=memory)
        loss = output.sum()
        loss.backward()
        assert target.grad is not None
        assert memory.grad is not None
        for name, parameter in decoder.named_parameters():
            assert parameter.grad is not None, f"No gradient for {name}"

    def test_target_mask_passed_through_all_layers(
        self,
        transformer_decoder_factory: Callable[..., TransformerDecoder],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        decoder = transformer_decoder_factory(number_of_layers=2)
        decoder.eval()
        target = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        memory = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        causal_mask = torch.triu(
            torch.ones(TARGET_LENGTH, TARGET_LENGTH, dtype=torch.bool),
            diagonal=1,
        )
        output_causal = decoder(
            target=target,
            memory=memory,
            target_mask=causal_mask,
        )
        # Modify last target token and verify first token is unchanged
        modified_target = target.clone()
        modified_target[:, -1, :] = 0.0
        output_modified = decoder(
            target=modified_target,
            memory=memory,
            target_mask=causal_mask,
        )
        # First token cannot attend to last token across all layers
        assert torch.allclose(
            output_causal[0, :, 0, :], output_modified[0, :, 0, :], atol=1e-5
        )
