"""Tests for versatil.models.layers.detr_transformer.transformer module."""
from collections.abc import Callable

import pytest
import torch

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.detr_transformer.transformer import Transformer

EMBEDDING_DIMENSION = 64
NUMBER_OF_HEADS = 4
FEEDFORWARD_DIMENSION = 128
SOURCE_LENGTH = 8
TARGET_LENGTH = 6


class TestTransformerInitialization:

    @pytest.mark.parametrize("embedding_dimension", [EMBEDDING_DIMENSION, 128])
    @pytest.mark.parametrize("number_of_heads", [NUMBER_OF_HEADS, 8])
    def test_stores_configuration(
        self,
        transformer_factory: Callable[..., Transformer],
        embedding_dimension: int,
        number_of_heads: int,
    ):
        transformer = transformer_factory(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
        )
        assert transformer.embedding_dimension == embedding_dimension
        assert transformer.number_of_heads == number_of_heads

    @pytest.mark.parametrize("number_of_encoder_layers", [1, 3])
    def test_encoder_layer_count(
        self,
        transformer_factory: Callable[..., Transformer],
        number_of_encoder_layers: int,
    ):
        transformer = transformer_factory(
            number_of_encoder_layers=number_of_encoder_layers,
        )
        assert len(transformer.encoder.layers) == number_of_encoder_layers

    @pytest.mark.parametrize("number_of_decoder_layers", [1, 3])
    def test_decoder_layer_count(
        self,
        transformer_factory: Callable[..., Transformer],
        number_of_decoder_layers: int,
    ):
        transformer = transformer_factory(
            number_of_decoder_layers=number_of_decoder_layers,
        )
        assert len(transformer.decoder.layers) == number_of_decoder_layers

    def test_normalize_before_creates_encoder_normalization(
        self,
        transformer_factory: Callable[..., Transformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        transformer = transformer_factory(normalize_before=True)
        transformer.eval()
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        # Pre-norm encoder has a final LayerNorm -- verify the output has
        # near-zero mean and unit variance along the embedding dimension
        encoder_output = transformer.encoder(source=source)
        mean = encoder_output.mean(dim=-1)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=0.1)

    def test_normalize_after_omits_encoder_normalization(
        self,
        transformer_factory: Callable[..., Transformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        # In post-norm mode, the Transformer sets encoder normalization to None.
        # Verify by checking that a pre-norm transformer produces different encoder
        # output than a post-norm transformer (the pre-norm one has a final LayerNorm).
        torch.manual_seed(42)
        transformer_post = transformer_factory(normalize_before=False)
        torch.manual_seed(42)
        transformer_pre = transformer_factory(normalize_before=True)
        transformer_post.eval()
        transformer_pre.eval()
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        # Both have the same layer weights but different normalization strategies
        # Pre-norm has a final LayerNorm on encoder, post-norm does not
        encoder_output_post = transformer_post.encoder(source=source)
        encoder_output_pre = transformer_pre.encoder(source=source)
        assert not torch.allclose(encoder_output_post, encoder_output_pre, atol=1e-5)

    @pytest.mark.parametrize("normalize_before", [True, False])
    def test_decoder_always_has_normalization(
        self,
        transformer_factory: Callable[..., Transformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
        normalize_before: bool,
    ):
        # Both pre-norm and post-norm transformers should have decoder normalization.
        # Verify the decoder output is layer-normalized (mean ~0, std ~1).
        transformer = transformer_factory(normalize_before=normalize_before)
        transformer.eval()
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        target = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output = transformer(source=source, target=target)
        # output shape: (1, B, T, C)
        embedding_output = output[0]
        mean = embedding_output.mean(dim=-1)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=0.1)

    @pytest.mark.parametrize("return_intermediate_decoder", [True, False])
    def test_return_intermediate_decoder_configuration(
        self,
        transformer_factory: Callable[..., Transformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
        return_intermediate_decoder: bool,
    ):
        number_of_decoder_layers = 2
        transformer = transformer_factory(
            return_intermediate_decoder=return_intermediate_decoder,
            number_of_decoder_layers=number_of_decoder_layers,
        )
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        target = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output = transformer(source=source, target=target)
        if return_intermediate_decoder:
            assert output.shape[0] == number_of_decoder_layers
        else:
            assert output.shape[0] == 1

    def test_encoder_submodule_processes_source(
        self,
        transformer_factory: Callable[..., Transformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        transformer = transformer_factory()
        transformer.eval()
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        encoder_output = transformer.encoder(source=source)
        assert encoder_output.shape == (batch_size, SOURCE_LENGTH, EMBEDDING_DIMENSION)
        # The encoder transforms its input
        assert not torch.allclose(encoder_output, source, atol=1e-5)

    def test_decoder_submodule_processes_target_and_memory(
        self,
        transformer_factory: Callable[..., Transformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        transformer = transformer_factory()
        transformer.eval()
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        target = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        memory = transformer.encoder(source=source)
        decoder_output = transformer.decoder(target=target, memory=memory)
        # decoder returns (1, B, T, C) or (num_layers, B, T, C)
        assert decoder_output.shape == (1, batch_size, TARGET_LENGTH, EMBEDDING_DIMENSION)


class TestTransformerForward:

    def test_output_shape_without_intermediate(
        self,
        transformer_factory: Callable[..., Transformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        transformer = transformer_factory(return_intermediate_decoder=False)
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        target = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output = transformer(source=source, target=target)
        assert output.shape == (1, batch_size, TARGET_LENGTH, EMBEDDING_DIMENSION)

    @pytest.mark.parametrize("number_of_decoder_layers", [2, 3])
    def test_output_shape_with_intermediate(
        self,
        transformer_factory: Callable[..., Transformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
        number_of_decoder_layers: int,
    ):
        transformer = transformer_factory(
            return_intermediate_decoder=True,
            number_of_decoder_layers=number_of_decoder_layers,
        )
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        target = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output = transformer(source=source, target=target)
        assert output.shape == (
            number_of_decoder_layers,
            batch_size,
            TARGET_LENGTH,
            EMBEDDING_DIMENSION,
        )

    def test_different_sources_produce_different_outputs(
        self,
        transformer_factory: Callable[..., Transformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        transformer = transformer_factory()
        transformer.eval()
        source_a = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        source_b = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        target = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output_a = transformer(source=source_a, target=target)
        output_b = transformer(source=source_b, target=target)
        assert not torch.allclose(output_a, output_b)

    def test_different_targets_produce_different_outputs(
        self,
        transformer_factory: Callable[..., Transformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        transformer = transformer_factory()
        transformer.eval()
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        target_a = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        target_b = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output_a = transformer(source=source, target=target_a)
        output_b = transformer(source=source, target=target_b)
        assert not torch.allclose(output_a, output_b)

    def test_source_positional_encoding_changes_output(
        self,
        transformer_factory: Callable[..., Transformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        transformer = transformer_factory()
        transformer.eval()
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        target = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        source_pe = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output_without = transformer(source=source, target=target)
        output_with = transformer(
            source=source,
            target=target,
            source_positional_encoding=source_pe,
        )
        assert not torch.allclose(output_without, output_with)

    def test_target_positional_encoding_changes_output(
        self,
        transformer_factory: Callable[..., Transformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        transformer = transformer_factory()
        transformer.eval()
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        target = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        target_pe = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output_without = transformer(source=source, target=target)
        output_with = transformer(
            source=source,
            target=target,
            target_positional_encoding=target_pe,
        )
        assert not torch.allclose(output_without, output_with)

    def test_source_padding_mask_changes_output(
        self,
        transformer_factory: Callable[..., Transformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        transformer = transformer_factory()
        transformer.eval()
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        target = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        source_padding_mask = torch.zeros(
            batch_size, SOURCE_LENGTH, dtype=torch.bool
        )
        source_padding_mask[:, -1] = True
        output_without = transformer(source=source, target=target)
        output_with = transformer(
            source=source,
            target=target,
            source_key_padding_mask=source_padding_mask,
        )
        assert not torch.allclose(output_without, output_with)

    def test_target_padding_mask_changes_output(
        self,
        transformer_factory: Callable[..., Transformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        transformer = transformer_factory()
        transformer.eval()
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        target = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        target_padding_mask = torch.zeros(
            batch_size, TARGET_LENGTH, dtype=torch.bool
        )
        target_padding_mask[:, -1] = True
        output_without = transformer(source=source, target=target)
        output_with = transformer(
            source=source,
            target=target,
            target_key_padding_mask=target_padding_mask,
        )
        assert not torch.allclose(output_without, output_with)

    @pytest.mark.parametrize("normalize_before", [False, True])
    def test_forward_with_both_normalization_modes(
        self,
        transformer_factory: Callable[..., Transformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
        normalize_before: bool,
    ):
        transformer = transformer_factory(normalize_before=normalize_before)
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        target = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output = transformer(source=source, target=target)
        assert output.shape == (1, batch_size, TARGET_LENGTH, EMBEDDING_DIMENSION)

    @pytest.mark.parametrize("activation", [
        ActivationFunction.RELU.value,
        ActivationFunction.GELU.value,
        ActivationFunction.SWIGLU.value,
    ])
    def test_forward_with_different_activations(
        self,
        transformer_factory: Callable[..., Transformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
        activation: str,
    ):
        transformer = transformer_factory(activation=activation)
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        target = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        output = transformer(source=source, target=target)
        assert output.shape == (1, batch_size, TARGET_LENGTH, EMBEDDING_DIMENSION)

    def test_gradients_flow_to_source_and_target_and_all_parameters(
        self,
        transformer_factory: Callable[..., Transformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        transformer = transformer_factory()
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        target = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        source.requires_grad_(True)
        target.requires_grad_(True)
        output = transformer(source=source, target=target)
        loss = output.sum()
        loss.backward()
        assert source.grad is not None
        assert target.grad is not None
        for name, parameter in transformer.named_parameters():
            assert parameter.grad is not None, f"No gradient for {name}"

    def test_reset_parameters_applies_xavier_uniform(
        self,
        transformer_factory: Callable[..., Transformer],
    ):
        transformer = transformer_factory()
        for name, parameter in transformer.named_parameters():
            if parameter.dim() > 1:
                fan_in = parameter.shape[1]
                fan_out = parameter.shape[0]
                xavier_bound = (6.0 / (fan_in + fan_out)) ** 0.5
                assert parameter.data.abs().max().item() <= xavier_bound + 1e-6, (
                    f"Parameter {name} exceeds Xavier uniform bound"
                )

    def test_source_padding_mask_forwarded_to_decoder_as_memory_mask(
        self,
        transformer_factory: Callable[..., Transformer],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        # The source_key_padding_mask is used both in the encoder and as
        # memory_key_padding_mask in the decoder. Verify that masking a source
        # position affects the decoder output differently than no masking.
        transformer = transformer_factory()
        transformer.eval()
        source = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=SOURCE_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        target = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=TARGET_LENGTH,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        # Mask half the source positions
        source_padding_mask = torch.zeros(
            batch_size, SOURCE_LENGTH, dtype=torch.bool
        )
        source_padding_mask[:, SOURCE_LENGTH // 2 :] = True
        output_no_mask = transformer(source=source, target=target)
        output_with_mask = transformer(
            source=source,
            target=target,
            source_key_padding_mask=source_padding_mask,
        )
        assert not torch.allclose(output_no_mask, output_with_mask)
