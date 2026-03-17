from collections.abc import Callable

import pytest
import torch

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.detr_transformer.attention import FlashAttention
from versatil.models.layers.detr_transformer.transformer import Transformer
from versatil.models.layers.detr_transformer.transformer_decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
)
from versatil.models.layers.detr_transformer.transformer_encoder import (
    TransformerEncoder,
    TransformerEncoderLayer,
)

EMBEDDING_DIMENSION = 64
NUMBER_OF_HEADS = 4
FEEDFORWARD_DIMENSION = 128
SOURCE_LENGTH = 8
TARGET_LENGTH = 6


@pytest.fixture
def flash_attention_factory() -> Callable[..., FlashAttention]:
    def factory(
        embedding_dimension: int = EMBEDDING_DIMENSION,
        number_of_heads: int = NUMBER_OF_HEADS,
        dropout: float = 0.0,
    ) -> FlashAttention:
        return FlashAttention(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            dropout=dropout,
        )

    return factory


@pytest.fixture
def encoder_layer_factory() -> Callable[..., TransformerEncoderLayer]:
    def factory(
        embedding_dimension: int = EMBEDDING_DIMENSION,
        number_of_heads: int = NUMBER_OF_HEADS,
        feedforward_dimension: int = FEEDFORWARD_DIMENSION,
        dropout: float = 0.0,
        activation: str = ActivationFunction.RELU.value,
        normalize_before: bool = False,
    ) -> TransformerEncoderLayer:
        return TransformerEncoderLayer(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            feedforward_dimension=feedforward_dimension,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before,
        )

    return factory


@pytest.fixture
def decoder_layer_factory() -> Callable[..., TransformerDecoderLayer]:
    def factory(
        embedding_dimension: int = EMBEDDING_DIMENSION,
        number_of_heads: int = NUMBER_OF_HEADS,
        feedforward_dimension: int = FEEDFORWARD_DIMENSION,
        dropout: float = 0.0,
        activation: str = ActivationFunction.RELU.value,
        normalize_before: bool = False,
    ) -> TransformerDecoderLayer:
        return TransformerDecoderLayer(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            feedforward_dimension=feedforward_dimension,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before,
        )

    return factory


@pytest.fixture
def transformer_encoder_factory(
    encoder_layer_factory: Callable[..., TransformerEncoderLayer],
) -> Callable[..., TransformerEncoder]:
    def factory(
        embedding_dimension: int = EMBEDDING_DIMENSION,
        number_of_heads: int = NUMBER_OF_HEADS,
        feedforward_dimension: int = FEEDFORWARD_DIMENSION,
        dropout: float = 0.0,
        activation: str = ActivationFunction.RELU.value,
        normalize_before: bool = False,
        number_of_layers: int = 2,
        use_normalization: bool = False,
    ) -> TransformerEncoder:
        layer = encoder_layer_factory(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            feedforward_dimension=feedforward_dimension,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before,
        )
        normalization = (
            torch.nn.LayerNorm(embedding_dimension) if use_normalization else None
        )
        return TransformerEncoder(
            encoder_layer=layer,
            number_of_layers=number_of_layers,
            normalization=normalization,
        )

    return factory


@pytest.fixture
def transformer_decoder_factory(
    decoder_layer_factory: Callable[..., TransformerDecoderLayer],
) -> Callable[..., TransformerDecoder]:
    def factory(
        embedding_dimension: int = EMBEDDING_DIMENSION,
        number_of_heads: int = NUMBER_OF_HEADS,
        feedforward_dimension: int = FEEDFORWARD_DIMENSION,
        dropout: float = 0.0,
        activation: str = ActivationFunction.RELU.value,
        normalize_before: bool = False,
        number_of_layers: int = 2,
        use_normalization: bool = True,
        return_intermediate: bool = False,
    ) -> TransformerDecoder:
        layer = decoder_layer_factory(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            feedforward_dimension=feedforward_dimension,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before,
        )
        normalization = (
            torch.nn.LayerNorm(embedding_dimension) if use_normalization else None
        )
        return TransformerDecoder(
            decoder_layer=layer,
            number_of_layers=number_of_layers,
            normalization=normalization,
            return_intermediate=return_intermediate,
        )

    return factory


@pytest.fixture
def transformer_factory() -> Callable[..., Transformer]:
    def factory(
        embedding_dimension: int = EMBEDDING_DIMENSION,
        number_of_heads: int = NUMBER_OF_HEADS,
        number_of_encoder_layers: int = 2,
        number_of_decoder_layers: int = 2,
        feedforward_dimension: int = FEEDFORWARD_DIMENSION,
        dropout: float = 0.0,
        activation: str = ActivationFunction.RELU.value,
        normalize_before: bool = False,
        return_intermediate_decoder: bool = False,
    ) -> Transformer:
        return Transformer(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_encoder_layers=number_of_encoder_layers,
            number_of_decoder_layers=number_of_decoder_layers,
            feedforward_dimension=feedforward_dimension,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before,
            return_intermediate_decoder=return_intermediate_decoder,
        )

    return factory
