"""Shared fixtures for free_transformer layer tests."""
from collections.abc import Callable

import pytest

from versatil.models.layers.free_transformer.binary_mapper import BinaryMapper
from versatil.models.layers.free_transformer.free_transformer import (
    FreeTransformer,
    FreeTransformerLatentEncoder,
    LatentConditionedDecoderLayer,
)
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType
from versatil.models.layers.normalization.constants import NormalizationType


@pytest.fixture
def binary_mapper_factory() -> Callable[..., BinaryMapper]:
    """Factory for BinaryMapper instances."""

    def factory(
        latent_bits: int = 4,
        embedding_dimension: int = 32,
    ) -> BinaryMapper:
        return BinaryMapper(
            latent_bits=latent_bits,
            embedding_dimension=embedding_dimension,
        )

    return factory


@pytest.fixture
def latent_conditioned_decoder_layer_factory() -> (
    Callable[..., LatentConditionedDecoderLayer]
):
    """Factory for LatentConditionedDecoderLayer instances."""

    def factory(
        embedding_dimension: int = 32,
        number_of_heads: int = 4,
        latent_dim: int = 16,
        number_of_key_value_heads: int = 2,
        feedforward_dimension: int | None = None,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation: str = ActivationFunction.GELU.value,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        attention_type: str = AttentionType.GROUPED_QUERY.value,
        bias: bool = True,
        normalization_epsilon: float = 1e-6,
        autoregressive: bool = True,
    ) -> LatentConditionedDecoderLayer:
        return LatentConditionedDecoderLayer(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            latent_dim=latent_dim,
            number_of_key_value_heads=number_of_key_value_heads,
            feedforward_dimension=feedforward_dimension,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation=activation,
            normalization_type=normalization_type,
            attention_type=attention_type,
            bias=bias,
            normalization_epsilon=normalization_epsilon,
            autoregressive=autoregressive,
        )

    return factory


@pytest.fixture
def latent_encoder_factory() -> Callable[..., FreeTransformerLatentEncoder]:
    """Factory for FreeTransformerLatentEncoder instances."""

    def factory(
        embedding_dimension: int = 32,
        number_of_layers: int = 1,
        number_of_heads: int = 4,
        number_of_key_value_heads: int = 2,
        feedforward_dimension: int | None = None,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation: str = ActivationFunction.GELU.value,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        attention_type: str = AttentionType.GROUPED_QUERY.value,
        bias: bool = True,
        normalization_epsilon: float = 1e-6,
        use_global_latent: bool = False,
    ) -> FreeTransformerLatentEncoder:
        return FreeTransformerLatentEncoder(
            embedding_dimension=embedding_dimension,
            number_of_layers=number_of_layers,
            number_of_heads=number_of_heads,
            number_of_key_value_heads=number_of_key_value_heads,
            feedforward_dimension=feedforward_dimension,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation=activation,
            normalization_type=normalization_type,
            attention_type=attention_type,
            bias=bias,
            normalization_epsilon=normalization_epsilon,
            use_global_latent=use_global_latent,
        )

    return factory


@pytest.fixture
def free_transformer_factory() -> Callable[..., FreeTransformer]:
    """Factory for FreeTransformer instances with small defaults for testing."""

    def factory(
        latent_bits: int = 4,
        latent_dim: int | None = None,
        number_of_decoder_layers: int = 4,
        number_of_encoder_layers: int = 1,
        embedding_dimension: int = 32,
        number_of_heads: int = 4,
        number_of_key_value_heads: int = 2,
        feedforward_dimension: int | None = None,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        attention_type: str = AttentionType.GROUPED_QUERY.value,
        activation: str = ActivationFunction.GELU.value,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        positional_encoding_type: str | None = None,
        maximum_sequence_length: int = 64,
        bias: bool = True,
        use_global_latent: bool = False,
        normalization_epsilon: float = 1e-6,
        initializer_range: float = 0.02,
    ) -> FreeTransformer:
        return FreeTransformer(
            latent_bits=latent_bits,
            latent_dim=latent_dim,
            number_of_decoder_layers=number_of_decoder_layers,
            number_of_encoder_layers=number_of_encoder_layers,
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_key_value_heads=number_of_key_value_heads,
            feedforward_dimension=feedforward_dimension,
            dropout=dropout,
            attention_dropout=attention_dropout,
            attention_type=attention_type,
            activation=activation,
            normalization_type=normalization_type,
            positional_encoding_type=positional_encoding_type,
            maximum_sequence_length=maximum_sequence_length,
            bias=bias,
            use_global_latent=use_global_latent,
            normalization_epsilon=normalization_epsilon,
            initializer_range=initializer_range,
        )

    return factory
