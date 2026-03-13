"""Tests for versatil.models.decoding.decoders.factory.discrete_detr_action_transformer module."""
import re
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from versatil.data.tokenization import Tokenizer
from versatil.models.decoding.action_heads.single_output import ActionHead
from versatil.models.decoding.constants import DecoderOutputKey, FeatureType
from versatil.models.decoding.decoders.base import ActionDecoder
from versatil.models.decoding.decoders.factory.discrete_detr_action_transformer import (
    DiscreteDETRActionTransformer,
)
from versatil.models.layers.activation import ActivationFunction


EMBEDDING_DIMENSION = 32
NUMBER_OF_HEADS = 2
NUMBER_OF_ENCODER_LAYERS = 1
NUMBER_OF_DECODER_LAYERS = 1
FEEDFORWARD_DIMENSION = 64
MAX_SEQ_LEN = 16
PREDICTION_HORIZON = 8
OBSERVATION_HORIZON = 1
BATCH_SIZE = 2
POSITION_DIM = 3
SPATIAL_HEIGHT = 4
SPATIAL_WIDTH = 4


@pytest.fixture
def detr_decoder_factory(
    mock_action_space_factory: Callable[..., MagicMock],
    mock_observation_space_factory: Callable[..., MagicMock],
    action_head_factory: Callable[..., ActionHead],
) -> Callable[..., DiscreteDETRActionTransformer]:
    """Factory for DiscreteDETRActionTransformer with small dimensions."""

    def factory(
        embedding_dimension: int = EMBEDDING_DIMENSION,
        number_of_encoder_layers: int = NUMBER_OF_ENCODER_LAYERS,
        number_of_decoder_layers: int = NUMBER_OF_DECODER_LAYERS,
        number_of_heads: int = NUMBER_OF_HEADS,
        feedforward_dimension: int = FEEDFORWARD_DIMENSION,
        max_seq_len: int = MAX_SEQ_LEN,
        prediction_horizon: int = PREDICTION_HORIZON,
        observation_horizon: int = OBSERVATION_HORIZON,
        position_dim: int = POSITION_DIM,
        activation: str = ActivationFunction.RELU.value,
        dropout_rate: float = 0.0,
        normalize_before: bool = False,
        temperature: float = 1.0,
        learnable_temperature: bool = False,
        deterministic: bool = True,
        input_keys: list[str] | None = None,
    ) -> DiscreteDETRActionTransformer:
        if input_keys is None:
            input_keys = ["rgb_features"]
        action_space = mock_action_space_factory(position_dim=position_dim)
        observation_space = mock_observation_space_factory()
        action_heads = {
            DecoderOutputKey.ACTION_LOGITS.value: action_head_factory(
                input_dim=embedding_dimension
            ),
        }
        return DiscreteDETRActionTransformer(
            action_heads=action_heads,
            input_keys=input_keys,
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device="cpu",
            max_seq_len=max_seq_len,
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            feedforward_dimension=feedforward_dimension,
            number_of_encoder_layers=number_of_encoder_layers,
            number_of_decoder_layers=number_of_decoder_layers,
            activation=activation,
            dropout_rate=dropout_rate,
            normalize_before=normalize_before,
            temperature=temperature,
            learnable_temperature=learnable_temperature,
            deterministic=deterministic,
        )

    return factory



class TestDiscreteDETRInitialization:

    def test_inherits_from_action_decoder(
        self,
        detr_decoder_factory: Callable[..., DiscreteDETRActionTransformer],
    ):
        decoder = detr_decoder_factory()
        assert isinstance(decoder, ActionDecoder)

    def test_supports_tokenized_actions(
        self,
        detr_decoder_factory: Callable[..., DiscreteDETRActionTransformer],
    ):
        decoder = detr_decoder_factory()
        assert decoder.supports_tokenized_actions is True

    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    @pytest.mark.parametrize("number_of_encoder_layers", [1, 2])
    def test_stores_configuration(
        self,
        detr_decoder_factory: Callable[..., DiscreteDETRActionTransformer],
        embedding_dimension: int,
        number_of_encoder_layers: int,
    ):
        decoder = detr_decoder_factory(
            embedding_dimension=embedding_dimension,
            number_of_encoder_layers=number_of_encoder_layers,
        )
        assert decoder.embedding_dimension == embedding_dimension
        assert decoder.number_of_encoder_layers == number_of_encoder_layers

    def test_rejects_non_logits_action_heads(
        self,
        mock_action_space_factory: Callable[..., MagicMock],
        mock_observation_space_factory: Callable[..., MagicMock],
        action_head_factory: Callable[..., ActionHead],
    ):
        action_space = mock_action_space_factory(position_dim=POSITION_DIM)
        observation_space = mock_observation_space_factory()
        wrong_action_heads = {
            "position_action": action_head_factory(
                input_dim=EMBEDDING_DIMENSION
            ),
        }
        with pytest.raises(
            ValueError,
            match=(
                f"DiscreteDETRActionTransformer only supports DecoderOutputKey.ACTION_LOGITS.value in action_heads."
                f" Make sure to use key {DecoderOutputKey.ACTION_LOGITS.value} in your hydra config."
            ),
        ):
            DiscreteDETRActionTransformer(
                action_heads=wrong_action_heads,
                input_keys=["rgb_features"],
                action_space=action_space,
                observation_space=observation_space,
                observation_horizon=OBSERVATION_HORIZON,
                prediction_horizon=PREDICTION_HORIZON,
                device="cpu",
                max_seq_len=MAX_SEQ_LEN,
                embedding_dimension=EMBEDDING_DIMENSION,
                number_of_heads=NUMBER_OF_HEADS,
                feedforward_dimension=FEEDFORWARD_DIMENSION,
                number_of_encoder_layers=NUMBER_OF_ENCODER_LAYERS,
                number_of_decoder_layers=NUMBER_OF_DECODER_LAYERS,
            )

    def test_temperature_is_parameter(
        self,
        detr_decoder_factory: Callable[..., DiscreteDETRActionTransformer],
    ):
        decoder = detr_decoder_factory(temperature=2.0)
        assert isinstance(decoder.temperature, nn.Parameter)
        assert decoder.temperature.item() == pytest.approx(2.0)

    @pytest.mark.parametrize("learnable_temperature, expected_requires_grad", [
        (True, True),
        (False, False),
    ])
    def test_learnable_temperature(
        self,
        detr_decoder_factory: Callable[..., DiscreteDETRActionTransformer],
        learnable_temperature: bool,
        expected_requires_grad: bool,
    ):
        decoder = detr_decoder_factory(
            learnable_temperature=learnable_temperature,
        )
        assert decoder.temperature.requires_grad is expected_requires_grad

    def test_decoder_input_requires_spatial_features(
        self,
        detr_decoder_factory: Callable[..., DiscreteDETRActionTransformer],
    ):
        decoder = detr_decoder_factory()
        assert FeatureType.SPATIAL.value in decoder.decoder_input.required_types

    def test_decoder_input_requires_actions(
        self,
        detr_decoder_factory: Callable[..., DiscreteDETRActionTransformer],
    ):
        decoder = detr_decoder_factory()
        assert decoder.decoder_input.requires_actions is True

    def test_token_embedding_initially_none(
        self,
        detr_decoder_factory: Callable[..., DiscreteDETRActionTransformer],
    ):
        decoder = detr_decoder_factory()
        assert decoder.token_embedding is None

    def test_vocab_size_initially_none(
        self,
        detr_decoder_factory: Callable[..., DiscreteDETRActionTransformer],
    ):
        decoder = detr_decoder_factory()
        assert decoder.vocab_size is None

    def test_creates_action_decoder_transformer(
        self,
        detr_decoder_factory: Callable[..., DiscreteDETRActionTransformer],
    ):
        decoder = detr_decoder_factory()
        assert hasattr(decoder, "action_decoder")
        assert decoder.action_decoder is not None

    def test_creates_learnable_query(
        self,
        detr_decoder_factory: Callable[..., DiscreteDETRActionTransformer],
    ):
        decoder = detr_decoder_factory()
        assert hasattr(decoder, "learnable_query")
        assert decoder.learnable_query.weight.shape == (
            MAX_SEQ_LEN,
            EMBEDDING_DIMENSION,
        )


VOCAB_SIZE = 32


class TestDiscreteDETRSetTokenizer:

    def test_raises_without_tokenizer(
        self,
        detr_decoder_factory: Callable[..., DiscreteDETRActionTransformer],
    ):
        decoder = detr_decoder_factory()
        with pytest.raises(
            ValueError,
            match=re.escape(
                "DiscreteDETRActionTransformer Decoder requires a tokenizer for tokenized action prediction."
            ),
        ):
            decoder.set_tokenizer(tokenizer=None)

    def test_raises_with_none_action_tokenizer(
        self,
        detr_decoder_factory: Callable[..., DiscreteDETRActionTransformer],
    ):
        decoder = detr_decoder_factory()
        tokenizer = MagicMock(spec=Tokenizer)
        tokenizer.action_tokenizer = None
        with pytest.raises(
            ValueError,
            match=re.escape(
                "DiscreteDETRActionTransformer Decoder requires a tokenizer for tokenized action prediction."
            ),
        ):
            decoder.set_tokenizer(tokenizer=tokenizer)

    def test_sets_vocab_size(
        self,
        detr_decoder_factory: Callable[..., DiscreteDETRActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
    ):
        decoder = detr_decoder_factory()
        tokenizer = mock_tokenizer_factory(vocab_size=VOCAB_SIZE)
        decoder.set_tokenizer(tokenizer=tokenizer)
        assert decoder.vocab_size == VOCAB_SIZE

    def test_creates_token_embedding(
        self,
        detr_decoder_factory: Callable[..., DiscreteDETRActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
    ):
        decoder = detr_decoder_factory()
        tokenizer = mock_tokenizer_factory(vocab_size=VOCAB_SIZE)
        decoder.set_tokenizer(tokenizer=tokenizer)
        assert decoder.token_embedding is not None

    def test_ties_output_weights(
        self,
        detr_decoder_factory: Callable[..., DiscreteDETRActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
    ):
        decoder = detr_decoder_factory()
        tokenizer = mock_tokenizer_factory(vocab_size=VOCAB_SIZE)
        decoder.set_tokenizer(tokenizer=tokenizer)
        lm_head = decoder.action_heads[DecoderOutputKey.ACTION_LOGITS.value].output_proj
        assert isinstance(decoder.token_embedding, nn.Embedding)
        assert lm_head.weight is decoder.token_embedding.weight

    def test_updates_action_head_output_dim(
        self,
        detr_decoder_factory: Callable[..., DiscreteDETRActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
    ):
        decoder = detr_decoder_factory()
        tokenizer = mock_tokenizer_factory(vocab_size=VOCAB_SIZE)
        decoder.set_tokenizer(tokenizer=tokenizer)
        head = decoder.action_heads[DecoderOutputKey.ACTION_LOGITS.value]
        assert head.output_dim == VOCAB_SIZE
        assert head.output_proj.out_features == VOCAB_SIZE

    def test_stores_action_tokenizer(
        self,
        detr_decoder_factory: Callable[..., DiscreteDETRActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
    ):
        decoder = detr_decoder_factory()
        tokenizer = mock_tokenizer_factory(vocab_size=VOCAB_SIZE)
        decoder.set_tokenizer(tokenizer=tokenizer)
        assert decoder.tokenizer is tokenizer.action_tokenizer


class TestDiscreteDETRForward:

    def test_returns_logits_for_training(
        self,
        detr_decoder_factory: Callable[..., DiscreteDETRActionTransformer],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = detr_decoder_factory()
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        outputs = decoder(features=features)
        assert DecoderOutputKey.ACTION_LOGITS.value in outputs

    def test_output_logits_shape(
        self,
        detr_decoder_factory: Callable[..., DiscreteDETRActionTransformer],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = detr_decoder_factory()
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        outputs = decoder(features=features)
        logits = outputs[DecoderOutputKey.ACTION_LOGITS.value]
        head = decoder.action_heads[DecoderOutputKey.ACTION_LOGITS.value]
        assert logits.shape == (BATCH_SIZE, MAX_SEQ_LEN, head.output_dim)


class TestDiscreteDETRDecodeActions:

    def test_output_shape(
        self,
        detr_decoder_factory: Callable[..., DiscreteDETRActionTransformer],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = detr_decoder_factory()
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        input_tokens, positional_encodings, padding_mask = (
            decoder.input_sequence_builder(features)
        )
        action_embeddings = decoder._decode_actions(
            input_tokens=input_tokens,
            positional_encodings=positional_encodings,
            padding_mask=padding_mask,
        )
        assert action_embeddings.shape == (
            BATCH_SIZE,
            MAX_SEQ_LEN,
            EMBEDDING_DIMENSION,
        )
