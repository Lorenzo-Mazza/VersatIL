"""Tests for versatil.models.decoding.decoders.factory.gpt_action_transformer module."""
import re
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from versatil.data.tokenization import Tokenizer
from versatil.models.decoding.action_heads.single_output import ActionHead
from versatil.models.decoding.constants import DecoderOutputKey, LatentKey
from versatil.models.decoding.decoders.base import ActionDecoder
from versatil.models.decoding.decoders.factory.gpt_action_transformer import (
    GPTActionTransformer,
)
from versatil.models.decoding.transformer_input_builder import TransformerInputBuilder
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType, PositionalEncodingType
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.transformer.autoregressive_decoder import GPTDecoder


EMBEDDING_DIMENSION = 32
NUMBER_OF_HEADS = 2
NUMBER_OF_LAYERS = 1
FEEDFORWARD_DIMENSION = 64
MAX_SEQ_LEN = 64
PREDICTION_HORIZON = 4
BATCH_SIZE = 2
POSITION_DIM = 3
SPATIAL_HEIGHT = 4
SPATIAL_WIDTH = 4
VOCAB_SIZE = 32
ACTION_TOKEN_LEN = 8


@pytest.fixture
def gpt_transformer_factory(
    mock_action_space_factory: Callable[..., MagicMock],
    mock_observation_space_factory: Callable[..., MagicMock],
    action_head_factory: Callable[..., ActionHead],
) -> Callable[..., GPTActionTransformer]:
    """Factory for GPTActionTransformer instances with small dimensions."""

    def factory(
        input_keys: list[str] | None = None,
        position_dim: int = POSITION_DIM,
        observation_horizon: int = 1,
        prediction_horizon: int = PREDICTION_HORIZON,
        embedding_dimension: int = EMBEDDING_DIMENSION,
        number_of_heads: int = NUMBER_OF_HEADS,
        number_of_key_value_heads: int | None = None,
        feedforward_dimension: int | None = FEEDFORWARD_DIMENSION,
        number_of_layers: int = NUMBER_OF_LAYERS,
        activation: str = ActivationFunction.SWIGLU.value,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        attention_type: str = AttentionType.MULTI_HEAD.value,
        dropout_rate: float = 0.1,
        attention_dropout: float = 0.0,
        positional_encoding_type: str | None = PositionalEncodingType.ROPE.value,
        max_seq_len: int = MAX_SEQ_LEN,
        temperature: float = 1.0,
        learnable_temperature: bool = False,
        deterministic: bool = True,
        device: str = "cpu",
    ) -> GPTActionTransformer:
        if input_keys is None:
            input_keys = ["rgb_features"]
        action_space = mock_action_space_factory(position_dim=position_dim)
        observation_space = mock_observation_space_factory()
        action_heads = {
            DecoderOutputKey.ACTION_LOGITS.value: action_head_factory(
                input_dim=embedding_dimension
            )
        }
        return GPTActionTransformer(
            action_heads=action_heads,
            input_keys=input_keys,
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            max_seq_len=max_seq_len,
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_key_value_heads=number_of_key_value_heads,
            feedforward_dimension=feedforward_dimension,
            number_of_layers=number_of_layers,
            activation=activation,
            normalization_type=normalization_type,
            attention_type=attention_type,
            dropout_rate=dropout_rate,
            attention_dropout=attention_dropout,
            positional_encoding_type=positional_encoding_type,
            temperature=temperature,
            learnable_temperature=learnable_temperature,
            deterministic=deterministic,
        )

    return factory


class TestGPTActionTransformerInitialization:

    def test_inherits_from_action_decoder(
        self,
        gpt_transformer_factory: Callable[..., GPTActionTransformer],
    ):
        decoder = gpt_transformer_factory()
        assert isinstance(decoder, ActionDecoder)

    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    @pytest.mark.parametrize("number_of_layers", [1, 2])
    def test_stores_configuration(
        self,
        gpt_transformer_factory: Callable[..., GPTActionTransformer],
        embedding_dimension: int,
        number_of_layers: int,
    ):
        decoder = gpt_transformer_factory(
            embedding_dimension=embedding_dimension,
            number_of_layers=number_of_layers,
            number_of_heads=NUMBER_OF_HEADS,
            feedforward_dimension=FEEDFORWARD_DIMENSION,
            activation=ActivationFunction.SWIGLU.value,
            normalization_type=NormalizationType.RMS_NORM.value,
            attention_type=AttentionType.MULTI_HEAD.value,
            dropout_rate=0.05,
            attention_dropout=0.02,
            positional_encoding_type=PositionalEncodingType.ROPE.value,
            max_seq_len=MAX_SEQ_LEN,
            temperature=0.8,
            deterministic=False,
            device="cpu",
        )
        assert decoder.embedding_dimension == embedding_dimension
        assert decoder.number_of_layers == number_of_layers
        assert decoder.number_of_heads == NUMBER_OF_HEADS
        assert decoder.feedforward_dimension == FEEDFORWARD_DIMENSION
        assert decoder.activation == ActivationFunction.SWIGLU.value
        assert decoder.normalization_type == NormalizationType.RMS_NORM.value
        assert decoder.attention_type == AttentionType.MULTI_HEAD.value
        assert decoder.dropout_rate == 0.05
        assert decoder.attention_dropout == 0.02
        assert decoder.positional_encoding_type == PositionalEncodingType.ROPE.value
        assert decoder.max_seq_len == MAX_SEQ_LEN
        assert decoder.deterministic is False
        assert next(decoder.parameters()).device.type == "cpu"

    def test_creates_components(
        self,
        gpt_transformer_factory: Callable[..., GPTActionTransformer],
    ):
        decoder = gpt_transformer_factory()
        assert isinstance(decoder.input_sequence_builder, TransformerInputBuilder)
        assert isinstance(decoder.gpt_decoder, GPTDecoder)

    def test_decoder_input_requires_actions(
        self,
        gpt_transformer_factory: Callable[..., GPTActionTransformer],
    ):
        decoder = gpt_transformer_factory()
        assert decoder.decoder_input.requires_actions is True

    def test_invalid_action_heads_key_raises(
        self,
        mock_action_space_factory: Callable[..., MagicMock],
        mock_observation_space_factory: Callable[..., MagicMock],
        action_head_factory: Callable[..., ActionHead],
    ):
        action_space = mock_action_space_factory(position_dim=POSITION_DIM)
        observation_space = mock_observation_space_factory()
        wrong_action_heads = {
            "wrong_key": action_head_factory(input_dim=EMBEDDING_DIMENSION)
        }
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"GPTActionTransformer only supports DecoderOutputKey.ACTION_LOGITS.value in action_heads."
                f" Make sure to use key {DecoderOutputKey.ACTION_LOGITS.value} in your hydra config."
            ),
        ):
            GPTActionTransformer(
                action_heads=wrong_action_heads,
                input_keys=["rgb_features"],
                action_space=action_space,
                observation_space=observation_space,
                observation_horizon=1,
                prediction_horizon=PREDICTION_HORIZON,
                device="cpu",
                embedding_dimension=EMBEDDING_DIMENSION,
                number_of_heads=NUMBER_OF_HEADS,
                number_of_layers=NUMBER_OF_LAYERS,
                feedforward_dimension=FEEDFORWARD_DIMENSION,
                max_seq_len=MAX_SEQ_LEN,
            )

    def test_supports_tokenized_actions_flag(
        self,
        gpt_transformer_factory: Callable[..., GPTActionTransformer],
    ):
        decoder = gpt_transformer_factory()
        assert decoder.supports_tokenized_actions is True

    def test_token_embedding_initially_none(
        self,
        gpt_transformer_factory: Callable[..., GPTActionTransformer],
    ):
        decoder = gpt_transformer_factory()
        assert decoder.token_embedding is None

    def test_temperature_is_parameter(
        self,
        gpt_transformer_factory: Callable[..., GPTActionTransformer],
    ):
        decoder = gpt_transformer_factory(temperature=0.5)
        assert isinstance(decoder.temperature, nn.Parameter)
        torch.testing.assert_close(
            decoder.temperature,
            torch.tensor(0.5, dtype=torch.float32),
            atol=1e-6,
            rtol=1e-6,
        )


class TestGPTActionTransformerSetTokenizer:

    def test_raises_without_tokenizer(
        self,
        gpt_transformer_factory: Callable[..., GPTActionTransformer],
    ):
        decoder = gpt_transformer_factory()
        with pytest.raises(
            ValueError,
            match=re.escape(
                "GPTActionTransformer requires a tokenizer for tokenized action prediction."
            ),
        ):
            decoder.set_tokenizer(tokenizer=None)

    def test_raises_with_none_action_tokenizer(
        self,
        gpt_transformer_factory: Callable[..., GPTActionTransformer],
    ):
        decoder = gpt_transformer_factory()
        tokenizer = MagicMock(spec=Tokenizer)
        tokenizer.action_tokenizer = None
        with pytest.raises(
            ValueError,
            match=re.escape(
                "GPTActionTransformer requires a tokenizer for tokenized action prediction."
            ),
        ):
            decoder.set_tokenizer(tokenizer=tokenizer)

    def test_sets_vocab_size(
        self,
        gpt_transformer_factory: Callable[..., GPTActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
    ):
        decoder = gpt_transformer_factory()
        tokenizer = mock_tokenizer_factory(vocab_size=64)
        decoder.set_tokenizer(tokenizer=tokenizer)
        assert decoder.vocab_size == 64

    def test_creates_token_embedding(
        self,
        gpt_transformer_factory: Callable[..., GPTActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
    ):
        decoder = gpt_transformer_factory()
        tokenizer = mock_tokenizer_factory(vocab_size=VOCAB_SIZE)
        decoder.set_tokenizer(tokenizer=tokenizer)
        assert decoder.token_embedding is not None

    def test_ties_output_weights(
        self,
        gpt_transformer_factory: Callable[..., GPTActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
    ):
        decoder = gpt_transformer_factory()
        tokenizer = mock_tokenizer_factory(vocab_size=VOCAB_SIZE)
        decoder.set_tokenizer(tokenizer=tokenizer)
        lm_head = decoder.action_heads[DecoderOutputKey.ACTION_LOGITS.value].output_proj
        # embedding_dimension == output_proj.in_features, so token_embedding is nn.Embedding
        assert isinstance(decoder.token_embedding, nn.Embedding)
        assert lm_head.weight is decoder.token_embedding.weight


class TestGPTActionTransformerForward:

    def test_training_output_keys(
        self,
        gpt_transformer_factory: Callable[..., GPTActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
        tokenized_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = gpt_transformer_factory()
        decoder.set_tokenizer(tokenizer=mock_tokenizer_factory())
        features = flat_feature_factory(
            batch_size=BATCH_SIZE,
            feature_dim=EMBEDDING_DIMENSION,
        )
        actions = tokenized_actions_factory(batch_size=BATCH_SIZE)
        predictions = decoder(features=features, actions=actions)
        assert DecoderOutputKey.ACTION_LOGITS.value in predictions

    def test_training_output_shape(
        self,
        gpt_transformer_factory: Callable[..., GPTActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
        tokenized_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = gpt_transformer_factory()
        decoder.set_tokenizer(tokenizer=mock_tokenizer_factory(vocab_size=VOCAB_SIZE))
        features = flat_feature_factory(
            batch_size=BATCH_SIZE,
            feature_dim=EMBEDDING_DIMENSION,
        )
        actions = tokenized_actions_factory(
            batch_size=BATCH_SIZE,
            action_token_length=ACTION_TOKEN_LEN,
        )
        predictions = decoder(features=features, actions=actions)
        logits = predictions[DecoderOutputKey.ACTION_LOGITS.value]
        assert logits.shape == (BATCH_SIZE, ACTION_TOKEN_LEN, VOCAB_SIZE)

    def test_inference_output_keys(
        self,
        gpt_transformer_factory: Callable[..., GPTActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = gpt_transformer_factory()
        decoder.set_tokenizer(tokenizer=mock_tokenizer_factory())
        features = flat_feature_factory(
            batch_size=BATCH_SIZE,
            feature_dim=EMBEDDING_DIMENSION,
        )
        decoder.eval()
        predictions = decoder(features=features, actions=None)
        assert DecoderOutputKey.PREDICTED_ACTION_TOKENS.value in predictions

    def test_inference_output_shape(
        self,
        gpt_transformer_factory: Callable[..., GPTActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = gpt_transformer_factory(max_seq_len=MAX_SEQ_LEN)
        decoder.set_tokenizer(tokenizer=mock_tokenizer_factory())
        features = flat_feature_factory(
            batch_size=BATCH_SIZE,
            feature_dim=EMBEDDING_DIMENSION,
        )
        decoder.eval()
        predictions = decoder(features=features, actions=None)
        tokens = predictions[DecoderOutputKey.PREDICTED_ACTION_TOKENS.value]
        assert tokens.shape[0] == BATCH_SIZE
        # Inference generates (max_seq_len - prefix_len) tokens per step;
        # prefix_len = 1 for a single flat feature projected to 1 token
        prefix_len = 1
        expected_generated_tokens = MAX_SEQ_LEN - prefix_len
        assert tokens.shape[1] == expected_generated_tokens

    def test_forward_passes_latent_keys_from_features(
        self,
        gpt_transformer_factory: Callable[..., GPTActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
        tokenized_actions_factory: Callable[..., dict[str, torch.Tensor]],
        input_tensor_factory: Callable[..., torch.Tensor],
    ):
        decoder = gpt_transformer_factory()
        decoder.set_tokenizer(tokenizer=mock_tokenizer_factory())
        features = flat_feature_factory(
            batch_size=BATCH_SIZE,
            feature_dim=EMBEDDING_DIMENSION,
        )
        posterior_mu = input_tensor_factory(batch_size=BATCH_SIZE, input_dim=16)
        posterior_logvar = input_tensor_factory(batch_size=BATCH_SIZE, input_dim=16)
        features[LatentKey.POSTERIOR_MU.value] = posterior_mu
        features[LatentKey.POSTERIOR_LOGVAR.value] = posterior_logvar
        actions = tokenized_actions_factory(batch_size=BATCH_SIZE)
        predictions = decoder(features=features, actions=actions)
        assert LatentKey.POSTERIOR_MU.value in predictions
        assert LatentKey.POSTERIOR_LOGVAR.value in predictions
        assert torch.equal(predictions[LatentKey.POSTERIOR_MU.value], posterior_mu)
        assert torch.equal(predictions[LatentKey.POSTERIOR_LOGVAR.value], posterior_logvar)

    def test_raises_if_sequence_too_long(
        self,
        gpt_transformer_factory: Callable[..., GPTActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
        tokenized_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        # Use a very small max_seq_len so that spatial features + action tokens exceed it
        small_max_seq_len = 10
        decoder = gpt_transformer_factory(max_seq_len=small_max_seq_len)
        decoder.set_tokenizer(tokenizer=mock_tokenizer_factory())
        # Spatial features produce SPATIAL_HEIGHT * SPATIAL_WIDTH = 16 tokens, already > 10
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        actions = tokenized_actions_factory(
            batch_size=BATCH_SIZE,
            action_token_length=ACTION_TOKEN_LEN,
        )
        expected_token_length = SPATIAL_HEIGHT * SPATIAL_WIDTH + ACTION_TOKEN_LEN
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Input token length {expected_token_length} > max_seq_len {small_max_seq_len}. "
                "No room for any action tokens. "
                "Consider increasing max_seq_len or reducing feature token count."
            ),
        ):
            decoder(features=features, actions=actions)
