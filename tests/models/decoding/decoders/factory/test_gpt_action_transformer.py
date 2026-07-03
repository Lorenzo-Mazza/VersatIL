"""Tests for versatil.models.decoding.decoders.factory.gpt_action_transformer module."""

import re
import unittest.mock
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from versatil.data.constants import SampleKey
from versatil.data.tokenization import Tokenizer
from versatil.models.decoding.action_heads.single_output import ActionHead
from versatil.models.decoding.constants import (
    AlgorithmContextKey,
    DecoderOutputKey,
    LatentKey,
)
from versatil.models.decoding.decoders.base import ActionDecoder
from versatil.models.decoding.decoders.factory.gpt_action_transformer import (
    GPTActionTransformer,
)
from versatil.models.decoding.transformer_input_builder import TransformerInputBuilder
from versatil.models.encoding.encoders.constants import EncoderOutputKeys
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
ACTION_TOKEN_LENGTH = 8


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
                input_dimension=embedding_dimension
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
    @pytest.mark.unit
    def test_inherits_from_action_decoder(
        self,
        gpt_transformer_factory: Callable[..., GPTActionTransformer],
    ):
        decoder = gpt_transformer_factory()
        assert isinstance(decoder, ActionDecoder)

    @pytest.mark.unit
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

    @pytest.mark.unit
    def test_creates_components(
        self,
        gpt_transformer_factory: Callable[..., GPTActionTransformer],
    ):
        decoder = gpt_transformer_factory()
        assert isinstance(decoder.input_sequence_builder, TransformerInputBuilder)
        assert isinstance(decoder.gpt_decoder, GPTDecoder)

    @pytest.mark.unit
    def test_creates_learned_action_bos_embedding(
        self,
        gpt_transformer_factory: Callable[..., GPTActionTransformer],
    ):
        decoder = gpt_transformer_factory()
        assert decoder.action_bos_embedding.shape == (1, 1, EMBEDDING_DIMENSION)
        assert decoder.action_bos_embedding.requires_grad is True

    @pytest.mark.unit
    def test_decoder_input_requires_actions(
        self,
        gpt_transformer_factory: Callable[..., GPTActionTransformer],
    ):
        decoder = gpt_transformer_factory()
        assert decoder.decoder_input.requires_actions is True

    @pytest.mark.unit
    def test_invalid_action_heads_key_raises(
        self,
        mock_action_space_factory: Callable[..., MagicMock],
        mock_observation_space_factory: Callable[..., MagicMock],
        action_head_factory: Callable[..., ActionHead],
    ):
        action_space = mock_action_space_factory(position_dim=POSITION_DIM)
        observation_space = mock_observation_space_factory()
        wrong_action_heads = {
            "wrong_key": action_head_factory(input_dimension=EMBEDDING_DIMENSION)
        }
        with pytest.raises(
            ValueError,
            match=re.escape(
                "GPTActionTransformer with action_head_layout=vocabulary "
                "expects action_heads keys {'action_logits'}, got {'wrong_key'}."
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

    @pytest.mark.unit
    def test_requires_tokenized_actions_flag(
        self,
        gpt_transformer_factory: Callable[..., GPTActionTransformer],
    ):
        decoder = gpt_transformer_factory()
        assert decoder.requires_tokenized_actions is True

    @pytest.mark.unit
    def test_token_embedding_initially_none(
        self,
        gpt_transformer_factory: Callable[..., GPTActionTransformer],
    ):
        decoder = gpt_transformer_factory()
        assert decoder.token_embedding is None

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "learnable_temperature, expected_requires_grad",
        [
            (True, True),
            (False, False),
        ],
    )
    def test_temperature_is_parameter(
        self,
        gpt_transformer_factory: Callable[..., GPTActionTransformer],
        learnable_temperature: bool,
        expected_requires_grad: bool,
    ):
        decoder = gpt_transformer_factory(
            temperature=0.5,
            learnable_temperature=learnable_temperature,
        )
        assert isinstance(decoder.temperature, nn.Parameter)
        assert decoder.temperature.requires_grad is expected_requires_grad
        torch.testing.assert_close(
            decoder.temperature,
            torch.tensor(0.5, dtype=torch.float32),
            atol=1e-6,
            rtol=1e-6,
        )


class TestGPTActionTransformerSetTokenizer:
    @pytest.mark.unit
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

    @pytest.mark.unit
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

    @pytest.mark.unit
    def test_sets_vocab_size(
        self,
        gpt_transformer_factory: Callable[..., GPTActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
    ):
        base_vocab_size = 64
        decoder = gpt_transformer_factory()
        tokenizer = mock_tokenizer_factory(vocab_size=base_vocab_size)
        decoder.set_tokenizer(tokenizer=tokenizer)
        assert decoder.vocab_size == base_vocab_size + 1

    @pytest.mark.unit
    def test_creates_token_embedding(
        self,
        gpt_transformer_factory: Callable[..., GPTActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
    ):
        decoder = gpt_transformer_factory()
        tokenizer = mock_tokenizer_factory(vocab_size=VOCAB_SIZE)
        decoder.set_tokenizer(tokenizer=tokenizer)
        effective_vocab_size = VOCAB_SIZE + 1
        assert isinstance(decoder.token_embedding, nn.Embedding)
        assert decoder.token_embedding.num_embeddings == effective_vocab_size
        assert decoder.token_embedding.embedding_dim == EMBEDDING_DIMENSION

    @pytest.mark.unit
    def test_ties_output_weights_to_token_embedding(
        self,
        gpt_transformer_factory: Callable[..., GPTActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
    ):
        decoder = gpt_transformer_factory()
        tokenizer = mock_tokenizer_factory(vocab_size=VOCAB_SIZE)
        decoder.set_tokenizer(tokenizer=tokenizer)
        lm_head = decoder.action_heads[DecoderOutputKey.ACTION_LOGITS.value].output_proj
        with torch.no_grad():
            decoder.token_embedding.weight.data[0] = 999.0
        assert lm_head.weight.data[0, 0] == 999.0


class TestGPTActionTransformerForward:
    @pytest.mark.unit
    @pytest.mark.parametrize("actions_provided", [True, False])
    def test_raises_when_tokenizer_not_set(
        self,
        gpt_transformer_factory: Callable[..., GPTActionTransformer],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
        tokenized_actions_factory: Callable[..., dict[str, torch.Tensor]],
        actions_provided: bool,
    ):
        decoder = gpt_transformer_factory()
        features = flat_feature_factory(
            batch_size=BATCH_SIZE,
            feature_dim=EMBEDDING_DIMENSION,
        )
        actions = (
            tokenized_actions_factory(batch_size=BATCH_SIZE)
            if actions_provided
            else None
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                "GPTActionTransformer requires set_tokenizer() to be called before forward."
            ),
        ):
            decoder(features=features, actions=actions)

    @pytest.mark.unit
    def test_raises_when_training_actions_are_not_tokenized(
        self,
        gpt_transformer_factory: Callable[..., GPTActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = gpt_transformer_factory()
        decoder.set_tokenizer(tokenizer=mock_tokenizer_factory())
        features = flat_feature_factory(
            batch_size=BATCH_SIZE,
            feature_dim=EMBEDDING_DIMENSION,
        )
        actions = noisy_actions_factory(batch_size=BATCH_SIZE)
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"GPTActionTransformer training requires "
                f"'{SampleKey.TOKENIZED_ACTIONS.value}' in actions."
            ),
        ):
            decoder(features=features, actions=actions)

    @pytest.mark.unit
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

    @pytest.mark.unit
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
            action_token_length=ACTION_TOKEN_LENGTH,
        )
        predictions = decoder(features=features, actions=actions)
        logits = predictions[DecoderOutputKey.ACTION_LOGITS.value]
        effective_vocab_size = VOCAB_SIZE + 1
        assert logits.shape == (BATCH_SIZE, ACTION_TOKEN_LENGTH, effective_vocab_size)

    @pytest.mark.unit
    def test_forward_passes_full_feature_dict_to_input_builder(
        self,
        gpt_transformer_factory: Callable[..., GPTActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
        tokenized_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = gpt_transformer_factory(input_keys=["validated_feature"])
        decoder.set_tokenizer(tokenizer=mock_tokenizer_factory(vocab_size=VOCAB_SIZE))
        feature_tokens = torch.ones(BATCH_SIZE, 2, EMBEDDING_DIMENSION)
        predictions = {
            DecoderOutputKey.ACTION_LOGITS.value: torch.ones(
                BATCH_SIZE,
                ACTION_TOKEN_LENGTH,
                VOCAB_SIZE + 1,
            )
        }
        features = {
            "validated_feature": torch.ones(BATCH_SIZE, EMBEDDING_DIMENSION),
            LatentKey.POSTERIOR_MU.value: torch.zeros(BATCH_SIZE, 4),
            AlgorithmContextKey.TIMESTEP.value: torch.ones(BATCH_SIZE),
        }
        actions = tokenized_actions_factory(
            batch_size=BATCH_SIZE,
            action_token_length=ACTION_TOKEN_LENGTH,
            vocab_size=VOCAB_SIZE,
        )
        input_builder_mock = MagicMock(
            spec=decoder.input_sequence_builder.forward,
            return_value=(feature_tokens, None, None),
        )
        training_mock = MagicMock(
            spec=decoder._forward_training,
            return_value=predictions,
        )
        with (
            unittest.mock.patch.object(
                decoder.input_sequence_builder,
                "forward",
                input_builder_mock,
            ),
            unittest.mock.patch.object(decoder, "_forward_training", training_mock),
        ):
            output = decoder(features=features, actions=actions)

        input_builder_mock.assert_called_once_with(features)
        training_mock.assert_called_once_with(
            feature_tokens=feature_tokens,
            feature_token_mask=None,
            actions=actions,
        )
        torch.testing.assert_close(
            output[DecoderOutputKey.ACTION_LOGITS.value],
            predictions[DecoderOutputKey.ACTION_LOGITS.value],
        )

    @pytest.mark.unit
    def test_training_ignores_padded_prefix_token_values(
        self,
        gpt_transformer_factory: Callable[..., GPTActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
        tokenized_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        input_key = "language_features"
        padding_mask_key = f"{input_key}_{EncoderOutputKeys.PADDING_MASK.value}"
        decoder = gpt_transformer_factory(
            input_keys=[input_key],
            dropout_rate=0.0,
            attention_dropout=0.0,
        )
        decoder.set_tokenizer(tokenizer=mock_tokenizer_factory(vocab_size=VOCAB_SIZE))
        decoder.eval()
        prefix_padding_mask = torch.tensor(
            [[False, False, True, True], [False, False, True, True]],
            dtype=torch.bool,
        )
        first_prefix = torch.ones(BATCH_SIZE, 4, EMBEDDING_DIMENSION)
        second_prefix = first_prefix.clone()
        second_prefix[:, 2:, :] = 99.0
        first_features = {
            input_key: first_prefix,
            padding_mask_key: prefix_padding_mask,
        }
        second_features = {
            input_key: second_prefix,
            padding_mask_key: prefix_padding_mask,
        }
        actions = tokenized_actions_factory(
            batch_size=BATCH_SIZE,
            action_token_length=ACTION_TOKEN_LENGTH,
            vocab_size=VOCAB_SIZE,
        )

        with torch.no_grad():
            first_predictions = decoder(features=first_features, actions=actions)
            second_predictions = decoder(features=second_features, actions=actions)

        torch.testing.assert_close(
            first_predictions[DecoderOutputKey.ACTION_LOGITS.value],
            second_predictions[DecoderOutputKey.ACTION_LOGITS.value],
        )

    @pytest.mark.unit
    @pytest.mark.parametrize("deterministic", [True, False])
    def test_inference_output_keys(
        self,
        gpt_transformer_factory: Callable[..., GPTActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
        deterministic: bool,
    ):
        decoder = gpt_transformer_factory(deterministic=deterministic)
        decoder.set_tokenizer(tokenizer=mock_tokenizer_factory())
        features = flat_feature_factory(
            batch_size=BATCH_SIZE,
            feature_dim=EMBEDDING_DIMENSION,
        )
        decoder.eval()
        predictions = decoder(features=features, actions=None)
        assert DecoderOutputKey.PREDICTED_ACTION_TOKENS.value in predictions

    @pytest.mark.unit
    @pytest.mark.parametrize("deterministic", [True, False])
    def test_inference_output_shape(
        self,
        gpt_transformer_factory: Callable[..., GPTActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
        deterministic: bool,
    ):
        decoder = gpt_transformer_factory(
            max_seq_len=MAX_SEQ_LEN, deterministic=deterministic
        )
        decoder.set_tokenizer(tokenizer=mock_tokenizer_factory())
        features = flat_feature_factory(
            batch_size=BATCH_SIZE,
            feature_dim=EMBEDDING_DIMENSION,
        )
        decoder.eval()
        predictions = decoder(features=features, actions=None)
        tokens = predictions[DecoderOutputKey.PREDICTED_ACTION_TOKENS.value]
        assert tokens.shape[0] == BATCH_SIZE
        # Inference generates up to (max_seq_len - prefix_len - BOS) tokens;
        # prefix_len = 1 for a single flat feature projected to 1 token.
        # May terminate early if EOS is generated for all batch items.
        prefix_len = 1
        max_generated_tokens = min(
            decoder.tokenizer.max_token_len,
            MAX_SEQ_LEN - prefix_len - 1,
        )
        assert 1 <= tokens.shape[1] <= max_generated_tokens

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "max_token_len, max_seq_len, expected_steps",
        [
            (3, 16, 3),
            (256, 5, 2),
        ],
    )
    def test_inference_caps_generation_by_tokenizer_and_context(
        self,
        gpt_transformer_factory: Callable[..., GPTActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
        max_token_len: int,
        max_seq_len: int,
        expected_steps: int,
    ):
        prefix_length = 2
        decoder = gpt_transformer_factory(max_seq_len=max_seq_len)
        decoder.set_tokenizer(
            tokenizer=mock_tokenizer_factory(max_token_len=max_token_len)
        )
        feature_tokens = torch.zeros(
            BATCH_SIZE,
            prefix_length,
            EMBEDDING_DIMENSION,
        )
        predictions = {
            DecoderOutputKey.PREDICTED_ACTION_TOKENS.value: torch.zeros(
                BATCH_SIZE,
                expected_steps,
                dtype=torch.long,
            )
        }
        generation_mock = MagicMock(
            spec=decoder._run_cached_autoregressive_generation,
            return_value=predictions,
        )

        with unittest.mock.patch.object(
            decoder,
            "_run_cached_autoregressive_generation",
            generation_mock,
        ):
            output = decoder._forward_inference(
                feature_tokens=feature_tokens,
                feature_token_mask=None,
            )

        generation_mock.assert_called_once()
        assert (
            generation_mock.call_args.kwargs["max_generation_steps"] == expected_steps
        )
        torch.testing.assert_close(
            output[DecoderOutputKey.PREDICTED_ACTION_TOKENS.value],
            predictions[DecoderOutputKey.PREDICTED_ACTION_TOKENS.value],
        )

    @pytest.mark.unit
    def test_forward_does_not_pass_variational_metadata_through(
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
        posterior_mu = input_tensor_factory(batch_size=BATCH_SIZE, input_dimension=16)
        posterior_logvar = input_tensor_factory(
            batch_size=BATCH_SIZE, input_dimension=16
        )
        features[LatentKey.POSTERIOR_MU.value] = posterior_mu
        features[LatentKey.POSTERIOR_LOGVAR.value] = posterior_logvar
        actions = tokenized_actions_factory(batch_size=BATCH_SIZE)
        predictions = decoder(features=features, actions=actions)
        assert LatentKey.POSTERIOR_MU.value not in predictions
        assert LatentKey.POSTERIOR_LOGVAR.value not in predictions

    @pytest.mark.unit
    def test_inference_terminates_early_on_eos(
        self,
        gpt_transformer_factory: Callable[..., GPTActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = gpt_transformer_factory()
        tokenizer = mock_tokenizer_factory(vocab_size=VOCAB_SIZE)
        decoder.set_tokenizer(tokenizer=tokenizer)
        decoder.eval()
        eos_token_id = tokenizer.action_tokenizer.eos_token_id
        head = decoder.action_heads[DecoderOutputKey.ACTION_LOGITS.value]
        eos_logits = torch.full(
            (BATCH_SIZE, 1, tokenizer.action_tokenizer.vocab_size),
            fill_value=-100.0,
        )
        eos_logits[:, :, eos_token_id] = 100.0
        features = flat_feature_factory(
            batch_size=BATCH_SIZE,
            feature_dim=EMBEDDING_DIMENSION,
        )
        with (
            torch.no_grad(),
            unittest.mock.patch.object(
                head,
                "forward",
                return_value=eos_logits,
            ) as head_forward_mock,
        ):
            predictions = decoder(features=features, actions=None)
        tokens = predictions[DecoderOutputKey.PREDICTED_ACTION_TOKENS.value]
        head_forward_mock.assert_called_once()
        assert tokens.shape[1] == 1
        assert (tokens == eos_token_id).all()

    @pytest.mark.unit
    def test_causal_masking_future_tokens_do_not_affect_past_predictions(
        self,
        gpt_transformer_factory: Callable[..., GPTActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
        tokenized_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = gpt_transformer_factory()
        tokenizer = mock_tokenizer_factory(vocab_size=VOCAB_SIZE)
        decoder.set_tokenizer(tokenizer=tokenizer)
        decoder.eval()
        features = flat_feature_factory(
            batch_size=BATCH_SIZE,
            feature_dim=EMBEDDING_DIMENSION,
        )
        actions_original = tokenized_actions_factory(
            batch_size=BATCH_SIZE,
            action_token_length=ACTION_TOKEN_LENGTH,
            vocab_size=VOCAB_SIZE,
        )
        with torch.no_grad():
            logits_original = decoder(features=features, actions=actions_original)
        actions_modified = {
            key: tensor.clone() for key, tensor in actions_original.items()
        }
        # Modify a middle token (index 3). Due to next-token prediction shift,
        # logit[i] predicts A[i] using A[0..i-1]. So modifying A[3] should:
        # - Leave logits[0..3] unchanged (they don't attend to A[3])
        # - Change logits[4+] (they attend to A[3])
        modified_index = 3
        original_token_ids = actions_modified[SampleKey.TOKENIZED_ACTIONS.value][
            :, modified_index
        ]
        actions_modified[SampleKey.TOKENIZED_ACTIONS.value][:, modified_index] = (
            original_token_ids + 1
        ) % VOCAB_SIZE
        with torch.no_grad():
            logits_modified = decoder(features=features, actions=actions_modified)
        original = logits_original[DecoderOutputKey.ACTION_LOGITS.value]
        modified = logits_modified[DecoderOutputKey.ACTION_LOGITS.value]
        split = modified_index + 1
        torch.testing.assert_close(original[:, :split, :], modified[:, :split, :])
        assert not torch.equal(original[:, split:, :], modified[:, split:, :])

    @pytest.mark.unit
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
            action_token_length=ACTION_TOKEN_LENGTH,
        )
        expected_token_length = SPATIAL_HEIGHT * SPATIAL_WIDTH + 1 + ACTION_TOKEN_LENGTH
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Input token length {expected_token_length} > max_seq_len {small_max_seq_len}. "
                "No room for any action tokens. "
                "Consider increasing max_seq_len or reducing feature token count."
            ),
        ):
            decoder(features=features, actions=actions)

    @pytest.mark.unit
    def test_inference_raises_if_prefix_fills_sequence(
        self,
        gpt_transformer_factory: Callable[..., GPTActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        small_max_seq_len = SPATIAL_HEIGHT * SPATIAL_WIDTH
        decoder = gpt_transformer_factory(max_seq_len=small_max_seq_len)
        decoder.set_tokenizer(tokenizer=mock_tokenizer_factory())
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Input prefix token length {small_max_seq_len} plus BOS token >= "
                f"max_seq_len {small_max_seq_len}. No room for generated action tokens. "
                "Consider increasing max_seq_len or reducing feature token count."
            ),
        ):
            decoder(features=features, actions=None)

    @pytest.mark.unit
    def test_training_does_not_use_generation_cache(
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
        with unittest.mock.patch.object(
            decoder.gpt_decoder, "forward", wraps=decoder.gpt_decoder.forward
        ) as mock_forward:
            decoder(features=features, actions=actions)
            call_kwargs = mock_forward.call_args.kwargs
            assert call_kwargs.get("generation_cache") is None

    @pytest.mark.unit
    def test_inference_uses_generation_cache(
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
        with unittest.mock.patch.object(
            decoder.gpt_decoder, "forward", wraps=decoder.gpt_decoder.forward
        ) as mock_forward:
            decoder(features=features, actions=None)
            # First call is prefill, should have generation_cache
            first_call_kwargs = mock_forward.call_args_list[0].kwargs
            assert first_call_kwargs.get("generation_cache") is not None

    @pytest.mark.unit
    def test_inference_feeds_bos_after_prefix_prefill(
        self,
        gpt_transformer_factory: Callable[..., GPTActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = gpt_transformer_factory(max_seq_len=3)
        decoder.set_tokenizer(tokenizer=mock_tokenizer_factory())
        features = flat_feature_factory(
            batch_size=BATCH_SIZE,
            feature_dim=EMBEDDING_DIMENSION,
        )
        decoder.eval()
        with unittest.mock.patch.object(
            decoder.gpt_decoder, "forward", wraps=decoder.gpt_decoder.forward
        ) as mock_forward:
            decoder(features=features, actions=None)
        assert len(mock_forward.call_args_list) == 2
        second_call_hidden_states = mock_forward.call_args_list[1].kwargs[
            "hidden_states"
        ]
        expected_bos = decoder.action_bos_embedding.expand(BATCH_SIZE, -1, -1)
        torch.testing.assert_close(second_call_hidden_states, expected_bos)
