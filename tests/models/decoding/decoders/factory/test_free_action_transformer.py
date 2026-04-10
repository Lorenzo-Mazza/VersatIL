"""Tests for versatil.models.decoding.decoders.factory.free_action_transformer module."""

import re
import unittest.mock
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from versatil.configs.experiment import ExperimentConfig
from versatil.data.constants import SampleKey
from versatil.data.tokenization import Tokenizer
from versatil.models.decoding.action_heads.single_output import ActionHead
from versatil.models.decoding.constants import DecoderOutputKey, LatentKey
from versatil.models.decoding.decoders.base import ActionDecoder
from versatil.models.decoding.decoders.factory.free_action_transformer import (
    FreeActionTransformer,
)
from versatil.models.decoding.transformer_input_builder import TransformerInputBuilder
from versatil.models.layers.free_transformer.free_transformer import FreeTransformer
from versatil.training.callbacks import LatentVisualizationCallback

EMBEDDING_DIMENSION = 32
NUMBER_OF_HEADS = 2
NUMBER_OF_DECODER_LAYERS = 2
NUMBER_OF_ENCODER_LAYERS = 1
LATENT_BITS = 4
MAX_SEQ_LEN = 64
PREDICTION_HORIZON = 4
POSITION_DIM = 3
BATCH_SIZE = 2
SPATIAL_HEIGHT = 4
SPATIAL_WIDTH = 4
VOCAB_SIZE = 32
ACTION_TOKEN_LENGTH = 8


@pytest.fixture
def free_transformer_factory(
    mock_action_space_factory: Callable[..., MagicMock],
    mock_observation_space_factory: Callable[..., MagicMock],
    action_head_factory: Callable[..., ActionHead],
) -> Callable[..., FreeActionTransformer]:
    """Factory for FreeActionTransformer instances with small dimensions."""

    def factory(
        input_keys: list[str] | None = None,
        position_dim: int = POSITION_DIM,
        observation_horizon: int = 1,
        prediction_horizon: int = PREDICTION_HORIZON,
        embedding_dimension: int = EMBEDDING_DIMENSION,
        number_of_heads: int = NUMBER_OF_HEADS,
        number_of_decoder_layers: int = NUMBER_OF_DECODER_LAYERS,
        number_of_encoder_layers: int = NUMBER_OF_ENCODER_LAYERS,
        latent_bits: int = LATENT_BITS,
        max_seq_len: int = MAX_SEQ_LEN,
        temperature: float = 1.0,
        learnable_temperature: bool = False,
        deterministic: bool = True,
        use_global_latent: bool = True,
        device: str = "cpu",
    ) -> FreeActionTransformer:
        if input_keys is None:
            input_keys = ["rgb_features"]
        action_space = mock_action_space_factory(position_dim=position_dim)
        observation_space = mock_observation_space_factory()
        action_heads = {
            DecoderOutputKey.ACTION_LOGITS.value: action_head_factory(
                input_dim=embedding_dimension
            )
        }
        return FreeActionTransformer(
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
            number_of_decoder_layers=number_of_decoder_layers,
            number_of_encoder_layers=number_of_encoder_layers,
            latent_bits=latent_bits,
            temperature=temperature,
            learnable_temperature=learnable_temperature,
            deterministic=deterministic,
            use_global_latent=use_global_latent,
        )

    return factory


class TestFreeActionTransformerInitialization:
    def test_inherits_from_action_decoder(
        self,
        free_transformer_factory: Callable[..., FreeActionTransformer],
    ):
        decoder = free_transformer_factory()
        assert isinstance(decoder, ActionDecoder)

    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    @pytest.mark.parametrize("number_of_decoder_layers", [2, 4])
    def test_stores_configuration(
        self,
        free_transformer_factory: Callable[..., FreeActionTransformer],
        embedding_dimension: int,
        number_of_decoder_layers: int,
    ):
        decoder = free_transformer_factory(
            embedding_dimension=embedding_dimension,
            number_of_decoder_layers=number_of_decoder_layers,
        )
        assert decoder.embedding_dimension == embedding_dimension
        assert decoder.number_of_decoder_layers == number_of_decoder_layers

    def test_creates_components(
        self,
        free_transformer_factory: Callable[..., FreeActionTransformer],
    ):
        decoder = free_transformer_factory()
        assert isinstance(decoder.input_sequence_builder, TransformerInputBuilder)
        assert isinstance(decoder.free_transformer, FreeTransformer)

    def test_decoder_input_requires_actions(
        self,
        free_transformer_factory: Callable[..., FreeActionTransformer],
    ):
        decoder = free_transformer_factory()
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
            "position_action": action_head_factory(input_dim=EMBEDDING_DIMENSION)
        }
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"FreeActionTransformer only supports DecoderOutputKey.ACTION_LOGITS.value in action_heads."
                f" Make sure to use key {DecoderOutputKey.ACTION_LOGITS.value} in your hydra config."
            ),
        ):
            FreeActionTransformer(
                action_heads=wrong_action_heads,
                input_keys=["rgb_features"],
                action_space=action_space,
                observation_space=observation_space,
                observation_horizon=1,
                prediction_horizon=PREDICTION_HORIZON,
                device="cpu",
                max_seq_len=MAX_SEQ_LEN,
                embedding_dimension=EMBEDDING_DIMENSION,
                number_of_heads=NUMBER_OF_HEADS,
                number_of_decoder_layers=NUMBER_OF_DECODER_LAYERS,
                number_of_encoder_layers=NUMBER_OF_ENCODER_LAYERS,
                latent_bits=LATENT_BITS,
            )

    def test_supports_tokenized_actions_flag(
        self,
        free_transformer_factory: Callable[..., FreeActionTransformer],
    ):
        decoder = free_transformer_factory()
        assert decoder.supports_tokenized_actions is True

    def test_token_embedding_initially_none(
        self,
        free_transformer_factory: Callable[..., FreeActionTransformer],
    ):
        decoder = free_transformer_factory()
        assert decoder.token_embedding is None

    @pytest.mark.parametrize(
        "learnable_temperature, expected_requires_grad",
        [
            (True, True),
            (False, False),
        ],
    )
    def test_temperature_is_parameter(
        self,
        free_transformer_factory: Callable[..., FreeActionTransformer],
        learnable_temperature: bool,
        expected_requires_grad: bool,
    ):
        decoder = free_transformer_factory(
            learnable_temperature=learnable_temperature,
        )
        assert isinstance(decoder.temperature, nn.Parameter)
        assert decoder.temperature.requires_grad is expected_requires_grad


class TestFreeActionTransformerSetTokenizer:
    def test_raises_without_tokenizer(
        self,
        free_transformer_factory: Callable[..., FreeActionTransformer],
    ):
        decoder = free_transformer_factory()
        with pytest.raises(
            ValueError,
            match=re.escape(
                "FreeActionTransformer requires a tokenizer for tokenized action prediction."
            ),
        ):
            decoder.set_tokenizer(tokenizer=None)

    def test_raises_with_none_action_tokenizer(
        self,
        free_transformer_factory: Callable[..., FreeActionTransformer],
    ):
        decoder = free_transformer_factory()
        tokenizer = MagicMock(spec=Tokenizer)
        tokenizer.action_tokenizer = None
        with pytest.raises(
            ValueError,
            match=re.escape(
                "FreeActionTransformer requires a tokenizer for tokenized action prediction."
            ),
        ):
            decoder.set_tokenizer(tokenizer=tokenizer)

    def test_sets_vocab_size(
        self,
        free_transformer_factory: Callable[..., FreeActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
    ):
        base_vocab_size = 64
        decoder = free_transformer_factory()
        tokenizer = mock_tokenizer_factory(vocab_size=base_vocab_size)
        decoder.set_tokenizer(tokenizer=tokenizer)
        assert decoder.vocab_size == base_vocab_size + 1

    def test_creates_token_embedding(
        self,
        free_transformer_factory: Callable[..., FreeActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
    ):
        decoder = free_transformer_factory()
        tokenizer = mock_tokenizer_factory(vocab_size=VOCAB_SIZE)
        decoder.set_tokenizer(tokenizer=tokenizer)
        effective_vocab_size = VOCAB_SIZE + 1
        assert isinstance(decoder.token_embedding, nn.Embedding)
        assert decoder.token_embedding.num_embeddings == effective_vocab_size
        assert decoder.token_embedding.embedding_dim == EMBEDDING_DIMENSION

    def test_ties_output_weights_to_token_embedding(
        self,
        free_transformer_factory: Callable[..., FreeActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
    ):
        decoder = free_transformer_factory()
        tokenizer = mock_tokenizer_factory(vocab_size=VOCAB_SIZE)
        decoder.set_tokenizer(tokenizer=tokenizer)
        lm_head = decoder.action_heads[DecoderOutputKey.ACTION_LOGITS.value].output_proj
        with torch.no_grad():
            decoder.token_embedding.weight.data[0] = 999.0
        assert lm_head.weight.data[0, 0] == 999.0


class TestFreeActionTransformerForward:
    def test_training_output_keys(
        self,
        free_transformer_factory: Callable[..., FreeActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
        tokenized_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = free_transformer_factory()
        tokenizer = mock_tokenizer_factory(vocab_size=VOCAB_SIZE)
        decoder.set_tokenizer(tokenizer=tokenizer)
        decoder.train()
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        actions = tokenized_actions_factory(
            batch_size=BATCH_SIZE,
            action_token_length=ACTION_TOKEN_LENGTH,
            vocab_size=VOCAB_SIZE,
        )
        predictions = decoder(features=features, actions=actions)
        expected_keys = {
            DecoderOutputKey.ACTION_LOGITS.value,
            DecoderOutputKey.BINARY_LOGITS.value,
            DecoderOutputKey.LATENT_CODES.value,
            LatentKey.POSTERIOR_LATENT.value,
        }
        assert set(predictions.keys()) == expected_keys

    def test_training_output_shape(
        self,
        free_transformer_factory: Callable[..., FreeActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
        tokenized_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = free_transformer_factory()
        tokenizer = mock_tokenizer_factory(vocab_size=VOCAB_SIZE)
        decoder.set_tokenizer(tokenizer=tokenizer)
        decoder.train()
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        actions = tokenized_actions_factory(
            batch_size=BATCH_SIZE,
            action_token_length=ACTION_TOKEN_LENGTH,
            vocab_size=VOCAB_SIZE,
        )
        predictions = decoder(features=features, actions=actions)
        logits = predictions[DecoderOutputKey.ACTION_LOGITS.value]
        effective_vocab_size = VOCAB_SIZE + 1
        assert logits.shape == (BATCH_SIZE, ACTION_TOKEN_LENGTH, effective_vocab_size)

    @pytest.mark.parametrize("deterministic", [True, False])
    def test_inference_output_keys(
        self,
        free_transformer_factory: Callable[..., FreeActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
        deterministic: bool,
    ):
        decoder = free_transformer_factory(deterministic=deterministic)
        tokenizer = mock_tokenizer_factory(vocab_size=VOCAB_SIZE)
        decoder.set_tokenizer(tokenizer=tokenizer)
        decoder.eval()
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        with torch.no_grad():
            predictions = decoder(features=features, actions=None)
        expected_keys = {
            DecoderOutputKey.PREDICTED_ACTION_TOKENS.value,
            DecoderOutputKey.LATENT_CODES.value,
            LatentKey.POSTERIOR_LATENT.value,
        }
        assert set(predictions.keys()) == expected_keys

    @pytest.mark.parametrize("deterministic", [True, False])
    def test_inference_output_shape(
        self,
        free_transformer_factory: Callable[..., FreeActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
        deterministic: bool,
    ):
        decoder = free_transformer_factory(deterministic=deterministic)
        tokenizer = mock_tokenizer_factory(vocab_size=VOCAB_SIZE)
        decoder.set_tokenizer(tokenizer=tokenizer)
        decoder.eval()
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        with torch.no_grad():
            predictions = decoder(features=features, actions=None)
        predicted_tokens = predictions[DecoderOutputKey.PREDICTED_ACTION_TOKENS.value]
        # Spatial features produce SPATIAL_HEIGHT * SPATIAL_WIDTH = 16 prefix tokens
        prefix_length = SPATIAL_HEIGHT * SPATIAL_WIDTH
        max_generated_length = MAX_SEQ_LEN - prefix_length
        assert predicted_tokens.shape[0] == BATCH_SIZE
        assert 1 <= predicted_tokens.shape[1] <= max_generated_length

    def test_inference_terminates_early_on_eos(
        self,
        free_transformer_factory: Callable[..., FreeActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = free_transformer_factory()
        tokenizer = mock_tokenizer_factory(vocab_size=VOCAB_SIZE)
        decoder.set_tokenizer(tokenizer=tokenizer)
        decoder.eval()
        eos_token_id = tokenizer.action_tokenizer.eos_token_id
        head = decoder.action_heads[DecoderOutputKey.ACTION_LOGITS.value]
        with torch.no_grad():
            head.output_proj.weight.data.zero_()
            # Set EOS embedding row to large value so dot product with any input strongly favors EOS
            head.output_proj.weight.data[eos_token_id] = 100.0
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        with torch.no_grad():
            predictions = decoder(features=features, actions=None)
        tokens = predictions[DecoderOutputKey.PREDICTED_ACTION_TOKENS.value]
        assert tokens.shape[1] == 1
        assert (tokens == eos_token_id).all()

    def test_raises_if_sequence_too_long(
        self,
        free_transformer_factory: Callable[..., FreeActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
        tokenized_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        # Use a very small max_seq_len that will be exceeded by feature + action tokens
        small_max_seq_len = 4
        decoder = free_transformer_factory(max_seq_len=small_max_seq_len)
        tokenizer = mock_tokenizer_factory(vocab_size=VOCAB_SIZE)
        decoder.set_tokenizer(tokenizer=tokenizer)
        decoder.eval()
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        # feature tokens: 4*4 = 16, action tokens: 8, total: 24 > small_max_seq_len=4
        actions = tokenized_actions_factory(
            batch_size=BATCH_SIZE,
            action_token_length=ACTION_TOKEN_LENGTH,
            vocab_size=VOCAB_SIZE,
        )
        expected_token_length = SPATIAL_HEIGHT * SPATIAL_WIDTH + ACTION_TOKEN_LENGTH
        with (
            pytest.raises(
                ValueError,
                match=re.escape(
                    f"Input token length {expected_token_length} > max_seq_len {small_max_seq_len}. "
                    "No room for any action tokens. "
                    "Consider increasing max_seq_len or reducing feature token count."
                ),
            ),
            torch.no_grad(),
        ):
            decoder(features=features, actions=actions)

    def test_training_does_not_use_generation_cache(
        self,
        free_transformer_factory: Callable[..., FreeActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
        tokenized_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = free_transformer_factory()
        decoder.set_tokenizer(tokenizer=mock_tokenizer_factory(vocab_size=VOCAB_SIZE))
        decoder.train()
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        actions = tokenized_actions_factory(
            batch_size=BATCH_SIZE,
            action_token_length=ACTION_TOKEN_LENGTH,
            vocab_size=VOCAB_SIZE,
        )
        with unittest.mock.patch.object(
            decoder.free_transformer, "forward", wraps=decoder.free_transformer.forward
        ) as mock_forward:
            decoder(features=features, actions=actions)
            call_kwargs = mock_forward.call_args.kwargs
            assert call_kwargs.get("generation_cache") is None

    def test_inference_uses_generation_cache(
        self,
        free_transformer_factory: Callable[..., FreeActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = free_transformer_factory()
        decoder.set_tokenizer(tokenizer=mock_tokenizer_factory(vocab_size=VOCAB_SIZE))
        decoder.eval()
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        with unittest.mock.patch.object(
            decoder.free_transformer, "forward", wraps=decoder.free_transformer.forward
        ) as mock_forward:
            with torch.no_grad():
                decoder(features=features, actions=None)
            first_call_kwargs = mock_forward.call_args_list[0].kwargs
            assert first_call_kwargs.get("generation_cache") is not None


def test_auxiliary_output_keys(
    free_transformer_factory: Callable[..., FreeActionTransformer],
):
    decoder = free_transformer_factory()
    assert decoder.get_auxiliary_output_keys() == {
        DecoderOutputKey.BINARY_LOGITS.value,
        DecoderOutputKey.LATENT_CODES.value,
        SampleKey.TOKENIZED_ACTIONS.value,
    }


def test_get_callbacks_returns_latent_visualization(
    free_transformer_factory: Callable[..., FreeActionTransformer],
):
    decoder = free_transformer_factory()
    experiment_config = MagicMock(spec=ExperimentConfig)
    experiment_config.val_every = 3
    callbacks = decoder.get_callbacks(experiment_config=experiment_config)
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], LatentVisualizationCallback)
    assert callbacks[0].log_every_n_epochs == 3
