"""Tests for versatil.models.decoding.decoders.factory.moe_free_action_transformer module."""

import re
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from versatil.data.tokenization import Tokenizer
from versatil.models.decoding.action_heads.moe import MoEHead
from versatil.models.decoding.action_heads.single_output import ActionHead
from versatil.models.decoding.constants import DecoderOutputKey
from versatil.models.decoding.decoders.factory.free_action_transformer import (
    FreeActionTransformer,
)
from versatil.models.decoding.decoders.factory.moe_free_action_transformer import (
    MoEFreeActionTransformer,
)

EMBEDDING_DIMENSION = 32
NUMBER_OF_HEADS = 2
NUMBER_OF_DECODER_LAYERS = 2
NUMBER_OF_ENCODER_LAYERS = 1
LATENT_BITS = 4
MAX_SEQ_LEN = 64
PREDICTION_HORIZON = 4
BATCH_SIZE = 2
POSITION_DIM = 3
SPATIAL_HEIGHT = 4
SPATIAL_WIDTH = 4
VOCAB_SIZE = 32
ACTION_TOKEN_LENGTH = 8
NUM_EXPERTS = 3


@pytest.fixture
def moe_head_factory(
    action_head_factory: Callable[..., ActionHead],
) -> Callable[..., MoEHead]:
    """Factory for MoEHead instances with pre-instantiated expert ActionHeads."""

    def factory(
        input_dim: int = EMBEDDING_DIMENSION,
        num_experts: int = NUM_EXPERTS,
    ) -> MoEHead:
        experts = [action_head_factory(input_dim=input_dim) for _ in range(num_experts)]
        return MoEHead(experts=experts)

    return factory


@pytest.fixture
def moe_free_transformer_factory(
    mock_action_space_factory: Callable[..., MagicMock],
    mock_observation_space_factory: Callable[..., MagicMock],
    moe_head_factory: Callable[..., MoEHead],
) -> Callable[..., MoEFreeActionTransformer]:
    """Factory for MoEFreeActionTransformer instances with small dimensions."""

    def factory(
        embedding_dimension: int = EMBEDDING_DIMENSION,
        num_experts: int = NUM_EXPERTS,
        number_of_heads: int = NUMBER_OF_HEADS,
        number_of_decoder_layers: int = NUMBER_OF_DECODER_LAYERS,
        number_of_encoder_layers: int = NUMBER_OF_ENCODER_LAYERS,
        latent_bits: int = LATENT_BITS,
        max_seq_len: int = MAX_SEQ_LEN,
        prediction_horizon: int = PREDICTION_HORIZON,
        position_dim: int = POSITION_DIM,
        deterministic: bool = True,
        input_keys: list[str] | None = None,
    ) -> MoEFreeActionTransformer:
        if input_keys is None:
            input_keys = ["rgb_features"]
        action_space = mock_action_space_factory(position_dim=position_dim)
        observation_space = mock_observation_space_factory()
        moe_head = moe_head_factory(
            input_dim=embedding_dimension, num_experts=num_experts
        )
        action_heads = {DecoderOutputKey.ACTION_LOGITS.value: moe_head}
        return MoEFreeActionTransformer(
            action_heads=action_heads,
            input_keys=input_keys,
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=1,
            prediction_horizon=prediction_horizon,
            device="cpu",
            max_seq_len=max_seq_len,
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_decoder_layers=number_of_decoder_layers,
            number_of_encoder_layers=number_of_encoder_layers,
            latent_bits=latent_bits,
            deterministic=deterministic,
        )

    return factory


class TestMoEFreeActionTransformerInitialization:
    def test_inherits_from_free_action_transformer(
        self,
        moe_free_transformer_factory: Callable[..., MoEFreeActionTransformer],
    ):
        decoder = moe_free_transformer_factory()
        assert isinstance(decoder, FreeActionTransformer)

    def test_moe_action_head_is_set(
        self,
        moe_free_transformer_factory: Callable[..., MoEFreeActionTransformer],
    ):
        decoder = moe_free_transformer_factory()
        assert isinstance(decoder.moe_action_head, MoEHead)
        assert (
            decoder.moe_action_head
            is decoder.action_heads[DecoderOutputKey.ACTION_LOGITS.value]
        )

    def test_expert_gating_projection_initially_none(
        self,
        moe_free_transformer_factory: Callable[..., MoEFreeActionTransformer],
    ):
        decoder = moe_free_transformer_factory()
        assert decoder.expert_gating_projection is None

    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    @pytest.mark.parametrize("number_of_decoder_layers", [2, 4])
    def test_stores_configuration(
        self,
        moe_free_transformer_factory: Callable[..., MoEFreeActionTransformer],
        embedding_dimension: int,
        number_of_decoder_layers: int,
    ):
        decoder = moe_free_transformer_factory(
            embedding_dimension=embedding_dimension,
            number_of_decoder_layers=number_of_decoder_layers,
        )
        assert decoder.embedding_dimension == embedding_dimension
        assert decoder.number_of_decoder_layers == number_of_decoder_layers


class TestMoEFreeActionTransformerSetTokenizer:
    def test_raises_without_tokenizer(
        self,
        moe_free_transformer_factory: Callable[..., MoEFreeActionTransformer],
    ):
        decoder = moe_free_transformer_factory()
        with pytest.raises(
            ValueError,
            match=re.escape(
                "MoEFreeActionTransformer requires a tokenizer for tokenized action prediction."
            ),
        ):
            decoder.set_tokenizer(tokenizer=None)

    def test_raises_without_action_tokenizer(
        self,
        moe_free_transformer_factory: Callable[..., MoEFreeActionTransformer],
    ):
        decoder = moe_free_transformer_factory()
        tokenizer = MagicMock(spec=Tokenizer)
        tokenizer.action_tokenizer = None
        with pytest.raises(
            ValueError,
            match=re.escape(
                "MoEFreeActionTransformer requires a tokenizer for tokenized action prediction."
            ),
        ):
            decoder.set_tokenizer(tokenizer=tokenizer)

    def test_sets_vocab_size(
        self,
        moe_free_transformer_factory: Callable[..., MoEFreeActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
    ):
        decoder = moe_free_transformer_factory()
        tokenizer = mock_tokenizer_factory(vocab_size=VOCAB_SIZE)
        decoder.set_tokenizer(tokenizer=tokenizer)
        effective_vocab_size = VOCAB_SIZE + 1
        assert decoder.vocab_size == effective_vocab_size

    def test_creates_expert_gating_projection(
        self,
        moe_free_transformer_factory: Callable[..., MoEFreeActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
    ):
        decoder = moe_free_transformer_factory()
        tokenizer = mock_tokenizer_factory(vocab_size=VOCAB_SIZE)
        decoder.set_tokenizer(tokenizer=tokenizer)
        assert isinstance(decoder.expert_gating_projection, nn.Linear)
        assert decoder.expert_gating_projection.in_features == EMBEDDING_DIMENSION
        assert decoder.expert_gating_projection.out_features == NUM_EXPERTS

    def test_expert_output_dims_set(
        self,
        moe_free_transformer_factory: Callable[..., MoEFreeActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
    ):
        decoder = moe_free_transformer_factory()
        tokenizer = mock_tokenizer_factory(vocab_size=VOCAB_SIZE)
        decoder.set_tokenizer(tokenizer=tokenizer)
        effective_vocab_size = VOCAB_SIZE + 1
        for expert in decoder.moe_action_head.experts:
            assert expert.output_dim == effective_vocab_size
            assert expert.output_proj.out_features == effective_vocab_size

    def test_creates_token_embedding(
        self,
        moe_free_transformer_factory: Callable[..., MoEFreeActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
    ):
        decoder = moe_free_transformer_factory()
        tokenizer = mock_tokenizer_factory(vocab_size=VOCAB_SIZE)
        decoder.set_tokenizer(tokenizer=tokenizer)
        assert isinstance(decoder.token_embedding, nn.Embedding)
        effective_vocab_size = VOCAB_SIZE + 1
        assert decoder.token_embedding.num_embeddings == effective_vocab_size
        assert decoder.token_embedding.embedding_dim == EMBEDDING_DIMENSION

    def test_stores_action_tokenizer(
        self,
        moe_free_transformer_factory: Callable[..., MoEFreeActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
    ):
        decoder = moe_free_transformer_factory()
        tokenizer = mock_tokenizer_factory(vocab_size=VOCAB_SIZE)
        decoder.set_tokenizer(tokenizer=tokenizer)
        assert decoder.tokenizer is tokenizer.action_tokenizer


class TestMoEFreeActionTransformerForward:
    def test_raises_if_sequence_too_long(
        self,
        moe_free_transformer_factory: Callable[..., MoEFreeActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
        tokenized_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        # H*W=16 feature tokens + 8 action tokens = 24 > max_seq_len=20
        small_max_seq_len = 20
        decoder = moe_free_transformer_factory(max_seq_len=small_max_seq_len)
        decoder.set_tokenizer(tokenizer=mock_tokenizer_factory(vocab_size=VOCAB_SIZE))
        decoder.train()
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        actions = tokenized_actions_factory()
        expected_token_length = (
            SPATIAL_HEIGHT * SPATIAL_WIDTH + 8
        )  # feature + action tokens
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Input token length {expected_token_length} > max_seq_len {small_max_seq_len}."
            ),
        ):
            decoder(features=features, actions=actions)

    def test_training_output_keys(
        self,
        moe_free_transformer_factory: Callable[..., MoEFreeActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
        tokenized_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = moe_free_transformer_factory()
        decoder.set_tokenizer(tokenizer=mock_tokenizer_factory(vocab_size=VOCAB_SIZE))
        decoder.train()
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        actions = tokenized_actions_factory()
        outputs = decoder(features=features, actions=actions)
        expected_keys = {
            DecoderOutputKey.ACTION_LOGITS.value,
            DecoderOutputKey.BINARY_LOGITS.value,
            DecoderOutputKey.LATENT_CODES.value,
            DecoderOutputKey.ROUTING_WEIGHTS.value,
        }
        assert set(outputs.keys()) == expected_keys

    def test_training_output_shape(
        self,
        moe_free_transformer_factory: Callable[..., MoEFreeActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
        tokenized_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = moe_free_transformer_factory()
        decoder.set_tokenizer(tokenizer=mock_tokenizer_factory(vocab_size=VOCAB_SIZE))
        decoder.train()
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        actions = tokenized_actions_factory()
        outputs = decoder(features=features, actions=actions)
        logits = outputs[DecoderOutputKey.ACTION_LOGITS.value]
        effective_vocab_size = VOCAB_SIZE + 1
        assert logits.shape == (BATCH_SIZE, ACTION_TOKEN_LENGTH, effective_vocab_size)

    def test_training_routing_weights_shape(
        self,
        moe_free_transformer_factory: Callable[..., MoEFreeActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
        tokenized_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = moe_free_transformer_factory()
        decoder.set_tokenizer(tokenizer=mock_tokenizer_factory(vocab_size=VOCAB_SIZE))
        decoder.train()
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        actions = tokenized_actions_factory()
        outputs = decoder(features=features, actions=actions)
        routing_weights = outputs[DecoderOutputKey.ROUTING_WEIGHTS.value]
        assert routing_weights.shape[-1] == NUM_EXPERTS
        assert routing_weights.shape[0] == BATCH_SIZE

    def test_inference_output_keys(
        self,
        moe_free_transformer_factory: Callable[..., MoEFreeActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = moe_free_transformer_factory()
        decoder.set_tokenizer(tokenizer=mock_tokenizer_factory(vocab_size=VOCAB_SIZE))
        decoder.eval()
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        with torch.no_grad():
            outputs = decoder(features=features, actions=None)
        routing_key = (
            f"{DecoderOutputKey.ACTION_LOGITS.value}"
            f"_{DecoderOutputKey.ROUTING_WEIGHTS.value}"
        )
        expected_keys = {
            DecoderOutputKey.PREDICTED_ACTION_TOKENS.value,
            routing_key,
        }
        assert set(outputs.keys()) == expected_keys

    def test_inference_output_shape(
        self,
        moe_free_transformer_factory: Callable[..., MoEFreeActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = moe_free_transformer_factory()
        decoder.set_tokenizer(tokenizer=mock_tokenizer_factory(vocab_size=VOCAB_SIZE))
        decoder.eval()
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        with torch.no_grad():
            outputs = decoder(features=features, actions=None)
        predicted_tokens = outputs[DecoderOutputKey.PREDICTED_ACTION_TOKENS.value]
        assert predicted_tokens.shape[0] == BATCH_SIZE
        prefix_length = SPATIAL_HEIGHT * SPATIAL_WIDTH
        max_generated_length = MAX_SEQ_LEN - prefix_length
        assert 1 <= predicted_tokens.shape[1] <= max_generated_length
        assert predicted_tokens.dtype == torch.long

    def test_inference_terminates_early_on_eos(
        self,
        moe_free_transformer_factory: Callable[..., MoEFreeActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = moe_free_transformer_factory()
        tokenizer = mock_tokenizer_factory(vocab_size=VOCAB_SIZE)
        decoder.set_tokenizer(tokenizer=tokenizer)
        decoder.eval()
        eos_token_id = tokenizer.action_tokenizer.eos_token_id
        with torch.no_grad():
            for expert in decoder.moe_action_head.experts:
                expert.output_proj.weight.data.zero_()
                expert.output_proj.bias.data.zero_()
                expert.output_proj.bias.data[eos_token_id] = 100.0
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        with torch.no_grad():
            outputs = decoder(features=features, actions=None)
        tokens = outputs[DecoderOutputKey.PREDICTED_ACTION_TOKENS.value]
        assert tokens.shape[1] == 1
        assert (tokens == eos_token_id).all()

    def test_inference_routing_weights_shape(
        self,
        moe_free_transformer_factory: Callable[..., MoEFreeActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = moe_free_transformer_factory()
        decoder.set_tokenizer(tokenizer=mock_tokenizer_factory(vocab_size=VOCAB_SIZE))
        decoder.eval()
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        with torch.no_grad():
            outputs = decoder(features=features, actions=None)
        routing_key = (
            f"{DecoderOutputKey.ACTION_LOGITS.value}"
            f"_{DecoderOutputKey.ROUTING_WEIGHTS.value}"
        )
        routing_weights = outputs[routing_key]
        assert routing_weights.shape[0] == BATCH_SIZE
        assert routing_weights.shape[-1] == NUM_EXPERTS

    def test_different_features_produce_different_routing(
        self,
        moe_free_transformer_factory: Callable[..., MoEFreeActionTransformer],
        mock_tokenizer_factory: Callable[..., MagicMock],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = moe_free_transformer_factory()
        decoder.set_tokenizer(tokenizer=mock_tokenizer_factory(vocab_size=VOCAB_SIZE))
        decoder.eval()
        features_a = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        features_b = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        with torch.no_grad():
            outputs_a = decoder(features=features_a, actions=None)
            outputs_b = decoder(features=features_b, actions=None)
        routing_key = (
            f"{DecoderOutputKey.ACTION_LOGITS.value}"
            f"_{DecoderOutputKey.ROUTING_WEIGHTS.value}"
        )
        assert not torch.equal(
            outputs_a[routing_key],
            outputs_b[routing_key],
        )
