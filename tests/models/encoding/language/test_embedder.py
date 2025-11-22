"""Tests for Embedder encoder."""
import pytest
import torch

from refactoring.data.constants import TOKENIZED_OBSERVATIONS_KEY
from refactoring.models.encoding.encoders.constants import EncoderOutputKeys
from refactoring.models.encoding.encoders.language.embedder import Embedder


@pytest.fixture
def token_inputs_factory():
    """Factory for creating tokenized inputs with customizable shape."""
    def factory(batch_size=4, seq_len=100, vocab_size=256):
        return {
            TOKENIZED_OBSERVATIONS_KEY: torch.randint(0, vocab_size, (batch_size, seq_len))
        }
    return factory


@pytest.mark.unit
class TestEmbedder:

    def test_embedder_can_be_instantiated(self):
        encoder = Embedder(
            input_keys=[TOKENIZED_OBSERVATIONS_KEY],
            vocab_size=256,
            embedding_dim=128,
            max_token_len=512,
        )
        assert isinstance(encoder, Embedder)
        assert encoder.vocab_size == 256
        assert encoder.embedding_dim == 128
        assert encoder.max_token_len == 512

    def test_output_specification(self):
        encoder = Embedder(
            input_keys=[TOKENIZED_OBSERVATIONS_KEY],
            vocab_size=256,
            embedding_dim=128,
            max_token_len=512,
        )
        output_spec = encoder.get_output_specification()
        assert output_spec.features == [EncoderOutputKeys.TOKEN_EMBEDDING.value]
        assert output_spec.dimensions == {EncoderOutputKeys.TOKEN_EMBEDDING.value: (512, 128)}

    def test_get_vocab_size(self):
        encoder = Embedder(
            input_keys=[TOKENIZED_OBSERVATIONS_KEY],
            vocab_size=256,
            embedding_dim=128,
            max_token_len=512,
        )
        assert encoder.get_vocab_size() == 256

    def test_forward_basic(self, token_inputs_factory):
        encoder = Embedder(
            input_keys=[TOKENIZED_OBSERVATIONS_KEY],
            vocab_size=256,
            embedding_dim=128,
            max_token_len=512,
            device="cpu",
        )

        inputs = token_inputs_factory(batch_size=2, seq_len=100)
        output = encoder(inputs)

        assert EncoderOutputKeys.TOKEN_EMBEDDING.value in output
        assert output[EncoderOutputKeys.TOKEN_EMBEDDING.value].shape == (2, 100, 128)

    def test_forward_max_length(self, token_inputs_factory):
        encoder = Embedder(
            input_keys=[TOKENIZED_OBSERVATIONS_KEY],
            vocab_size=256,
            embedding_dim=128,
            max_token_len=512,
            device="cpu",
        )

        inputs = token_inputs_factory(batch_size=2, seq_len=512)
        output = encoder(inputs)

        assert EncoderOutputKeys.TOKEN_EMBEDDING.value in output
        assert output[EncoderOutputKeys.TOKEN_EMBEDDING.value].shape == (2, 512, 128)

    def test_forward_exceeds_max_length(self, token_inputs_factory):
        encoder = Embedder(
            input_keys=[TOKENIZED_OBSERVATIONS_KEY],
            vocab_size=256,
            embedding_dim=128,
            max_token_len=512,
        )

        inputs = token_inputs_factory(batch_size=2, seq_len=600)
        with pytest.raises(ValueError, match="Sequence length 600 exceeds max_token_len 512"):
            encoder(inputs)

    def test_forward_wrong_shape(self):
        encoder = Embedder(
            input_keys=[TOKENIZED_OBSERVATIONS_KEY],
            vocab_size=256,
            embedding_dim=128,
            max_token_len=512,
        )

        tokens = torch.randint(0, 256, (2, 100, 10))
        with pytest.raises(ValueError, match="Expected tokenized input to have shape"):
            encoder({TOKENIZED_OBSERVATIONS_KEY: tokens})

    def test_requires_exactly_one_input_key(self):
        with pytest.raises(ValueError, match="requires exactly one input key"):
            Embedder(
                input_keys=["key1", "key2"],
                vocab_size=256,
                embedding_dim=128,
                max_token_len=512,
            )

    def test_frozen_weights(self):
        encoder = Embedder(
            input_keys=[TOKENIZED_OBSERVATIONS_KEY],
            vocab_size=256,
            embedding_dim=128,
            max_token_len=512,
            frozen=True,
        )

        for param in encoder.parameters():
            assert param.requires_grad is False

    def test_unfrozen_weights(self):
        encoder = Embedder(
            input_keys=[TOKENIZED_OBSERVATIONS_KEY],
            vocab_size=256,
            embedding_dim=128,
            max_token_len=512,
            frozen=False,
        )

        for param in encoder.parameters():
            assert param.requires_grad is True

    def test_different_vocab_sizes(self, token_inputs_factory):
        for vocab_size in [128, 256, 512, 1024]:
            encoder = Embedder(
                input_keys=[TOKENIZED_OBSERVATIONS_KEY],
                vocab_size=vocab_size,
                embedding_dim=128,
                max_token_len=512,
                device="cpu",
            )
            assert encoder.vocab_size == vocab_size
            assert encoder.get_vocab_size() == vocab_size

            inputs = token_inputs_factory(batch_size=2, seq_len=100, vocab_size=vocab_size)
            output = encoder(inputs)
            assert output[EncoderOutputKeys.TOKEN_EMBEDDING.value].shape == (2, 100, 128)

    def test_different_embedding_dims(self, token_inputs_factory):
        for embed_dim in [64, 128, 256, 512]:
            encoder = Embedder(
                input_keys=[TOKENIZED_OBSERVATIONS_KEY],
                vocab_size=256,
                embedding_dim=embed_dim,
                max_token_len=512,
                device="cpu",
            )
            assert encoder.embedding_dim == embed_dim

            inputs = token_inputs_factory(batch_size=2, seq_len=100)
            output = encoder(inputs)
            assert output[EncoderOutputKeys.TOKEN_EMBEDDING.value].shape == (2, 100, embed_dim)

    def test_single_batch_item(self, token_inputs_factory):
        encoder = Embedder(
            input_keys=[TOKENIZED_OBSERVATIONS_KEY],
            vocab_size=256,
            embedding_dim=128,
            max_token_len=512,
            device="cpu",
        )

        inputs = token_inputs_factory(batch_size=1, seq_len=100)
        output = encoder(inputs)
        assert output[EncoderOutputKeys.TOKEN_EMBEDDING.value].shape == (1, 100, 128)

    def test_device_placement(self):
        encoder = Embedder(
            input_keys=[TOKENIZED_OBSERVATIONS_KEY],
            vocab_size=256,
            embedding_dim=128,
            max_token_len=512,
            device="cpu",
        )

        assert encoder.device == torch.device("cpu")
        assert encoder.embedding.weight.device == torch.device("cpu")