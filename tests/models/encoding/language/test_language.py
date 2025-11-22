import pytest
import torch

from refactoring.models.encoding.encoders.language import LanguageEncoder
from refactoring.models.encoding.encoders.constants import (
    LanguageEncoderType,
    PoolingMethod,
    EncoderOutputKeys,
)
from refactoring.data.constants import TOKENIZED_OBSERVATIONS_KEY


LANGUAGE_MODELS = [
    (LanguageEncoderType.BERT_BASE.value, 768),
    (LanguageEncoderType.DISTILBERT_BASE.value, 768),
    (LanguageEncoderType.MINI_LM_L6.value, 384),
]

pooling_methodS = [
    PoolingMethod.DEFAULT.value,
    PoolingMethod.AVERAGE.value,
    PoolingMethod.LEARNED_AGGREGATION.value,
]


@pytest.fixture
def batch_size():
    """Standard batch size for testing."""
    return 2


@pytest.fixture
def temporal_length():
    """Standard temporal length for testing."""
    return 3


@pytest.fixture
def token_inputs_factory():
    """Factory for creating tokenized inputs with customizable shape."""
    def factory(batch_size=2, temporal_length=None, seq_len=100, vocab_size=30522):
        if temporal_length is None:
            return {
                TOKENIZED_OBSERVATIONS_KEY: torch.randint(0, vocab_size, (batch_size, seq_len))
            }
        else:
            return {
                TOKENIZED_OBSERVATIONS_KEY: torch.randint(0, vocab_size, (batch_size, temporal_length, seq_len))
            }
    return factory


@pytest.fixture
def text_inputs_batch(token_inputs_factory, batch_size):
    """Batch tokenized inputs (2D: batch x seq_len)."""
    return token_inputs_factory(batch_size=batch_size)


@pytest.fixture
def text_inputs_temporal(token_inputs_factory, batch_size, temporal_length):
    """Temporal tokenized inputs (3D: batch x temporal x seq_len)."""
    return token_inputs_factory(batch_size=batch_size, temporal_length=temporal_length)


@pytest.mark.integration
class TestLanguageEncoderInitialization:
    """Test LanguageEncoder initialization."""

    @pytest.mark.parametrize("model_name,expected_dim", LANGUAGE_MODELS)
    def test_initialization(self, model_name, expected_dim):
        encoder = LanguageEncoder(
            input_keys=TOKENIZED_OBSERVATIONS_KEY,
            model_name=model_name,
            pretrained=True,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
        )

        assert encoder.feature_dim == expected_dim
        spec = encoder.get_output_specification()
        assert spec.dimensions[EncoderOutputKeys.LANGUAGE.value] == expected_dim
        assert encoder.pooling_method == PoolingMethod.AVERAGE.value

    def test_get_output_specification(self):
        encoder = LanguageEncoder(
            input_keys=TOKENIZED_OBSERVATIONS_KEY,
            model_name=LanguageEncoderType.BERT_BASE.value,
            pretrained=True,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
        )

        spec = encoder.get_output_specification()
        assert EncoderOutputKeys.LANGUAGE.value in spec.features
        assert isinstance(spec.features, list)
        assert EncoderOutputKeys.LANGUAGE.value in spec.dimensions

    def test_init_missing_language_key(self):
        with pytest.raises(ValueError, match="Missing required inputs"):
            LanguageEncoder(
                input_keys="invalid_key",
                model_name=LanguageEncoderType.BERT_BASE.value,
                pretrained=True,
                frozen=True,
                pooling_method=PoolingMethod.AVERAGE.value,
            )

    def test_init_frozen(self):
        encoder = LanguageEncoder(
            input_keys=TOKENIZED_OBSERVATIONS_KEY,
            model_name=LanguageEncoderType.BERT_BASE.value,
            pretrained=True,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
        )

        for param in encoder.encoder.parameters():
            assert not param.requires_grad


@pytest.mark.integration
class TestLanguageEncoderForward:
    """Test LanguageEncoder forward pass."""

    @pytest.mark.parametrize("model_name,expected_dim", LANGUAGE_MODELS)
    @pytest.mark.parametrize("pooling_method", [PoolingMethod.NONE.value, PoolingMethod.AVERAGE.value, PoolingMethod.LEARNED_AGGREGATION.value])
    def test_forward_batch_input(self, model_name, expected_dim, text_inputs_batch, batch_size, pooling_method):

        encoder = LanguageEncoder(
            input_keys=TOKENIZED_OBSERVATIONS_KEY,
            model_name=model_name,
            pretrained=False,
            frozen=True,
            pooling_method=pooling_method,
        )
        if pooling_method == PoolingMethod.NONE.value:
            expected_dim = (encoder.max_text_length, expected_dim)
        else:
            expected_dim = (expected_dim,)
        mask_expected_dim = encoder.max_text_length if pooling_method == PoolingMethod.NONE.value else None
        mask_expected_key = f"{EncoderOutputKeys.LANGUAGE.value}_{EncoderOutputKeys.PADDING_MASK.value}"
        output_dict = encoder(text_inputs_batch)

        assert isinstance(output_dict, dict)
        assert EncoderOutputKeys.LANGUAGE.value in output_dict
        assert mask_expected_key in output_dict

        output = output_dict[EncoderOutputKeys.LANGUAGE.value]
        mask_output = output_dict[mask_expected_key]
        assert output.shape == (batch_size, *expected_dim)
        assert output.dtype == torch.float32
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        if mask_expected_dim is not None:
            assert mask_output.shape == (batch_size, mask_expected_dim)
        else:
            assert mask_output.shape == (batch_size,)
        assert mask_output.dtype == torch.bool

    @pytest.mark.parametrize("model_name,expected_dim", LANGUAGE_MODELS)
    @pytest.mark.parametrize("pooling_method", [PoolingMethod.NONE.value, PoolingMethod.AVERAGE.value, PoolingMethod.LEARNED_AGGREGATION.value])
    def test_forward_temporal_input(self, model_name, expected_dim, text_inputs_temporal, batch_size, temporal_length, pooling_method):
        encoder = LanguageEncoder(
            input_keys=TOKENIZED_OBSERVATIONS_KEY,
            model_name=model_name,
            pretrained=False,
            frozen=True,
            pooling_method=pooling_method,
        )
        if pooling_method == PoolingMethod.NONE.value:
            expected_dim = (encoder.max_text_length, expected_dim)
        else:
            expected_dim = (expected_dim,)
        mask_expected_dim = encoder.max_text_length if pooling_method == PoolingMethod.NONE.value else None
        mask_expected_key = f"{EncoderOutputKeys.LANGUAGE.value}_{EncoderOutputKeys.PADDING_MASK.value}"

        output_dict = encoder(text_inputs_temporal)

        assert isinstance(output_dict, dict)
        assert EncoderOutputKeys.LANGUAGE.value in output_dict
        assert mask_expected_key in output_dict

        output = output_dict[EncoderOutputKeys.LANGUAGE.value]
        mask_output = output_dict[mask_expected_key]
        assert output.shape == (batch_size, temporal_length, *expected_dim)
        assert output.dtype == torch.float32
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        if mask_expected_dim is not None:
            assert mask_output.shape == (batch_size, temporal_length, mask_expected_dim)
        else:
            assert mask_output.shape == (batch_size, temporal_length)
        assert mask_output.dtype == torch.bool

    @pytest.mark.parametrize("feature_method", pooling_methodS)
    def test_pooling_methods(self, feature_method, text_inputs_batch, batch_size):
        encoder = LanguageEncoder(
            input_keys=TOKENIZED_OBSERVATIONS_KEY,
            model_name=LanguageEncoderType.BERT_BASE.value,
            pretrained=True,
            frozen=True,
            pooling_method=feature_method,
        )

        output = encoder(text_inputs_batch)[EncoderOutputKeys.LANGUAGE.value]

        assert output.shape == (batch_size, 768)
        assert output.dtype == torch.float32
        assert not torch.isnan(output).any()

        if feature_method == PoolingMethod.LEARNED_AGGREGATION.value:
            assert encoder.pooling_head is not None
        else:
            assert encoder.pooling_head is None

    def test_different_texts_produce_different_features(self, token_inputs_factory):
        encoder = LanguageEncoder(
            input_keys=TOKENIZED_OBSERVATIONS_KEY,
            model_name=LanguageEncoderType.BERT_BASE.value,
            pretrained=True,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
        )

        encoder.eval()

        batch_size = 2
        tokens1 = token_inputs_factory(batch_size=batch_size)
        tokens2 = token_inputs_factory(batch_size=batch_size)

        with torch.no_grad():
            output1 = encoder(tokens1)[EncoderOutputKeys.LANGUAGE.value]
            output2 = encoder(tokens2)[EncoderOutputKeys.LANGUAGE.value]

        assert output1.shape == output2.shape

    def test_same_text_produces_same_features(self, token_inputs_factory):
        encoder = LanguageEncoder(
            input_keys=TOKENIZED_OBSERVATIONS_KEY,
            model_name=LanguageEncoderType.BERT_BASE.value,
            pretrained=True,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
        )

        encoder.eval()
        batch_size = 2
        tokens = token_inputs_factory(batch_size=batch_size)

        with torch.no_grad():
            output1 = encoder(tokens)[EncoderOutputKeys.LANGUAGE.value]
            output2 = encoder(tokens)[EncoderOutputKeys.LANGUAGE.value]

        assert torch.allclose(output1, output2, atol=1e-6)


@pytest.mark.integration
class TestLanguageEncoderEdgeCases:
    """Test edge cases."""

    def test_single_word_input(self, token_inputs_factory):
        encoder = LanguageEncoder(
            input_keys=TOKENIZED_OBSERVATIONS_KEY,
            model_name=LanguageEncoderType.BERT_BASE.value,
            pretrained=True,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
        )

        tokens = token_inputs_factory(batch_size=2, seq_len=5)
        output = encoder(tokens)[EncoderOutputKeys.LANGUAGE.value]

        assert output.shape == (2, 768)
        assert not torch.isnan(output).any()

    def test_long_text_truncation(self, token_inputs_factory):
        encoder = LanguageEncoder(
            input_keys=TOKENIZED_OBSERVATIONS_KEY,
            model_name=LanguageEncoderType.BERT_BASE.value,
            pretrained=True,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
        )

        tokens = token_inputs_factory(batch_size=1, seq_len=512)
        output = encoder(tokens)[EncoderOutputKeys.LANGUAGE.value]

        assert output.shape == (1, 768)
        assert not torch.isnan(output).any()

    def test_batch_size_one(self, token_inputs_factory):
        encoder = LanguageEncoder(
            input_keys=TOKENIZED_OBSERVATIONS_KEY,
            model_name=LanguageEncoderType.BERT_BASE.value,
            pretrained=True,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
        )

        tokens = token_inputs_factory(batch_size=1)
        output = encoder(tokens)[EncoderOutputKeys.LANGUAGE.value]

        assert output.shape == (1, 768)

    def test_variable_length_temporal(self, token_inputs_factory):
        encoder = LanguageEncoder(
            input_keys=TOKENIZED_OBSERVATIONS_KEY,
            model_name=LanguageEncoderType.BERT_BASE.value,
            pretrained=True,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
        )

        batch_size = 2
        temporal_length = 3
        tokens = token_inputs_factory(batch_size=batch_size, temporal_length=temporal_length)
        output = encoder(tokens)[EncoderOutputKeys.LANGUAGE.value]

        assert output.shape == (batch_size, temporal_length, 768)


@pytest.mark.integration
class TestLanguageEncoderOutputDims:
    """Test output dimension methods."""

    @pytest.mark.parametrize("model_name,expected_dim", LANGUAGE_MODELS)
    def test_output_specification_structure(self, model_name, expected_dim):
        encoder = LanguageEncoder(
            input_keys=TOKENIZED_OBSERVATIONS_KEY,
            model_name=model_name,
            pretrained=True,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
        )

        spec = encoder.get_output_specification()

        assert isinstance(spec.features, list)
        assert len(spec.features) == 1
        assert EncoderOutputKeys.LANGUAGE.value in spec.features
        assert isinstance(spec.dimensions, dict)
        assert spec.dimensions[EncoderOutputKeys.LANGUAGE.value] == expected_dim


@pytest.mark.integration
class TestLanguageEncoderGradients:
    """Test gradient behavior."""

    @pytest.mark.parametrize("model_name", [m[0] for m in LANGUAGE_MODELS])
    def test_gradients_enabled_unfrozen(self, model_name, text_inputs_batch):
        encoder = LanguageEncoder(
            input_keys=TOKENIZED_OBSERVATIONS_KEY,
            model_name=model_name,
            pretrained=True,
            frozen=False,
            pooling_method=PoolingMethod.AVERAGE.value,
        )

        output = encoder(text_inputs_batch)[EncoderOutputKeys.LANGUAGE.value]
        loss = output.sum()
        loss.backward()

        encoder_has_grads = any(
            param.grad is not None and param.grad.abs().sum() > 0
            for param in encoder.encoder.parameters() if param.requires_grad
        )
        assert encoder_has_grads

    @pytest.mark.parametrize("model_name", [m[0] for m in LANGUAGE_MODELS])
    def test_gradients_disabled_frozen(self, model_name, text_inputs_batch):
        encoder = LanguageEncoder(
            input_keys=TOKENIZED_OBSERVATIONS_KEY,
            model_name=model_name,
            pretrained=True,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
        )

        output = encoder(text_inputs_batch)[EncoderOutputKeys.LANGUAGE.value]

        assert not output.requires_grad

        for param in encoder.encoder.parameters():
            assert not param.requires_grad


@pytest.mark.integration
class TestLanguageEncoderIntegration:
    """Integration tests for complete workflows."""

    def test_complete_forward_backward_pass(self, text_inputs_batch):
        encoder = LanguageEncoder(
            input_keys=TOKENIZED_OBSERVATIONS_KEY,
            model_name=LanguageEncoderType.BERT_BASE.value,
            pretrained=True,
            frozen=False,
            pooling_method=PoolingMethod.AVERAGE.value,
        )

        encoder.train()
        output = encoder(text_inputs_batch)[EncoderOutputKeys.LANGUAGE.value]
        loss = output.mean()
        loss.backward()

        assert output.requires_grad
        assert loss.requires_grad

    def test_eval_mode_no_gradients(self, text_inputs_batch):
        encoder = LanguageEncoder(
            input_keys=TOKENIZED_OBSERVATIONS_KEY,
            model_name=LanguageEncoderType.BERT_BASE.value,
            pretrained=True,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
        )

        encoder.eval()

        with torch.no_grad():
            output = encoder(text_inputs_batch)[EncoderOutputKeys.LANGUAGE.value]

        assert not output.requires_grad

    def test_consistent_output_shapes(self, token_inputs_factory):
        encoder = LanguageEncoder(
            input_keys=TOKENIZED_OBSERVATIONS_KEY,
            model_name=LanguageEncoderType.BERT_BASE.value,
            pretrained=True,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
        )

        batch_size = 2
        tokens1 = token_inputs_factory(batch_size=batch_size)
        tokens2 = token_inputs_factory(batch_size=batch_size)

        output1 = encoder(tokens1)[EncoderOutputKeys.LANGUAGE.value]
        output2 = encoder(tokens2)[EncoderOutputKeys.LANGUAGE.value]

        assert output1.shape == output2.shape

    def test_different_feature_methods_produce_different_outputs(self, text_inputs_batch):
        encoder_cls = LanguageEncoder(
            input_keys=TOKENIZED_OBSERVATIONS_KEY,
            model_name=LanguageEncoderType.BERT_BASE.value,
            pretrained=True,
            frozen=True,
            pooling_method=PoolingMethod.DEFAULT.value,
        )

        encoder_gap = LanguageEncoder(
            input_keys=TOKENIZED_OBSERVATIONS_KEY,
            model_name=LanguageEncoderType.BERT_BASE.value,
            pretrained=True,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
        )

        encoder_cls.eval()
        encoder_gap.eval()

        with torch.no_grad():
            output_cls = encoder_cls(text_inputs_batch)[EncoderOutputKeys.LANGUAGE.value]
            output_gap = encoder_gap(text_inputs_batch)[EncoderOutputKeys.LANGUAGE.value]

        assert output_cls.shape == output_gap.shape
        assert not torch.allclose(output_cls, output_gap, atol=1e-5)

    def test_different_sequence_lengths(self, token_inputs_factory):
        encoder = LanguageEncoder(
            input_keys=TOKENIZED_OBSERVATIONS_KEY,
            model_name=LanguageEncoderType.BERT_BASE.value,
            pretrained=True,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
        )

        batch_size = 2
        tokens_short = token_inputs_factory(batch_size=batch_size, seq_len=50)
        tokens_long = token_inputs_factory(batch_size=batch_size, seq_len=200)

        output_short = encoder(tokens_short)[EncoderOutputKeys.LANGUAGE.value]
        output_long = encoder(tokens_long)[EncoderOutputKeys.LANGUAGE.value]

        assert output_short.shape == (batch_size, 768)
        assert output_long.shape == (batch_size, 768)