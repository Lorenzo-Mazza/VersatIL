import pytest
import torch

from refactoring.models.encoding.encoders.language import LanguageEncoder
from refactoring.models.encoding.encoders.constants import (
    LanguageEncoderType,
    PoolingMethod,
    EncoderOutputKeys,
)
from refactoring.data.constants import LANGUAGE_KEY


LANGUAGE_MODELS = [
    (LanguageEncoderType.BERT_BASE.value, 768),
    (LanguageEncoderType.DISTILBERT_BASE.value, 768),
    (LanguageEncoderType.MINI_LM_L6.value, 384),
]

FEATURE_EXTRACTION_METHODS = [
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
def text_inputs_batch(batch_size):
    """Sample text inputs for batch (1D list)."""
    return [
        "pick up the red block",
        "move the robot arm to the left",
    ][:batch_size]


@pytest.fixture
def text_inputs_temporal(batch_size, temporal_length):
    """Sample text inputs for temporal sequences (2D list: batch × time)."""
    texts = [
        "pick up the red block",
        "move the robot arm to the left",
    ]
    return [[texts[i % len(texts)]] * temporal_length for i in range(batch_size)]


@pytest.mark.integration
class TestLanguageEncoderInitialization:
    """Test LanguageEncoder initialization."""

    @pytest.mark.parametrize("model_name,expected_dim", LANGUAGE_MODELS)
    def test_initialization(self, model_name, expected_dim):
        """Test LanguageEncoder initialization with different models."""
        encoder = LanguageEncoder(
            input_keys=LANGUAGE_KEY,
            model_name=model_name,
            pretrained=True,
            frozen=True,
            feature_extraction_method=PoolingMethod.AVERAGE.value,
        )

        assert encoder.feature_dim == expected_dim
        spec = encoder.get_output_specification()
        assert spec.dimensions[EncoderOutputKeys.LANGUAGE.value] == expected_dim
        assert encoder.feature_extraction_method == PoolingMethod.AVERAGE.value

    def test_get_output_specification(self):
        """Test get_output_specification returns correct structure."""
        encoder = LanguageEncoder(
            input_keys=LANGUAGE_KEY,
            model_name=LanguageEncoderType.BERT_BASE.value,
            pretrained=True,
            frozen=True,
            feature_extraction_method=PoolingMethod.AVERAGE.value,
        )

        spec = encoder.get_output_specification()
        assert EncoderOutputKeys.LANGUAGE.value in spec.features
        assert isinstance(spec.features, list)
        assert EncoderOutputKeys.LANGUAGE.value in spec.dimensions

    def test_init_missing_language_key(self):
        """Test initialization without language key raises error."""
        with pytest.raises(ValueError, match="Missing required inputs"):
            LanguageEncoder(
                input_keys="invalid_key",
                model_name=LanguageEncoderType.BERT_BASE.value,
                pretrained=True,
                frozen=True,
                feature_extraction_method=PoolingMethod.AVERAGE.value,
            )

    def test_init_frozen(self):
        """Test initialization with frozen weights."""
        encoder = LanguageEncoder(
            input_keys=LANGUAGE_KEY,
            model_name=LanguageEncoderType.BERT_BASE.value,
            pretrained=True,
            frozen=True,
            feature_extraction_method=PoolingMethod.AVERAGE.value,
        )

        for param in encoder.encoder.parameters():
            assert not param.requires_grad

    def test_init_custom_max_length(self):
        """Test initialization with custom max length."""
        max_len = 128
        encoder = LanguageEncoder(
            input_keys=LANGUAGE_KEY,
            model_name=LanguageEncoderType.BERT_BASE.value,
            pretrained=True,
            frozen=True,
            feature_extraction_method=PoolingMethod.AVERAGE.value,
            max_length=max_len,
        )

        assert encoder.max_length == max_len


@pytest.mark.integration
class TestLanguageEncoderForward:
    """Test LanguageEncoder forward pass."""

    @pytest.mark.parametrize("model_name,expected_dim", LANGUAGE_MODELS)
    def test_forward_batch_input(self, model_name, expected_dim, text_inputs_batch, batch_size):
        """Test forward pass with batch input (1D list of strings)."""
        encoder = LanguageEncoder(
            input_keys=LANGUAGE_KEY,
            model_name=model_name,
            pretrained=True,
            frozen=True,
            feature_extraction_method=PoolingMethod.AVERAGE.value,
        )

        input_dict = {LANGUAGE_KEY: text_inputs_batch}
        output_dict = encoder(input_dict)

        assert isinstance(output_dict, dict)
        assert EncoderOutputKeys.LANGUAGE.value in output_dict

        output = output_dict[EncoderOutputKeys.LANGUAGE.value]
        assert output.shape == (batch_size, expected_dim)
        assert output.dtype == torch.float32
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    @pytest.mark.parametrize("model_name,expected_dim", LANGUAGE_MODELS)
    def test_forward_temporal_input(self, model_name, expected_dim, text_inputs_temporal, batch_size, temporal_length):
        """Test forward pass with temporal input (2D list of strings)."""
        encoder = LanguageEncoder(
            input_keys=LANGUAGE_KEY,
            model_name=model_name,
            pretrained=True,
            frozen=True,
            feature_extraction_method=PoolingMethod.AVERAGE.value,
        )

        input_dict = {LANGUAGE_KEY: text_inputs_temporal}
        output_dict = encoder(input_dict)

        assert isinstance(output_dict, dict)
        assert EncoderOutputKeys.LANGUAGE.value in output_dict

        output = output_dict[EncoderOutputKeys.LANGUAGE.value]
        assert output.shape == (batch_size, temporal_length, expected_dim)
        assert output.dtype == torch.float32
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    @pytest.mark.parametrize("feature_method", FEATURE_EXTRACTION_METHODS)
    def test_feature_extraction_methods(self, feature_method, text_inputs_batch, batch_size):
        """Test different feature extraction methods."""
        encoder = LanguageEncoder(
            input_keys=LANGUAGE_KEY,
            model_name=LanguageEncoderType.BERT_BASE.value,
            pretrained=True,
            frozen=True,
            feature_extraction_method=feature_method,
        )

        input_dict = {LANGUAGE_KEY: text_inputs_batch}
        output = encoder(input_dict)[EncoderOutputKeys.LANGUAGE.value]

        assert output.shape == (batch_size, 768)
        assert output.dtype == torch.float32
        assert not torch.isnan(output).any()

        if feature_method == PoolingMethod.LEARNED_AGGREGATION.value:
            assert encoder.pooling_head is not None
        else:
            assert encoder.pooling_head is None

    def test_different_texts_produce_different_features(self, batch_size):
        """Test different text inputs produce different features."""
        encoder = LanguageEncoder(
            input_keys=LANGUAGE_KEY,
            model_name=LanguageEncoderType.BERT_BASE.value,
            pretrained=True,
            frozen=True,
            feature_extraction_method=PoolingMethod.AVERAGE.value,
        )

        encoder.eval()

        text1 = ["pick up the red block"] * batch_size
        text2 = ["move the robot arm to the left"] * batch_size

        with torch.no_grad():
            output1 = encoder({LANGUAGE_KEY: text1})[EncoderOutputKeys.LANGUAGE.value]
            output2 = encoder({LANGUAGE_KEY: text2})[EncoderOutputKeys.LANGUAGE.value]

        assert output1.shape == output2.shape
        assert not torch.allclose(output1, output2, atol=1e-5)

    def test_same_text_produces_same_features(self, batch_size):
        """Test same text produces consistent features in eval mode."""
        encoder = LanguageEncoder(
            input_keys=LANGUAGE_KEY,
            model_name=LanguageEncoderType.BERT_BASE.value,
            pretrained=True,
            frozen=True,
            feature_extraction_method=PoolingMethod.AVERAGE.value,
        )

        encoder.eval()
        text = ["pick up the red block"] * batch_size

        with torch.no_grad():
            output1 = encoder({LANGUAGE_KEY: text})[EncoderOutputKeys.LANGUAGE.value]
            output2 = encoder({LANGUAGE_KEY: text})[EncoderOutputKeys.LANGUAGE.value]

        assert torch.allclose(output1, output2, atol=1e-6)


@pytest.mark.integration
class TestLanguageEncoderEdgeCases:
    """Test edge cases."""

    def test_single_word_input(self):
        """Test encoder handles single word inputs."""
        encoder = LanguageEncoder(
            input_keys=LANGUAGE_KEY,
            model_name=LanguageEncoderType.BERT_BASE.value,
            pretrained=True,
            frozen=True,
            feature_extraction_method=PoolingMethod.AVERAGE.value,
        )

        single_words = ["pick", "move"]
        input_dict = {LANGUAGE_KEY: single_words}
        output = encoder(input_dict)[EncoderOutputKeys.LANGUAGE.value]

        assert output.shape == (2, 768)
        assert not torch.isnan(output).any()

    def test_long_text_truncation(self):
        """Test encoder properly truncates long text."""
        max_len = 20
        encoder = LanguageEncoder(
            input_keys=LANGUAGE_KEY,
            model_name=LanguageEncoderType.BERT_BASE.value,
            pretrained=True,
            frozen=True,
            feature_extraction_method=PoolingMethod.AVERAGE.value,
            max_length=max_len,
        )

        long_text = [" ".join(["word"] * 100)]
        input_dict = {LANGUAGE_KEY: long_text}
        output = encoder(input_dict)[EncoderOutputKeys.LANGUAGE.value]

        assert output.shape == (1, 768)
        assert not torch.isnan(output).any()

    def test_empty_string(self):
        """Test encoder handles empty string."""
        encoder = LanguageEncoder(
            input_keys=LANGUAGE_KEY,
            model_name=LanguageEncoderType.BERT_BASE.value,
            pretrained=True,
            frozen=True,
            feature_extraction_method=PoolingMethod.AVERAGE.value,
        )

        empty_texts = ["", "move the arm"]
        input_dict = {LANGUAGE_KEY: empty_texts}
        output = encoder(input_dict)[EncoderOutputKeys.LANGUAGE.value]

        assert output.shape == (2, 768)
        assert not torch.isnan(output).any()

    def test_batch_size_one(self):
        """Test encoder works with batch size 1."""
        encoder = LanguageEncoder(
            input_keys=LANGUAGE_KEY,
            model_name=LanguageEncoderType.BERT_BASE.value,
            pretrained=True,
            frozen=True,
            feature_extraction_method=PoolingMethod.AVERAGE.value,
        )

        input_dict = {LANGUAGE_KEY: ["pick up the block"]}
        output = encoder(input_dict)[EncoderOutputKeys.LANGUAGE.value]

        assert output.shape == (1, 768)

    def test_variable_length_temporal(self, batch_size, temporal_length):
        """Test encoder handles temporal sequences with different texts."""
        encoder = LanguageEncoder(
            input_keys=LANGUAGE_KEY,
            model_name=LanguageEncoderType.BERT_BASE.value,
            pretrained=True,
            frozen=True,
            feature_extraction_method=PoolingMethod.AVERAGE.value,
        )

        texts = [
            [f"instruction at time {t}" for t in range(temporal_length)]
            for _ in range(batch_size)
        ]
        input_dict = {LANGUAGE_KEY: texts}
        output = encoder(input_dict)[EncoderOutputKeys.LANGUAGE.value]

        assert output.shape == (batch_size, temporal_length, 768)


@pytest.mark.integration
class TestLanguageEncoderOutputDims:
    """Test output dimension methods."""

    @pytest.mark.parametrize("model_name,expected_dim", LANGUAGE_MODELS)
    def test_output_specification_structure(self, model_name, expected_dim):
        """Test get_output_specification returns proper structure."""
        encoder = LanguageEncoder(
            input_keys=LANGUAGE_KEY,
            model_name=model_name,
            pretrained=True,
            frozen=True,
            feature_extraction_method=PoolingMethod.AVERAGE.value,
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
        """Test gradients flow when not frozen."""
        encoder = LanguageEncoder(
            input_keys=LANGUAGE_KEY,
            model_name=model_name,
            pretrained=True,
            frozen=False,
            feature_extraction_method=PoolingMethod.AVERAGE.value,
        )

        input_dict = {LANGUAGE_KEY: text_inputs_batch}
        output = encoder(input_dict)[EncoderOutputKeys.LANGUAGE.value]
        loss = output.sum()
        loss.backward()

        encoder_has_grads = any(
            param.grad is not None and param.grad.abs().sum() > 0
            for param in encoder.encoder.parameters() if param.requires_grad
        )
        assert encoder_has_grads

    @pytest.mark.parametrize("model_name", [m[0] for m in LANGUAGE_MODELS])
    def test_gradients_disabled_frozen(self, model_name, text_inputs_batch):
        """Test gradients don't flow when frozen."""
        encoder = LanguageEncoder(
            input_keys=LANGUAGE_KEY,
            model_name=model_name,
            pretrained=True,
            frozen=True,
            feature_extraction_method=PoolingMethod.AVERAGE.value,
        )

        input_dict = {LANGUAGE_KEY: text_inputs_batch}
        output = encoder(input_dict)[EncoderOutputKeys.LANGUAGE.value]

        assert not output.requires_grad

        for param in encoder.encoder.parameters():
            assert not param.requires_grad


@pytest.mark.integration
class TestLanguageEncoderIntegration:
    """Integration tests for complete workflows."""

    def test_complete_forward_backward_pass(self, text_inputs_batch):
        """Test complete forward and backward pass."""
        encoder = LanguageEncoder(
            input_keys=LANGUAGE_KEY,
            model_name=LanguageEncoderType.BERT_BASE.value,
            pretrained=True,
            frozen=False,
            feature_extraction_method=PoolingMethod.AVERAGE.value,
        )

        encoder.train()
        input_dict = {LANGUAGE_KEY: text_inputs_batch}
        output = encoder(input_dict)[EncoderOutputKeys.LANGUAGE.value]
        loss = output.mean()
        loss.backward()

        assert output.requires_grad
        assert loss.requires_grad

    def test_eval_mode_no_gradients(self, text_inputs_batch):
        """Test eval mode doesn't compute gradients."""
        encoder = LanguageEncoder(
            input_keys=LANGUAGE_KEY,
            model_name=LanguageEncoderType.BERT_BASE.value,
            pretrained=True,
            frozen=True,
            feature_extraction_method=PoolingMethod.AVERAGE.value,
        )

        encoder.eval()
        input_dict = {LANGUAGE_KEY: text_inputs_batch}

        with torch.no_grad():
            output = encoder(input_dict)[EncoderOutputKeys.LANGUAGE.value]

        assert not output.requires_grad

    def test_consistent_output_shapes(self, batch_size):
        """Test output shapes are consistent across multiple forward passes."""
        encoder = LanguageEncoder(
            input_keys=LANGUAGE_KEY,
            model_name=LanguageEncoderType.BERT_BASE.value,
            pretrained=True,
            frozen=True,
            feature_extraction_method=PoolingMethod.AVERAGE.value,
        )

        texts1 = ["pick up the block"] * batch_size
        texts2 = ["move the arm"] * batch_size

        input1 = {LANGUAGE_KEY: texts1}
        input2 = {LANGUAGE_KEY: texts2}

        output1 = encoder(input1)[EncoderOutputKeys.LANGUAGE.value]
        output2 = encoder(input2)[EncoderOutputKeys.LANGUAGE.value]

        assert output1.shape == output2.shape

    def test_different_feature_methods_produce_different_outputs(self, text_inputs_batch):
        """Test different feature extraction methods produce different features."""
        encoder_cls = LanguageEncoder(
            input_keys=LANGUAGE_KEY,
            model_name=LanguageEncoderType.BERT_BASE.value,
            pretrained=True,
            frozen=True,
            feature_extraction_method=PoolingMethod.DEFAULT.value,
        )

        encoder_gap = LanguageEncoder(
            input_keys=LANGUAGE_KEY,
            model_name=LanguageEncoderType.BERT_BASE.value,
            pretrained=True,
            frozen=True,
            feature_extraction_method=PoolingMethod.AVERAGE.value,
        )

        encoder_cls.eval()
        encoder_gap.eval()

        input_dict = {LANGUAGE_KEY: text_inputs_batch}

        with torch.no_grad():
            output_cls = encoder_cls(input_dict)[EncoderOutputKeys.LANGUAGE.value]
            output_gap = encoder_gap(input_dict)[EncoderOutputKeys.LANGUAGE.value]

        assert output_cls.shape == output_gap.shape
        assert not torch.allclose(output_cls, output_gap, atol=1e-5)

    def test_special_characters_handling(self):
        """Test encoder handles special characters."""
        encoder = LanguageEncoder(
            input_keys=LANGUAGE_KEY,
            model_name=LanguageEncoderType.BERT_BASE.value,
            pretrained=True,
            frozen=True,
            feature_extraction_method=PoolingMethod.AVERAGE.value,
        )

        special_texts = [
            "Pick up the 5kg object!",
            "Move @20cm/s to position (10, -5)",
            "Grasp object #3 & place it",
        ]
        input_dict = {LANGUAGE_KEY: special_texts}
        output = encoder(input_dict)[EncoderOutputKeys.LANGUAGE.value]

        assert output.shape == (3, 768)
        assert not torch.isnan(output).any()

    def test_unicode_text_handling(self):
        """Test encoder handles unicode text."""
        encoder = LanguageEncoder(
            input_keys=LANGUAGE_KEY,
            model_name=LanguageEncoderType.BERT_BASE.value,
            pretrained=True,
            frozen=True,
            feature_extraction_method=PoolingMethod.AVERAGE.value,
        )

        unicode_texts = [
            "Pick up the 红色 block",
            "Déplacer le bras robotique",
        ]
        input_dict = {LANGUAGE_KEY: unicode_texts}
        output = encoder(input_dict)[EncoderOutputKeys.LANGUAGE.value]

        assert output.shape == (2, 768)
        assert not torch.isnan(output).any()