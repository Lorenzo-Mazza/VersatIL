import pytest
import torch
import numpy as np

from refactoring.models.encoding.encoders.multimodal import LanguageProprioTokenizerEncoder
from refactoring.models.encoding.encoders.constants import (
    LanguageEncoderType,
    EncoderOutputKeys,
)
from refactoring.data.constants import (
    LANGUAGE_KEY,
    PROPRIO_OBS_ROBOT_FRAME_KEY,
    PROPRIO_OBS_CAMERA_FRAME_KEY,
)
from refactoring.data.tokenize.tokenizer import Tokenizer
from refactoring.data.tokenize.binning_tokenizer import BinningTokenizer


LANGUAGE_MODELS = [
    (LanguageEncoderType.GEMMA_2B.value, 2048),
    (LanguageEncoderType.BERT_BASE.value, 768),
]


@pytest.fixture
def batch_size():
    """Standard batch size for testing."""
    return 2


@pytest.fixture
def proprio_dim():
    """Proprioceptive dimension (position + quaternion)."""
    return 7


@pytest.fixture
def num_bins():
    """Number of bins for binning tokenizer."""
    return 256


@pytest.fixture
def max_token_len():
    """Maximum token length."""
    return 128


@pytest.fixture
def proprio_robot_2d(batch_size, proprio_dim):
    """2D proprioceptive data in robot frame (B, D)."""
    np.random.seed(42)
    return torch.randn(batch_size, proprio_dim)


@pytest.fixture
def proprio_robot_3d(batch_size, proprio_dim):
    """3D proprioceptive data in robot frame (B, T, D)."""
    np.random.seed(42)
    return torch.randn(batch_size, 5, proprio_dim)


@pytest.fixture
def proprio_camera_2d(batch_size, proprio_dim):
    """2D proprioceptive data in camera frame (B, D)."""
    np.random.seed(43)
    return torch.randn(batch_size, proprio_dim)


@pytest.fixture
def proprio_camera_3d(batch_size, proprio_dim):
    """3D proprioceptive data in camera frame (B, T, D)."""
    np.random.seed(43)
    return torch.randn(batch_size, 5, proprio_dim)


@pytest.fixture
def text_inputs_batch(batch_size):
    """Sample text inputs for batch."""
    return [
        "pick up the red block",
        "move the robot arm to the left",
    ][:batch_size]


@pytest.fixture
def fitted_tokenizer_robot(proprio_dim, num_bins):
    """Fitted tokenizer for robot frame only."""
    tokenizer = Tokenizer(device=torch.device("cpu"))

    np.random.seed(42)
    normalized_data = np.random.randn(100, proprio_dim).astype(np.float32) * 0.5

    binning_tok_robot = BinningTokenizer(num_bins=num_bins, device=torch.device("cpu"))
    binning_tok_robot.fit(normalized_data)

    tokenizer.tokenizers[PROPRIO_OBS_ROBOT_FRAME_KEY] = binning_tok_robot

    return tokenizer


@pytest.fixture
def fitted_tokenizer_both(proprio_dim, num_bins):
    """Fitted tokenizer for both robot and camera frames."""
    tokenizer = Tokenizer(device=torch.device("cpu"))

    np.random.seed(42)
    normalized_data_robot = np.random.randn(100, proprio_dim).astype(np.float32) * 0.5
    normalized_data_camera = np.random.randn(100, proprio_dim).astype(np.float32) * 0.5

    binning_tok_robot = BinningTokenizer(num_bins=num_bins, device=torch.device("cpu"))
    binning_tok_robot.fit(normalized_data_robot)

    binning_tok_camera = BinningTokenizer(num_bins=num_bins, device=torch.device("cpu"))
    binning_tok_camera.fit(normalized_data_camera)

    tokenizer.tokenizers[PROPRIO_OBS_ROBOT_FRAME_KEY] = binning_tok_robot
    tokenizer.tokenizers[PROPRIO_OBS_CAMERA_FRAME_KEY] = binning_tok_camera

    return tokenizer


class TestLanguageProprioTokenizerInitialization:
    """Test LanguageProprioTokenizerEncoder initialization."""

    def test_initialization_robot_frame_only(self):
        """Test initialization with robot frame only."""
        encoder = LanguageProprioTokenizerEncoder(
            input_keys=[PROPRIO_OBS_ROBOT_FRAME_KEY],
            pretrained=True,
            frozen=True,
            language_model_name=LanguageEncoderType.BERT_BASE.value,
            max_token_len=128,
            device="cpu",
        )

        assert encoder.max_token_len == 128
        assert encoder.device_str == "cpu"
        assert encoder.lm_model_name == LanguageEncoderType.BERT_BASE.value
        assert encoder.binning_tokenizer_robot is None
        assert encoder.binning_tokenizer_camera is None

    def test_initialization_both_frames(self):
        """Test initialization with both robot and camera frames."""
        encoder = LanguageProprioTokenizerEncoder(
            input_keys=[PROPRIO_OBS_ROBOT_FRAME_KEY, PROPRIO_OBS_CAMERA_FRAME_KEY],
            pretrained=True,
            frozen=True,
            language_model_name=LanguageEncoderType.GEMMA_2B.value,
            max_token_len=128,
            device="cpu",
        )

        assert encoder.binning_tokenizer_robot is None
        assert encoder.binning_tokenizer_camera is None

    def test_initialization_with_language(self):
        """Test initialization with language key included."""
        encoder = LanguageProprioTokenizerEncoder(
            input_keys=[LANGUAGE_KEY, PROPRIO_OBS_ROBOT_FRAME_KEY],
            pretrained=True,
            frozen=True,
            language_model_name=LanguageEncoderType.BERT_BASE.value,
            max_token_len=128,
            device="cpu",
        )

        assert LANGUAGE_KEY in encoder.input_specification.keys

    def test_initialization_frozen(self):
        """Test initialization with frozen weights."""
        encoder = LanguageProprioTokenizerEncoder(
            input_keys=[PROPRIO_OBS_ROBOT_FRAME_KEY],
            pretrained=True,
            frozen=True,
            language_model_name=LanguageEncoderType.BERT_BASE.value,
            max_token_len=128,
            device="cpu",
        )

        for param in encoder.language_model.parameters():
            assert not param.requires_grad

    def test_initialization_custom_max_token_len(self):
        """Test initialization with custom max token length."""
        max_len = 256
        encoder = LanguageProprioTokenizerEncoder(
            input_keys=[PROPRIO_OBS_ROBOT_FRAME_KEY],
            pretrained=True,
            frozen=True,
            language_model_name=LanguageEncoderType.BERT_BASE.value,
            max_token_len=max_len,
            device="cpu",
        )

        assert encoder.max_token_len == max_len

    def test_get_output_specification(self):
        """Test get_output_specification returns correct structure."""
        encoder = LanguageProprioTokenizerEncoder(
            input_keys=[PROPRIO_OBS_ROBOT_FRAME_KEY],
            pretrained=True,
            frozen=True,
            language_model_name=LanguageEncoderType.BERT_BASE.value,
            max_token_len=128,
            device="cpu",
        )

        spec = encoder.get_output_specification()
        assert EncoderOutputKeys.LANGUAGE.value in spec.features
        assert isinstance(spec.features, list)
        assert EncoderOutputKeys.LANGUAGE.value in spec.dimensions

    def test_initialization_without_proprio_frames_raises_error(self):
        """Test initialization without any proprio frames raises validation error."""
        with pytest.raises(ValueError, match="At least one from"):
            encoder = LanguageProprioTokenizerEncoder(
                input_keys=[LANGUAGE_KEY],
                pretrained=True,
                frozen=True,
                language_model_name=LanguageEncoderType.BERT_BASE.value,
                max_token_len=128,
                device="cpu",
            )


class TestSetTokenizer:
    """Test set_tokenizer method."""

    def test_set_tokenizer_robot_only(self, fitted_tokenizer_robot):
        """Test setting tokenizer with robot frame only."""
        encoder = LanguageProprioTokenizerEncoder(
            input_keys=[PROPRIO_OBS_ROBOT_FRAME_KEY],
            pretrained=True,
            frozen=True,
            language_model_name=LanguageEncoderType.BERT_BASE.value,
            max_token_len=128,
            device="cpu",
        )

        encoder.set_tokenizer(fitted_tokenizer_robot)

        assert encoder.binning_tokenizer_robot is not None
        assert encoder.binning_tokenizer_robot._is_fitted is True
        assert encoder.binning_tokenizer_camera is None

    def test_set_tokenizer_both_frames(self, fitted_tokenizer_both):
        """Test setting tokenizer with both frames."""
        encoder = LanguageProprioTokenizerEncoder(
            input_keys=[PROPRIO_OBS_ROBOT_FRAME_KEY, PROPRIO_OBS_CAMERA_FRAME_KEY],
            pretrained=True,
            frozen=True,
            language_model_name=LanguageEncoderType.BERT_BASE.value,
            max_token_len=128,
            device="cpu",
        )

        encoder.set_tokenizer(fitted_tokenizer_both)

        assert encoder.binning_tokenizer_robot is not None
        assert encoder.binning_tokenizer_camera is not None
        assert encoder.binning_tokenizer_robot._is_fitted is True
        assert encoder.binning_tokenizer_camera._is_fitted is True

    def test_set_tokenizer_missing_raises_error(self):
        """Test setting tokenizer without required keys raises error."""
        encoder = LanguageProprioTokenizerEncoder(
            input_keys=[PROPRIO_OBS_ROBOT_FRAME_KEY],
            pretrained=True,
            frozen=True,
            language_model_name=LanguageEncoderType.BERT_BASE.value,
            max_token_len=128,
            device="cpu",
        )

        empty_tokenizer = Tokenizer(device=torch.device("cpu"))

        with pytest.raises(ValueError, match="Tokenizer must contain at least one"):
            encoder.set_tokenizer(empty_tokenizer)


class TestForwardPass:
    """Test forward pass with different scenarios."""

    def test_forward_robot_frame_with_language(
        self,
        fitted_tokenizer_robot,
        text_inputs_batch,
        proprio_robot_2d,
        batch_size,
        max_token_len,
    ):
        """Test forward with robot frame and language."""
        encoder = LanguageProprioTokenizerEncoder(
            input_keys=[LANGUAGE_KEY, PROPRIO_OBS_ROBOT_FRAME_KEY],
            pretrained=True,
            frozen=True,
            language_model_name=LanguageEncoderType.BERT_BASE.value,
            max_token_len=max_token_len,
            device="cpu",
        )

        encoder.set_tokenizer(fitted_tokenizer_robot)

        inputs = {
            LANGUAGE_KEY: text_inputs_batch,
            PROPRIO_OBS_ROBOT_FRAME_KEY: proprio_robot_2d,
        }

        output = encoder(inputs)

        assert EncoderOutputKeys.LANGUAGE.value in output
        assert EncoderOutputKeys.TOKEN_MASK.value in output

        embeddings = output[EncoderOutputKeys.LANGUAGE.value]
        assert embeddings.shape == (batch_size, max_token_len, encoder.embed_dim)
        assert embeddings.dtype == torch.float32
        assert not torch.isnan(embeddings).any()

        token_mask = output[EncoderOutputKeys.TOKEN_MASK.value]
        assert token_mask.shape == (batch_size, max_token_len)
        assert token_mask.dtype == torch.bool

    def test_forward_robot_frame_without_language(
        self,
        fitted_tokenizer_robot,
        proprio_robot_2d,
        batch_size,
        max_token_len,
    ):
        """Test forward with robot frame but no language (optional)."""
        encoder = LanguageProprioTokenizerEncoder(
            input_keys=[PROPRIO_OBS_ROBOT_FRAME_KEY],
            pretrained=True,
            frozen=True,
            language_model_name=LanguageEncoderType.BERT_BASE.value,
            max_token_len=max_token_len,
            device="cpu",
        )

        encoder.set_tokenizer(fitted_tokenizer_robot)

        inputs = {
            PROPRIO_OBS_ROBOT_FRAME_KEY: proprio_robot_2d,
        }

        output = encoder(inputs)

        assert EncoderOutputKeys.LANGUAGE.value in output
        embeddings = output[EncoderOutputKeys.LANGUAGE.value]
        assert embeddings.shape == (batch_size, max_token_len, encoder.embed_dim)
        assert not torch.isnan(embeddings).any()

    def test_forward_both_frames_with_language(
        self,
        fitted_tokenizer_both,
        text_inputs_batch,
        proprio_robot_2d,
        proprio_camera_2d,
        batch_size,
        max_token_len,
    ):
        """Test forward with both frames and language."""
        encoder = LanguageProprioTokenizerEncoder(
            input_keys=[LANGUAGE_KEY, PROPRIO_OBS_ROBOT_FRAME_KEY, PROPRIO_OBS_CAMERA_FRAME_KEY],
            pretrained=True,
            frozen=True,
            language_model_name=LanguageEncoderType.BERT_BASE.value,
            max_token_len=max_token_len,
            device="cpu",
        )

        encoder.set_tokenizer(fitted_tokenizer_both)

        inputs = {
            LANGUAGE_KEY: text_inputs_batch,
            PROPRIO_OBS_ROBOT_FRAME_KEY: proprio_robot_2d,
            PROPRIO_OBS_CAMERA_FRAME_KEY: proprio_camera_2d,
        }

        output = encoder(inputs)

        embeddings = output[EncoderOutputKeys.LANGUAGE.value]
        assert embeddings.shape == (batch_size, max_token_len, encoder.embed_dim)
        assert not torch.isnan(embeddings).any()

    def test_forward_3d_proprio(
        self,
        fitted_tokenizer_robot,
        text_inputs_batch,
        proprio_robot_3d,
        batch_size,
        max_token_len,
    ):
        """Test forward with 3D proprioceptive data (takes last timestep)."""
        encoder = LanguageProprioTokenizerEncoder(
            input_keys=[LANGUAGE_KEY, PROPRIO_OBS_ROBOT_FRAME_KEY],
            pretrained=True,
            frozen=True,
            language_model_name=LanguageEncoderType.BERT_BASE.value,
            max_token_len=max_token_len,
            device="cpu",
        )

        encoder.set_tokenizer(fitted_tokenizer_robot)

        inputs = {
            LANGUAGE_KEY: text_inputs_batch,
            PROPRIO_OBS_ROBOT_FRAME_KEY: proprio_robot_3d,
        }

        output = encoder(inputs)

        embeddings = output[EncoderOutputKeys.LANGUAGE.value]
        assert embeddings.shape == (batch_size, max_token_len, encoder.embed_dim)

    def test_forward_camera_frame_only(
        self,
        fitted_tokenizer_both,
        text_inputs_batch,
        proprio_camera_2d,
        batch_size,
        max_token_len,
    ):
        """Test forward with camera frame only."""
        encoder = LanguageProprioTokenizerEncoder(
            input_keys=[LANGUAGE_KEY, PROPRIO_OBS_CAMERA_FRAME_KEY],
            pretrained=True,
            frozen=True,
            language_model_name=LanguageEncoderType.BERT_BASE.value,
            max_token_len=max_token_len,
            device="cpu",
        )

        encoder.set_tokenizer(fitted_tokenizer_both)

        inputs = {
            LANGUAGE_KEY: text_inputs_batch,
            PROPRIO_OBS_CAMERA_FRAME_KEY: proprio_camera_2d,
        }

        output = encoder(inputs)

        embeddings = output[EncoderOutputKeys.LANGUAGE.value]
        assert embeddings.shape == (batch_size, max_token_len, encoder.embed_dim)


class TestParametrized:
    """Parametrized tests with different configurations."""

    @pytest.mark.parametrize("model_name,embed_dim", LANGUAGE_MODELS)
    def test_different_language_models(
        self,
        model_name,
        embed_dim,
        fitted_tokenizer_robot,
        text_inputs_batch,
        proprio_robot_2d,
        batch_size,
        max_token_len,
    ):
        """Test with different language models."""
        encoder = LanguageProprioTokenizerEncoder(
            input_keys=[LANGUAGE_KEY, PROPRIO_OBS_ROBOT_FRAME_KEY],
            pretrained=True,
            frozen=True,
            language_model_name=model_name,
            max_token_len=max_token_len,
            device="cpu",
        )

        encoder.set_tokenizer(fitted_tokenizer_robot)

        inputs = {
            LANGUAGE_KEY: text_inputs_batch,
            PROPRIO_OBS_ROBOT_FRAME_KEY: proprio_robot_2d,
        }

        output = encoder(inputs)

        embeddings = output[EncoderOutputKeys.LANGUAGE.value]
        assert embeddings.shape == (batch_size, max_token_len, embed_dim)

    @pytest.mark.parametrize("num_bins_param", [64, 128, 256])
    def test_different_num_bins(
        self,
        num_bins_param,
        text_inputs_batch,
        proprio_robot_2d,
        batch_size,
        max_token_len,
        proprio_dim,
    ):
        """Test with different number of bins."""
        tokenizer = Tokenizer(device=torch.device("cpu"))

        np.random.seed(42)
        normalized_data = np.random.randn(100, proprio_dim).astype(np.float32) * 0.5

        binning_tok = BinningTokenizer(num_bins=num_bins_param, device=torch.device("cpu"))
        binning_tok.fit(normalized_data)

        tokenizer.tokenizers[PROPRIO_OBS_ROBOT_FRAME_KEY] = binning_tok

        encoder = LanguageProprioTokenizerEncoder(
            input_keys=[LANGUAGE_KEY, PROPRIO_OBS_ROBOT_FRAME_KEY],
            pretrained=True,
            frozen=True,
            language_model_name=LanguageEncoderType.BERT_BASE.value,
            max_token_len=max_token_len,
            device="cpu",
        )

        encoder.set_tokenizer(tokenizer)

        inputs = {
            LANGUAGE_KEY: text_inputs_batch,
            PROPRIO_OBS_ROBOT_FRAME_KEY: proprio_robot_2d,
        }

        output = encoder(inputs)
        assert EncoderOutputKeys.LANGUAGE.value in output

    @pytest.mark.parametrize("max_len", [64, 128, 256])
    def test_different_max_token_len(
        self,
        max_len,
        fitted_tokenizer_robot,
        text_inputs_batch,
        proprio_robot_2d,
        batch_size,
    ):
        """Test with different max token lengths."""
        encoder = LanguageProprioTokenizerEncoder(
            input_keys=[LANGUAGE_KEY, PROPRIO_OBS_ROBOT_FRAME_KEY],
            pretrained=True,
            frozen=True,
            language_model_name=LanguageEncoderType.BERT_BASE.value,
            max_token_len=max_len,
            device="cpu",
        )

        encoder.set_tokenizer(fitted_tokenizer_robot)

        inputs = {
            LANGUAGE_KEY: text_inputs_batch,
            PROPRIO_OBS_ROBOT_FRAME_KEY: proprio_robot_2d,
        }

        output = encoder(inputs)

        embeddings = output[EncoderOutputKeys.LANGUAGE.value]
        assert embeddings.shape[1] == max_len

    @pytest.mark.parametrize("T", [1, 3, 5])
    def test_temporal_proprio(
        self,
        T,
        fitted_tokenizer_robot,
        batch_size,
        max_token_len,
        proprio_dim,
    ):
        """Test with temporal proprioceptive sequences."""
        encoder = LanguageProprioTokenizerEncoder(
            input_keys=[PROPRIO_OBS_ROBOT_FRAME_KEY],
            pretrained=True,
            frozen=True,
            language_model_name=LanguageEncoderType.BERT_BASE.value,
            max_token_len=max_token_len,
            device="cpu",
        )

        encoder.set_tokenizer(fitted_tokenizer_robot)

        proprio_temporal = torch.randn(batch_size, T, proprio_dim)

        inputs = {
            PROPRIO_OBS_ROBOT_FRAME_KEY: proprio_temporal,
        }

        output = encoder(inputs)

        embeddings = output[EncoderOutputKeys.LANGUAGE.value]
        assert embeddings.shape == (batch_size, max_token_len, encoder.embed_dim)

    @pytest.mark.parametrize("T", [1, 3, 5])
    def test_temporal_proprio_and_language(
        self,
        T,
        fitted_tokenizer_robot,
        batch_size,
        max_token_len,
        proprio_dim,
    ):
        """Test with temporal proprio and temporal language."""
        encoder = LanguageProprioTokenizerEncoder(
            input_keys=[LANGUAGE_KEY, PROPRIO_OBS_ROBOT_FRAME_KEY],
            pretrained=True,
            frozen=True,
            language_model_name=LanguageEncoderType.BERT_BASE.value,
            max_token_len=max_token_len,
            device="cpu",
        )

        encoder.set_tokenizer(fitted_tokenizer_robot)

        proprio_temporal = torch.randn(batch_size, T, proprio_dim)
        language_temporal = [["instruction " + str(t) for t in range(T)] for _ in range(batch_size)]

        inputs = {
            LANGUAGE_KEY: language_temporal,
            PROPRIO_OBS_ROBOT_FRAME_KEY: proprio_temporal,
        }

        output = encoder(inputs)

        embeddings = output[EncoderOutputKeys.LANGUAGE.value]
        assert embeddings.shape == (batch_size, max_token_len, encoder.embed_dim)

    @pytest.mark.parametrize("T", [3, 5])
    def test_temporal_both_frames(
        self,
        T,
        fitted_tokenizer_both,
        batch_size,
        max_token_len,
        proprio_dim,
    ):
        """Test with temporal robot and camera frames."""
        encoder = LanguageProprioTokenizerEncoder(
            input_keys=[PROPRIO_OBS_ROBOT_FRAME_KEY, PROPRIO_OBS_CAMERA_FRAME_KEY],
            pretrained=True,
            frozen=True,
            language_model_name=LanguageEncoderType.BERT_BASE.value,
            max_token_len=max_token_len,
            device="cpu",
        )

        encoder.set_tokenizer(fitted_tokenizer_both)

        proprio_robot_temporal = torch.randn(batch_size, T, proprio_dim)
        proprio_camera_temporal = torch.randn(batch_size, T, proprio_dim)

        inputs = {
            PROPRIO_OBS_ROBOT_FRAME_KEY: proprio_robot_temporal,
            PROPRIO_OBS_CAMERA_FRAME_KEY: proprio_camera_temporal,
        }

        output = encoder(inputs)

        embeddings = output[EncoderOutputKeys.LANGUAGE.value]
        assert embeddings.shape == (batch_size, max_token_len, encoder.embed_dim)


class TestTemporalMismatch:
    """Test temporal dimension mismatch errors."""

    def test_mismatched_robot_camera_T_raises_error(
        self,
        fitted_tokenizer_both,
        batch_size,
        proprio_dim,
    ):
        """Test robot and camera frames with different T raises error."""
        encoder = LanguageProprioTokenizerEncoder(
            input_keys=[PROPRIO_OBS_ROBOT_FRAME_KEY, PROPRIO_OBS_CAMERA_FRAME_KEY],
            pretrained=True,
            frozen=True,
            language_model_name=LanguageEncoderType.BERT_BASE.value,
            max_token_len=128,
            device="cpu",
        )

        encoder.set_tokenizer(fitted_tokenizer_both)

        proprio_robot_temporal = torch.randn(batch_size, 3, proprio_dim)
        proprio_camera_temporal = torch.randn(batch_size, 5, proprio_dim)

        inputs = {
            PROPRIO_OBS_ROBOT_FRAME_KEY: proprio_robot_temporal,
            PROPRIO_OBS_CAMERA_FRAME_KEY: proprio_camera_temporal,
        }

        with pytest.raises(ValueError, match="Robot and camera proprio must have same T"):
            encoder(inputs)

    def test_mismatched_proprio_language_T_raises_error(
        self,
        fitted_tokenizer_robot,
        batch_size,
        proprio_dim,
    ):
        """Test proprio and language with different T raises error."""
        encoder = LanguageProprioTokenizerEncoder(
            input_keys=[LANGUAGE_KEY, PROPRIO_OBS_ROBOT_FRAME_KEY],
            pretrained=True,
            frozen=True,
            language_model_name=LanguageEncoderType.BERT_BASE.value,
            max_token_len=128,
            device="cpu",
        )

        encoder.set_tokenizer(fitted_tokenizer_robot)

        proprio_temporal = torch.randn(batch_size, 3, proprio_dim)
        language_temporal = [["instruction " + str(t) for t in range(5)] for _ in range(batch_size)]

        inputs = {
            LANGUAGE_KEY: language_temporal,
            PROPRIO_OBS_ROBOT_FRAME_KEY: proprio_temporal,
        }

        with pytest.raises(ValueError, match="Language and proprio must have same T"):
            encoder(inputs)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_forward_before_set_tokenizer_raises_error(
        self,
        text_inputs_batch,
        proprio_robot_2d,
    ):
        """Test forward without setting tokenizer raises error."""
        encoder = LanguageProprioTokenizerEncoder(
            input_keys=[LANGUAGE_KEY, PROPRIO_OBS_ROBOT_FRAME_KEY],
            pretrained=True,
            frozen=True,
            language_model_name=LanguageEncoderType.BERT_BASE.value,
            max_token_len=128,
            device="cpu",
        )

        inputs = {
            LANGUAGE_KEY: text_inputs_batch,
            PROPRIO_OBS_ROBOT_FRAME_KEY: proprio_robot_2d,
        }

        with pytest.raises(RuntimeError, match="No binning tokenizer set"):
            encoder(inputs)

    def test_forward_missing_proprio_frame_raises_error(
        self,
        fitted_tokenizer_robot,
        text_inputs_batch,
    ):
        """Test forward without required proprio frame raises error."""
        encoder = LanguageProprioTokenizerEncoder(
            input_keys=[LANGUAGE_KEY, PROPRIO_OBS_ROBOT_FRAME_KEY],
            pretrained=True,
            frozen=True,
            language_model_name=LanguageEncoderType.BERT_BASE.value,
            max_token_len=128,
            device="cpu",
        )

        encoder.set_tokenizer(fitted_tokenizer_robot)

        inputs = {
            LANGUAGE_KEY: text_inputs_batch,
        }

        with pytest.raises(ValueError, match="At least one proprio frame must be present"):
            encoder(inputs)

    def test_empty_language_uses_default_task(
        self,
        fitted_tokenizer_robot,
        proprio_robot_2d,
        batch_size,
        max_token_len,
    ):
        """Test with empty language string uses 'Task: ' prefix."""
        encoder = LanguageProprioTokenizerEncoder(
            input_keys=[LANGUAGE_KEY, PROPRIO_OBS_ROBOT_FRAME_KEY],
            pretrained=True,
            frozen=True,
            language_model_name=LanguageEncoderType.BERT_BASE.value,
            max_token_len=max_token_len,
            device="cpu",
        )

        encoder.set_tokenizer(fitted_tokenizer_robot)

        inputs = {
            LANGUAGE_KEY: ["", ""],
            PROPRIO_OBS_ROBOT_FRAME_KEY: proprio_robot_2d,
        }

        output = encoder(inputs)

        embeddings = output[EncoderOutputKeys.LANGUAGE.value]
        assert embeddings.shape == (batch_size, max_token_len, encoder.embed_dim)
        assert not torch.isnan(embeddings).any()

    def test_batch_size_one(
        self,
        fitted_tokenizer_robot,
        max_token_len,
        proprio_dim,
    ):
        """Test with batch size of 1."""
        encoder = LanguageProprioTokenizerEncoder(
            input_keys=[LANGUAGE_KEY, PROPRIO_OBS_ROBOT_FRAME_KEY],
            pretrained=True,
            frozen=True,
            language_model_name=LanguageEncoderType.BERT_BASE.value,
            max_token_len=max_token_len,
            device="cpu",
        )

        encoder.set_tokenizer(fitted_tokenizer_robot)

        inputs = {
            LANGUAGE_KEY: ["pick up the block"],
            PROPRIO_OBS_ROBOT_FRAME_KEY: torch.randn(1, proprio_dim),
        }

        output = encoder(inputs)

        embeddings = output[EncoderOutputKeys.LANGUAGE.value]
        assert embeddings.shape == (1, max_token_len, encoder.embed_dim)


@pytest.mark.integration
class TestIntegration:
    """Integration tests for complete workflows."""

    def test_different_texts_produce_different_features(
        self,
        fitted_tokenizer_robot,
        proprio_robot_2d,
        batch_size,
    ):
        """Test different text inputs produce different features."""
        encoder = LanguageProprioTokenizerEncoder(
            input_keys=[LANGUAGE_KEY, PROPRIO_OBS_ROBOT_FRAME_KEY],
            pretrained=True,
            frozen=True,
            language_model_name=LanguageEncoderType.BERT_BASE.value,
            max_token_len=128,
            device="cpu",
        )

        encoder.set_tokenizer(fitted_tokenizer_robot)
        encoder.eval()

        text1 = ["pick up the red block"] * batch_size
        text2 = ["move the robot arm to the left"] * batch_size

        with torch.no_grad():
            output1 = encoder({
                LANGUAGE_KEY: text1,
                PROPRIO_OBS_ROBOT_FRAME_KEY: proprio_robot_2d,
            })[EncoderOutputKeys.LANGUAGE.value]

            output2 = encoder({
                LANGUAGE_KEY: text2,
                PROPRIO_OBS_ROBOT_FRAME_KEY: proprio_robot_2d,
            })[EncoderOutputKeys.LANGUAGE.value]

        assert output1.shape == output2.shape
        assert not torch.allclose(output1, output2, atol=1e-5)

    def test_same_inputs_produce_same_features(
        self,
        fitted_tokenizer_robot,
        text_inputs_batch,
        proprio_robot_2d,
    ):
        """Test same inputs produce consistent features in eval mode."""
        encoder = LanguageProprioTokenizerEncoder(
            input_keys=[LANGUAGE_KEY, PROPRIO_OBS_ROBOT_FRAME_KEY],
            pretrained=True,
            frozen=True,
            language_model_name=LanguageEncoderType.BERT_BASE.value,
            max_token_len=128,
            device="cpu",
        )

        encoder.set_tokenizer(fitted_tokenizer_robot)
        encoder.eval()

        inputs = {
            LANGUAGE_KEY: text_inputs_batch,
            PROPRIO_OBS_ROBOT_FRAME_KEY: proprio_robot_2d,
        }

        with torch.no_grad():
            output1 = encoder(inputs)[EncoderOutputKeys.LANGUAGE.value]
            output2 = encoder(inputs)[EncoderOutputKeys.LANGUAGE.value]

        assert torch.allclose(output1, output2, atol=1e-6)

    def test_no_gradient_flow_frozen(
        self,
        fitted_tokenizer_robot,
        text_inputs_batch,
        proprio_robot_2d,
    ):
        """Test gradients don't flow when frozen."""
        encoder = LanguageProprioTokenizerEncoder(
            input_keys=[LANGUAGE_KEY, PROPRIO_OBS_ROBOT_FRAME_KEY],
            pretrained=True,
            frozen=True,
            language_model_name=LanguageEncoderType.BERT_BASE.value,
            max_token_len=128,
            device="cpu",
        )

        encoder.set_tokenizer(fitted_tokenizer_robot)

        inputs = {
            LANGUAGE_KEY: text_inputs_batch,
            PROPRIO_OBS_ROBOT_FRAME_KEY: proprio_robot_2d,
        }

        output = encoder(inputs)[EncoderOutputKeys.LANGUAGE.value]

        assert not output.requires_grad

        for param in encoder.language_model.parameters():
            assert not param.requires_grad