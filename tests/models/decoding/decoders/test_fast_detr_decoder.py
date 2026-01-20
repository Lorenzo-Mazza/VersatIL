"""Tests for FAST Decoder for tokenized action prediction."""
import numpy as np
import pytest
import torch

from versatil.data.task import ActionSpace, ObservationSpace
from versatil.data.constants import (
    ACTION_KEY,
    GRIPPER_ACTION_KEY,
    IS_PAD_ACTION_KEY,
    ORIENTATION_ACTION_KEY,
    POSITION_ACTION_KEY,
    Cameras,
    GripperType,
    OrientationRepresentation,
)
from versatil.data.tokenization.tokenizer import Tokenizer
from versatil.models.decoding.constants import PREDICTED_ACTION_TOKENS_KEY
from versatil.models.decoding.decoders.factory.fast_detr_decoder import FASTDETRDecoder


@pytest.fixture
def device():
    """Get available device."""
    return "cpu"


@pytest.fixture
def batch_size():
    """Default batch size."""
    return 2


@pytest.fixture
def observation_horizon():
    """Default observation horizon."""
    return 1


@pytest.fixture
def prediction_horizon():
    """Default prediction horizon."""
    return 10


@pytest.fixture
def embedding_dimension():
    """Default embedding dimension."""
    return 32


@pytest.fixture
def use_pretrained_weights():
    """Whether to use pretrained FAST weights.

    False: Faster tests, vocab_size=1024 (custom fitted)
    True: Slower tests, vocab_size=2048 (pretrained)
    """
    return False


@pytest.fixture
def vocab_size(use_pretrained_weights):
    """Vocabulary size depends on whether pretrained weights are used.

    Pretrained FAST: vocab_size = 2048
    Custom fitted: vocab_size = 1024
    """
    return 2048 if use_pretrained_weights else 1024


@pytest.fixture
def action_space():
    """Create default action space configuration."""
    return ActionSpace(
        has_position=True,
        position_dim=3,
        has_orientation=True,
        orientation_dim=4,
        orientation_repr=OrientationRepresentation.QUATERNION.value,
        has_gripper=True,
        gripper_type=GripperType.BINARY.value,
        gripper_dim=1,
        predict_in_camera_frame=False,
        deltas_as_actions=False,
    )


@pytest.fixture
def observation_space():
    """Create default observation space configuration."""
    return ObservationSpace(
        use_proprioceptive_data=False,
        use_proprio_base_frame=False,
        use_proprio_camera_frame=False,
        use_gripper_state=False,
        gripper_type=GripperType.BINARY.value,
        camera_keys=[Cameras.LEFT.value],
        use_language=False,
    )


@pytest.fixture
def spatial_features(batch_size, device):
    """Single spatial feature."""
    return {
        "rgb_left_features": torch.randn(batch_size, 128, 4, 4, device=device)
    }


@pytest.fixture
def actions_dict(batch_size, prediction_horizon, action_space, device):
    """Create ground-truth actions dictionary."""
    actions = {}

    if action_space.has_position:
        actions[POSITION_ACTION_KEY] = torch.randn(
            batch_size, prediction_horizon, action_space.position_dim, device=device
        )

    if action_space.has_orientation:
        actions[ORIENTATION_ACTION_KEY] = torch.randn(
            batch_size, prediction_horizon, action_space.orientation_dim, device=device
        )

    if action_space.has_gripper:
        actions[GRIPPER_ACTION_KEY] = torch.randint(
            0, 2, (batch_size, prediction_horizon, action_space.gripper_dim), device=device
        ).float()

    actions[IS_PAD_ACTION_KEY] = torch.zeros(
        batch_size, prediction_horizon, 1, dtype=torch.bool, device=device
    )

    return actions


@pytest.fixture
def tokenizer(action_space, prediction_horizon, device, use_pretrained_weights):
    """Create FAST tokenizer for testing."""
    tokenizer_obj = Tokenizer(device=torch.device(device))
    total_action_dim = action_space.get_total_action_dim()
    dummy_action_chunks = np.random.randn(100, prediction_horizon, total_action_dim) * 2 - 1
    tokenizer_obj.fit_action_tokenizer(
        action_chunks=dummy_action_chunks,
        use_pretrained_weights=use_pretrained_weights,
    )
    sample_actions = torch.randn(1, prediction_horizon, total_action_dim, device=device)
    tokenizer_obj.tokenize({ACTION_KEY: sample_actions})
    return tokenizer_obj


@pytest.mark.unit
class TestFASTDETRDecoderInitialization:
    """Test FASTDETRDecoder initialization."""

    def test_init_basic(
        self,
        action_space,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        embedding_dimension,
        vocab_size,
    ):
        """Test basic initialization."""
        decoder = FASTDETRDecoder(
            input_keys=["rgb_features"],
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            vocab_size=vocab_size,
            embedding_dimension=embedding_dimension,
        )

        assert decoder.embedding_dimension == embedding_dimension
        assert decoder.prediction_horizon == prediction_horizon
        assert decoder.observation_horizon == observation_horizon
        assert decoder.vocab_size == vocab_size
        # FAST doesn't use BOS token
        assert decoder.eos_token_id == 1
        assert decoder.pad_token_id == 0

    @pytest.mark.parametrize("embedding_dimension,vocab_size", [
        (32, 128),
        (64, 256),
    ])
    def test_init_custom_params(
        self,
        action_space,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        embedding_dimension,
        vocab_size,
    ):
        """Test initialization with custom parameters."""
        decoder = FASTDETRDecoder(
            input_keys=["rgb_features"],
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            vocab_size=vocab_size,
            embedding_dimension=embedding_dimension,
            number_of_heads=2,
            feedforward_dimension=128,
            number_of_encoder_layers=2,
            number_of_decoder_layers=2,
        )

        assert decoder.embedding_dimension == embedding_dimension
        assert decoder.vocab_size == vocab_size

    def test_special_tokens_configurable(
        self,
        action_space,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        vocab_size,
        embedding_dimension,
    ):
        """Test that special tokens are configurable."""
        custom_eos = 11
        custom_pad = 2

        decoder = FASTDETRDecoder(
            input_keys=["rgb_features"],
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            vocab_size=vocab_size,
            embedding_dimension=embedding_dimension,
            eos_token_id=custom_eos,
            pad_token_id=custom_pad,
            number_of_heads=2,
            feedforward_dimension=128,
            number_of_encoder_layers=2,
            number_of_decoder_layers=2,

        )

        assert decoder.eos_token_id == custom_eos
        assert decoder.pad_token_id == custom_pad


@pytest.mark.unit
class TestFASTDETRDecoderTokenizer:
    """Test FASTDETRDecoder tokenizer handling."""

    def test_set_tokenizer_success(
        self,
        action_space,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        vocab_size,
        embedding_dimension,
        tokenizer,
    ):
        """Test setting valid tokenizer."""
        decoder = FASTDETRDecoder(
            input_keys=["rgb_features"],
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            vocab_size=vocab_size,
            embedding_dimension=embedding_dimension,
            number_of_heads=2,
            feedforward_dimension=128,
            number_of_encoder_layers=2,
            number_of_decoder_layers=2,

        )

        decoder.set_tokenizer(tokenizer)
        assert decoder.tokenizer == tokenizer

    def test_set_tokenizer_wrong_vocab_size_raises_error(
        self,
        action_space,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        vocab_size,
        embedding_dimension,
        tokenizer,
    ):
        """Test that mismatched vocab size raises ValueError."""
        decoder = FASTDETRDecoder(
            input_keys=["rgb_features"],
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            vocab_size=vocab_size + 1000,
            embedding_dimension=embedding_dimension,
            number_of_heads=2,
            feedforward_dimension=128,
            number_of_encoder_layers=2,
            number_of_decoder_layers=2,

        )

        with pytest.raises(ValueError, match="vocab_size.*doesn't match"):
            decoder.set_tokenizer(tokenizer)

    def test_forward_without_tokenizer_raises_error(
        self,
        action_space,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        vocab_size,
        embedding_dimension,
        spatial_features,
    ):
        """Test that forward without tokenizer raises RuntimeError."""
        decoder = FASTDETRDecoder(
            input_keys=["rgb_left_features"],
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            vocab_size=vocab_size,
            embedding_dimension=embedding_dimension,
            number_of_heads=2,
            feedforward_dimension=128,
            number_of_encoder_layers=2,
            number_of_decoder_layers=2,

        )

        with pytest.raises(RuntimeError, match="Tokenizer not set"):
            decoder(spatial_features, actions=None)


@pytest.mark.unit
class TestFASTDETRDecoderForwardPass:
    """Test FASTDETRDecoder forward pass."""

    def test_forward_training_with_actions(
        self,
        action_space,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        vocab_size,
        embedding_dimension,
        spatial_features,
        actions_dict,
        tokenizer,
        batch_size,
    ):
        """Test forward pass during training with action tokenization."""
        decoder = FASTDETRDecoder(
            input_keys=["rgb_left_features"],
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            vocab_size=vocab_size,
            embedding_dimension=embedding_dimension,
            number_of_heads=2,
            feedforward_dimension=64,
            number_of_encoder_layers=2,
            number_of_decoder_layers=2,

        )
        decoder.set_tokenizer(tokenizer)

        predictions = decoder(spatial_features, actions=actions_dict)

        # Training returns PREDICTED_ACTION_TOKENS_KEY (logits) and target tokens
        assert PREDICTED_ACTION_TOKENS_KEY in predictions
        assert f"{PREDICTED_ACTION_TOKENS_KEY}_target" in predictions
        assert predictions[PREDICTED_ACTION_TOKENS_KEY].shape[0] == batch_size
        assert predictions[PREDICTED_ACTION_TOKENS_KEY].shape[2] == vocab_size

    def test_forward_inference_without_actions(
        self,
        action_space,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        vocab_size,
        embedding_dimension,
        spatial_features,
        tokenizer,
        batch_size,
    ):
        """Test forward pass during inference with detokenization."""
        decoder = FASTDETRDecoder(
            input_keys=["rgb_left_features"],
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            vocab_size=vocab_size,
            embedding_dimension=embedding_dimension,
            number_of_heads=2,
            feedforward_dimension=128,
            number_of_encoder_layers=2,
            number_of_decoder_layers=2,

        )
        decoder.set_tokenizer(tokenizer)

        predictions = decoder(spatial_features, actions=None)

        # Inference returns continuous actions
        assert POSITION_ACTION_KEY in predictions
        assert ORIENTATION_ACTION_KEY in predictions
        assert GRIPPER_ACTION_KEY in predictions

        assert predictions[POSITION_ACTION_KEY].shape == (
            batch_size, prediction_horizon, action_space.position_dim
        )
        assert predictions[ORIENTATION_ACTION_KEY].shape == (
            batch_size, prediction_horizon, action_space.orientation_dim
        )
        assert predictions[GRIPPER_ACTION_KEY].shape == (
            batch_size, prediction_horizon, action_space.gripper_dim
        )

    def test_tokenize_actions_removes_padding(
        self,
        action_space,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        vocab_size,
        embedding_dimension,
        spatial_features,
        batch_size,
        tokenizer,
    ):
        """Test that padding timesteps are removed before tokenization."""
        decoder = FASTDETRDecoder(
            input_keys=["rgb_left_features"],
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            vocab_size=vocab_size,
            embedding_dimension=embedding_dimension,
            number_of_heads=2,
            feedforward_dimension=128,
            number_of_encoder_layers=2,
            number_of_decoder_layers=2,

        )
        decoder.set_tokenizer(tokenizer)

        actions = {
            POSITION_ACTION_KEY: torch.randn(batch_size, prediction_horizon, 3, device=device),
            ORIENTATION_ACTION_KEY: torch.randn(batch_size, prediction_horizon, 4, device=device),
            GRIPPER_ACTION_KEY: torch.randn(batch_size, prediction_horizon, 1, device=device),
            IS_PAD_ACTION_KEY: torch.zeros(batch_size, prediction_horizon, 1, dtype=torch.bool, device=device),
        }

        # Mark last 5 timesteps as padding for first sample
        actions[IS_PAD_ACTION_KEY][0, 5:] = True

        predictions = decoder(spatial_features, actions=actions)

        assert PREDICTED_ACTION_TOKENS_KEY in predictions


@pytest.mark.unit
class TestFASTDETRDecoderTokenizationDetokenization:
    """Test tokenization and detokenization methods."""

    def test_tokenize_actions_adds_eos(
        self,
        action_space,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        vocab_size,
        embedding_dimension,
        batch_size,
        tokenizer,
    ):
        """Test _tokenize_actions creates PREDICTED_ACTION_TOKENS_KEY with EOS (no BOS in FAST)."""
        decoder = FASTDETRDecoder(
            input_keys=["rgb_features"],
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            vocab_size=vocab_size,
            embedding_dimension=embedding_dimension,
            number_of_heads=2,
            feedforward_dimension=128,
            number_of_encoder_layers=2,
            number_of_decoder_layers=2,

        )
        decoder.set_tokenizer(tokenizer)

        actions = {
            POSITION_ACTION_KEY: torch.randn(batch_size, prediction_horizon, 3, device=device),
            ORIENTATION_ACTION_KEY: torch.randn(batch_size, prediction_horizon, 4, device=device),
            GRIPPER_ACTION_KEY: torch.randn(batch_size, prediction_horizon, 1, device=device),
        }

        tokenized = decoder._tokenize_actions(actions)

        assert PREDICTED_ACTION_TOKENS_KEY in tokenized
        assert IS_PAD_ACTION_KEY in tokenized
        assert tokenized[PREDICTED_ACTION_TOKENS_KEY].dtype == torch.long

        # FAST doesn't use BOS - check that last non-pad token is EOS
        token_ids = tokenized[PREDICTED_ACTION_TOKENS_KEY]
        is_pad = tokenized[IS_PAD_ACTION_KEY]
        # Find last non-pad token for first sample
        non_pad_mask = ~is_pad[0]
        non_pad_indices = torch.where(non_pad_mask)[0]
        if len(non_pad_indices) > 0:
            last_token_idx = non_pad_indices[-1]
            assert token_ids[0, last_token_idx] == decoder.eos_token_id

    def test_detokenize_predictions_splits_actions(
        self,
        action_space,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        vocab_size,
        embedding_dimension,
        batch_size,
        tokenizer,
    ):
        """Test _detokenize_predictions splits actions correctly."""
        decoder = FASTDETRDecoder(
            input_keys=["rgb_features"],
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            vocab_size=vocab_size,
            embedding_dimension=embedding_dimension,
            number_of_heads=2,
            feedforward_dimension=128,
            number_of_encoder_layers=2,
            number_of_decoder_layers=2,

        )
        decoder.set_tokenizer(tokenizer)

        # Simulate generated tokens (FAST doesn't use BOS, only EOS)
        token_ids = torch.randint(
            2, vocab_size, (batch_size, 20), device=device
        )
        token_ids[:, -1] = decoder.eos_token_id

        detokenized = decoder._detokenize_predictions(token_ids)

        assert POSITION_ACTION_KEY in detokenized
        assert ORIENTATION_ACTION_KEY in detokenized
        assert GRIPPER_ACTION_KEY in detokenized
        assert detokenized[POSITION_ACTION_KEY].shape[-1] == action_space.position_dim
        assert detokenized[ORIENTATION_ACTION_KEY].shape[-1] == action_space.orientation_dim
        assert detokenized[GRIPPER_ACTION_KEY].shape[-1] == action_space.gripper_dim


@pytest.mark.unit
class TestFASTDETRDecoderParametrized:
    """Parametrized tests for FASTDETRDecoder with different configurations."""

    @pytest.mark.parametrize("prediction_horizon", [10, 20])
    def test_different_prediction_horizons(
        self,
        action_space,
        observation_space,
        prediction_horizon,
        device,
        vocab_size,
        embedding_dimension,
        batch_size,
    ):
        """Test FASTDETRDecoder with different prediction horizons."""
        observation_horizon = 1

        tokenizer_obj = Tokenizer(device=torch.device(device))
        total_action_dim = action_space.get_total_action_dim()
        dummy_action_chunks = np.random.randn(100, prediction_horizon, total_action_dim) * 2 - 1
        tokenizer_obj.fit_action_tokenizer(
            action_chunks=dummy_action_chunks,
            use_pretrained_weights=False,
        )
        sample_actions = torch.randn(1, prediction_horizon, total_action_dim, device=device)
        tokenizer_obj.tokenize({ACTION_KEY: sample_actions})

        decoder = FASTDETRDecoder(
            input_keys=["rgb_left_features"],
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            vocab_size=vocab_size,
            embedding_dimension=embedding_dimension,
            number_of_heads=2,
            feedforward_dimension=128,
            number_of_encoder_layers=2,
            number_of_decoder_layers=2,

        )
        decoder.set_tokenizer(tokenizer_obj)

        spatial_features = {
            "rgb_left_features": torch.randn(batch_size, 128, 4, 4, device=device)
        }

        predictions = decoder(spatial_features, actions=None)

        assert predictions[POSITION_ACTION_KEY].shape[1] == prediction_horizon

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_different_batch_sizes(
        self,
        action_space,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        vocab_size,
        embedding_dimension,
        batch_size,
        tokenizer,
    ):
        """Test FASTDETRDecoder with different batch sizes."""
        decoder = FASTDETRDecoder(
            input_keys=["rgb_left_features"],
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            vocab_size=vocab_size,
            embedding_dimension=embedding_dimension,
            number_of_heads=2,
            feedforward_dimension=128,
            number_of_encoder_layers=2,
            number_of_decoder_layers=2,

        )
        decoder.set_tokenizer(tokenizer)

        features = {
            "rgb_left_features": torch.randn(batch_size, 128, 4, 4, device=device)
        }

        predictions = decoder(features, actions=None)

        assert predictions[POSITION_ACTION_KEY].shape[0] == batch_size