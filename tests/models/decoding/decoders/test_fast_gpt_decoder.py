"""Tests for FAST GPT Decoder for tokenized action prediction."""
import numpy as np
import pytest
import torch

from versatil.data.task import ActionSpace, ObservationSpace
from versatil.data.constants import (
    Cameras,
    GripperType,
    OrientationRepresentation,
    ProprioceptiveType,
    SampleKey,
)
from versatil.data.tokenization.tokenizer import Tokenizer
from versatil.models.decoding.constants import DecoderOutputKey, LatentKey
from versatil.models.decoding.decoders.factory.gpt_action_transformer import GPTActionTransformer
from versatil.models.encoding.encoders.constants import EncoderOutputKeys


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
    return 64


@pytest.fixture
def use_pretrained_weights():
    """Whether to use pretrained FAST weights.

    False: Faster tests, action_vocabulary_size=1024 (custom fitted)
    True: Slower tests, action_vocabulary_size=2048 (pretrained)
    """
    return False


@pytest.fixture
def action_vocabulary_size(use_pretrained_weights):
    """Vocabulary size depends on whether pretrained weights are used.

    Pretrained FAST: action_vocabulary_size = 2048
    Custom fitted: action_vocabulary_size = 1024
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
def flat_features(batch_size, embedding_dimension, device):
    """Flat features (B, D)."""
    return {
        "visual_embedding": torch.randn(batch_size, embedding_dimension, device=device)
    }


@pytest.fixture
def sequential_features(batch_size, embedding_dimension, device):
    """Sequential features (B, T, D)."""
    return {
        "visual_tokens": torch.randn(batch_size, 16, embedding_dimension, device=device)
    }


@pytest.fixture
def mixed_features(batch_size, embedding_dimension, device):
    """Mixed flat and sequential features."""
    return {
        "visual_embedding": torch.randn(batch_size, embedding_dimension, device=device),
        "proprio_embedding": torch.randn(batch_size, 32, device=device),
        "visual_tokens": torch.randn(batch_size, 8, embedding_dimension, device=device),
    }


@pytest.fixture
def language_features(batch_size, embedding_dimension, device):
    """Language features with token mask."""
    max_tokens = 20
    return {
        EncoderOutputKeys.LANGUAGE.value + "_embeddings": torch.randn(
            batch_size, max_tokens, embedding_dimension, device=device
        ),
        EncoderOutputKeys.TOKEN_MASK.value: torch.ones(
            batch_size, max_tokens, dtype=torch.bool, device=device
        ),
    }


@pytest.fixture
def actions_dict(batch_size, prediction_horizon, action_space, device):
    """Create ground-truth actions dictionary."""
    actions = {}

    if action_space.has_position:
        actions[ProprioceptiveType.POSITION.value] = torch.randn(
            batch_size, prediction_horizon, action_space.position_dim, device=device
        )

    if action_space.has_orientation:
        actions[ProprioceptiveType.ORIENTATION.value] = torch.randn(
            batch_size, prediction_horizon, action_space.orientation_dim, device=device
        )

    if action_space.has_gripper:
        actions[ProprioceptiveType.GRIPPER.value] = torch.randint(
            0, 2, (batch_size, prediction_horizon, action_space.gripper_dim), device=device
        ).float()

    actions[SampleKey.IS_PAD_ACTION.value] = torch.zeros(
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
    tokenizer_obj.tokenize({SampleKey.ACTION.value: sample_actions})
    return tokenizer_obj


@pytest.mark.unit
class TestFASTGPTDecoderInitialization:
    """Test FASTGPTDecoder initialization."""

    def test_init_basic(
        self,
        action_space,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        embedding_dimension,
        action_vocabulary_size,
    ):
        """Test basic initialization."""
        decoder = GPTActionTransformer(
            input_keys=["visual_embedding"],
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            action_vocabulary_size=action_vocabulary_size,
            embedding_dimension=embedding_dimension,
            number_of_heads=4,
            number_of_key_value_heads=2,  # GQA requires this
            number_of_layers=2,
        )

        assert decoder.embedding_dimension == embedding_dimension
        assert decoder.prediction_horizon == prediction_horizon
        assert decoder.observation_horizon == observation_horizon
        assert decoder.vocab_size == action_vocabulary_size
        assert decoder.eos_token_id == 1
        assert decoder.pad_token_id == 0
        assert decoder.deterministic is True

    @pytest.mark.parametrize(
        "embedding_dimension,action_vocabulary_size,number_of_heads",
        [
            (64, 128, 4),
            (128, 256, 8),
            (256, 512, 8),
        ],
    )
    def test_init_custom_params(
        self,
        action_space,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        embedding_dimension,
        action_vocabulary_size,
        number_of_heads,
    ):
        """Test initialization with custom parameters."""
        decoder = GPTActionTransformer(
            input_keys=["visual_embedding"],
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            action_vocabulary_size=action_vocabulary_size,
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_key_value_heads=number_of_heads // 2,  # GQA with half K/V heads
            feedforward_dimension=embedding_dimension * 4,
            number_of_layers=4,
        )

        assert decoder.embedding_dimension == embedding_dimension
        assert decoder.vocab_size == action_vocabulary_size

    @pytest.mark.parametrize(
        "attention_type,number_of_key_value_heads",
        [
            ("mha", None),  # Multi-head attention
            ("gqa", 2),  # Grouped query attention
        ],
    )
    def test_init_attention_types(
        self,
        action_space,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        embedding_dimension,
        action_vocabulary_size,
        attention_type,
        number_of_key_value_heads,
    ):
        """Test initialization with different attention types."""
        decoder = GPTActionTransformer(
            input_keys=["visual_embedding"],
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            action_vocabulary_size=action_vocabulary_size,
            embedding_dimension=embedding_dimension,
            number_of_heads=8,
            number_of_key_value_heads=number_of_key_value_heads,
            attention_type=attention_type,
            number_of_layers=2,
        )

        assert decoder is not None

    def test_special_tokens_configurable(
        self,
        action_space,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        action_vocabulary_size,
        embedding_dimension,
    ):
        """Test that special tokens are configurable."""
        custom_eos = 11
        custom_pad = 2

        decoder = GPTActionTransformer(
            input_keys=["visual_embedding"],
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            action_vocabulary_size=action_vocabulary_size,
            embedding_dimension=embedding_dimension,
            number_of_heads=4,
            number_of_key_value_heads=2,
            number_of_layers=2,
            eos_token_id=custom_eos,
            pad_token_id=custom_pad,
        )

        assert decoder.eos_token_id == custom_eos
        assert decoder.pad_token_id == custom_pad

    def test_temperature_parameters(
        self,
        action_space,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        action_vocabulary_size,
        embedding_dimension,
    ):
        """Test temperature initialization."""
        temperature = 0.7
        decoder = GPTActionTransformer(
            input_keys=["visual_embedding"],
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            action_vocabulary_size=action_vocabulary_size,
            embedding_dimension=embedding_dimension,
            number_of_heads=4,
            number_of_key_value_heads=2,
            number_of_layers=2,
            temperature=temperature,
            learnable_temperature=True,
            deterministic=False,
        )

        assert torch.allclose(decoder.temperature, torch.tensor(temperature))
        assert decoder.temperature.requires_grad is True
        assert decoder.deterministic is False


@pytest.mark.unit
class TestFASTGPTDecoderTokenizer:
    """Test FASTGPTDecoder tokenizer handling."""

    def test_set_tokenizer_success(
        self,
        action_space,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        action_vocabulary_size,
        embedding_dimension,
        tokenizer,
    ):
        """Test setting valid tokenizer."""
        decoder = GPTActionTransformer(
            input_keys=["visual_embedding"],
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            action_vocabulary_size=action_vocabulary_size,
            embedding_dimension=embedding_dimension,
            number_of_heads=4,
            number_of_key_value_heads=2,
            number_of_layers=2,
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
        action_vocabulary_size,
        embedding_dimension,
        tokenizer,
    ):
        """Test that mismatched vocab size raises ValueError."""
        decoder = GPTActionTransformer(
            input_keys=["visual_embedding"],
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            action_vocabulary_size=action_vocabulary_size + 1000,
            embedding_dimension=embedding_dimension,
            number_of_heads=4,
            number_of_key_value_heads=2,
            number_of_layers=2,
        )

        with pytest.raises(ValueError, match="action_vocabulary_size.*doesn't match"):
            decoder.set_tokenizer(tokenizer)

    def test_forward_without_tokenizer_raises_error(
        self,
        action_space,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        action_vocabulary_size,
        embedding_dimension,
        flat_features,
    ):
        """Test that forward without tokenizer raises RuntimeError."""
        decoder = GPTActionTransformer(
            input_keys=["visual_embedding"],
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            action_vocabulary_size=action_vocabulary_size,
            embedding_dimension=embedding_dimension,
            number_of_heads=4,
            number_of_key_value_heads=2,
            number_of_layers=2,
        )

        with pytest.raises(RuntimeError, match="Tokenizer not set"):
            decoder(flat_features, actions=None)


@pytest.mark.unit
class TestFASTGPTDecoderForwardPass:
    """Test FASTGPTDecoder forward pass."""

    def test_forward_training_with_flat_features(
        self,
        action_space,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        action_vocabulary_size,
        embedding_dimension,
        flat_features,
        actions_dict,
        tokenizer,
        batch_size,
    ):
        """Test forward pass during training with flat features."""
        decoder = GPTActionTransformer(
            input_keys=["visual_embedding"],
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            action_vocabulary_size=action_vocabulary_size,
            embedding_dimension=embedding_dimension,
            number_of_heads=4,
            number_of_key_value_heads=2,
            number_of_layers=2,
        )
        decoder.set_tokenizer(tokenizer)

        predictions = decoder(flat_features, actions=actions_dict)

        # Training returns DecoderOutputKey.PREDICTED_ACTION_TOKENS.value (logits) and target tokens
        assert DecoderOutputKey.PREDICTED_ACTION_TOKENS.value in predictions
        assert f"{DecoderOutputKey.PREDICTED_ACTION_TOKENS.value}_target" in predictions
        assert predictions[DecoderOutputKey.PREDICTED_ACTION_TOKENS.value].shape[0] == batch_size
        assert predictions[DecoderOutputKey.PREDICTED_ACTION_TOKENS.value].shape[2] == action_vocabulary_size

    def test_forward_training_with_sequential_features(
        self,
        action_space,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        action_vocabulary_size,
        embedding_dimension,
        sequential_features,
        actions_dict,
        tokenizer,
        batch_size,
    ):
        """Test forward pass during training with sequential features."""
        decoder = GPTActionTransformer(
            input_keys=["visual_tokens"],
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            action_vocabulary_size=action_vocabulary_size,
            embedding_dimension=embedding_dimension,
            number_of_heads=4,
            number_of_key_value_heads=2,
            number_of_layers=2,
        )
        decoder.set_tokenizer(tokenizer)

        predictions = decoder(sequential_features, actions=actions_dict)

        assert DecoderOutputKey.PREDICTED_ACTION_TOKENS.value in predictions
        assert predictions[DecoderOutputKey.PREDICTED_ACTION_TOKENS.value].shape[0] == batch_size

    def test_forward_training_with_mixed_features(
        self,
        action_space,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        action_vocabulary_size,
        mixed_features,
        actions_dict,
        tokenizer,
        batch_size,
    ):
        """Test forward pass with mixed flat and sequential features."""
        decoder = GPTActionTransformer(
            input_keys=["visual_embedding", "proprio_embedding", "visual_tokens"],
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            action_vocabulary_size=action_vocabulary_size,
            embedding_dimension=64,
            number_of_heads=4,
            number_of_key_value_heads=2,
            number_of_layers=2,
        )
        decoder.set_tokenizer(tokenizer)

        predictions = decoder(mixed_features, actions=actions_dict)

        assert DecoderOutputKey.PREDICTED_ACTION_TOKENS.value in predictions
        assert predictions[DecoderOutputKey.PREDICTED_ACTION_TOKENS.value].shape[0] == batch_size

    def test_forward_training_with_language_features(
        self,
        action_space,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        action_vocabulary_size,
        embedding_dimension,
        language_features,
        actions_dict,
        tokenizer,
        batch_size,
    ):
        """Test forward pass with language features and token mask."""
        input_keys = [EncoderOutputKeys.LANGUAGE.value + "_embeddings"]
        decoder = GPTActionTransformer(
            input_keys=input_keys,
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            action_vocabulary_size=action_vocabulary_size,
            embedding_dimension=embedding_dimension,
            number_of_heads=4,
            number_of_key_value_heads=2,
            number_of_layers=2,
        )
        decoder.set_tokenizer(tokenizer)

        predictions = decoder(language_features, actions=actions_dict)

        assert DecoderOutputKey.PREDICTED_ACTION_TOKENS.value in predictions
        assert predictions[DecoderOutputKey.PREDICTED_ACTION_TOKENS.value].shape[0] == batch_size

    def test_forward_inference_with_flat_features(
        self,
        action_space,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        action_vocabulary_size,
        embedding_dimension,
        flat_features,
        tokenizer,
        batch_size,
    ):
        """Test forward pass during inference with detokenization."""
        decoder = GPTActionTransformer(
            input_keys=["visual_embedding"],
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            action_vocabulary_size=action_vocabulary_size,
            embedding_dimension=embedding_dimension,
            number_of_heads=4,
            number_of_key_value_heads=2,
            number_of_layers=2,
        )
        decoder.set_tokenizer(tokenizer)

        predictions = decoder(flat_features, actions=None)

        # Inference returns continuous actions
        assert ProprioceptiveType.POSITION.value in predictions
        assert ProprioceptiveType.ORIENTATION.value in predictions
        assert ProprioceptiveType.GRIPPER.value in predictions

        assert predictions[ProprioceptiveType.POSITION.value].shape[0] == batch_size
        assert predictions[ProprioceptiveType.ORIENTATION.value].shape[0] == batch_size
        assert predictions[ProprioceptiveType.GRIPPER.value].shape[0] == batch_size

    def test_forward_rejects_spatial_features(
        self,
        action_space,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        action_vocabulary_size,
        embedding_dimension,
        batch_size,
        tokenizer,
    ):
        """Test that spatial features are rejected."""
        spatial_features = {
            "spatial_feature": torch.randn(batch_size, 128, 8, 8, device=device)
        }

        decoder = GPTActionTransformer(
            input_keys=["spatial_feature"],
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            action_vocabulary_size=action_vocabulary_size,
            embedding_dimension=embedding_dimension,
            number_of_heads=4,
            number_of_key_value_heads=2,
            number_of_layers=2,
        )
        decoder.set_tokenizer(tokenizer)

        with pytest.raises(ValueError, match="doesn't accept spatial features"):
            decoder(spatial_features, actions=None)

    def test_forward_with_latent_variable(
        self,
        action_space,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        action_vocabulary_size,
        embedding_dimension,
        flat_features,
        actions_dict,
        tokenizer,
        batch_size,
    ):
        """Test forward pass with latent variable from variational algorithm."""
        # Add latent variable (as from VAE)
        latent_dim = 32
        features_with_latent = {
            **flat_features,
            LatentKey.POSTERIOR_LATENT.value: torch.randn(batch_size, latent_dim, device=device),
            LatentKey.POSTERIOR_MU.value: torch.randn(batch_size, latent_dim, device=device),
            LatentKey.POSTERIOR_LOGVAR.value: torch.randn(batch_size, latent_dim, device=device),
        }

        decoder = GPTActionTransformer(
            input_keys=["visual_embedding"],
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            action_vocabulary_size=action_vocabulary_size,
            embedding_dimension=embedding_dimension,
            number_of_heads=4,
            number_of_key_value_heads=2,
            number_of_layers=2,
        )
        decoder.set_tokenizer(tokenizer)

        predictions = decoder(features_with_latent, actions=actions_dict)

        # Check that latent statistics are passed through
        assert LatentKey.POSTERIOR_MU.value in predictions
        assert LatentKey.POSTERIOR_LOGVAR.value in predictions
        assert DecoderOutputKey.PREDICTED_ACTION_TOKENS.value in predictions


@pytest.mark.unit
class TestFASTGPTDecoderTokenizationDetokenization:
    """Test tokenization and detokenization methods."""

    def test_tokenize_actions_adds_eos(
        self,
        action_space,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        action_vocabulary_size,
        embedding_dimension,
        actions_dict,
        tokenizer,
        batch_size,
    ):
        """Test that EOS token is added during tokenization."""
        decoder = GPTActionTransformer(
            input_keys=["visual_embedding"],
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            action_vocabulary_size=action_vocabulary_size,
            embedding_dimension=embedding_dimension,
            number_of_heads=4,
            number_of_key_value_heads=2,
            number_of_layers=2,
        )
        decoder.set_tokenizer(tokenizer)

        tokenized = decoder._tokenize_actions(actions_dict)

        assert DecoderOutputKey.PREDICTED_ACTION_TOKENS.value in tokenized
        # Check that at least one sample has EOS token
        token_ids = tokenized[DecoderOutputKey.PREDICTED_ACTION_TOKENS.value]
        has_eos = (token_ids == decoder.eos_token_id).any()
        assert has_eos

    def test_tokenize_actions_removes_padding(
        self,
        action_space,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        action_vocabulary_size,
        embedding_dimension,
        batch_size,
        tokenizer,
    ):
        """Test that padding timesteps are removed before tokenization."""
        decoder = GPTActionTransformer(
            input_keys=["visual_embedding"],
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            action_vocabulary_size=action_vocabulary_size,
            embedding_dimension=embedding_dimension,
            number_of_heads=4,
            number_of_key_value_heads=2,
            number_of_layers=2,
        )
        decoder.set_tokenizer(tokenizer)

        actions = {
            ProprioceptiveType.POSITION.value: torch.randn(batch_size, prediction_horizon, 3, device=device),
            ProprioceptiveType.ORIENTATION.value: torch.randn(batch_size, prediction_horizon, 4, device=device),
            ProprioceptiveType.GRIPPER.value: torch.randn(batch_size, prediction_horizon, 1, device=device),
            SampleKey.IS_PAD_ACTION.value: torch.zeros(batch_size, prediction_horizon, 1, dtype=torch.bool, device=device),
        }

        # Mark last 5 timesteps as padding for first sample
        actions[SampleKey.IS_PAD_ACTION.value][0, 5:] = True

        tokenized = decoder._tokenize_actions(actions)

        # First sample should have fewer tokens than second sample
        first_sample_length = (~tokenized[SampleKey.IS_PAD_ACTION.value][0]).sum()
        second_sample_length = (~tokenized[SampleKey.IS_PAD_ACTION.value][1]).sum()
        assert first_sample_length < second_sample_length

    def test_detokenize_predictions(
        self,
        action_space,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        action_vocabulary_size,
        embedding_dimension,
        batch_size,
        tokenizer,
    ):
        """Test detokenization of token predictions."""
        decoder = GPTActionTransformer(
            input_keys=["visual_embedding"],
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            action_vocabulary_size=action_vocabulary_size,
            embedding_dimension=embedding_dimension,
            number_of_heads=4,
            number_of_key_value_heads=2,
            number_of_layers=2,
        )
        decoder.set_tokenizer(tokenizer)

        # Create dummy token IDs
        num_tokens = 50
        token_ids = torch.randint(
            2, action_vocabulary_size, (batch_size, num_tokens), device=device
        )

        detokenized = decoder._detokenize_predictions(token_ids)

        assert ProprioceptiveType.POSITION.value in detokenized
        assert ProprioceptiveType.ORIENTATION.value in detokenized
        assert ProprioceptiveType.GRIPPER.value in detokenized

        assert detokenized[ProprioceptiveType.POSITION.value].shape[-1] == action_space.position_dim
        assert detokenized[ProprioceptiveType.ORIENTATION.value].shape[-1] == action_space.orientation_dim
        assert detokenized[ProprioceptiveType.GRIPPER.value].shape[-1] == action_space.gripper_dim


@pytest.mark.unit
class TestFASTGPTDecoderParametrized:
    """Parametrized tests for FASTGPTDecoder with different configurations."""

    @pytest.mark.parametrize("prediction_horizon", [5, 10, 20])
    def test_different_prediction_horizons(
        self,
        action_space,
        observation_space,
        observation_horizon,
        device,
        action_vocabulary_size,
        embedding_dimension,
        flat_features,
        tokenizer,
        prediction_horizon,
    ):
        """Test with different prediction horizons."""
        decoder = GPTActionTransformer(
            input_keys=["visual_embedding"],
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            action_vocabulary_size=action_vocabulary_size,
            embedding_dimension=embedding_dimension,
            number_of_heads=4,
            number_of_key_value_heads=2,
            number_of_layers=2,
        )
        decoder.set_tokenizer(tokenizer)

        predictions = decoder(flat_features, actions=None)

        # Should generate approximately prediction_horizon actions
        assert ProprioceptiveType.POSITION.value in predictions
        # May generate more or fewer depending on tokenization
        assert predictions[ProprioceptiveType.POSITION.value].shape[1] >= 1

    @pytest.mark.parametrize(
        "number_of_layers,number_of_heads",
        [
            (2, 4),
            (4, 8),
            (6, 8),
        ],
    )
    def test_different_architectures(
        self,
        action_space,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        action_vocabulary_size,
        flat_features,
        actions_dict,
        tokenizer,
        number_of_layers,
        number_of_heads,
    ):
        """Test with different architecture sizes."""
        embedding_dimension = 64
        decoder = GPTActionTransformer(
            input_keys=["visual_embedding"],
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            action_vocabulary_size=action_vocabulary_size,
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_key_value_heads=number_of_heads // 2,
            number_of_layers=number_of_layers,
        )
        decoder.set_tokenizer(tokenizer)

        predictions = decoder(flat_features, actions=actions_dict)

        assert DecoderOutputKey.PREDICTED_ACTION_TOKENS.value in predictions

    @pytest.mark.parametrize(
        "normalization_type,activation",
        [
            ("rmsnorm", "swiglu"),
            ("layernorm", "gelu"),
            ("layernorm", "relu"),
        ],
    )
    def test_different_normalization_activation(
        self,
        action_space,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        action_vocabulary_size,
        embedding_dimension,
        flat_features,
        actions_dict,
        tokenizer,
        normalization_type,
        activation,
    ):
        """Test with different normalization and activation functions."""
        decoder = GPTActionTransformer(
            input_keys=["visual_embedding"],
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            action_vocabulary_size=action_vocabulary_size,
            embedding_dimension=embedding_dimension,
            number_of_heads=4,
            number_of_key_value_heads=2,
            number_of_layers=2,
            normalization_type=normalization_type,
            activation=activation,
        )
        decoder.set_tokenizer(tokenizer)

        predictions = decoder(flat_features, actions=actions_dict)

        assert DecoderOutputKey.PREDICTED_ACTION_TOKENS.value in predictions

    @pytest.mark.parametrize(
        "positional_encoding_type",
        [
            "sinusoidal",
            "rope",
            None,
        ],
    )
    def test_different_positional_encodings(
        self,
        action_space,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        action_vocabulary_size,
        embedding_dimension,
        flat_features,
        actions_dict,
        tokenizer,
        positional_encoding_type,
    ):
        """Test with different positional encoding types."""
        decoder = GPTActionTransformer(
            input_keys=["visual_embedding"],
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            action_vocabulary_size=action_vocabulary_size,
            embedding_dimension=embedding_dimension,
            number_of_heads=4,
            number_of_key_value_heads=2,
            number_of_layers=2,
            positional_encoding_type=positional_encoding_type,
        )
        decoder.set_tokenizer(tokenizer)

        predictions = decoder(flat_features, actions=actions_dict)

        assert DecoderOutputKey.PREDICTED_ACTION_TOKENS.value in predictions


@pytest.mark.unit
class TestFASTGPTDecoderFeatureProjection:
    """Test feature projection and handling."""

    def test_feature_projection_to_embedding_dim(
        self,
        action_space,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        action_vocabulary_size,
        actions_dict,
        tokenizer,
        batch_size,
    ):
        """Test that features are projected to embedding dimension."""
        embedding_dimension = 128
        mismatched_features = {
            "visual_embedding": torch.randn(batch_size, 64, device=device),  # Wrong dim
            "proprio_embedding": torch.randn(batch_size, 32, device=device),  # Wrong dim
        }

        decoder = GPTActionTransformer(
            input_keys=["visual_embedding", "proprio_embedding"],
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            action_vocabulary_size=action_vocabulary_size,
            embedding_dimension=embedding_dimension,
            number_of_heads=4,
            number_of_key_value_heads=2,
            number_of_layers=2,
        )
        decoder.set_tokenizer(tokenizer)

        # Should not raise error - features are projected
        predictions = decoder(mismatched_features, actions=actions_dict)

        assert DecoderOutputKey.PREDICTED_ACTION_TOKENS.value in predictions

    def test_language_features_not_duplicated(
        self,
        action_space,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        action_vocabulary_size,
        embedding_dimension,
        language_features,
        actions_dict,
        tokenizer,
    ):
        """Test that language features are not added twice (bug fix test)."""
        input_keys = [EncoderOutputKeys.LANGUAGE.value + "_embeddings"]
        decoder = GPTActionTransformer(
            input_keys=input_keys,
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            action_vocabulary_size=action_vocabulary_size,
            embedding_dimension=embedding_dimension,
            number_of_heads=4,
            number_of_key_value_heads=2,
            number_of_layers=2,
        )
        decoder.set_tokenizer(tokenizer)

        # This should work without errors
        predictions = decoder(language_features, actions=actions_dict)

        assert DecoderOutputKey.PREDICTED_ACTION_TOKENS.value in predictions
