"""Tests for Free Transformer decoder."""
import pytest
import torch

from refactoring.models.decoding.decoders.factory.free_transformer import FreeTransformerDecoder
from refactoring.models.decoding.action_heads import ActionHead
from refactoring.models.decoding.action_heads.blocks import MLPBlock
from refactoring.data.task import ActionSpace, ObservationSpace
from refactoring.data.constants import (
    POSITION_ACTION_KEY,
    ORIENTATION_ACTION_KEY,
    GRIPPER_ACTION_KEY,
    IS_PAD_ACTION_KEY,
    Cameras,
    OrientationRepresentation,
    GripperType,
)
from refactoring.models.decoding.constants import LATENT_KEY, BINARY_LOGITS_KEY


@pytest.fixture
def device():
    """Get available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


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
    return 256


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
        use_proprioceptive_data=True,
        use_proprio_base_frame=True,
        use_proprio_camera_frame=False,
        use_gripper_state=True,
        gripper_type=GripperType.BINARY.value,
        camera_keys=[],
        use_language=False,
    )


@pytest.fixture
def action_heads(action_space, embedding_dimension, device):
    """Create action heads for all action modalities."""
    heads = {}

    if action_space.has_position:
        heads[POSITION_ACTION_KEY] = ActionHead(
            input_dim=embedding_dimension,
            output_dim=action_space.position_dim,
            blocks=[
                MLPBlock(
                    input_dim=embedding_dimension,
                    hidden_dims=[128],
                    output_dim=embedding_dimension,
                    activation="relu",
                    dropout=0.1,
                    normalization=True,
                )
            ]
        ).to(device)
    if action_space.has_orientation:
        heads[ORIENTATION_ACTION_KEY] = ActionHead(
            input_dim=embedding_dimension,
            output_dim=action_space.orientation_dim,
            blocks=[
                MLPBlock(
                    input_dim=embedding_dimension,
                    hidden_dims=[128],
                    output_dim=embedding_dimension,
                    activation="relu",
                    dropout=0.1,
                    normalization=True,
                )
            ]
        ).to(device)
    if action_space.has_gripper:
        heads[GRIPPER_ACTION_KEY] = ActionHead(
            input_dim=embedding_dimension,
            output_dim=action_space.gripper_dim,
            blocks=[]
        ).to(device)

    return heads


@pytest.fixture
def flat_features_single(batch_size, device):
    """Single flat feature."""
    return {
        "proprioceptive_features": torch.randn(batch_size, 128, device=device)
    }


@pytest.fixture
def flat_features_temporal(batch_size, observation_horizon, device):
    """Temporal flat features."""
    return {
        "proprioceptive_features": torch.randn(batch_size, observation_horizon, 128, device=device)
    }


@pytest.fixture
def flat_features_mismatched(batch_size, device):
    """Flat features with different dimensions."""
    return {
        "language_embedding": torch.randn(batch_size, 64, device=device),
        "proprioceptive_features": torch.randn(batch_size, 128, device=device),
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
        batch_size, prediction_horizon, dtype=torch.bool, device=device
    )

    return actions


@pytest.mark.unit
class TestFreeTransformerInitialization:
    """Test Free Transformer decoder initialization."""

    def test_init_basic(
        self,
        action_space,
        observation_space,
        action_heads,
        observation_horizon,
        prediction_horizon,
        device,
        embedding_dimension,
    ):
        """Test basic initialization."""
        decoder = FreeTransformerDecoder(
            input_keys=["proprioceptive_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
            latent_bits=8,
        ).to(device)

        assert decoder.embedding_dimension == embedding_dimension
        assert decoder.prediction_horizon == prediction_horizon
        assert decoder.latent_bits == 8
        assert decoder.latent_dim == 2**8
        assert len(decoder.action_heads) == 3

        assert hasattr(decoder, "encoder")
        assert hasattr(decoder, "decoder")
        assert hasattr(decoder, "action_queries")
        assert hasattr(decoder, "action_embedding")

    def test_init_custom_latent_bits(
        self,
        action_space,
        observation_space,
        action_heads,
        observation_horizon,
        prediction_horizon,
        device,
        embedding_dimension,
    ):
        """Test initialization with custom latent bits."""
        for latent_bits in [8, 12, 16]:
            decoder = FreeTransformerDecoder(
                input_keys=["proprioceptive_features"],
                action_space=action_space,
                action_heads=action_heads,
                observation_space=observation_space,
                observation_horizon=observation_horizon,
                prediction_horizon=prediction_horizon,
                device=device,
                embedding_dimension=embedding_dimension,
                latent_bits=latent_bits,
            ).to(device)

            assert decoder.latent_bits == latent_bits
            assert decoder.latent_dim == 2**latent_bits

    def test_init_custom_architecture_params(
        self,
        action_space,
        observation_space,
        action_heads,
        observation_horizon,
        prediction_horizon,
        device,
    ):
        """Test initialization with custom architecture parameters."""
        decoder = FreeTransformerDecoder(
            input_keys=["proprioceptive_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=512,
            number_of_heads=16,
            feedforward_dimension=2048,
            number_of_decoder_layers=8,
            number_of_encoder_layers=2,
            dropout_rate=0.2,
            latent_bits=16,
        ).to(device)

        assert decoder.embedding_dimension == 512
        assert decoder.latent_bits == 16


@pytest.mark.unit
class TestFreeTransformerFeaturePreparation:
    """Test Free Transformer feature preparation methods."""

    def test_prepare_flat_features_single(
        self,
        action_space,
        observation_space,
        action_heads,
        observation_horizon,
        prediction_horizon,
        device,
        embedding_dimension,
        flat_features_single,
    ):
        """Test preparing single flat feature."""
        decoder = FreeTransformerDecoder(
            input_keys=["proprioceptive_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        ).to(device)

        flat_features = decoder._prepare_sequential_features(flat_features_single)

        batch_size = flat_features_single["proprioceptive_features"].shape[0]
        assert flat_features.shape == (batch_size, 1, embedding_dimension)

    def test_prepare_flat_features_dimension_mismatch(
        self,
        action_space,
        observation_space,
        action_heads,
        observation_horizon,
        prediction_horizon,
        device,
        embedding_dimension,
        flat_features_mismatched,
    ):
        """Test preparing flat features with mismatched dimensions."""
        decoder = FreeTransformerDecoder(
            input_keys=["language_embedding", "proprioceptive_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        ).to(device)

        flat_features = decoder._prepare_sequential_features(flat_features_mismatched)

        batch_size = flat_features_mismatched["language_embedding"].shape[0]
        assert flat_features.shape == (batch_size, 1, embedding_dimension * 2)

    def test_spatial_features_raises_error(
        self,
        action_space,
        observation_space,
        action_heads,
        observation_horizon,
        prediction_horizon,
        device,
        embedding_dimension,
        batch_size,
    ):
        """Test that spatial features raise error."""
        decoder = FreeTransformerDecoder(
            input_keys=["rgb_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        ).to(device)

        spatial_features = {
            "rgb_features": torch.randn(batch_size, 2048, 7, 7, device=device)
        }

        with pytest.raises(ValueError, match="does not support spatial features"):
            decoder(spatial_features, actions=None)


@pytest.mark.unit
class TestFreeTransformerForwardPass:
    """Test Free Transformer forward pass."""

    def test_forward_training_with_actions(
        self,
        action_space,
        observation_space,
        action_heads,
        observation_horizon,
        prediction_horizon,
        device,
        embedding_dimension,
        flat_features_single,
        actions_dict,
    ):
        """Test forward pass during training with actions."""
        decoder = FreeTransformerDecoder(
            input_keys=["proprioceptive_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
            latent_bits=8,
        ).to(device)

        decoder.train()
        predictions = decoder(flat_features_single, actions=actions_dict)

        batch_size = flat_features_single["proprioceptive_features"].shape[0]

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

        assert LATENT_KEY in predictions
        assert BINARY_LOGITS_KEY in predictions

        assert predictions[LATENT_KEY].shape == (batch_size, prediction_horizon, decoder.latent_dim)
        assert predictions[BINARY_LOGITS_KEY].shape == (batch_size, prediction_horizon, decoder.latent_bits)

    def test_forward_inference_without_actions(
        self,
        action_space,
        observation_space,
        action_heads,
        observation_horizon,
        prediction_horizon,
        device,
        embedding_dimension,
        flat_features_single,
    ):
        """Test forward pass during inference without actions."""
        decoder = FreeTransformerDecoder(
            input_keys=["proprioceptive_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
            latent_bits=8,
        ).to(device)

        decoder.eval()
        predictions = decoder(flat_features_single, actions=None)

        batch_size = flat_features_single["proprioceptive_features"].shape[0]

        assert POSITION_ACTION_KEY in predictions
        assert ORIENTATION_ACTION_KEY in predictions
        assert GRIPPER_ACTION_KEY in predictions

        assert LATENT_KEY in predictions
        assert BINARY_LOGITS_KEY not in predictions

        assert predictions[LATENT_KEY].shape == (batch_size, prediction_horizon, decoder.latent_dim)


    def test_forward_with_temporal_features(
        self,
        action_space,
        observation_space,
        action_heads,
        prediction_horizon,
        device,
        embedding_dimension,
        batch_size,
        actions_dict,
    ):
        """Test forward pass with temporal flat features."""
        observation_horizon = 3
        temporal_features = {
            "proprioceptive_features": torch.randn(batch_size, observation_horizon, 128, device=device)
        }

        decoder = FreeTransformerDecoder(
            input_keys=["proprioceptive_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        ).to(device)

        decoder.train()
        predictions = decoder(temporal_features, actions=actions_dict)

        assert POSITION_ACTION_KEY in predictions
        assert predictions[POSITION_ACTION_KEY].shape == (
            batch_size, prediction_horizon, action_space.position_dim
        )


@pytest.mark.unit
class TestFreeTransformerLatentEncoding:
    """Test Free Transformer latent encoding methods."""

    def test_encode_latent(
        self,
        action_space,
        observation_space,
        action_heads,
        observation_horizon,
        prediction_horizon,
        device,
        embedding_dimension,
        actions_dict,
    ):
        """Test latent encoding from ground-truth actions."""
        decoder = FreeTransformerDecoder(
            input_keys=["proprioceptive_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
            latent_bits=8,
        ).to(device)

        latent_codes, binary_logits = decoder._encode_latent(actions_dict)

        batch_size = actions_dict[POSITION_ACTION_KEY].shape[0]

        assert latent_codes.shape == (batch_size, prediction_horizon, decoder.latent_dim)
        assert binary_logits.shape == (batch_size, prediction_horizon, decoder.latent_bits)

        assert torch.allclose(latent_codes.sum(dim=-1), torch.ones_like(latent_codes.sum(dim=-1)))

    def test_sample_prior_latent(
        self,
        action_space,
        observation_space,
        action_heads,
        observation_horizon,
        prediction_horizon,
        device,
        embedding_dimension,
        batch_size,
    ):
        """Test sampling latent from uniform prior."""
        decoder = FreeTransformerDecoder(
            input_keys=["proprioceptive_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
            latent_bits=8,
        ).to(device)

        latent_codes = decoder._sample_prior_latent(batch_size)

        assert latent_codes.shape == (batch_size, prediction_horizon, decoder.latent_dim)

        assert torch.allclose(latent_codes.sum(dim=-1), torch.ones_like(latent_codes.sum(dim=-1)))

        unique_indices = (latent_codes == 1.0).nonzero()
        assert unique_indices.shape[0] == batch_size * prediction_horizon


@pytest.mark.unit
class TestFreeTransformerActionHeads:
    """Test Free Transformer action head application."""

    def test_apply_action_heads(
        self,
        action_space,
        observation_space,
        action_heads,
        observation_horizon,
        prediction_horizon,
        device,
        embedding_dimension,
        batch_size,
    ):
        """Test applying action heads to embeddings."""
        decoder = FreeTransformerDecoder(
            input_keys=["proprioceptive_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        ).to(device)

        action_embeddings = torch.randn(batch_size, prediction_horizon, embedding_dimension, device=device)

        predictions = decoder._apply_action_heads(action_embeddings)

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


@pytest.mark.unit
class TestFreeTransformerParametrized:
    """Parametrized tests for Free Transformer with different configurations."""

    @pytest.mark.parametrize("latent_bits", [8, 12, 16])
    def test_different_latent_bits(
        self,
        action_space,
        observation_space,
        latent_bits,
        device,
        flat_features_single,
        actions_dict,
    ):
        """Test Free Transformer with different latent bits."""
        embedding_dimension = 256
        observation_horizon = 1
        prediction_horizon = 10

        action_heads = {
            POSITION_ACTION_KEY: ActionHead(
                input_dim=embedding_dimension,
                output_dim=action_space.position_dim,
                blocks=[],
            ),
            ORIENTATION_ACTION_KEY: ActionHead(
                input_dim=embedding_dimension,
                output_dim=action_space.orientation_dim,
                blocks=[],
            ),
            GRIPPER_ACTION_KEY: ActionHead(
                input_dim=embedding_dimension,
                output_dim=action_space.gripper_dim,
                blocks=[],
            ),
        }

        decoder = FreeTransformerDecoder(
            input_keys=["proprioceptive_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
            latent_bits=latent_bits,
        ).to(device)

        decoder.train()
        predictions = decoder(flat_features_single, actions=actions_dict)

        assert predictions[LATENT_KEY].shape[-1] == 2**latent_bits
        assert predictions[BINARY_LOGITS_KEY].shape[-1] == latent_bits

    @pytest.mark.parametrize("prediction_horizon", [1, 10, 50])
    def test_different_prediction_horizons(
        self,
        action_space,
        observation_space,
        prediction_horizon,
        device,
        batch_size,
    ):
        """Test Free Transformer with different prediction horizons."""
        embedding_dimension = 256
        observation_horizon = 1

        action_heads = {
            POSITION_ACTION_KEY: ActionHead(
                input_dim=embedding_dimension,
                output_dim=action_space.position_dim,
                blocks=[],
            ),
            ORIENTATION_ACTION_KEY: ActionHead(
                input_dim=embedding_dimension,
                output_dim=action_space.orientation_dim,
                blocks=[],
            ),
            GRIPPER_ACTION_KEY: ActionHead(
                input_dim=embedding_dimension,
                output_dim=action_space.gripper_dim,
                blocks=[],
            ),
        }

        decoder = FreeTransformerDecoder(
            input_keys=["proprioceptive_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        ).to(device)

        features = {
            "proprioceptive_features": torch.randn(batch_size, 128, device=device)
        }

        decoder.eval()
        predictions = decoder(features, actions=None)

        assert predictions[POSITION_ACTION_KEY].shape == (
            batch_size, prediction_horizon, action_space.position_dim
        )

    @pytest.mark.parametrize("embedding_dimension", [128, 256, 512])
    def test_different_embedding_dimensions(
        self,
        action_space,
        observation_space,
        embedding_dimension,
        device,
        batch_size,
    ):
        """Test Free Transformer with different embedding dimensions."""
        observation_horizon = 1
        prediction_horizon = 10

        action_heads = {
            POSITION_ACTION_KEY: ActionHead(
                input_dim=embedding_dimension,
                output_dim=action_space.position_dim,
                blocks=[],
            ),
            ORIENTATION_ACTION_KEY: ActionHead(
                input_dim=embedding_dimension,
                output_dim=action_space.orientation_dim,
                blocks=[],
            ),
            GRIPPER_ACTION_KEY: ActionHead(
                input_dim=embedding_dimension,
                output_dim=action_space.gripper_dim,
                blocks=[],
            ),
        }

        decoder = FreeTransformerDecoder(
            input_keys=["proprioceptive_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        ).to(device)

        features = {
            "proprioceptive_features": torch.randn(batch_size, 128, device=device)
        }

        decoder.eval()
        predictions = decoder(features, actions=None)

        assert predictions[POSITION_ACTION_KEY].shape == (
            batch_size, prediction_horizon, action_space.position_dim
        )
