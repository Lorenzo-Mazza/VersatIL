"""Tests for Action Chunking Transformer (ACT) decoder."""
import pytest
import torch
import warnings

from refactoring.models.decoding.decoders.factory.act import ACT
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
from refactoring.models.decoding.constants import MU_KEY, LOGVAR_KEY, LATENT_KEY
from refactoring.models.decoding.action_heads import AttentionBlock, ResidualBlock


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
        camera_keys=[Cameras.LEFT.value, Cameras.RIGHT.value],
        use_language=False,
    )


@pytest.fixture
def action_heads(action_space, embedding_dimension):
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
        )
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
        )
    if action_space.has_gripper:
        heads[GRIPPER_ACTION_KEY] = ActionHead(
            input_dim=embedding_dimension,
            output_dim=action_space.gripper_dim,
            blocks=[]
        )

    return heads


@pytest.fixture
def spatial_features_single(batch_size, device):
    """Single spatial feature with standard ResNet50 dimensions."""
    return {
        "rgb_left_features": torch.randn(batch_size, 2048, 7, 7, device=device)
    }


@pytest.fixture
def spatial_features_multi_camera(batch_size, device):
    """Multiple spatial features with same dimensions."""
    return {
        "rgb_left_features": torch.randn(batch_size, 2048, 7, 7, device=device),
        "rgb_right_features": torch.randn(batch_size, 2048, 7, 7, device=device),
    }


@pytest.fixture
def spatial_features_mismatched(batch_size, device):
    """Spatial features with different channel dimensions."""
    return {
        "rgb_left_features": torch.randn(batch_size, 2048, 7, 7, device=device),
        "rgb_right_features": torch.randn(batch_size, 2048, 7, 7, device=device),
        "depth_features": torch.randn(batch_size, 512, 7, 7, device=device),
    }


@pytest.fixture
def flat_features_single(batch_size, device):
    """Single flat feature."""
    return {
        "proprioceptive_features": torch.randn(batch_size, 128, device=device)
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


def create_act_decoder(
    input_keys,
    action_space,
    action_heads,
    observation_space,
    observation_horizon,
    prediction_horizon,
    device,
    **kwargs
):
    """Helper function to create ACT decoder and move to device."""
    decoder = ACT(
        input_keys=input_keys,
        action_space=action_space,
        action_heads=action_heads,
        observation_space=observation_space,
        observation_horizon=observation_horizon,
        prediction_horizon=prediction_horizon,
        device=device,
        **kwargs
    )
    return decoder


@pytest.mark.unit
class TestACTInitialization:
    """Test ACT decoder initialization."""

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
        decoder = ACT(
            input_keys=["rgb_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        )

        assert decoder.embedding_dimension == embedding_dimension
        assert decoder.prediction_horizon == prediction_horizon
        assert decoder.observation_horizon == observation_horizon
        assert len(decoder.action_heads) == 3  # position, orientation, gripper

        # Check feature projection utilities exist
        assert hasattr(decoder, "spatial_feature_concatenator")
        assert hasattr(decoder, "flat_feature_projection")

        # VAE is now handled at algorithm level, not decoder level
        assert not hasattr(decoder, "vae")
        assert not hasattr(decoder, "use_vae_latent")

    def test_init_minimal_params(
        self,
        action_space,
        observation_space,
        action_heads,
        observation_horizon,
        prediction_horizon,
        device,
        embedding_dimension,
    ):
        """Test initialization with minimal parameters."""
        decoder = ACT(
            input_keys=["rgb_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        )

        # VAE components should not exist in decoder (moved to algorithm level)
        assert not hasattr(decoder, "vae")
        assert not hasattr(decoder, "use_vae_latent")

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
        decoder = ACT(
            input_keys=["rgb_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=512,
            number_of_heads=16,
            feedforward_dimension=2048,
            number_of_encoder_layers=8,
            number_of_decoder_layers=8,
            dropout_rate=0.2,
        )

        assert decoder.embedding_dimension == 512
        assert decoder.number_of_heads == 16
        assert decoder.feedforward_dimension == 2048
        assert decoder.number_of_encoder_layers == 8
        assert decoder.number_of_decoder_layers == 8
        assert decoder.dropout_rate == 0.2


@pytest.mark.unit
class TestACTFeaturePreparation:
    """Test ACT feature preparation methods."""

    def test_prepare_spatial_features_single(
        self,
        action_space,
        observation_space,
        action_heads,
        observation_horizon,
        prediction_horizon,
        device,
        embedding_dimension,
        spatial_features_single,
    ):
        """Test preparing single spatial feature."""
        decoder = ACT(
            input_keys=["rgb_left_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        )

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            spatial_features = decoder._tokenize_visual_features(spatial_features_single)

        # Should project 2048 channels to embedding_dimension
        batch_size = spatial_features_single["rgb_left_features"].shape[0]
        assert spatial_features.shape == (batch_size, embedding_dimension, 7, 7)

        # Should warn about projection
        assert len(warning_list) == 1
        assert "2048 channels" in str(warning_list[0].message)

    def test_prepare_spatial_features_multi_camera(
        self,
        action_space,
        observation_space,
        action_heads,
        observation_horizon,
        prediction_horizon,
        device,
        embedding_dimension,
        spatial_features_multi_camera,
    ):
        """Test preparing multiple spatial features with same dimensions."""
        decoder = ACT(
            input_keys=["rgb_left_features", "rgb_right_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        )

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            spatial_features = decoder._tokenize_visual_features(spatial_features_multi_camera)

        # Should concatenate along width (dim=3)
        batch_size = spatial_features_multi_camera["rgb_left_features"].shape[0]
        assert spatial_features.shape == (batch_size, embedding_dimension, 7, 14)  # width doubled

        # Should warn about projection for both features
        assert len(warning_list) == 2

    def test_prepare_spatial_features_dimension_mismatch(
        self,
        action_space,
        observation_space,
        action_heads,
        observation_horizon,
        prediction_horizon,
        device,
        embedding_dimension,
        spatial_features_mismatched,
    ):
        """Test preparing spatial features with mismatched dimensions."""
        decoder = ACT(
            input_keys=["rgb_left_features", "rgb_right_features", "depth_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        )

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            spatial_features = decoder._tokenize_visual_features(spatial_features_mismatched)

        # All features should be projected to embedding_dimension and concatenated
        batch_size = spatial_features_mismatched["rgb_left_features"].shape[0]
        assert spatial_features.shape == (batch_size, embedding_dimension, 7, 21)  # width tripled

        # Should warn about projection for all features
        assert len(warning_list) == 3
        # Check that depth feature warning mentions 512 channels
        depth_warning_found = False
        for warning in warning_list:
            if "depth_features" in str(warning.message) and "512 channels" in str(warning.message):
                depth_warning_found = True
                break
        assert depth_warning_found

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
        decoder = ACT(
            input_keys=["proprioceptive_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        )

        flat_features = decoder._prepare_flat_features(flat_features_single)

        # Should project 128-dim to embedding_dimension
        batch_size = flat_features_single["proprioceptive_features"].shape[0]
        assert flat_features.shape == (batch_size, embedding_dimension)

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
        decoder = ACT(
            input_keys=["language_embedding", "proprioceptive_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        )

        flat_features = decoder._prepare_flat_features(flat_features_mismatched)

        # Both features projected to embedding_dimension, concatenated, then projected back
        batch_size = flat_features_mismatched["language_embedding"].shape[0]
        # language (64 -> 256) + proprio (128 -> 256) = 512, then 512 -> 256
        assert flat_features.shape == (batch_size, embedding_dimension)

    def test_prepare_flat_features_none_when_no_flat_features(
        self,
        action_space,
        observation_space,
        action_heads,
        observation_horizon,
        prediction_horizon,
        device,
        embedding_dimension,
        spatial_features_single,
    ):
        """Test that flat features return None when no flat features present."""
        decoder = ACT(
            input_keys=["rgb_left_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        )

        # Only spatial features, no flat features
        flat_features = decoder._prepare_flat_features(spatial_features_single)
        assert flat_features is None


@pytest.mark.unit
class TestACTForwardPass:
    """Test ACT forward pass."""

    def test_forward_training_with_actions(
        self,
        action_space,
        observation_space,
        action_heads,
        observation_horizon,
        prediction_horizon,
        device,
        embedding_dimension,
        spatial_features_single,
        actions_dict,
    ):
        """Test forward pass during training with actions."""
        observation_space.use_proprioceptive_data = False
        decoder = ACT(
            input_keys=["rgb_left_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore projection warnings
            predictions = decoder(spatial_features_single, actions=actions_dict)

        batch_size = spatial_features_single["rgb_left_features"].shape[0]

        # Check all action predictions exist
        assert POSITION_ACTION_KEY in predictions
        assert ORIENTATION_ACTION_KEY in predictions
        assert GRIPPER_ACTION_KEY in predictions

        # Check shapes
        assert predictions[POSITION_ACTION_KEY].shape == (
            batch_size, prediction_horizon, action_space.position_dim
        )
        assert predictions[ORIENTATION_ACTION_KEY].shape == (
            batch_size, prediction_horizon, action_space.orientation_dim
        )
        assert predictions[GRIPPER_ACTION_KEY].shape == (batch_size, prediction_horizon, action_space.gripper_dim)

        # VAE statistics should NOT be in decoder output (handled at algorithm level)
        assert MU_KEY not in predictions
        assert LOGVAR_KEY not in predictions

    def test_forward_inference_without_actions(
        self,
        action_space,
        observation_space,
        action_heads,
        observation_horizon,
        prediction_horizon,
        device,
        embedding_dimension,
        spatial_features_single,
    ):
        """Test forward pass during inference without actions."""
        decoder = ACT(
            input_keys=["rgb_left_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictions = decoder(spatial_features_single, actions=None)

        batch_size = spatial_features_single["rgb_left_features"].shape[0]

        # Check all action predictions exist
        assert POSITION_ACTION_KEY in predictions
        assert ORIENTATION_ACTION_KEY in predictions
        assert GRIPPER_ACTION_KEY in predictions

        # Decoder should not include VAE statistics (handled at algorithm level)
        assert MU_KEY not in predictions
        assert LOGVAR_KEY not in predictions

    def test_forward_with_latent_from_algorithm(
        self,
        action_space,
        observation_space,
        action_heads,
        observation_horizon,
        prediction_horizon,
        device,
        embedding_dimension,
        spatial_features_single,
        actions_dict,
    ):
        """Test forward pass with latent embedding from algorithm layer."""
        decoder = ACT(
            input_keys=["rgb_left_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        )

        batch_size = spatial_features_single["rgb_left_features"].shape[0]

        # Simulate algorithm providing latent embedding
        features_with_latent = {
            **spatial_features_single,
            LATENT_KEY: torch.randn(batch_size, embedding_dimension, device=device),
            MU_KEY: torch.randn(batch_size, 32, device=device),
            LOGVAR_KEY: torch.randn(batch_size, 32, device=device),
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictions = decoder(features_with_latent, actions=actions_dict)

        # Decoder should preserve latent-related keys from algorithm
        assert MU_KEY in predictions
        assert LOGVAR_KEY in predictions

    def test_forward_with_flat_and_spatial_features(
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
        """Test forward pass with both flat and spatial features."""
        # Combine spatial and flat features
        features = {
            "rgb_left_features": torch.randn(batch_size, 2048, 7, 7, device=device),
            "language_embedding": torch.randn(batch_size, 64, device=device),
            "proprioceptive_features": torch.randn(batch_size, 128, device=device),
        }

        decoder = ACT(
            input_keys=["rgb_left_features", "language_embedding", "proprioceptive_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictions = decoder(features, actions=None)

        # Should successfully process both feature types
        assert POSITION_ACTION_KEY in predictions
        assert ORIENTATION_ACTION_KEY in predictions
        assert GRIPPER_ACTION_KEY in predictions

    def test_forward_multi_camera_dimension_mismatch(
        self,
        action_space,
        observation_space,
        action_heads,
        observation_horizon,
        prediction_horizon,
        device,
        embedding_dimension,
        spatial_features_mismatched,
    ):
        """Test forward pass with multi-camera dimension mismatch."""
        decoder = ACT(
            input_keys=["rgb_left_features", "rgb_right_features", "depth_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        )

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            predictions = decoder(spatial_features_mismatched, actions=None)

        # Should successfully handle dimension mismatches
        assert POSITION_ACTION_KEY in predictions

        # Should have warnings about projection
        assert len(warning_list) > 0
        assert any("SpatialProjectionFusion" in str(warning.message) for warning in warning_list)


# VAE encoding tests removed - VAE is now handled at Algorithm level, not Decoder level
# See integration tests for Algorithm + LatentActionEncoder + Decoder


@pytest.mark.unit
class TestACTActionHeads:
    """Test ACT action head application."""

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
        decoder = ACT(
            input_keys=["rgb_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        )

        # Create action embeddings
        action_embeddings = torch.randn(batch_size, prediction_horizon, embedding_dimension, device=device)

        predictions = decoder._apply_action_heads(action_embeddings)

        # Check all heads were applied
        assert POSITION_ACTION_KEY in predictions
        assert ORIENTATION_ACTION_KEY in predictions
        assert GRIPPER_ACTION_KEY in predictions

        # Check output shapes match action space
        assert predictions[POSITION_ACTION_KEY].shape == (
            batch_size, prediction_horizon, action_space.position_dim
        )
        assert predictions[ORIENTATION_ACTION_KEY].shape == (
            batch_size, prediction_horizon, action_space.orientation_dim
        )
        assert predictions[GRIPPER_ACTION_KEY].shape == (batch_size, prediction_horizon, action_space.gripper_dim)


@pytest.mark.unit
class TestACTEdgeCases:
    """Test ACT edge cases and error handling."""

    def test_no_spatial_features_raises_error(
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
        """Test that missing spatial features raises error."""
        decoder = ACT(
            input_keys=["proprioceptive_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        )

        # Only flat features, no spatial features
        with pytest.raises(ValueError, match="No spatial features found"):
            decoder._tokenize_visual_features(flat_features_single)

    def test_temporal_spatial_features(
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
        """Test handling of temporal spatial features (B, T, C, H, W)."""
        # Create temporal spatial features
        temporal_length = 3
        features = {
            "rgb_features": torch.randn(batch_size, temporal_length, 2048, 7, 7, device=device)
        }

        decoder = ACT(
            input_keys=["rgb_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=temporal_length,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spatial_features = decoder._tokenize_visual_features(features)

        # ACT only uses most recent timestep, not full temporal history
        expected_batch = batch_size
        assert spatial_features.shape[0] == expected_batch

    def test_temporal_flat_features_proprioceptive(
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
        """Test handling of temporal proprioceptive features (B, T, D)."""
        temporal_length = 3
        features = {
            "proprioceptive_features": torch.randn(batch_size, temporal_length, 128, device=device)
        }

        decoder = ACT(
            input_keys=["proprioceptive_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=temporal_length,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            flat_features = decoder._prepare_flat_features(features)

        # ACT only uses most recent timestep: (B, T, D) -> (B, embedding_dimension)
        assert flat_features is not None
        expected_batch = batch_size
        assert flat_features.shape == (expected_batch, embedding_dimension)

    def test_temporal_flat_features_language(
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
        """Test handling of temporal language features (B, T, D)."""
        temporal_length = 3
        features = {
            "language_embedding": torch.randn(batch_size, temporal_length, 512, device=device)
        }

        decoder = ACT(
            input_keys=["language_embedding"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=temporal_length,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            flat_features = decoder._prepare_flat_features(features)

        # ACT only uses most recent timestep
        assert flat_features is not None
        assert flat_features.shape[0] == batch_size

    def test_temporal_spatial_and_flat_features(
        self,
        action_space,
        observation_space,
        action_heads,
        observation_horizon,
        prediction_horizon,
        device,
        embedding_dimension,
        batch_size,
        actions_dict,
    ):
        """Test forward pass with both temporal spatial and temporal flat features."""
        temporal_length = 3
        features = {
            "rgb_features": torch.randn(batch_size, temporal_length, 2048, 7, 7, device=device),
            "proprioceptive_features": torch.randn(batch_size, temporal_length, 128, device=device),
        }

        decoder = ACT(
            input_keys=["rgb_features", "proprioceptive_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=temporal_length,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictions = decoder(features, actions=None)

        # Should successfully process both temporal feature types
        assert POSITION_ACTION_KEY in predictions
        assert ORIENTATION_ACTION_KEY in predictions
        assert GRIPPER_ACTION_KEY in predictions

        # Check output shapes - ACT only uses most recent timestep (batch size unchanged)
        expected_batch = batch_size
        assert predictions[POSITION_ACTION_KEY].shape == (
            expected_batch, prediction_horizon, action_space.position_dim
        )

    def test_mixed_temporal_and_static_flat_features(
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
        """Test handling mix of temporal (B, T, D) and static (B, D) flat features."""
        temporal_length = 3
        features = {
            "rgb_features": torch.randn(batch_size, 2048, 7, 7, device=device),
            "proprioceptive_features": torch.randn(batch_size, temporal_length, 128, device=device),
            "language_embedding": torch.randn(batch_size, 512, device=device),
        }

        decoder = ACT(
            input_keys=["rgb_features", "proprioceptive_features", "language_embedding"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        )

        with pytest.raises(ValueError, match="ACT expects single-frame observation"):
            decoder._prepare_flat_features(features)



@pytest.mark.integration
class TestACTWithDifferentActionHeads:
    """Integration tests for ACT with different action head configurations."""

    def test_act_with_simple_linear_heads(
        self,
        action_space,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        embedding_dimension,
        batch_size,
    ):
        """Test ACT with simple linear projection heads (no blocks)."""
        # Create simple action heads with no processing blocks
        action_heads = {
            POSITION_ACTION_KEY: ActionHead(
                input_dim=embedding_dimension,
                output_dim=action_space.position_dim,
                blocks=[],  # Empty - just linear projection
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

        decoder = ACT(
            input_keys=["rgb_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        )

        features = {
            "rgb_features": torch.randn(batch_size, 2048, 7, 7, device=device)
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictions = decoder(features, actions=None)

        # Check outputs
        assert POSITION_ACTION_KEY in predictions
        assert ORIENTATION_ACTION_KEY in predictions
        assert GRIPPER_ACTION_KEY in predictions

    def test_act_with_mlp_heads(
        self,
        action_space,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        embedding_dimension,
        batch_size,
    ):
        """Test ACT with MLP processing in action heads."""
        # Create action heads with MLP blocks
        action_heads = {
            POSITION_ACTION_KEY: ActionHead(
                input_dim=embedding_dimension,
                output_dim=action_space.position_dim,
                blocks=[
                    MLPBlock(
                        input_dim=embedding_dimension,
                        hidden_dims=[512, 256],
                        output_dim=embedding_dimension,
                        activation="relu",
                        dropout=0.1,
                        normalization=True,
                    )
                ],
            ),
            ORIENTATION_ACTION_KEY: ActionHead(
                input_dim=embedding_dimension,
                output_dim=action_space.orientation_dim,
                blocks=[
                    MLPBlock(
                        input_dim=embedding_dimension,
                        hidden_dims=[256, 128],
                        output_dim=embedding_dimension,
                        activation="gelu",
                        dropout=0.1,
                        normalization=True,
                    )
                ],
            ),
            GRIPPER_ACTION_KEY: ActionHead(
                input_dim=embedding_dimension,
                output_dim=action_space.gripper_dim,
                blocks=[
                    MLPBlock(
                        input_dim=embedding_dimension,
                        hidden_dims=[128],
                        output_dim=embedding_dimension,
                        activation="silu",
                        dropout=0.0,
                        normalization=False,
                    )
                ],
            ),
        }

        decoder = ACT(
            input_keys=["rgb_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        )

        features = {
            "rgb_features": torch.randn(batch_size, 2048, 7, 7, device=device)
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictions = decoder(features, actions=None)

        # Check outputs have correct shapes
        assert predictions[POSITION_ACTION_KEY].shape == (
            batch_size, prediction_horizon, action_space.position_dim
        )
        assert predictions[ORIENTATION_ACTION_KEY].shape == (
            batch_size, prediction_horizon, action_space.orientation_dim
        )
        assert predictions[GRIPPER_ACTION_KEY].shape == (
            batch_size, prediction_horizon, action_space.gripper_dim
        )

    def test_act_with_attention_heads(
        self,
        action_space,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        embedding_dimension,
        batch_size,
        actions_dict,
    ):
        """Test ACT with self-attention in action heads."""
        # Create action heads with attention blocks
        action_heads = {
            POSITION_ACTION_KEY: ActionHead(
                input_dim=embedding_dimension,
                output_dim=action_space.position_dim,
                blocks=[
                    AttentionBlock(
                        embedding_dimension=embedding_dimension,
                        num_heads=8,
                        dropout=0.1,
                    ),
                    MLPBlock(
                        input_dim=embedding_dimension,
                        hidden_dims=[128],
                        output_dim=embedding_dimension,
                        activation="relu",
                    ),
                ],
            ),
            ORIENTATION_ACTION_KEY: ActionHead(
                input_dim=embedding_dimension,
                output_dim=action_space.orientation_dim,
                blocks=[
                    AttentionBlock(
                        embedding_dimension=embedding_dimension,
                        num_heads=4,
                        dropout=0.1,
                    ),
                ],
            ),
            GRIPPER_ACTION_KEY: ActionHead(
                input_dim=embedding_dimension,
                output_dim=action_space.gripper_dim,
                blocks=[],  # Simple head for binary gripper
            ),
        }

        observation_space.use_proprioceptive_data = False
        decoder = ACT(
            input_keys=["rgb_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        )

        features = {
            "rgb_features": torch.randn(batch_size, 2048, 7, 7, device=device)
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Test training with actions
            predictions = decoder(features, actions=actions_dict)

        # Check all outputs exist
        assert POSITION_ACTION_KEY in predictions
        assert ORIENTATION_ACTION_KEY in predictions
        assert GRIPPER_ACTION_KEY in predictions
        # VAE statistics NOT expected from decoder (handled at algorithm level)
        assert MU_KEY not in predictions
        assert LOGVAR_KEY not in predictions

    def test_act_with_residual_heads(
        self,
        action_space,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        embedding_dimension,
        batch_size,
    ):
        """Test ACT with residual connections in action heads."""
        # Create action heads with residual blocks
        action_heads = {
            POSITION_ACTION_KEY: ActionHead(
                input_dim=embedding_dimension,
                output_dim=action_space.position_dim,
                blocks=[
                    ResidualBlock(
                        MLPBlock(
                            input_dim=embedding_dimension,
                            hidden_dims=[256],
                            output_dim=embedding_dimension,
                            activation="gelu",
                        ),
                        dropout=0.1,
                    ),
                    ResidualBlock(
                        MLPBlock(
                            input_dim=embedding_dimension,
                            hidden_dims=[128],
                            output_dim=embedding_dimension,
                            activation="relu",
                        ),
                        dropout=0.1,
                    ),
                ],
            ),
            ORIENTATION_ACTION_KEY: ActionHead(
                input_dim=embedding_dimension,
                output_dim=action_space.orientation_dim,
                blocks=[
                    ResidualBlock(
                        MLPBlock(
                            input_dim=embedding_dimension,
                            hidden_dims=[256, 128],
                            output_dim=embedding_dimension,
                        ),
                        dropout=0.0,
                    ),
                ],
            ),
            GRIPPER_ACTION_KEY: ActionHead(
                input_dim=embedding_dimension,
                output_dim=action_space.gripper_dim,
                blocks=[],
            ),
        }

        decoder = ACT(
            input_keys=["rgb_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        )

        features = {
            "rgb_features": torch.randn(batch_size, 2048, 7, 7, device=device)
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictions = decoder(features, actions=None)

        # Check outputs
        assert predictions[POSITION_ACTION_KEY].shape == (
            batch_size, prediction_horizon, action_space.position_dim
        )

    def test_act_with_mixed_head_architectures(
        self,
        action_space,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        embedding_dimension,
        batch_size,
        actions_dict,
    ):
        """Test ACT with different architectures for each action modality."""

        # Each action modality gets a different head architecture
        action_heads = {
            # Position: Complex head with attention + residual MLP
            POSITION_ACTION_KEY: ActionHead(
                input_dim=embedding_dimension,
                output_dim=action_space.position_dim,
                blocks=[
                    AttentionBlock(
                        embedding_dimension=embedding_dimension,
                        num_heads=8,
                        dropout=0.1,
                    ),
                    ResidualBlock(
                        MLPBlock(
                            input_dim=embedding_dimension,
                            hidden_dims=[512, 256],
                            output_dim=embedding_dimension,
                            activation="gelu",
                            dropout=0.1,
                        )
                    ),
                ],
            ),
            # Orientation: Simple MLP
            ORIENTATION_ACTION_KEY: ActionHead(
                input_dim=embedding_dimension,
                output_dim=action_space.orientation_dim,
                blocks=[
                    MLPBlock(
                        input_dim=embedding_dimension,
                        hidden_dims=[256],
                        output_dim=embedding_dimension,
                        activation="relu",
                    )
                ],
            ),
            # Gripper: Direct linear projection (no blocks)
            GRIPPER_ACTION_KEY: ActionHead(
                input_dim=embedding_dimension,
                output_dim=action_space.gripper_dim,
                blocks=[],
            ),
        }

        observation_space.use_proprioceptive_data = False
        decoder = ACT(
            input_keys=["rgb_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        )

        features = {
            "rgb_features": torch.randn(batch_size, 2048, 7, 7, device=device)
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Test both training and inference
            train_predictions = decoder(features, actions=actions_dict)
            inference_predictions = decoder(features, actions=None)

        # Check training predictions
        assert POSITION_ACTION_KEY in train_predictions
        assert ORIENTATION_ACTION_KEY in train_predictions
        assert GRIPPER_ACTION_KEY in train_predictions

        # Check inference predictions
        assert POSITION_ACTION_KEY in inference_predictions
        assert ORIENTATION_ACTION_KEY in inference_predictions
        assert GRIPPER_ACTION_KEY in inference_predictions

        # Check shapes match
        assert train_predictions[POSITION_ACTION_KEY].shape == inference_predictions[POSITION_ACTION_KEY].shape


@pytest.mark.unit
class TestACTEncoderInputPrepending:
    """Test ACT encoder input prepending (latent + proprio tokens following DETR)."""

    def test_encoder_input_with_spatial_only(
        self,
        action_space,
        observation_space,
        action_heads,
        observation_horizon,
        prediction_horizon,
        device,
        embedding_dimension,
        batch_size,
        spatial_features_single,
    ):
        """Test encoder input with only spatial features (no latent, no proprio)."""
        # Disable proprioceptive features
        observation_space.use_proprioceptive_data = False

        decoder = ACT(
            input_keys=["rgb_left_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        )

        spatial_features = decoder._tokenize_visual_features(spatial_features_single)
        flat_features = decoder._prepare_flat_features(spatial_features_single)

        # Flat features should be None
        assert flat_features is None

        # Prepare encoder input
        encoder_input, positional_encoding = decoder._prepare_encoder_input(
            spatial_features,
            latent_embedding=None,
            flat_features=None,
            batch_size=batch_size,
        )

        # No additional tokens prepended: encoder_input should only have spatial features
        # Spatial features are (B, C, H, W) = (2, 256, 7, 7) -> flattened to (49, 2, 256)
        expected_seq_len = 7 * 7  # H * W
        assert encoder_input.shape[0] == expected_seq_len  # No additional tokens
        assert encoder_input.shape[1] == batch_size
        assert encoder_input.shape[2] == embedding_dimension

    def test_encoder_input_with_latent_only(
        self,
        action_space,
        observation_space,
        action_heads,
        observation_horizon,
        prediction_horizon,
        device,
        embedding_dimension,
        batch_size,
        spatial_features_single,
    ):
        """Test encoder input with latent token only (no proprio)."""

        # Disable proprioceptive features
        observation_space.use_proprioceptive_data = False

        decoder = ACT(
            input_keys=["rgb_left_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        )

        spatial_features = decoder._tokenize_visual_features(spatial_features_single)
        latent_embedding = torch.randn(batch_size, embedding_dimension, device=device)

        # Prepare encoder input
        encoder_input, positional_encoding = decoder._prepare_encoder_input(
            spatial_features,
            latent_embedding=latent_embedding,
            flat_features=None,
            batch_size=batch_size,
        )

        # Should prepend 1 latent token: (1 + H*W, B, D)
        expected_seq_len = 1 + 7 * 7  # 1 latent + spatial
        assert encoder_input.shape[0] == expected_seq_len
        assert encoder_input.shape[1] == batch_size
        assert encoder_input.shape[2] == embedding_dimension

        # Positional encoding should match
        assert positional_encoding.shape[0] == expected_seq_len

    def test_encoder_input_with_proprio_only(
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
        """Test encoder input with proprio token only (no latent)."""
        # Enable proprioceptive features
        observation_space.use_proprioceptive_data = True

        # Features with both spatial and flat (proprio)
        features = {
            "rgb_left_features": torch.randn(batch_size, 2048, 7, 7, device=device),
            "proprioceptive_features": torch.randn(batch_size, 128, device=device),
        }

        decoder = ACT(
            input_keys=["rgb_left_features", "proprioceptive_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spatial_features = decoder._tokenize_visual_features(features)
            flat_features = decoder._prepare_flat_features(features)

        # Flat features should exist
        assert flat_features is not None
        assert flat_features.shape == (batch_size, embedding_dimension)

        # Prepare encoder input (no latent)
        encoder_input, positional_encoding = decoder._prepare_encoder_input(
            spatial_features,
            latent_embedding=None,
            flat_features=flat_features,
            batch_size=batch_size,
        )

        # Should prepend 1 proprio token: (1 + H*W, B, D)
        expected_seq_len = 1 + 7 * 7  # 1 proprio + spatial
        assert encoder_input.shape[0] == expected_seq_len
        assert encoder_input.shape[1] == batch_size
        assert encoder_input.shape[2] == embedding_dimension

        # Positional encoding should match
        assert positional_encoding.shape[0] == expected_seq_len

    def test_encoder_input_with_latent_and_proprio(
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
        """Test encoder input with both latent and proprio tokens."""

        # Enable proprioceptive features
        observation_space.use_proprioceptive_data = True

        # Features with both spatial and flat (proprio)
        features = {
            "rgb_left_features": torch.randn(batch_size, 2048, 7, 7, device=device),
            "proprioceptive_features": torch.randn(batch_size, 128, device=device),
        }

        decoder = ACT(
            input_keys=["rgb_left_features", "proprioceptive_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spatial_features = decoder._tokenize_visual_features(features)
            flat_features = decoder._prepare_flat_features(features)

        latent_embedding = torch.randn(batch_size, embedding_dimension, device=device)

        # Prepare encoder input (both latent and proprio)
        encoder_input, positional_encoding = decoder._prepare_encoder_input(
            spatial_features,
            latent_embedding=latent_embedding,
            flat_features=flat_features,
            batch_size=batch_size,
        )

        # Should prepend 2 tokens (latent + proprio): (2 + H*W, B, D)
        expected_seq_len = 2 + 7 * 7  # 2 additional + spatial
        assert encoder_input.shape[0] == expected_seq_len
        assert encoder_input.shape[1] == batch_size
        assert encoder_input.shape[2] == embedding_dimension

        # Positional encoding should match
        assert positional_encoding.shape[0] == expected_seq_len

    def test_forward_with_proprio_features(
        self,
        action_space,
        observation_space,
        action_heads,
        observation_horizon,
        prediction_horizon,
        device,
        embedding_dimension,
        batch_size,
        actions_dict,
    ):
        """Test full forward pass with proprioceptive features."""
        # Enable proprioceptive features
        observation_space.use_proprioceptive_data = True

        # Features with both spatial and flat (proprio)
        features = {
            "rgb_left_features": torch.randn(batch_size, 2048, 7, 7, device=device),
            "rgb_right_features": torch.randn(batch_size, 2048, 7, 7, device=device),
            "proprioceptive_features": torch.randn(batch_size, 128, device=device),
        }

        decoder = ACT(
            input_keys=["rgb_left_features", "rgb_right_features", "proprioceptive_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Training forward
            predictions = decoder(features, actions=actions_dict)

        # Check all action predictions exist
        assert POSITION_ACTION_KEY in predictions
        assert ORIENTATION_ACTION_KEY in predictions
        assert GRIPPER_ACTION_KEY in predictions

        # Check shapes
        assert predictions[POSITION_ACTION_KEY].shape == (
            batch_size, prediction_horizon, action_space.position_dim
        )
        assert predictions[ORIENTATION_ACTION_KEY].shape == (
            batch_size, prediction_horizon, action_space.orientation_dim
        )
        assert predictions[GRIPPER_ACTION_KEY].shape == (
            batch_size, prediction_horizon, action_space.gripper_dim
        )

    def test_forward_with_latent_and_proprio(
        self,
        action_space,
        observation_space,
        action_heads,
        observation_horizon,
        prediction_horizon,
        device,
        embedding_dimension,
        batch_size,
        actions_dict,
    ):
        """Test full forward pass with both latent and proprioceptive features."""

        # Enable proprioceptive features
        observation_space.use_proprioceptive_data = True

        # Features with spatial, flat (proprio), and latent
        features = {
            "rgb_left_features": torch.randn(batch_size, 2048, 7, 7, device=device),
            "proprioceptive_features": torch.randn(batch_size, 128, device=device),
            LATENT_KEY: torch.randn(batch_size, embedding_dimension, device=device),
            MU_KEY: torch.randn(batch_size, 32, device=device),
            LOGVAR_KEY: torch.randn(batch_size, 32, device=device),
        }

        decoder = ACT(
            input_keys=["rgb_left_features", "proprioceptive_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Training forward with latent and proprio
            predictions = decoder(features, actions=actions_dict)

        # Check all action predictions exist
        assert POSITION_ACTION_KEY in predictions
        assert ORIENTATION_ACTION_KEY in predictions
        assert GRIPPER_ACTION_KEY in predictions

        # Decoder should preserve latent-related keys from algorithm
        assert MU_KEY in predictions
        assert LOGVAR_KEY in predictions

        # Check shapes
        assert predictions[POSITION_ACTION_KEY].shape == (
            batch_size, prediction_horizon, action_space.position_dim
        )


@pytest.mark.unit
class TestACTParametrized:
    """Parametrized tests for ACT with different horizons and dimensions."""

    @pytest.mark.parametrize("prediction_horizon", [1, 10, 50])
    def test_different_prediction_horizons(
        self,
        action_space,
        observation_space,
        prediction_horizon,
        device,
    ):
        """Test ACT with different prediction horizons."""
        embedding_dimension = 256
        batch_size = 2
        observation_horizon = 1

        # Create simple action heads
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

        decoder = ACT(
            input_keys=["rgb_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        )

        # Create features (no temporal dimension for observation_horizon=1)
        features = {
            "rgb_features": torch.randn(batch_size, 2048, 7, 7, device=device)
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictions = decoder(features, actions=None)

        # Check output shapes
        assert predictions[POSITION_ACTION_KEY].shape == (
            batch_size, prediction_horizon, action_space.position_dim
        )
        assert predictions[ORIENTATION_ACTION_KEY].shape == (
            batch_size, prediction_horizon, action_space.orientation_dim
        )
        assert predictions[GRIPPER_ACTION_KEY].shape == (
            batch_size, prediction_horizon, action_space.gripper_dim
        )

    @pytest.mark.parametrize("embedding_dimension", [128, 256, 512])
    def test_different_embedding_dimensions(
        self,
        action_space,
        observation_space,
        embedding_dimension,
        device,
    ):
        """Test ACT with different embedding dimensions."""
        observation_horizon = 1
        prediction_horizon = 10
        batch_size = 2

        # Create action heads with matching input dimension
        action_heads = {
            POSITION_ACTION_KEY: ActionHead(
                input_dim=embedding_dimension,
                output_dim=action_space.position_dim,
                blocks=[
                    MLPBlock(
                        input_dim=embedding_dimension,
                        hidden_dims=[embedding_dimension // 2],
                        output_dim=embedding_dimension,
                    )
                ],
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

        decoder = ACT(
            input_keys=["rgb_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        )

        features = {
            "rgb_features": torch.randn(batch_size, 2048, 7, 7, device=device)
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictions = decoder(features, actions=None)

        # Check output shapes are correct
        assert predictions[POSITION_ACTION_KEY].shape == (
            batch_size, prediction_horizon, action_space.position_dim
        )

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_different_batch_sizes(
        self,
        action_space,
        observation_space,
        batch_size,
        device,
    ):
        """Test ACT with different batch sizes."""
        observation_horizon = 1
        prediction_horizon = 10
        embedding_dimension = 256

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

        decoder = ACT(
            input_keys=["rgb_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        )

        features = {
            "rgb_features": torch.randn(batch_size, 2048, 7, 7, device=device)
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictions = decoder(features, actions=None)

        # Check batch dimension
        assert predictions[POSITION_ACTION_KEY].shape[0] == batch_size
        assert predictions[ORIENTATION_ACTION_KEY].shape[0] == batch_size
        assert predictions[GRIPPER_ACTION_KEY].shape[0] == batch_size

    @pytest.mark.parametrize("prediction_horizon,embedding_dimension", [
        (10, 256),
        (20, 128),
        (50, 512),
    ])
    def test_combined_parameters(
        self,
        action_space,
        observation_space,
        prediction_horizon,
        embedding_dimension,
        device,
    ):
        """Test ACT with combined parameter variations."""
        batch_size = 4
        observation_horizon = 1

        action_heads = {
            POSITION_ACTION_KEY: ActionHead(
                input_dim=embedding_dimension,
                output_dim=action_space.position_dim,
                blocks=[
                    MLPBlock(
                        input_dim=embedding_dimension,
                        hidden_dims=[embedding_dimension // 2],
                        output_dim=embedding_dimension,
                        activation="gelu",
                    )
                ],
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

        decoder = ACT(
            input_keys=["rgb_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        )

        features = {
            "rgb_features": torch.randn(batch_size, 2048, 7, 7, device=device)
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictions = decoder(features, actions=None)

        # Verify all outputs
        assert predictions[POSITION_ACTION_KEY].shape == (
            batch_size, prediction_horizon, action_space.position_dim
        )
        assert predictions[ORIENTATION_ACTION_KEY].shape == (
            batch_size, prediction_horizon, action_space.orientation_dim
        )
        assert predictions[GRIPPER_ACTION_KEY].shape == (
            batch_size, prediction_horizon, action_space.gripper_dim
        )
        assert not torch.isnan(predictions[POSITION_ACTION_KEY]).any()
        assert not torch.isnan(predictions[ORIENTATION_ACTION_KEY]).any()
        assert not torch.isnan(predictions[GRIPPER_ACTION_KEY]).any()