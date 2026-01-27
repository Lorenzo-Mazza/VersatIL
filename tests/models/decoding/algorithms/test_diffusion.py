"""Tests for action-decoding algorithms (BC, Diffusion, Flow Matching)."""
import pytest
import torch
from typing import Dict, Optional

from versatil.models.decoding.algorithm.diffusion import Diffusion
from versatil.models.decoding.decoders.base import ActionDecoder, DecoderInput
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.data.constants import (
    POSITION_ACTION_KEY,
    ORIENTATION_ACTION_KEY,
    GRIPPER_ACTION_KEY,
    OrientationRepresentation,
    GripperType,
    Cameras,
)
from versatil.models.decoding.constants import (
    PredictionType,
    BetaSchedule,
    VarianceType,
)
from versatil.models.layers.denoising.diffusion_process import SchedulerType


# Mock decoder for testing algorithms
class MockActionDecoder(ActionDecoder):
    """Mock action decoder for testing algorithms."""

    def __init__(
        self,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        feature_dim: int = 128,
        prediction_horizon: int = 10,
        device: str = "cpu",
    ):
        """Initialize mock decoder."""
        decoder_input = DecoderInput(keys=["features"])

        # Create simple action heads
        from versatil.models.decoding.action_heads import ActionHead

        action_heads = {}
        if action_space.has_position:
            action_heads[POSITION_ACTION_KEY] = ActionHead(
                input_dim=feature_dim,
                output_dim=action_space.position_dim,
                blocks=[],
            )
        if action_space.has_orientation:
            action_heads[ORIENTATION_ACTION_KEY] = ActionHead(
                input_dim=feature_dim,
                output_dim=action_space.orientation_dim,
                blocks=[],
            )
        if action_space.has_gripper:
            action_heads[GRIPPER_ACTION_KEY] = ActionHead(
                input_dim=feature_dim,
                output_dim=action_space.gripper_dim,
                blocks=[],
            )

        super().__init__(
            decoder_input=decoder_input,
            observation_space=observation_space,
            action_space=action_space,
            action_heads=action_heads,
            device=device,
            observation_horizon=1,
            prediction_horizon=prediction_horizon,
        )

        self.feature_dim = feature_dim

        # Create simple projection layers for mock predictions
        self.feature_proj = torch.nn.Linear(feature_dim, feature_dim).to(self.device)

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        actions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Mock forward pass."""
        # Extract batch size from features
        if "features" in features:
            feature_tensor = features["features"]
        else:
            feature_tensor = next(iter(features.values()))
        batch_size = feature_tensor.shape[0]

        # Project features
        projected = self.feature_proj(feature_tensor)

        # Generate mock predictions
        predictions = {}

        if self.use_position_actions:
            predictions[POSITION_ACTION_KEY] = torch.randn(
                batch_size,
                self.prediction_horizon,
                self.position_dim,
                device=self.device,
            )

        if self.use_orientation_actions:
            predictions[ORIENTATION_ACTION_KEY] = torch.randn(
                batch_size,
                self.prediction_horizon,
                self.orientation_dim,
                device=self.device,
            )

        if self.use_gripper_actions:
            predictions[GRIPPER_ACTION_KEY] = torch.randn(
                batch_size,
                self.prediction_horizon,
                self.gripper_dim,
                device=self.device,
            )

        return predictions


@pytest.fixture
def device():
    """Get available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def batch_size():
    """Default batch size."""
    return 4


@pytest.fixture
def prediction_horizon():
    """Default prediction horizon."""
    return 10


@pytest.fixture
def feature_dim():
    """Default feature dimension."""
    return 128


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
        camera_keys=[Cameras.LEFT.value],
        use_language=False,
    )


@pytest.fixture
def mock_decoder(observation_space, action_space, feature_dim, prediction_horizon, device):
    """Create mock decoder for testing."""
    return MockActionDecoder(
        observation_space=observation_space,
        action_space=action_space,
        feature_dim=feature_dim,
        prediction_horizon=prediction_horizon,
        device=device,
    )


@pytest.fixture
def features(batch_size, feature_dim, device):
    """Create mock features."""
    return {
        "features": torch.randn(batch_size, feature_dim, device=device),
    }


@pytest.fixture
def actions(batch_size, prediction_horizon, action_space, device):
    """Create mock actions."""
    actions_dict = {}

    if action_space.has_position:
        actions_dict[POSITION_ACTION_KEY] = torch.randn(
            batch_size, prediction_horizon, action_space.position_dim, device=device
        )

    if action_space.has_orientation:
        actions_dict[ORIENTATION_ACTION_KEY] = torch.randn(
            batch_size, prediction_horizon, action_space.orientation_dim, device=device
        )

    if action_space.has_gripper:
        actions_dict[GRIPPER_ACTION_KEY] = torch.randn(
            batch_size, prediction_horizon, action_space.gripper_dim, device=device
        )

    return actions_dict


@pytest.mark.unit
class TestDiffusion:
    """Tests for Diffusion algorithm."""

    @pytest.mark.parametrize("scheduler_type", [
        SchedulerType.DDPM.value,
        SchedulerType.DDIM.value,
    ])
    def test_instantiation_schedulers(self, scheduler_type):
        """Test that Diffusion algorithm can be instantiated with all scheduler types."""
        algo = Diffusion(scheduler_type=scheduler_type)
        assert algo is not None
        assert algo.noise_scheduler is not None

    def test_invalid_scheduler_type(self):
        """Test that invalid scheduler type raises error."""
        with pytest.raises(ValueError, match="Unknown scheduler_type"):
            Diffusion(scheduler_type="invalid")

    @pytest.mark.parametrize("prediction_type", [
        PredictionType.EPSILON.value,
        PredictionType.SAMPLE.value,
        PredictionType.VELOCITY.value,
    ])
    def test_prediction_types(self, mock_decoder, features, actions, prediction_type):
        """Test all prediction types."""
        from versatil.models.decoding.constants import TIMESTEP_KEY, NOISE_KEY, TARGET_DIFFUSION_KEY

        algo = Diffusion(
            num_train_timesteps=50,
            prediction_type=prediction_type,
        )
        outputs = algo.forward(mock_decoder, features, actions)

        # Check that outputs contain required keys
        assert NOISE_KEY in outputs
        assert TIMESTEP_KEY in outputs
        assert TARGET_DIFFUSION_KEY in outputs

    @pytest.mark.parametrize("beta_schedule", [
        BetaSchedule.LINEAR.value,
        BetaSchedule.SCALED_LINEAR.value,
        BetaSchedule.SQUAREDCOS_CAP_V2.value,
    ])
    def test_beta_schedules(self, beta_schedule):
        """Test all beta schedule types."""
        algo = Diffusion(beta_schedule=beta_schedule)
        assert algo is not None
        assert algo.noise_scheduler is not None

    @pytest.mark.parametrize("variance_type", [
        VarianceType.FIXED_SMALL.value,
        VarianceType.FIXED_LARGE.value,
    ])
    def test_variance_types_ddpm(self, variance_type):
        """Test variance types for DDPM scheduler."""
        algo = Diffusion(
            scheduler_type=SchedulerType.DDPM.value,
            scheduler_variance_type=variance_type,
        )
        assert algo is not None
        assert algo.noise_scheduler is not None

    @pytest.mark.parametrize("num_train_timesteps,num_inference_steps", [
        (50, 10),
        (100, 20),
        (1000, 50),
    ])
    def test_timestep_configurations(self, num_train_timesteps, num_inference_steps):
        """Test different timestep configurations."""
        algo = Diffusion(
            num_train_timesteps=num_train_timesteps,
            num_inference_steps=num_inference_steps,
        )
        assert algo.num_inference_steps == num_inference_steps
        assert algo.noise_scheduler.config.num_train_timesteps == num_train_timesteps

    def test_forward_pass(self, mock_decoder, features, actions):
        """Test forward pass during training."""
        from versatil.models.decoding.constants import TIMESTEP_KEY, NOISE_KEY, TARGET_DIFFUSION_KEY

        algo = Diffusion(num_train_timesteps=100)
        outputs = algo.forward(mock_decoder, features, actions)

        # Check that outputs contain noise and timesteps
        assert NOISE_KEY in outputs
        assert TIMESTEP_KEY in outputs
        assert TARGET_DIFFUSION_KEY in outputs

        # Check that timesteps are in valid range
        timesteps = outputs[TIMESTEP_KEY]
        assert torch.all(timesteps >= 0)
        assert torch.all(timesteps < 100)

    def test_forward_requires_actions(self, mock_decoder, features):
        """Test that forward pass requires actions."""
        algo = Diffusion()
        with pytest.raises(ValueError, match="requires actions"):
            algo.forward(mock_decoder, features, actions=None)

    @pytest.mark.parametrize("scheduler_type", [
        SchedulerType.DDPM.value,
        SchedulerType.DDIM.value,
    ])
    def test_predict_with_schedulers(self, mock_decoder, features, scheduler_type):
        """Test prediction with different schedulers."""
        algo = Diffusion(
            scheduler_type=scheduler_type,
            num_inference_steps=3,  # Use fewer steps for testing
        )
        outputs = algo.predict(mock_decoder, features)

        # Check that outputs contain expected action keys
        assert POSITION_ACTION_KEY in outputs
        assert ORIENTATION_ACTION_KEY in outputs
        assert GRIPPER_ACTION_KEY in outputs

        # Check output shapes
        batch_size = features["features"].shape[0]
        prediction_horizon = mock_decoder.prediction_horizon

        assert outputs[POSITION_ACTION_KEY].shape == (
            batch_size,
            prediction_horizon,
            mock_decoder.position_dim,
        )
