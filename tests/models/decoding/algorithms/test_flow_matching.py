"""Tests for action-decoding algorithms (BC, Diffusion, Flow Matching)."""
import pytest
import torch
from typing import Dict, Optional

from versatil.models.decoding.algorithm.flow_matching import FlowMatching
from versatil.models.decoding.decoders.base import ActionDecoder, DecoderInput
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.data.constants import (
    Cameras,
    GripperType,
    OrientationRepresentation,
    ProprioceptiveType,
)
from versatil.models.decoding.constants import (
    ODESolver,
)


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
            action_heads[ProprioceptiveType.POSITION.value] = ActionHead(
                input_dim=feature_dim,
                output_dim=action_space.position_dim,
                blocks=[],
            )
        if action_space.has_orientation:
            action_heads[ProprioceptiveType.ORIENTATION.value] = ActionHead(
                input_dim=feature_dim,
                output_dim=action_space.orientation_dim,
                blocks=[],
            )
        if action_space.has_gripper:
            action_heads[ProprioceptiveType.GRIPPER.value] = ActionHead(
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
            predictions[ProprioceptiveType.POSITION.value] = torch.randn(
                batch_size,
                self.prediction_horizon,
                self.position_dim,
                device=self.device,
            )

        if self.use_orientation_actions:
            predictions[ProprioceptiveType.ORIENTATION.value] = torch.randn(
                batch_size,
                self.prediction_horizon,
                self.orientation_dim,
                device=self.device,
            )

        if self.use_gripper_actions:
            predictions[ProprioceptiveType.GRIPPER.value] = torch.randn(
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
        actions_dict[ProprioceptiveType.POSITION.value] = torch.randn(
            batch_size, prediction_horizon, action_space.position_dim, device=device
        )

    if action_space.has_orientation:
        actions_dict[ProprioceptiveType.ORIENTATION.value] = torch.randn(
            batch_size, prediction_horizon, action_space.orientation_dim, device=device
        )

    if action_space.has_gripper:
        actions_dict[ProprioceptiveType.GRIPPER.value] = torch.randn(
            batch_size, prediction_horizon, action_space.gripper_dim, device=device
        )

    return actions_dict


@pytest.mark.unit
class TestFlowMatching:
    """Tests for Flow Matching algorithm."""

    @pytest.mark.parametrize("ode_solver", [
        ODESolver.EULER.value,
        ODESolver.HEUN.value,
        ODESolver.RK4.value,
    ])
    def test_instantiation_solvers(self, ode_solver):
        """Test that Flow Matching can be instantiated with all ODE solvers."""
        algo = FlowMatching(ode_solver=ode_solver)
        assert algo is not None
        assert algo.ode_solver == ode_solver

    def test_invalid_ode_solver(self):
        """Test that invalid ODE solver raises error."""
        with pytest.raises(ValueError, match="Unknown ODE solver"):
            FlowMatching(ode_solver="invalid")

    @pytest.mark.parametrize("sigma", [0.0, 0.001, 0.01, 0.1])
    def test_sigma_values(self, sigma):
        """Test different sigma values for conditional flow matching."""
        algo = FlowMatching(sigma=sigma)
        assert algo is not None
        assert algo.flow_matcher.sigma == sigma

    @pytest.mark.parametrize("num_inference_steps", [5, 10, 20, 50])
    def test_inference_steps(self, num_inference_steps):
        """Test different numbers of inference steps."""
        algo = FlowMatching(num_inference_steps=num_inference_steps)
        assert algo.num_inference_steps == num_inference_steps

    def test_forward_pass(self, mock_decoder, features, actions):
        """Test forward pass during training."""
        algo = FlowMatching(sigma=0.0)
        outputs = algo.forward(mock_decoder, features, actions)

        # Check that outputs contain time and target velocity
        assert "time" in outputs
        assert "target_velocity" in outputs

        # Check that time is in [0, 1]
        time = outputs["time"]
        assert torch.all(time >= 0.0)
        assert torch.all(time <= 1.0)

    @pytest.mark.parametrize("sigma", [0.0, 0.01])
    def test_forward_with_sigma(self, mock_decoder, features, actions, sigma):
        """Test forward pass with different sigma values."""
        algo = FlowMatching(sigma=sigma)
        outputs = algo.forward(mock_decoder, features, actions)

        # Check that outputs contain required keys
        assert "time" in outputs
        assert "target_velocity" in outputs

    def test_forward_requires_actions(self, mock_decoder, features):
        """Test that forward pass requires actions."""
        algo = FlowMatching()
        with pytest.raises(ValueError, match="requires actions"):
            algo.forward(mock_decoder, features, actions=None)

    @pytest.mark.parametrize("ode_solver,num_steps", [
        (ODESolver.EULER.value, 5),
        (ODESolver.HEUN.value, 3),
        (ODESolver.RK4.value, 2),
    ])
    def test_predict_with_solvers(self, mock_decoder, features, ode_solver, num_steps):
        """Test prediction with all ODE solvers."""
        algo = FlowMatching(
            ode_solver=ode_solver,
            num_inference_steps=num_steps,
        )
        outputs = algo.predict(mock_decoder, features)

        # Check that outputs contain expected action keys
        assert ProprioceptiveType.POSITION.value in outputs
        assert ProprioceptiveType.ORIENTATION.value in outputs
        assert ProprioceptiveType.GRIPPER.value in outputs

        # Check output shapes
        batch_size = features["features"].shape[0]
        prediction_horizon = mock_decoder.prediction_horizon

        assert outputs[ProprioceptiveType.POSITION.value].shape == (
            batch_size,
            prediction_horizon,
            mock_decoder.position_dim,
        )

    @pytest.mark.parametrize("ode_solver", [
        ODESolver.EULER.value,
        ODESolver.HEUN.value,
        ODESolver.RK4.value,
    ])
    def test_predict_all_solvers_detailed(self, mock_decoder, features, ode_solver):
        """Test prediction with each solver individually with more detail."""
        algo = FlowMatching(
            ode_solver=ode_solver,
            num_inference_steps=5,
        )
        outputs = algo.predict(mock_decoder, features)

        # Check all action modalities
        assert ProprioceptiveType.POSITION.value in outputs
        assert ProprioceptiveType.ORIENTATION.value in outputs
        assert ProprioceptiveType.GRIPPER.value in outputs

        # Verify no NaN or Inf values
        for key, value in outputs.items():
            assert not torch.isnan(value).any(), f"NaN found in {key} for solver {ode_solver}"
            assert not torch.isinf(value).any(), f"Inf found in {key} for solver {ode_solver}"
