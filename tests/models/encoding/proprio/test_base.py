import pytest
import torch
import torch.nn as nn

from refactoring.models.encoding.encoders.proprioceptive import ProprioceptiveEncoder
from refactoring.models.layers.activation import ActivationFunction
from refactoring.models.encoding.encoders.constants import EncoderOutputKeys
from refactoring.data.constants import (
    PROPRIO_OBS_ROBOT_FRAME_KEY,
    PROPRIO_OBS_CAMERA_FRAME_KEY,
)


@pytest.fixture
def batch_size():
    """Standard batch size for testing."""
    return 8


@pytest.fixture
def temporal_length():
    """Standard temporal length for testing."""
    return 5


@pytest.fixture
def robot_frame_dim():
    """Typical robot frame proprioceptive dimension (position + quaternion)."""
    return 7


@pytest.fixture
def camera_frame_dim():
    """Typical camera frame proprioceptive dimension (position + quaternion)."""
    return 7


@pytest.fixture
def gripper_dim():
    """Gripper state dimension."""
    return 1


@pytest.fixture
def output_dim():
    """Standard output dimension for encoded features."""
    return 128


@pytest.fixture
def robot_frame_input_2d(batch_size, robot_frame_dim):
    """2D robot frame input (batch_size, features)."""
    return torch.randn(batch_size, robot_frame_dim)


@pytest.fixture
def robot_frame_input_3d(batch_size, temporal_length, robot_frame_dim):
    """3D robot frame input (batch_size, time_steps, features)."""
    return torch.randn(batch_size, temporal_length, robot_frame_dim)


@pytest.fixture
def camera_frame_input_2d(batch_size, camera_frame_dim):
    """2D camera frame input (batch_size, features)."""
    return torch.randn(batch_size, camera_frame_dim)


@pytest.fixture
def camera_frame_input_3d(batch_size, temporal_length, camera_frame_dim):
    """3D camera frame input (batch_size, time_steps, features)."""
    return torch.randn(batch_size, temporal_length, camera_frame_dim)


@pytest.fixture
def single_input_dict_2d(robot_frame_input_2d):
    """Input dictionary with only robot frame (2D)."""
    return {PROPRIO_OBS_ROBOT_FRAME_KEY: robot_frame_input_2d}


@pytest.fixture
def single_input_dict_3d(robot_frame_input_3d):
    """Input dictionary with only robot frame (3D temporal)."""
    return {PROPRIO_OBS_ROBOT_FRAME_KEY: robot_frame_input_3d}


@pytest.fixture
def dual_input_dict_2d(robot_frame_input_2d, camera_frame_input_2d):
    """Input dictionary with both robot and camera frames (2D)."""
    return {
        PROPRIO_OBS_ROBOT_FRAME_KEY: robot_frame_input_2d,
        PROPRIO_OBS_CAMERA_FRAME_KEY: camera_frame_input_2d,
    }


@pytest.fixture
def dual_input_dict_3d(robot_frame_input_3d, camera_frame_input_3d):
    """Input dictionary with both robot and camera frames (3D temporal)."""
    return {
        PROPRIO_OBS_ROBOT_FRAME_KEY: robot_frame_input_3d,
        PROPRIO_OBS_CAMERA_FRAME_KEY: camera_frame_input_3d,
    }


@pytest.mark.unit
class TestProprioceptiveEncoderInit:
    """Test ProprioceptiveEncoder initialization."""

    def test_init_single_input_key(self, output_dim):
        """Test initialization with single input key."""
        encoder = ProprioceptiveEncoder(
            input_keys=PROPRIO_OBS_ROBOT_FRAME_KEY,
            output_dim=output_dim,
        )
        assert encoder.output_dim == output_dim
        assert encoder.input_specification.keys == [PROPRIO_OBS_ROBOT_FRAME_KEY]
        assert encoder.network is None

    def test_init_multiple_input_keys(self, output_dim):
        """Test initialization with multiple input keys."""
        encoder = ProprioceptiveEncoder(
            input_keys=[PROPRIO_OBS_ROBOT_FRAME_KEY, PROPRIO_OBS_CAMERA_FRAME_KEY],
            output_dim=output_dim,
        )
        assert encoder.input_specification.keys == [PROPRIO_OBS_ROBOT_FRAME_KEY, PROPRIO_OBS_CAMERA_FRAME_KEY]

    def test_init_with_hidden_dims(self, output_dim):
        """Test initialization with hidden layers."""
        hidden_dims = [256, 128]
        encoder = ProprioceptiveEncoder(
            input_keys=PROPRIO_OBS_ROBOT_FRAME_KEY,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
        )
        assert encoder.hidden_dims == hidden_dims

    def test_init_with_activation(self, output_dim):
        """Test initialization with custom activation."""
        encoder = ProprioceptiveEncoder(
            input_keys=PROPRIO_OBS_ROBOT_FRAME_KEY,
            output_dim=output_dim,
            activation=ActivationFunction.GELU.value,
        )
        assert encoder.activation_fn == nn.GELU

    def test_init_with_dropout(self, output_dim):
        """Test initialization with dropout."""
        dropout = 0.2
        encoder = ProprioceptiveEncoder(
            input_keys=PROPRIO_OBS_ROBOT_FRAME_KEY,
            output_dim=output_dim,
            dropout=dropout,
        )
        assert encoder.dropout == dropout

    def test_init_frozen(self, output_dim):
        """Test initialization with frozen weights."""
        encoder = ProprioceptiveEncoder(
            input_keys=PROPRIO_OBS_ROBOT_FRAME_KEY,
            output_dim=output_dim,
            frozen=True,
        )
        assert encoder.frozen is True


@pytest.mark.unit
class TestProprioceptiveEncoderForward2D:
    """Test ProprioceptiveEncoder forward pass with 2D inputs."""

    def test_forward_single_input_no_hidden(self, single_input_dict_2d, output_dim, batch_size, robot_frame_dim):
        """Test forward with single input and no hidden layers."""
        encoder = ProprioceptiveEncoder(
            input_keys=PROPRIO_OBS_ROBOT_FRAME_KEY,
            output_dim=output_dim,
            hidden_dims=None,
        )
        output = encoder.forward(single_input_dict_2d, is_train=True)

        assert EncoderOutputKeys.PROPRIOCEPTIVE.value in output
        assert output[EncoderOutputKeys.PROPRIOCEPTIVE.value].shape == (batch_size, output_dim)
        assert encoder.network is not None

    def test_forward_single_input_with_hidden(self, single_input_dict_2d, output_dim, batch_size):
        """Test forward with single input and hidden layers."""
        hidden_dims = [256, 128]
        encoder = ProprioceptiveEncoder(
            input_keys=PROPRIO_OBS_ROBOT_FRAME_KEY,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
        )
        output = encoder.forward(single_input_dict_2d, is_train=True)

        assert EncoderOutputKeys.PROPRIOCEPTIVE.value in output
        assert output[EncoderOutputKeys.PROPRIOCEPTIVE.value].shape == (batch_size, output_dim)

    def test_forward_dual_input(self, dual_input_dict_2d, output_dim, batch_size):
        """Test forward with both robot and camera frame inputs."""
        encoder = ProprioceptiveEncoder(
            input_keys=[PROPRIO_OBS_ROBOT_FRAME_KEY, PROPRIO_OBS_CAMERA_FRAME_KEY],
            output_dim=output_dim,
            hidden_dims=[256],
        )
        output = encoder.forward(dual_input_dict_2d, is_train=True)

        assert EncoderOutputKeys.PROPRIOCEPTIVE.value in output
        assert output[EncoderOutputKeys.PROPRIOCEPTIVE.value].shape == (batch_size, output_dim)

    def test_forward_empty_hidden_dims(self, single_input_dict_2d, output_dim, batch_size):
        """Test forward with empty hidden_dims list (should create linear layer)."""
        encoder = ProprioceptiveEncoder(
            input_keys=PROPRIO_OBS_ROBOT_FRAME_KEY,
            output_dim=output_dim,
            hidden_dims=[],
        )
        output = encoder.forward(single_input_dict_2d, is_train=True)

        assert EncoderOutputKeys.PROPRIOCEPTIVE.value in output
        assert output[EncoderOutputKeys.PROPRIOCEPTIVE.value].shape == (batch_size, output_dim)


@pytest.mark.unit
class TestProprioceptiveEncoderForward3D:
    """Test ProprioceptiveEncoder forward pass with 3D temporal inputs."""

    def test_forward_single_input_temporal(self, single_input_dict_3d, output_dim, batch_size, temporal_length):
        """Test forward with temporal input (batch, time, features)."""
        encoder = ProprioceptiveEncoder(
            input_keys=PROPRIO_OBS_ROBOT_FRAME_KEY,
            output_dim=output_dim,
            hidden_dims=[256],
        )
        output = encoder.forward(single_input_dict_3d, is_train=True)

        assert EncoderOutputKeys.PROPRIOCEPTIVE.value in output
        assert output[EncoderOutputKeys.PROPRIOCEPTIVE.value].shape == (batch_size, temporal_length, output_dim)

    def test_forward_dual_input_temporal(self, dual_input_dict_3d, output_dim, batch_size, temporal_length):
        """Test forward with temporal dual inputs."""
        encoder = ProprioceptiveEncoder(
            input_keys=[PROPRIO_OBS_ROBOT_FRAME_KEY, PROPRIO_OBS_CAMERA_FRAME_KEY],
            output_dim=output_dim,
            hidden_dims=[256, 128],
        )
        output = encoder.forward(dual_input_dict_3d, is_train=True)

        assert EncoderOutputKeys.PROPRIOCEPTIVE.value in output
        assert output[EncoderOutputKeys.PROPRIOCEPTIVE.value].shape == (batch_size, temporal_length, output_dim)

    def test_forward_no_hidden_temporal(self, single_input_dict_3d, output_dim, batch_size, temporal_length):
        """Test forward with temporal input and no hidden layers."""
        encoder = ProprioceptiveEncoder(
            input_keys=PROPRIO_OBS_ROBOT_FRAME_KEY,
            output_dim=output_dim,
            hidden_dims=None,
        )
        output = encoder.forward(single_input_dict_3d, is_train=True)

        assert EncoderOutputKeys.PROPRIOCEPTIVE.value in output
        assert output[EncoderOutputKeys.PROPRIOCEPTIVE.value].shape == (batch_size, temporal_length, output_dim)


@pytest.mark.unit
class TestProprioceptiveEncoderOutputDimensions:
    """Test output dimension methods."""

    def test_get_output_dims(self, output_dim):
        """Test get_output_dims method."""
        encoder = ProprioceptiveEncoder(
            input_keys=PROPRIO_OBS_ROBOT_FRAME_KEY,
            output_dim=output_dim,
        )

        output_dims = encoder.get_output_dims()
        assert EncoderOutputKeys.PROPRIOCEPTIVE.value in output_dims
        assert output_dims[EncoderOutputKeys.PROPRIOCEPTIVE.value] == output_dim

    def test_get_output_specification(self, output_dim):
        """Test get_output_specification method."""
        encoder = ProprioceptiveEncoder(
            input_keys=PROPRIO_OBS_ROBOT_FRAME_KEY,
            output_dim=output_dim,
        )

        spec = encoder.get_output_specification()
        assert EncoderOutputKeys.PROPRIOCEPTIVE.value in spec.features
        assert spec.dimensions[EncoderOutputKeys.PROPRIOCEPTIVE.value] == output_dim


@pytest.mark.unit
class TestProprioceptiveEncoderNetworkBuilding:
    """Test network building."""

    def test_network_builds_lazily(self, output_dim):
        """Test that network is built lazily on first forward."""
        encoder = ProprioceptiveEncoder(
            input_keys=PROPRIO_OBS_ROBOT_FRAME_KEY,
            output_dim=output_dim,
        )

        assert encoder.network is None

        inputs = {PROPRIO_OBS_ROBOT_FRAME_KEY: torch.randn(2, 7)}
        _ = encoder.forward(inputs, is_train=True)

        assert encoder.network is not None

    def test_network_input_dim_inference(self, batch_size, robot_frame_dim, output_dim):
        """Test that network correctly infers input dimension."""
        encoder = ProprioceptiveEncoder(
            input_keys=PROPRIO_OBS_ROBOT_FRAME_KEY,
            output_dim=output_dim,
            hidden_dims=[128],
        )

        inputs = {PROPRIO_OBS_ROBOT_FRAME_KEY: torch.randn(batch_size, robot_frame_dim)}
        _ = encoder.forward(inputs, is_train=True)

        first_layer = encoder.network.layers[0]
        assert isinstance(first_layer, nn.Linear)
        assert first_layer.in_features == robot_frame_dim

    def test_network_concatenates_multiple_inputs(self, batch_size, robot_frame_dim, camera_frame_dim, output_dim):
        """Test that network correctly concatenates multiple input keys."""
        encoder = ProprioceptiveEncoder(
            input_keys=[PROPRIO_OBS_ROBOT_FRAME_KEY, PROPRIO_OBS_CAMERA_FRAME_KEY],
            output_dim=output_dim,
            hidden_dims=[128],
        )

        inputs = {
            PROPRIO_OBS_ROBOT_FRAME_KEY: torch.randn(batch_size, robot_frame_dim),
            PROPRIO_OBS_CAMERA_FRAME_KEY: torch.randn(batch_size, camera_frame_dim),
        }
        _ = encoder.forward(inputs, is_train=True)

        first_layer = encoder.network.layers[0]
        expected_dim = robot_frame_dim + camera_frame_dim
        assert first_layer.in_features == expected_dim


@pytest.mark.unit
class TestProprioceptiveEncoderEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_feature_input(self, batch_size, output_dim):
        """Test with single feature input (e.g., gripper state)."""
        single_feature_input = torch.randn(batch_size, 1)
        inputs = {PROPRIO_OBS_ROBOT_FRAME_KEY: single_feature_input}

        encoder = ProprioceptiveEncoder(
            input_keys=PROPRIO_OBS_ROBOT_FRAME_KEY,
            output_dim=output_dim,
            hidden_dims=[64],
        )
        output = encoder.forward(inputs, is_train=True)

        assert output[EncoderOutputKeys.PROPRIOCEPTIVE.value].shape == (batch_size, output_dim)

    def test_high_dimensional_input(self, batch_size, output_dim):
        """Test with high-dimensional input."""
        high_dim_input = torch.randn(batch_size, 50)
        inputs = {PROPRIO_OBS_ROBOT_FRAME_KEY: high_dim_input}

        encoder = ProprioceptiveEncoder(
            input_keys=PROPRIO_OBS_ROBOT_FRAME_KEY,
            output_dim=output_dim,
            hidden_dims=[256, 128],
        )
        output = encoder.forward(inputs, is_train=True)

        assert output[EncoderOutputKeys.PROPRIOCEPTIVE.value].shape == (batch_size, output_dim)

    def test_batch_size_one(self, output_dim, robot_frame_dim):
        """Test with batch size of 1."""
        inputs = {PROPRIO_OBS_ROBOT_FRAME_KEY: torch.randn(1, robot_frame_dim)}

        encoder = ProprioceptiveEncoder(
            input_keys=PROPRIO_OBS_ROBOT_FRAME_KEY,
            output_dim=output_dim,
        )
        output = encoder.forward(inputs, is_train=True)

        assert output[EncoderOutputKeys.PROPRIOCEPTIVE.value].shape == (1, output_dim)

    def test_large_batch_size(self, output_dim, robot_frame_dim):
        """Test with large batch size."""
        large_batch = 128
        inputs = {PROPRIO_OBS_ROBOT_FRAME_KEY: torch.randn(large_batch, robot_frame_dim)}

        encoder = ProprioceptiveEncoder(
            input_keys=PROPRIO_OBS_ROBOT_FRAME_KEY,
            output_dim=output_dim,
            hidden_dims=[256],
        )
        output = encoder.forward(inputs, is_train=True)

        assert output[EncoderOutputKeys.PROPRIOCEPTIVE.value].shape == (large_batch, output_dim)


@pytest.mark.unit
class TestProprioceptiveEncoderGradientFlow:
    """Test gradient flow through encoder."""

    def test_gradient_flow_unfrozen(self, single_input_dict_2d, output_dim):
        """Test that gradients flow through unfrozen encoder."""
        encoder = ProprioceptiveEncoder(
            input_keys=PROPRIO_OBS_ROBOT_FRAME_KEY,
            output_dim=output_dim,
            hidden_dims=[128],
            frozen=False,
        )

        inputs_with_grad = {k: v.requires_grad_(True) for k, v in single_input_dict_2d.items()}
        output = encoder.forward(inputs_with_grad, is_train=True)

        loss = output[EncoderOutputKeys.PROPRIOCEPTIVE.value].sum()
        loss.backward()

        for param in encoder.network.parameters():
            assert param.grad is not None

    def test_no_gradient_flow_frozen(self, single_input_dict_2d, output_dim):
        """Test that gradients don't flow through frozen encoder."""
        encoder = ProprioceptiveEncoder(
            input_keys=PROPRIO_OBS_ROBOT_FRAME_KEY,
            output_dim=output_dim,
            hidden_dims=[128],
            frozen=True,
        )

        inputs_with_grad = {k: v.requires_grad_(True) for k, v in single_input_dict_2d.items()}
        output = encoder.forward(inputs_with_grad, is_train=True)

        loss = output[EncoderOutputKeys.PROPRIOCEPTIVE.value].sum()
        loss.backward()

        for param in encoder.network.parameters():
            assert param.grad is None


@pytest.mark.unit
class TestProprioceptiveEncoderRealistic:
    """Test with realistic robot configurations."""

    def test_surgical_robot_single_arm(self, batch_size, output_dim):
        """Test with realistic surgical robot configuration (single arm)."""
        proprio_dim = 8
        inputs = {PROPRIO_OBS_ROBOT_FRAME_KEY: torch.randn(batch_size, proprio_dim)}

        encoder = ProprioceptiveEncoder(
            input_keys=PROPRIO_OBS_ROBOT_FRAME_KEY,
            output_dim=output_dim,
            hidden_dims=[256, 128],
            activation=ActivationFunction.RELU.value,
        )
        output = encoder.forward(inputs, is_train=True)

        assert output[EncoderOutputKeys.PROPRIOCEPTIVE.value].shape == (batch_size, output_dim)

    def test_dual_arm_robot(self, batch_size, output_dim):
        """Test with dual-arm robot configuration."""
        left_arm_dim = 7
        right_arm_dim = 7

        inputs = {
            PROPRIO_OBS_ROBOT_FRAME_KEY: torch.randn(batch_size, left_arm_dim),
            PROPRIO_OBS_CAMERA_FRAME_KEY: torch.randn(batch_size, right_arm_dim),
        }

        encoder = ProprioceptiveEncoder(
            input_keys=[PROPRIO_OBS_ROBOT_FRAME_KEY, PROPRIO_OBS_CAMERA_FRAME_KEY],
            output_dim=output_dim,
            hidden_dims=[512, 256],
        )
        output = encoder.forward(inputs, is_train=True)

        assert output[EncoderOutputKeys.PROPRIOCEPTIVE.value].shape == (batch_size, output_dim)

    def test_temporal_robot_trajectory(self, batch_size, output_dim):
        """Test with temporal trajectory (history of robot states)."""
        trajectory_length = 5
        proprio_dim = 7

        inputs = {
            PROPRIO_OBS_ROBOT_FRAME_KEY: torch.randn(batch_size, trajectory_length, proprio_dim)
        }

        encoder = ProprioceptiveEncoder(
            input_keys=PROPRIO_OBS_ROBOT_FRAME_KEY,
            output_dim=output_dim,
            hidden_dims=[256],
        )
        output = encoder.forward(inputs, is_train=True)

        assert output[EncoderOutputKeys.PROPRIOCEPTIVE.value].shape == (batch_size, trajectory_length, output_dim)