import pytest
import numpy as np
from unittest.mock import MagicMock

from refactoring.data.action_processor import ActionProcessor
from refactoring.data.constants import (
    POSITION_ACTION_KEY,
    ORIENTATION_ACTION_KEY,
    GRIPPER_ACTION_KEY,
    OrientationRepresentation,
    GripperType,
)


@pytest.fixture
def basic_action_config():
    """Basic action space config with quaternion orientation."""
    config = MagicMock()
    config.predict_in_camera_frame = False
    config.deltas_as_actions = True
    config.denoise_actions = False
    config.has_position = True
    config.has_orientation = True
    config.has_gripper = True
    config.position_dim = 3
    config.orientation_dim = 4
    config.gripper_dim = 1
    config.orientation_repr = OrientationRepresentation.QUATERNION.value
    config.gripper_type = GripperType.BINARY.value
    return config


@pytest.fixture
def roll_action_config():
    """Action config with roll orientation."""
    config = MagicMock()
    config.predict_in_camera_frame = False
    config.deltas_as_actions = True
    config.denoise_actions = False
    config.has_position = True
    config.has_orientation = True
    config.has_gripper = False
    config.position_dim = 3
    config.orientation_dim = 1
    config.gripper_dim = 0
    config.orientation_repr = OrientationRepresentation.ROLL.value
    config.gripper_type = None
    return config


@pytest.fixture
def euler_action_config():
    """Action config with Euler orientation."""
    config = MagicMock()
    config.predict_in_camera_frame = False
    config.deltas_as_actions = True
    config.denoise_actions = False
    config.has_position = True
    config.has_orientation = True
    config.has_gripper = False
    config.position_dim = 3
    config.orientation_dim = 3
    config.gripper_dim = 0
    config.orientation_repr = OrientationRepresentation.EULER.value
    config.gripper_type = None
    return config


@pytest.fixture
def continuous_gripper_config():
    """Action config with continuous gripper."""
    config = MagicMock()
    config.predict_in_camera_frame = False
    config.deltas_as_actions = True
    config.denoise_actions = False
    config.has_position = True
    config.has_orientation = False
    config.has_gripper = True
    config.position_dim = 3
    config.orientation_dim = 0
    config.gripper_dim = 1
    config.orientation_repr = None
    config.gripper_type = GripperType.CONTINUOUS.value
    return config


@pytest.fixture
def positions():
    """Position data (N=5, dim=3) with reproducible values."""
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)


@pytest.fixture
def quaternions():
    """Quaternions (N=5, dim=4) in (w,action_embedding,y,z) format."""
    return np.array([
        [1.0, 0.0, 0.0, 0.0],           # Identity
        [0.707, 0.0, 0.0, 0.707],       # 90° around Z
        [0.0, 0.0, 0.0, 1.0],           # 180° around Z
        [0.707, 0.0, 0.707, 0.0],       # 90° around Y
        [0.707, 0.707, 0.0, 0.0],       # 90° around X
    ], dtype=np.float32)


@pytest.fixture
def euler_angles():
    """Euler angles (N=5, dim=3) in radians."""
    return np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 0.1],
        [0.1, 0.1, 0.1],
    ], dtype=np.float32)


@pytest.fixture
def binary_gripper():
    """Binary gripper states (N=5, dim=1)."""
    return np.array([[0], [1], [1], [0], [1]], dtype=np.float32)


@pytest.fixture
def continuous_gripper():
    """Continuous gripper states (N=5, dim=1)."""
    return np.array([[0.0], [0.5], [1.0], [0.25], [0.75]], dtype=np.float32)


@pytest.fixture
def identity_rotation():
    """3x3 identity rotation matrix."""
    return np.eye(3, dtype=np.float32)


@pytest.fixture
def rotation_90_z():
    """90-degree rotation around Z-axis."""
    return np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ], dtype=np.float32)


@pytest.fixture
def rotation_180_z():
    """180-degree rotation around Z-axis."""
    return np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ], dtype=np.float32)



class TestActionProcessorInitialization:
    """Test ActionProcessor initialization."""

    def test_init_basic(self, basic_action_config):
        """Test basic initialization."""
        processor = ActionProcessor(basic_action_config)

        assert processor.predict_in_camera_frame == False
        assert processor.deltas_as_actions == True
        assert processor.has_position == True
        assert processor.has_orientation == True
        assert processor.has_gripper == True
        assert processor.position_dim == 3
        assert processor.orientation_dim == 4
        assert processor.gripper_dim == 1

    def test_init_no_orientation(self, basic_action_config):
        """Test initialization without orientation."""
        basic_action_config.has_orientation = False
        basic_action_config.orientation_dim = 0

        processor = ActionProcessor(basic_action_config)

        assert processor.has_orientation == False
        assert processor.orientation_dim == 0

    def test_init_no_gripper(self, basic_action_config):
        """Test initialization without gripper."""
        basic_action_config.has_gripper = False
        basic_action_config.gripper_dim = 0

        processor = ActionProcessor(basic_action_config)

        assert processor.has_gripper == False
        assert processor.gripper_dim == 0


class TestComputeActionsFromObservations:
    """Test action computation from observations."""

    def test_compute_position_deltas(self, basic_action_config, positions):
        """Test position-only delta computation."""
        basic_action_config.has_orientation = False
        basic_action_config.has_gripper = False
        basic_action_config.orientation_dim = 0
        basic_action_config.gripper_dim = 0

        processor = ActionProcessor(basic_action_config)

        curr_obs = positions[:-1]
        next_obs = positions[1:]

        actions = processor.compute_actions_from_observations(curr_obs, next_obs)

        assert POSITION_ACTION_KEY in actions
        assert actions[POSITION_ACTION_KEY].shape == (4, 3)

        # Verify first delta: [1,0,0] - [0,0,0] = [1,0,0]
        np.testing.assert_allclose(actions[POSITION_ACTION_KEY][0], [1.0, 0.0, 0.0])
        # Verify second delta: [1,1,0] - [1,0,0] = [0,1,0]
        np.testing.assert_allclose(actions[POSITION_ACTION_KEY][1], [0.0, 1.0, 0.0])

    def test_compute_position_absolute(self, basic_action_config, positions):
        """Test absolute position computation."""
        basic_action_config.has_orientation = False
        basic_action_config.has_gripper = False
        basic_action_config.deltas_as_actions = False

        processor = ActionProcessor(basic_action_config)

        curr_obs = positions[:-1]
        next_obs = positions[1:]

        actions = processor.compute_actions_from_observations(curr_obs, next_obs)

        # Should return next positions directly
        np.testing.assert_allclose(actions[POSITION_ACTION_KEY], next_obs)

    def test_compute_with_quaternion(self, basic_action_config):
        """Test with quaternion orientation."""
        basic_action_config.has_gripper = False
        processor = ActionProcessor(basic_action_config)

        # Observations: position + quaternion
        curr_obs = np.array([
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ], dtype=np.float32)

        next_obs = np.array([
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.707, 0.0, 0.0, 0.707],
        ], dtype=np.float32)

        actions = processor.compute_actions_from_observations(curr_obs, next_obs)

        assert POSITION_ACTION_KEY in actions
        assert ORIENTATION_ACTION_KEY in actions
        assert actions[POSITION_ACTION_KEY].shape == (2, 3)
        assert actions[ORIENTATION_ACTION_KEY].shape == (2, 4)

    def test_compute_with_gripper(self, basic_action_config, binary_gripper):
        """Test with gripper states."""
        processor = ActionProcessor(basic_action_config)

        curr_obs = np.array([[0, 0, 0, 1, 0, 0, 0]], dtype=np.float32)
        next_obs = np.array([[1, 0, 0, 1, 0, 0, 0]], dtype=np.float32)
        curr_gripper = binary_gripper[:1]
        next_gripper = binary_gripper[1:2]

        actions = processor.compute_actions_from_observations(
            curr_obs, next_obs, curr_gripper, next_gripper
        )

        assert GRIPPER_ACTION_KEY in actions
        assert actions[GRIPPER_ACTION_KEY].shape == (1, 1)


class TestComputeGripperActions:
    """Test gripper action computation."""

    def test_binary_gripper_returns_next_state(self, basic_action_config):
        """Binary gripper returns next state."""
        processor = ActionProcessor(basic_action_config)

        curr = np.array([[0], [1]], dtype=np.float32)
        next_gripper = np.array([[1], [0]], dtype=np.float32)

        actions = processor.compute_gripper_actions(curr, next_gripper)

        np.testing.assert_array_equal(actions, next_gripper)

    def test_continuous_gripper_with_deltas(self, continuous_gripper_config):
        """Continuous gripper computes deltas."""
        processor = ActionProcessor(continuous_gripper_config)

        curr = np.array([[0.2], [0.5]], dtype=np.float32)
        next_gripper = np.array([[0.7], [0.3]], dtype=np.float32)

        actions = processor.compute_gripper_actions(curr, next_gripper)

        expected = np.array([[0.5], [-0.2]], dtype=np.float32)
        np.testing.assert_allclose(actions, expected)

    def test_continuous_gripper_without_deltas(self, continuous_gripper_config):
        """Continuous gripper returns next state when deltas disabled."""
        continuous_gripper_config.deltas_as_actions = False
        processor = ActionProcessor(continuous_gripper_config)

        curr = np.array([[0.2]], dtype=np.float32)
        next_gripper = np.array([[0.7]], dtype=np.float32)

        actions = processor.compute_gripper_actions(curr, next_gripper)

        np.testing.assert_allclose(actions, next_gripper)

    def test_unsupported_gripper_type_raises_error(self, basic_action_config):
        """Unsupported gripper type raises ValueError."""
        basic_action_config.gripper_type = "invalid"
        processor = ActionProcessor(basic_action_config)

        with pytest.raises(ValueError, match="Unsupported gripper type"):
            processor.compute_gripper_actions(None, np.array([[1.0]]))


class TestOrientationDeltas:
    """Test orientation delta computation."""

    def test_quaternion_deltas_identity(self, basic_action_config):
        """Quaternion deltas with identical orientations."""
        processor = ActionProcessor(basic_action_config)

        quat = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

        deltas = processor._compute_quaternion_deltas(quat, quat)

        np.testing.assert_allclose(deltas, quat, atol=1e-6)

    def test_euler_deltas_identity(self, euler_action_config):
        """Euler deltas with identical angles."""
        processor = ActionProcessor(euler_action_config)

        euler = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

        deltas = processor._compute_euler_deltas(euler, euler)

        np.testing.assert_allclose(deltas, np.zeros((1, 3)), atol=1e-6)

    def test_euler_deltas_rotation(self, euler_action_config):
        """Euler deltas with simple rotation."""
        processor = ActionProcessor(euler_action_config)

        curr = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        next_euler = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

        deltas = processor._compute_euler_deltas(curr, next_euler)

        assert deltas.shape == (1, 3)
        # For small angles from identity, deltas ≈ next_euler
        np.testing.assert_allclose(deltas, next_euler, atol=0.01)

    def test_roll_deltas(self, roll_action_config):
        """Roll deltas are simple subtraction."""
        processor = ActionProcessor(roll_action_config)

        curr = np.array([[0.0], [0.5]], dtype=np.float32)
        next_roll = np.array([[0.3], [1.0]], dtype=np.float32)

        deltas = processor._compute_roll_deltas(curr, next_roll)

        expected = np.array([[0.3], [0.5]], dtype=np.float32)
        np.testing.assert_allclose(deltas, expected)

    def test_unsupported_orientation_raises_error(self, basic_action_config):
        """Unsupported orientation representation raises ValueError."""
        basic_action_config.orientation_repr = "invalid"
        processor = ActionProcessor(basic_action_config)

        with pytest.raises(ValueError, match="Unsupported orientation representation"):
            processor.compute_orientation_deltas(
                np.zeros((1, 4)), np.zeros((1, 4))
            )


class TestPositionDenoising:
    """Test position denoising."""

    def test_denoising_computes_threshold(self, basic_action_config):
        """Denoising computes 5th percentile threshold."""
        processor = ActionProcessor(basic_action_config)

        # Mix of small and large movements
        curr_pos = np.array([
            [0.0, 0.0, 0.0],
            [0.001, 0.001, 0.001],
            [0.002, 0.002, 0.002],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0],
        ], dtype=np.float32)

        next_pos = curr_pos + np.array([
            [0.001, 0.001, 0.001],
            [0.002, 0.002, 0.002],
            [0.001, 0.001, 0.001],
            [0.3, 0.3, 0.3],
            [0.4, 0.4, 0.4],
        ], dtype=np.float32)

        denoised_next, _ = processor.apply_position_denoising(
            next_pos.copy(), curr_pos.copy()
        )

        assert processor.action_denoising_threshold > 0

    def test_denoising_zeros_small_movements(self, basic_action_config):
        """Small movements are zeroed out."""
        processor = ActionProcessor(basic_action_config)

        curr_pos = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ], dtype=np.float32)

        next_pos = np.array([
            [0.001, 0.001, 0.001],  # Very small
            [1.5, 1.5, 1.5],        # Large
            [2.6, 2.6, 2.6],        # Large
        ], dtype=np.float32)

        denoised_next, _ = processor.apply_position_denoising(
            next_pos.copy(), curr_pos.copy()
        )

        # Small movement should be set to curr_pos
        np.testing.assert_allclose(denoised_next[0], curr_pos[0])

    def test_denoising_all_zeros(self, basic_action_config):
        """All zero movements set threshold to 0."""
        processor = ActionProcessor(basic_action_config)

        pos = np.zeros((5, 3), dtype=np.float32)

        denoised_next, _ = processor.apply_position_denoising(pos, pos)

        assert processor.action_denoising_threshold == 0.0


class TestOrientationDenoising:
    """Test orientation denoising."""

    def test_denoising_computes_threshold(self, basic_action_config, quaternions):
        """Orientation denoising computes threshold."""
        processor = ActionProcessor(basic_action_config)

        curr_ori = quaternions[:-1]
        next_ori = quaternions[1:]

        denoised_next, _ = processor.apply_orientation_denoising(
            next_ori.copy(), curr_ori.copy()
        )

        assert processor.orientation_denoising_threshold >= 0

    def test_denoising_with_roll(self, roll_action_config):
        """Orientation denoising works with roll."""
        processor = ActionProcessor(roll_action_config)

        curr_ori = np.array([[0.0], [0.5]], dtype=np.float32)
        next_ori = np.array([[0.1], [1.0]], dtype=np.float32)

        denoised_next, _ = processor.apply_orientation_denoising(
            next_ori.copy(), curr_ori.copy()
        )

        assert denoised_next.shape == next_ori.shape


class TestOrientationMagnitudes:
    """Test orientation magnitude computation."""

    def test_roll_magnitude_identical(self, roll_action_config):
        """Roll magnitude is zero for identical orientations."""
        processor = ActionProcessor(roll_action_config)

        ori = np.array([[0.5], [1.0]], dtype=np.float32)

        magnitudes = processor._compute_orientation_magnitudes(ori, ori)

        np.testing.assert_allclose(magnitudes, np.zeros(2), atol=1e-6)

    def test_quaternion_magnitude_identity(self, basic_action_config):
        """Quaternion magnitude is zero for identity."""
        processor = ActionProcessor(basic_action_config)

        quat = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

        magnitudes = processor._compute_orientation_magnitudes(quat, quat)

        np.testing.assert_allclose(magnitudes, np.zeros(1), atol=1e-6)


class TestRotateActions:
    """Test action rotation for augmentation."""

    def test_rotate_position_90_degrees(self, basic_action_config, rotation_90_z):
        """Rotate position by 90 degrees."""
        basic_action_config.has_orientation = False
        basic_action_config.has_gripper = False
        processor = ActionProcessor(basic_action_config)

        action_dict = {
            POSITION_ACTION_KEY: np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        }

        rotated = processor.rotate_actions(action_dict, rotation_90_z)

        # (1,0,0) rotated 90° → (0,1,0)
        expected = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(rotated[POSITION_ACTION_KEY], expected, atol=1e-6)

    def test_rotate_gripper_unchanged(self, basic_action_config, rotation_90_z):
        """Gripper actions are not rotated."""
        processor = ActionProcessor(basic_action_config)

        action_dict = {
            POSITION_ACTION_KEY: np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            GRIPPER_ACTION_KEY: np.array([[1.0]], dtype=np.float32)
        }

        rotated = processor.rotate_actions(action_dict, rotation_90_z)

        np.testing.assert_array_equal(
            rotated[GRIPPER_ACTION_KEY],
            action_dict[GRIPPER_ACTION_KEY]
        )

    def test_rotate_position_identity(self, basic_action_config, identity_rotation):
        """Identity rotation doesn't change position."""
        basic_action_config.has_orientation = False
        processor = ActionProcessor(basic_action_config)

        action_dict = {
            POSITION_ACTION_KEY: np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        }

        rotated = processor.rotate_actions(action_dict, identity_rotation)

        np.testing.assert_allclose(
            rotated[POSITION_ACTION_KEY],
            action_dict[POSITION_ACTION_KEY]
        )


class TestIntegration:
    """Integration tests."""

    def test_full_pipeline_with_denoising(self, basic_action_config):
        """Test complete pipeline with denoising enabled."""
        basic_action_config.denoise_actions = True
        processor = ActionProcessor(basic_action_config)

        curr_obs = np.array([
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ], dtype=np.float32)

        next_obs = np.array([
            [0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0],
            [1.6, 1.6, 1.6, 0.707, 0.0, 0.0, 0.707],
        ], dtype=np.float32)

        curr_gripper = np.array([[0], [1]], dtype=np.float32)
        next_gripper = np.array([[1], [0]], dtype=np.float32)

        actions = processor.compute_actions_from_observations(
            curr_obs, next_obs, curr_gripper, next_gripper
        )

        assert POSITION_ACTION_KEY in actions
        assert ORIENTATION_ACTION_KEY in actions
        assert GRIPPER_ACTION_KEY in actions

    @pytest.mark.parametrize("ori_repr,ori_dim", [
        (OrientationRepresentation.QUATERNION.value, 4),
        (OrientationRepresentation.EULER.value, 3),
        (OrientationRepresentation.ROLL.value, 1),
    ])
    def test_all_orientation_representations(self, basic_action_config, ori_repr, ori_dim):
        """Test all orientation representations."""
        basic_action_config.orientation_repr = ori_repr
        basic_action_config.orientation_dim = ori_dim
        basic_action_config.has_gripper = False
        processor = ActionProcessor(basic_action_config)

        pos = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        ori = np.random.rand(2, ori_dim).astype(np.float32)

        if ori_repr == OrientationRepresentation.QUATERNION.value:
            ori = ori / np.linalg.norm(ori, axis=1, keepdims=True)

        obs = np.concatenate([pos, ori], axis=1)

        actions = processor.compute_actions_from_observations(obs[:1], obs[1:])

        assert POSITION_ACTION_KEY in actions
        assert ORIENTATION_ACTION_KEY in actions
        assert actions[ORIENTATION_ACTION_KEY].shape == (1, ori_dim)