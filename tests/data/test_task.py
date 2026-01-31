"""Tests for task space module (ActionSpace, ObservationSpace, TaskSpace)."""
import pytest
from unittest.mock import MagicMock
from versatil.data.constants import (
    Cameras,
    GripperType,
    ObsKey,
    OrientationRepresentation,
    ProprioKey,
)
from versatil.data.task import ActionSpace, ObservationSpace, TaskSpace


@pytest.fixture
def action_space_factory():
    """Factory for creating ActionSpace instances."""
    def factory(**kwargs):
        defaults = {
            'has_position': True,
            'position_dim': 3,
            'has_orientation': False,
            'orientation_dim': 0,
            'orientation_repr': OrientationRepresentation.ROLL.value,
            'has_gripper': True,
            'gripper_type': GripperType.BINARY.value,
            'gripper_dim': 1,
            'use_gripper_class_weights': False,
            'predict_in_camera_frame': True,
            'deltas_as_actions': False,
            'denoise_actions': True,
            'predict_task_phases': False,
            'number_of_phases': 5,
        }
        defaults.update(kwargs)
        return ActionSpace(**defaults)
    return factory


@pytest.fixture
def observation_space_factory():
    """Factory for creating ObservationSpace instances."""
    def factory(**kwargs):
        defaults = {
            'use_proprioceptive_data': False,
            'use_proprio_base_frame': False,
            'use_proprio_camera_frame': False,
            'use_gripper_state': False,
            'gripper_type': GripperType.BINARY.value,
            'camera_keys': None,
            'use_language': False,
            'custom_obs_keys_to_column_names': None,
        }
        defaults.update(kwargs)
        return ObservationSpace(**defaults)
    return factory


@pytest.fixture
def task_space_factory(action_space_factory, observation_space_factory):
    """Factory for creating TaskSpace instances."""
    def factory(**kwargs):
        defaults = {
            'dataset_schema': MagicMock(),
            'dataloader': MagicMock(),
            'action_space': action_space_factory(),
            'observation_space': observation_space_factory(),
            'observation_horizon': 1,
            'prediction_horizon': 16,
        }
        defaults.update(kwargs)
        return TaskSpace(**defaults)
    return factory


@pytest.mark.unit
class TestActionSpace:
    """Test ActionSpace initialization and methods."""

    def test_init_defaults(self, action_space_factory):
        """Test ActionSpace initialization with default values."""
        action_space = action_space_factory()

        assert action_space.has_position is True
        assert action_space.position_dim == 3
        assert action_space.has_orientation is False
        assert action_space.orientation_dim == 0
        assert action_space.has_gripper is True
        assert action_space.gripper_dim == 1
        assert action_space.predict_in_camera_frame is True
        assert action_space.deltas_as_actions is False
        assert action_space.predict_task_phases is False
        assert action_space.number_of_phases == 5

    def test_init_custom_values(self, action_space_factory):
        """Test ActionSpace initialization with custom values."""
        action_space = action_space_factory(
            has_orientation=True,
            orientation_dim=3,
            orientation_repr=OrientationRepresentation.EULER.value,
            predict_in_camera_frame=False,
            task_has_phases=True,
        )

        assert action_space.has_orientation is True
        assert action_space.orientation_dim == 3
        assert action_space.orientation_repr == OrientationRepresentation.EULER.value
        assert action_space.predict_in_camera_frame is False
        assert action_space.predict_task_phases is True

    @pytest.mark.parametrize("has_pos,pos_dim,has_ori,ori_dim,has_grip,grip_dim,has_phases,n_phases,expected", [
        (True, 3, False, 0, True, 1, False, 5, 4),
        (True, 3, True, 3, True, 1, False, 5, 7),
        (True, 3, True, 4, True, 1, False, 5, 8),
        (True, 3, False, 0, False, 0, False, 5, 3),
        (False, 0, True, 3, True, 1, False, 5, 4),
        (True, 3, True, 3, True, 1, True, 5, 12),
        (True, 3, True, 3, True, 1, True, 3, 10),
        (False, 0, False, 0, False, 0, True, 7, 7),
    ])
    def test_get_total_action_dim(self, action_space_factory, has_pos, pos_dim, has_ori, ori_dim,
                                   has_grip, grip_dim, has_phases, n_phases, expected):
        """Test total action dimension calculation."""
        action_space = action_space_factory(
            has_position=has_pos,
            position_dim=pos_dim,
            has_orientation=has_ori,
            orientation_dim=ori_dim,
            has_gripper=has_grip,
            gripper_dim=grip_dim,
            task_has_phases=has_phases,
            number_of_phases=n_phases,
        )

        assert action_space.get_total_action_dim() == expected


    def test_get_required_zarr_keys_camera_frame(self, action_space_factory):
        """Test required zarr keys for camera frame prediction."""
        action_space = action_space_factory(
            has_position=True,
            has_gripper=True,
            predict_in_camera_frame=True,
        )

        keys = action_space.get_required_zarr_keys()

        assert ProprioKey.CAMERA_FRAME_CARTESIAN_TIP_POS.value in keys
        assert ProprioKey.GRIPPER_STATE.value in keys
        assert ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value not in keys

    def test_get_required_zarr_keys_robot_frame(self, action_space_factory):
        """Test required zarr keys for robot frame prediction."""
        action_space = action_space_factory(
            has_position=True,
            has_gripper=True,
            predict_in_camera_frame=False,
        )

        keys = action_space.get_required_zarr_keys()

        assert ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value in keys
        assert ProprioKey.GRIPPER_STATE.value in keys
        assert ProprioKey.CAMERA_FRAME_CARTESIAN_TIP_POS.value not in keys

    def test_get_required_zarr_keys_with_phases(self, action_space_factory):
        """Test required zarr keys include phase labels."""
        action_space = action_space_factory(
            has_position=True,
            task_has_phases=True,
        )

        keys = action_space.get_required_zarr_keys()

        assert ObsKey.PHASE_LABEL.value in keys

    def test_get_required_zarr_keys_no_position_no_gripper(self, action_space_factory):
        """Test required zarr keys when no position or gripper."""
        action_space = action_space_factory(
            has_position=False,
            has_gripper=False,
        )

        keys = action_space.get_required_zarr_keys()

        assert len(keys) == 0


@pytest.mark.unit
class TestObservationSpace:
    """Test ObservationSpace initialization and methods."""

    def test_init_defaults(self, observation_space_factory):
        """Test ObservationSpace initialization with default values."""
        obs_space = observation_space_factory()

        assert obs_space.use_proprioceptive_data is False
        assert obs_space.use_proprio_base_frame is False
        assert obs_space.use_proprio_camera_frame is False
        assert obs_space.use_gripper_state is False
        assert obs_space.camera_keys == []
        assert obs_space.use_language is False
        assert obs_space.custom_obs_keys == []

    def test_init_with_cameras(self, observation_space_factory):
        """Test ObservationSpace initialization with camera keys."""
        obs_space = observation_space_factory(
            camera_keys=[Cameras.LEFT.value, Cameras.RIGHT.value]
        )

        assert len(obs_space.camera_keys) == 2
        assert Cameras.LEFT.value in obs_space.camera_keys
        assert Cameras.RIGHT.value in obs_space.camera_keys

    def test_init_with_language(self, observation_space_factory):
        """Test ObservationSpace initialization with language."""
        obs_space = observation_space_factory(use_language=True)

        assert obs_space.use_language is True

    def test_get_required_zarr_keys_cameras_only(self, observation_space_factory):
        """Test required zarr keys with only cameras."""
        obs_space = observation_space_factory(
            camera_keys=[Cameras.LEFT.value, Cameras.DEPTH.value]
        )

        keys = obs_space.get_required_zarr_keys()

        assert Cameras.LEFT.value in keys
        assert Cameras.DEPTH.value in keys
        assert len(keys) == 2

    def test_get_required_zarr_keys_proprio_base_frame(self, observation_space_factory):
        """Test required zarr keys with proprioception in robot frame."""
        obs_space = observation_space_factory(
            use_proprioceptive_data=True,
            use_proprio_base_frame=True,
        )

        keys = obs_space.get_required_zarr_keys()

        assert ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value in keys
        assert ProprioKey.CAMERA_FRAME_CARTESIAN_TIP_POS.value not in keys

    def test_get_required_zarr_keys_proprio_camera_frame(self, observation_space_factory):
        """Test required zarr keys with proprioception in camera frame."""
        obs_space = observation_space_factory(
            use_proprioceptive_data=True,
            use_proprio_camera_frame=True,
        )

        keys = obs_space.get_required_zarr_keys()

        assert ProprioKey.CAMERA_FRAME_CARTESIAN_TIP_POS.value in keys
        assert ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value not in keys

    def test_get_required_zarr_keys_with_language(self, observation_space_factory):
        """Test required zarr keys include language."""
        obs_space = observation_space_factory(use_language=True)

        keys = obs_space.get_required_zarr_keys()

        assert ObsKey.LANGUAGE.value in keys

    def test_get_required_zarr_keys_with_gripper_state(self, observation_space_factory):
        """Test required zarr keys include gripper state."""
        obs_space = observation_space_factory(use_gripper_state=True)

        keys = obs_space.get_required_zarr_keys()

        assert ProprioKey.GRIPPER_STATE.value in keys

    def test_get_required_zarr_keys_with_custom_obs(self, observation_space_factory):
        """Test required zarr keys include custom observations."""
        obs_space = observation_space_factory(
            custom_obs_keys=['force_sensor', 'tactile_sensor']
        )

        keys = obs_space.get_required_zarr_keys()

        assert 'force_sensor' in keys
        assert 'tactile_sensor' in keys

    def test_get_required_zarr_keys_all_modalities(self, observation_space_factory):
        """Test required zarr keys with all modalities enabled."""
        obs_space = observation_space_factory(
            camera_keys=[Cameras.LEFT.value],
            use_proprio_base_frame=True,
            use_gripper_state=True,
            use_language=True,
            custom_obs_keys=['force'],
        )

        keys = obs_space.get_required_zarr_keys()

        assert Cameras.LEFT.value in keys
        assert ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value in keys
        assert ProprioKey.GRIPPER_STATE.value in keys
        assert ObsKey.LANGUAGE.value in keys
        assert 'force' in keys
        assert len(keys) == 5


@pytest.mark.unit
class TestTaskSpace:
    """Test TaskSpace initialization and validation."""

    def test_init_defaults(self, task_space_factory):
        """Test TaskSpace initialization with default values."""
        task_space = task_space_factory()

        assert task_space.observation_horizon == 1
        assert task_space.prediction_horizon == 16
        assert task_space.dataset_schema is not None
        assert task_space.dataloader is not None
        assert task_space.action_space is not None
        assert task_space.observation_space is not None

    def test_init_custom_horizons(self, task_space_factory):
        """Test TaskSpace initialization with custom horizons."""
        task_space = task_space_factory(
            observation_horizon=3,
            prediction_horizon=32,
        )

        assert task_space.observation_horizon == 3
        assert task_space.prediction_horizon == 32

    def test_validation_invalid_observation_horizon(self, task_space_factory):
        """Test validation fails for invalid observation_horizon."""
        task_space = task_space_factory(observation_horizon=0)

        with pytest.raises(ValueError, match="observation_horizon must be >= 1"):
            task_space.__post_init__()

    def test_validation_invalid_prediction_horizon(self, task_space_factory):
        """Test validation fails for invalid prediction_horizon."""
        task_space = task_space_factory(prediction_horizon=-5)

        with pytest.raises(ValueError, match="prediction_horizon must be >= 1"):
            task_space.__post_init__()

    @pytest.mark.parametrize("ori_dim", [2, 5, 10])
    def test_validation_invalid_orientation_dim(self, task_space_factory, action_space_factory, ori_dim):
        """Test validation fails for invalid orientation_dim."""
        action_space = action_space_factory(has_orientation=True, orientation_dim=ori_dim)
        task_space = task_space_factory(action_space=action_space)

        with pytest.raises(ValueError, match="orientation_dim must be one of"):
            task_space.__post_init__()

    @pytest.mark.parametrize("ori_dim", [1, 3, 4])
    def test_validation_valid_orientation_dim(self, task_space_factory, action_space_factory, ori_dim):
        """Test validation passes for valid orientation_dim."""
        action_space = action_space_factory(has_orientation=True, orientation_dim=ori_dim)
        task_space = task_space_factory(action_space=action_space)

        task_space.__post_init__()

    def test_validation_proprio_without_frame_fails(self, task_space_factory, observation_space_factory):
        """Test validation fails when use_proprioceptive_data but no frame specified."""
        obs_space = observation_space_factory(
            use_proprioceptive_data=True,
            use_proprio_base_frame=False,
            use_proprio_camera_frame=False,
        )
        task_space = task_space_factory(observation_space=obs_space)

        with pytest.raises(ValueError, match="one of.*use_proprio_base_frame or use_proprio_camera_frame"):
            task_space.__post_init__()

    def test_validation_proprio_with_base_frame_passes(self, task_space_factory, observation_space_factory):
        """Test validation passes when use_proprioceptive_data with base frame."""
        obs_space = observation_space_factory(
            use_proprioceptive_data=True,
            use_proprio_base_frame=True,
        )
        task_space = task_space_factory(observation_space=obs_space)

        task_space.__post_init__()

    def test_validation_proprio_with_camera_frame_passes(self, task_space_factory, observation_space_factory):
        """Test validation passes when use_proprioceptive_data with camera frame."""
        obs_space = observation_space_factory(
            use_proprioceptive_data=True,
            use_proprio_camera_frame=True,
        )
        task_space = task_space_factory(observation_space=obs_space)

        task_space.__post_init__()

    def test_validation_invalid_camera_key_fails(self, task_space_factory, observation_space_factory):
        """Test validation fails for invalid camera key."""
        obs_space = observation_space_factory(camera_keys=['invalid_camera'])
        task_space = task_space_factory(observation_space=obs_space)

        with pytest.raises(ValueError, match="Invalid camera key"):
            task_space.__post_init__()

    @pytest.mark.parametrize("camera_key", [Cameras.LEFT.value, Cameras.RIGHT.value, Cameras.DEPTH.value])
    def test_validation_valid_camera_keys_pass(self, task_space_factory, observation_space_factory, camera_key):
        """Test validation passes for valid camera keys."""
        obs_space = observation_space_factory(camera_keys=[camera_key])
        task_space = task_space_factory(observation_space=obs_space)

        task_space.__post_init__()


@pytest.mark.unit
class TestActionSpaceIntegration:
    """Integration tests for ActionSpace with various configurations."""

    def test_position_only_action_space(self, action_space_factory):
        """Test action space with only position."""
        action_space = action_space_factory(
            has_position=True,
            has_orientation=False,
            has_gripper=False,
        )

        assert action_space.get_total_action_dim() == 3
        keys = action_space.get_required_zarr_keys()
        assert ProprioKey.CAMERA_FRAME_CARTESIAN_TIP_POS.value in keys

    def test_full_action_space_with_phases(self, action_space_factory):
        """Test action space with all components including phases."""
        action_space = action_space_factory(
            has_position=True,
            position_dim=3,
            has_orientation=True,
            orientation_dim=4,
            has_gripper=True,
            gripper_dim=1,
            task_has_phases=True,
            number_of_phases=3,
        )

        assert action_space.get_total_action_dim() == 11
        keys = action_space.get_required_zarr_keys()
        assert ProprioKey.CAMERA_FRAME_CARTESIAN_TIP_POS.value in keys
        assert ProprioKey.GRIPPER_STATE.value in keys
        assert ObsKey.PHASE_LABEL.value in keys


@pytest.mark.unit
class TestObservationSpaceIntegration:
    """Integration tests for ObservationSpace with various configurations."""

    def test_minimal_observation_space(self, observation_space_factory):
        """Test minimal observation space with single camera."""
        obs_space = observation_space_factory(camera_keys=[Cameras.LEFT.value])

        keys = obs_space.get_required_zarr_keys()
        assert len(keys) == 1
        assert Cameras.LEFT.value in keys

    def test_multimodal_observation_space(self, observation_space_factory):
        """Test observation space with multiple modalities."""
        obs_space = observation_space_factory(
            camera_keys=[Cameras.LEFT.value, Cameras.RIGHT.value, Cameras.DEPTH.value],
            use_proprio_base_frame=True,
            use_gripper_state=True,
            use_language=True,
        )

        keys = obs_space.get_required_zarr_keys()
        assert Cameras.LEFT.value in keys
        assert Cameras.RIGHT.value in keys
        assert Cameras.DEPTH.value in keys
        assert ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value in keys
        assert ProprioKey.GRIPPER_STATE.value in keys
        assert ObsKey.LANGUAGE.value in keys
        assert len(keys) == 6