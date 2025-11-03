"""
Configurations for the experiment task.
The task defines what data the experiment will use at runtime, but not what data is in the dataset (see DatasetSchema for that).
This design is motivated by the fact that a single dataset can be used for multiple tasks, each requiring different data.
"""

from dataclasses import dataclass, field

from omegaconf import MISSING

from refactoring.configs.task.dataloader import DataloaderConfig
from refactoring.configs.task.dataset.schema import DatasetSchemaConfig
from refactoring.data.constants import (
    GRIPPER_STATE_OBS_KEY,
    LANGUAGE_KEY,
    PHASE_LABEL_KEY,
    PROPRIO_OBS_CAMERA_FRAME_KEY,
    PROPRIO_OBS_ROBOT_FRAME_KEY,
    VALID_CAMERAS,
    GripperType,
    OrientationRepresentation,
)


@dataclass
class ActionSpace:
    """Defines what actions the task will predict and how they are computed."""
    _target_: str = "refactoring.configs.task.task.ActionSpace"
    has_position: bool = True
    position_dim: int = 3
    has_orientation: bool = False
    orientation_dim: int = 0
    orientation_repr: str = OrientationRepresentation.ROLL.value
    has_gripper: bool = True
    gripper_type: str = GripperType.BINARY.value
    gripper_dim: int = 1
    use_gripper_class_weights: bool = False
    predict_in_camera_frame: bool = True
    deltas_as_actions: bool = False
    denoise_actions: bool = True
    custom_action_dims: dict[str, int] = field(default_factory=dict)
    task_has_phases: bool = False
    number_of_phases: int = 5

    def get_total_action_dim(self) -> int:
        """Calculate total action dimension.

        Returns:
            Total dimension of action space
        """
        total = 0
        if self.has_position:
            total += self.position_dim
        if self.has_orientation:
            total += self.orientation_dim
        if self.has_gripper:
            total += self.gripper_dim
        for dim in self.custom_action_dims.values():
            total += dim
        if self.task_has_phases:
            total += self.number_of_phases
        return total


    def get_required_zarr_keys(self) -> list[str]:
        """Get zarr keys needed for computing actions.

        Returns:
            List of keys to load from replay buffer
        """
        keys = []
        if self.has_position or self.has_orientation:
            if self.predict_in_camera_frame:
                keys.append(PROPRIO_OBS_CAMERA_FRAME_KEY)
            else:
                keys.append(PROPRIO_OBS_ROBOT_FRAME_KEY)
        if self.has_gripper:
            keys.append(GRIPPER_STATE_OBS_KEY)
        if self.task_has_phases:
            keys.append(PHASE_LABEL_KEY)
        return keys


@dataclass
class ObservationSpace:
    """Defines what observations the task will request and how they are processed."""
    _target_: str = "refactoring.configs.task.task.ObservationSpace"
    use_proprioceptive_data: bool = False
    use_proprio_base_frame: bool = False
    use_proprio_camera_frame: bool = False
    use_gripper_state: bool = False
    gripper_type: str = GripperType.BINARY.value
    camera_keys: list[str] = field(default_factory=list) # Have to be consistent with constants.data.Cameras.value
    use_language: bool = False
    custom_obs_keys: list[str] = field(default_factory=list)


    def get_required_zarr_keys(self) -> list[str]:
        """Get all zarr keys needed for this observation space at runtime.

        Returns:
            List of keys to load from replay buffer
        """
        keys = []
        keys.extend(self.camera_keys)
        if self.use_proprio_base_frame:
            keys.append(PROPRIO_OBS_ROBOT_FRAME_KEY)
        if self.use_proprio_camera_frame:
            keys.append(PROPRIO_OBS_CAMERA_FRAME_KEY)
        if self.use_language:
            keys.append(LANGUAGE_KEY)
        if self.use_gripper_state:
            keys.append(GRIPPER_STATE_OBS_KEY)
        for key in self.custom_obs_keys:
            keys.append(key)
        return keys


@dataclass
class TaskConfig:
    """Task-specific configuration."""
    #: Dataset schema configuration, defining what dataset the task uses
    dataset_schema: DatasetSchemaConfig = MISSING
    #: Data loading and preprocessing configuration
    dataloader: DataloaderConfig = MISSING
    #: Action space used by the task
    action_space: ActionSpace = field(default_factory=ActionSpace)
    #: Observation space used by the task
    observation_space: ObservationSpace = field(default_factory=ObservationSpace)
    #: Observation horizon, i.e. history size
    observation_horizon: int = 1
    #: Prediction horizon, i.e. chunk size
    prediction_horizon: int = 16


    def __post_init__(self):
        """Validate task configuration."""
        if self.observation_horizon < 1:
            raise ValueError(f"observation_horizon must be >= 1, got {self.observation_horizon}")
        if self.prediction_horizon < 1:
            raise ValueError(f"prediction_horizon must be >= 1, got {self.prediction_horizon}")
        if self.action_space.has_orientation:
            valid_ori_dims = {1, 3, 4}
            if self.action_space.orientation_dim not in valid_ori_dims:
                raise ValueError(
                    f"orientation_dim must be one of {valid_ori_dims}:"
                    f"1 for roll of RCM-constrained EE, 3 for euler, 4 for quaternion,"
                    f"got {self.action_space.orientation_dim}"
                )
        if self.observation_space.use_proprioceptive_data and not (self.observation_space.use_proprio_base_frame or self.observation_space.use_proprio_camera_frame):
            raise ValueError(
                "If use_proprioceptive_data is True, then one of"
                "use_proprio_base_frame or use_proprio_camera_frame must be True"
                )
        if self.observation_space.camera_keys:
            for cam in self.observation_space.camera_keys:
                if cam not in VALID_CAMERAS:
                    raise ValueError(f"Invalid camera key '{cam}', must be one of {VALID_CAMERAS}."
                                     f"To add custom camera keys, add them to constants.data.Cameras enum.")


