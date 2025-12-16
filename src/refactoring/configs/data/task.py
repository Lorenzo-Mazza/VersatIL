"""
Configurations for the experiment task.
The task defines what data the experiment will use at runtime, but not what data is in the dataset (see DatasetSchema for that).
This design is motivated by the fact that a single dataset can be used for multiple tasks, each requiring different data.
"""

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

from refactoring.configs import DataLoaderConfig
from refactoring.data.constants import (
    GripperType,
    OrientationRepresentation,
)


@dataclass
class ActionSpaceConfig:
    """Configuration for action space."""
    _target_: str = "refactoring.data.task.ActionSpace"
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
    use_precomputed_actions: bool = False


@dataclass
class ObservationSpaceConfig:
    """Configuration for observation space."""
    _target_: str = "refactoring.data.task.ObservationSpace"
    use_proprio_base_frame: bool = False
    use_proprio_camera_frame: bool = False
    use_gripper_state: bool = False
    gripper_type: str = GripperType.BINARY.value
    camera_keys: list[str] = field(default_factory=list) # Have to be consistent with constants.data.Cameras.value
    use_language: bool = False
    custom_obs_keys: list[str] = field(default_factory=list)


@dataclass
class TaskSpaceConfig:
    """Task space-specific configuration."""
    _target_: str = "refactoring.data.task.TaskSpace"
    #: Dataset schema configuration, defining what dataset the task uses
    dataset_schema: Any = MISSING
    #: Data loading and preprocessing configuration
    dataloader: DataLoaderConfig = MISSING
    #: Action space used by the task
    action_space: ActionSpaceConfig = field(default_factory=ActionSpaceConfig)
    #: Observation space used by the task
    observation_space: ObservationSpaceConfig = field(default_factory=ObservationSpaceConfig)
    #: Observation horizon, i.e. history size
    observation_horizon: int = 1
    #: Prediction horizon, i.e. chunk size
    prediction_horizon: int = 16



