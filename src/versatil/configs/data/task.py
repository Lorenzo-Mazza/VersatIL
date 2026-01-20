"""
Configurations for the experiment task.
The task defines what data the experiment will use at runtime, but not what data is in the dataset (see DatasetSchema for that).
This design is motivated by the fact that a single dataset can be used for multiple tasks, each requiring different data.
"""

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

from versatil.configs.data.dataloader import DataLoaderConfig


@dataclass
class ActionSpaceConfig:
    """Configuration for action space.

    Attributes:
        actions_metadata: Dict of all action metadata, indexed by zarr store key.
            Values are OnTheFlyActionMetadataConfig or PrecomputedActionMetadataConfig subclasses.
        use_gripper_class_weights: Whether to use class weights for binary gripper.
        denoise_actions: Whether to apply denoising to actions.
        denoising_percentile: Percentile for denoising threshold.
    """

    _target_: str = "versatil.data.task.ActionSpace"
    actions_metadata: dict[str, Any] = field(default_factory=dict)
    use_gripper_class_weights: bool = False
    denoise_actions: bool = True
    denoising_percentile: float = 15.0


@dataclass
class ObservationSpaceConfig:
    """Configuration for observation space.

    Attributes:
        observations_metadata: Dict of all observation metadata, indexed by zarr store key.
            Values are ObservationMetadataConfig subclasses or CameraMetadataConfig.
    """

    _target_: str = "versatil.data.task.ObservationSpace"
    observations_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskSpaceConfig:
    """Task space specific configuration for the experiment run.

    Attributes:
        dataset_schema: Dataset schema configuration, defining what dataset and zarr store the task uses.
        dataloader: Data loading and preprocessing configuration.
        action_space: Action space configuration used by the task at runtime.
        observation_space: Observation space configuration used by the task at runtime.
        observation_horizon: Number of history timesteps to include.
        prediction_horizon: Number of timesteps to predict (action chunk size).
    """

    _target_: str = "versatil.data.task.TaskSpace"
    dataset_schema: Any = MISSING
    dataloader: DataLoaderConfig = MISSING
    action_space: ActionSpaceConfig = MISSING
    observation_space: ObservationSpaceConfig = MISSING
    observation_horizon: int = 1
    prediction_horizon: int = 16
