from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

from refactoring.configs.experiment import ExperimentConfig
from refactoring.configs.inference import InferenceConfig
from refactoring.configs.policy import PolicyConfig
from refactoring.configs.data.task import TaskSpaceConfig
from refactoring.configs.training import TrainingConfig


@dataclass
class MainConfig:
    defaults: list[Any] = field(default_factory=lambda: [
        {"experiment": ExperimentConfig},
        {"task": "base"},
        {"training": "base"},
        {"policy": "base"},
        {"inference": "base"},    ])

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    task: TaskSpaceConfig = field(default_factory=TaskSpaceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
