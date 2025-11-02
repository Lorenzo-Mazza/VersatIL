from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

from refactoring.configs.experiment import ExperimentConfig
from refactoring.configs.inference import InferenceConfig
from refactoring.configs.policy import PolicyConfig
from refactoring.configs.task.task import TaskConfig
from refactoring.configs.training import TrainingConfig


@dataclass
class MainConfig:
    defaults: list[Any] = field(default_factory=lambda: [
        'experiment',
        'data',
        'task',
        'training',
        'policy',
        'inference',
    ])

    experiment: ExperimentConfig = MISSING
    task: TaskConfig = MISSING
    training: TrainingConfig = MISSING
    policy: PolicyConfig = MISSING
    inference: InferenceConfig = MISSING
