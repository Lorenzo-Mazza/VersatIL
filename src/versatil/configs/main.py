"""Main Hydra configuration dataclass composing all sub-configurations."""

from dataclasses import dataclass, field
from typing import Any

from versatil.configs.data.task import TaskSpaceConfig
from versatil.configs.experiment import ExperimentConfig
from versatil.configs.inference import InferenceConfig
from versatil.configs.policy import PolicyConfig
from versatil.configs.training import TrainingConfig


@dataclass
class MainConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            {"experiment": ExperimentConfig},
            {"task": "base"},
            {"training": "base"},
            {"policy": "base"},
            {"inference": "base"},
            {"optional quantization": None},
        ]
    )

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    task: TaskSpaceConfig = field(default_factory=TaskSpaceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    quantization: Any = None
