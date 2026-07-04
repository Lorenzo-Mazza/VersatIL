"""Main Hydra configuration dataclass for the `train` endpoint."""

from dataclasses import dataclass, field
from typing import Any

from versatil.configs.data.task import TaskSpaceConfig
from versatil.configs.experiment import ExperimentConfig
from versatil.configs.policy import PolicyConfig
from versatil.configs.training import TrainingConfig


@dataclass
class MainConfig:
    """Root Hydra config composed from experiment, task, training, policy, and inference configs.

    Attributes:
        defaults: Hydra defaults list composed into the run config.
        experiment: Experiment settings.
        task: Task settings.
        training: Training settings.
        policy: Policy settings.
        quantization: Quantization-aware training settings, or null.
    """

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
    quantization: Any = None
