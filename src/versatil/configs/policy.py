from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING

from versatil.configs.encoding.pipeline import EncodingPipelineConfig
from versatil.configs.loss import CompositeLossConfig
from versatil.configs.data.task import ActionSpaceConfig, ObservationSpaceConfig


@dataclass
class PolicyConfig:
    _target_: str = "versatil.models.policy.Policy"
    encoding_pipeline: EncodingPipelineConfig = MISSING
    algorithm: Any = MISSING
    decoder: Any = MISSING
    observation_space: ObservationSpaceConfig = "${task.observation_space}"  # type: ignore[assignment]
    action_space: ActionSpaceConfig = "${task.action_space}"  # type: ignore[assignment]
    prediction_horizon: int = "${task.prediction_horizon}"  # type: ignore[assignment]
    observation_horizon: int = "${task.observation_horizon}"  # type: ignore[assignment]
    device: str = "${experiment.device}"
    loss: CompositeLossConfig = MISSING
    validate_loss_keys: bool = (
        True  # Whether to validate loss keys against action space
    )
