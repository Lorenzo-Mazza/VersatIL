from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

from versatil.configs.data.task import ActionSpaceConfig, ObservationSpaceConfig
from versatil.configs.encoding.pipeline import EncodingPipelineConfig
from versatil.configs.loss import CompositeLossConfig


@dataclass
class PolicyConfig:
    """Hydra config for constructing a policy from encoding, algorithm, decoder, and loss configs."""

    _target_: str = "versatil.models.policy.Policy"
    encoding_pipeline: EncodingPipelineConfig = MISSING
    algorithm: Any = MISSING
    decoder: Any = MISSING
    observation_space: ObservationSpaceConfig = "${task.observation_space}"
    action_space: ActionSpaceConfig = "${task.action_space}"
    prediction_horizon: int = "${task.prediction_horizon}"
    observation_horizon: int = "${task.observation_horizon}"
    device: str = "${experiment.device}"
    loss: CompositeLossConfig = MISSING
    metadata_passthrough: dict[str, dict[str, str]] = field(default_factory=dict)
    validate_loss_keys: bool = (
        True  # Whether to validate loss keys against action space
    )
