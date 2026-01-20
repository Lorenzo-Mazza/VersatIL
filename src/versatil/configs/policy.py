from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING

from versatil.configs.encoding.pipeline import EncodingPipelineConfig
from versatil.configs.loss import CompositeLossConfig
from versatil.configs.data.task import ActionSpaceConfig, ObservationSpaceConfig


@dataclass
class PolicyConfig:
    _target_: str = "versatil.models.policy.Policy"
    #: Observation encoding pipeline configuration
    encoding_pipeline: EncodingPipelineConfig = MISSING
    #: Decoding algorithm configuration (e.g., diffusion, flow matching)
    algorithm: Any = MISSING
    #: Action decoder architecture configuration
    decoder: Any = MISSING
    #: Reference to task's observation space
    observation_space: ObservationSpaceConfig = "${task.observation_space}"  # type: ignore[assignment]
    #: Reference to task's action space
    action_space: ActionSpaceConfig = "${task.action_space}"  # type: ignore[assignment]
    #: Reference to task's prediction horizon, i.e. chunk size
    prediction_horizon: int = "${task.prediction_horizon}"  # type: ignore[assignment]
    #: Reference to task's observation horizon
    observation_horizon: int = "${task.observation_horizon}"  # type: ignore[assignment]
    #: Device to run the policy on (references experiment device)
    device: str = "${experiment.device}"
    #: Loss configuration
    loss: CompositeLossConfig = MISSING
    #: Whether to validate loss keys against action space
    validate_loss_keys: bool = True
