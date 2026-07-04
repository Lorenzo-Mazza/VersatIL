from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

from versatil.configs.data.task import ActionSpaceConfig, ObservationSpaceConfig
from versatil.configs.encoding.pipeline import EncodingPipelineConfig
from versatil.configs.loss import CompositeLossConfig


@dataclass
class PolicyConfig:
    """Hydra config for constructing a policy from encoding, algorithm, decoder, and loss configs.

    Attributes:
        _target_: Import path instantiated by Hydra.
        encoding_pipeline: Observation encoding pipeline.
        algorithm: Decoding algorithm (diffusion, flow matching, etc.).
        decoder: Action decoder architecture.
        observation_space: Observation space configuration.
        action_space: Action space configuration.
        prediction_horizon: Number of future actions to predict.
        observation_horizon: Number of past observations to condition on.
        device: Device to run on.
        loss: Loss module for training.
        metadata_passthrough: Mapping from source dictionaries to metadata keys for
            logging/visualization.
    """

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
