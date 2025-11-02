from dataclasses import dataclass

from omegaconf import MISSING

from refactoring.configs.decoding.algorithm import DecodingAlgorithmConfig
from refactoring.configs.decoding.decoder import DecodingNetworkConfig
from refactoring.configs.encoding.pipeline import EncodingPipelineConfig
from refactoring.configs.loss import CompositeLossConfig
from refactoring.configs.task.task import ActionSpace, ObservationSpace


@dataclass
class PolicyConfig:
    _target_: str = "refactoring.models.policy.Policy"
    #: Observation encoding pipeline configuration
    encoding_pipeline: EncodingPipelineConfig = MISSING
    #: Decoding algorithm configuration (e.g., diffusion, flow matching)
    algorithm: DecodingAlgorithmConfig = MISSING
    #: Action decoder architecture configuration
    decoder: DecodingNetworkConfig = MISSING
    #: Reference to task's observation space
    observation_space: ObservationSpace = "${task.observation_space}"  # type: ignore[assignment]
    #: Reference to task's action space
    action_space: ActionSpace = "${task.action_space}"  # type: ignore[assignment]
    #: Reference to task's prediction horizon, i.e. chunk size
    prediction_horizon: int = "${task.prediction_horizon}"  # type: ignore[assignment]
    #: Device to run the policy on (references experiment device)
    device: str = "${experiment.device}"
    #: Loss configuration
    loss: CompositeLossConfig = MISSING
    #: Whether to validate loss keys against action space
    validate_loss_keys: bool = True
