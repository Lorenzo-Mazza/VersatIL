"""Policy context used by post-training compression."""

from dataclasses import dataclass

from versatil.configs.main import MainConfig
from versatil.data.task import ObservationSpace
from versatil.data.tokenization.tokenizer import Tokenizer
from versatil.models.policy import Policy


@dataclass
class PolicyContext:
    """Loaded policy state used by compression workflows."""

    policy: Policy
    config: MainConfig
    tokenizer: Tokenizer | None
    observation_space: ObservationSpace
    observation_horizon: int
    checkpoint_path: str
    checkpoint_name: str
