"""Policy context used by post-training compression."""

from dataclasses import dataclass

from versatil.configs.main import MainConfig
from versatil.data.task import ObservationSpace
from versatil.data.tokenization.tokenizer import Tokenizer
from versatil.models.policy import Policy
from versatil.training.constants import PrecisionType


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
    precision: PrecisionType | None = None

    @property
    def autocast_precision(self) -> PrecisionType | None:
        """Reuse checkpoint autocast when compression precision is unset.

        Note:
            Keeping the original weight dtypes can leave FP32 inputs feeding BF16
            layers. Autocast handles these conversions during calibration and export.
        """
        if self.precision is not None:
            return None
        checkpoint_precision = PrecisionType(self.config.experiment.precision)
        return checkpoint_precision if checkpoint_precision.is_mixed() else None
