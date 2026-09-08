"""Observation and action metadata used by checkpoint loaders."""

from dataclasses import dataclass

from versatil.data.task import ActionSpace, ObservationSpace


@dataclass(frozen=True)
class CheckpointMetadata:
    """Describe the observation windows and action chunks used for inference.

    Attributes:
        observation_space: Observation keys, shapes and preprocessing requirements.
        action_space: Action components and their reconstruction settings.
        prediction_horizon: Number of future action steps produced per prediction.
        observation_horizon: Number of observation steps consumed by the decoder.
    """

    observation_space: ObservationSpace
    action_space: ActionSpace
    prediction_horizon: int
    observation_horizon: int
