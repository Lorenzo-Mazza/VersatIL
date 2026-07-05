"""Source contracts and batch container types."""

from dataclasses import dataclass
from typing import Protocol

import torch

from versatil.configs.data.dataloader import DataLoaderConfig
from versatil.data.raw.schemas.base import DatasetSchema
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.explainability.typedefs import (
    ActionBatch,
    ExplanationMetadataValue,
    ObservationBatch,
)


class DatasetTaskConfig(Protocol):
    """Task fields required by dataset explanation sources."""

    dataset_schema: DatasetSchema
    dataloader: DataLoaderConfig
    action_space: ActionSpace
    observation_space: ObservationSpace
    prediction_horizon: int
    observation_horizon: int


class DatasetExperimentConfig(Protocol):
    """Experiment fields required by dataset explanation sources."""

    seed: int


class DatasetRunnerConfig(Protocol):
    """Config fields required by dataset explanation sources."""

    task: DatasetTaskConfig
    experiment: DatasetExperimentConfig


@dataclass(frozen=True)
class ExplanationBatch:
    """A model-ready observation window plus metadata for saving explanations.

    Args:
        observation: Observation dictionary passed to the policy attribution
            functions.
        actions: Optional action dictionary from the source sample. Offline
            dataset mode provides normalized/tokenized action targets when the
            dataset contains them. Online inference mode sets this to ``None``.
        display_observation: Camera tensors used for visual overlays. These are
            kept separate because dataset batches may contain extra tokenized
            observation keys that are not displayable.
        metadata: Lightweight source metadata used for output filenames and
            reports.
        preprocess_observation: Whether the explainer should run policy
            normalization/tokenization before attribution.
    """

    observation: ObservationBatch
    actions: ActionBatch | None
    display_observation: dict[str, torch.Tensor]
    metadata: dict[str, ExplanationMetadataValue]
    preprocess_observation: bool


class ExplanationBatchConsumer(Protocol):
    """Consumes explanation batches produced by online inference hooks."""

    def explain_batch(self, batch: ExplanationBatch) -> None:
        """Generate and persist explanations for one batch."""
        ...
