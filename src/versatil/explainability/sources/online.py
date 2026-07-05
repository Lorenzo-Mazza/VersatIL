"""Online inference explanation source."""

import torch

from versatil.explainability.constants import ExplanationSourceType
from versatil.explainability.sources.typedefs import (
    ExplanationBatch,
    ExplanationBatchConsumer,
)
from versatil.explainability.typedefs import ObservationBatch, ObservationValue


class OnlineInferenceExplanationSource:
    """Adapts ready inference windows into explanation batches."""

    def __init__(
        self,
        consumer: ExplanationBatchConsumer,
        sample_stride: int = 1,
        max_samples: int | None = None,
    ) -> None:
        """Initialize the online inference adapter.

        Args:
            consumer: Object that generates explanations for each accepted
                inference window.
            sample_stride: Explain every Nth inference timestep.
            max_samples: Optional cap on the number of ready inference windows
                sent to the consumer.

        Raises:
            ValueError: If ``sample_stride`` or ``max_samples`` is invalid.
        """
        if sample_stride <= 0:
            raise ValueError(f"sample_stride must be positive. Got: {sample_stride}")
        if max_samples is not None and max_samples <= 0:
            raise ValueError(
                f"max_samples must be positive when set. Got: {max_samples}"
            )
        self.consumer = consumer
        self.sample_stride = sample_stride
        self.max_samples = max_samples
        self.explained_sample_count = 0

    def explain_observation_batch(
        self,
        observation: ObservationBatch,
        display_observation: dict[str, torch.Tensor],
        environment_indices: list[int],
        timestep: int,
    ) -> None:
        """Generate explanations for one ready online inference batch.

        Args:
            observation: The exact observation batch passed to
                ``PolicyRuntime.run_inference``.
            display_observation: Camera tensors for overlays.
            environment_indices: Environment indices represented by the batch
                rows.
            timestep: Inference client timestep.
        """
        if timestep % self.sample_stride != 0:
            return
        batch_size = len(environment_indices)
        accepted_sample_count = self._resolve_accepted_sample_count(
            batch_size=batch_size
        )
        if accepted_sample_count == 0:
            return
        accepted_environment_indices = environment_indices[:accepted_sample_count]
        accepted_observation = dict(observation)
        accepted_display_observation = display_observation
        if accepted_sample_count < batch_size:
            accepted_observation = self._slice_observation_batch(
                observation=observation,
                sample_count=accepted_sample_count,
                batch_size=batch_size,
            )
            accepted_display_observation = self._slice_display_observation(
                display_observation=display_observation,
                sample_count=accepted_sample_count,
                batch_size=batch_size,
            )
        self.consumer.explain_batch(
            batch=ExplanationBatch(
                observation=accepted_observation,
                actions=None,
                display_observation={
                    key: value.detach().cpu()
                    for key, value in accepted_display_observation.items()
                },
                metadata={
                    "source": ExplanationSourceType.ONLINE_INFERENCE.value,
                    "environment_indices": accepted_environment_indices,
                    "timestep": timestep,
                },
                preprocess_observation=True,
            )
        )
        self.explained_sample_count += accepted_sample_count

    def _resolve_accepted_sample_count(self, batch_size: int) -> int:
        """Return how many rows from the ready batch can still be explained."""
        if batch_size <= 0:
            return 0
        if self.max_samples is None:
            return batch_size
        remaining_samples = self.max_samples - self.explained_sample_count
        return min(batch_size, max(remaining_samples, 0))

    def _slice_observation_batch(
        self,
        observation: ObservationBatch,
        sample_count: int,
        batch_size: int,
    ) -> ObservationBatch:
        """Slice an online observation batch to the remaining sample budget."""
        return {
            key: self._slice_observation_value(
                key=key,
                value=value,
                sample_count=sample_count,
                batch_size=batch_size,
            )
            for key, value in observation.items()
        }

    def _slice_display_observation(
        self,
        display_observation: dict[str, torch.Tensor],
        sample_count: int,
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        """Slice display tensors to the remaining sample budget."""
        return {
            key: self._slice_tensor_value(
                key=key,
                value=value,
                sample_count=sample_count,
                batch_size=batch_size,
            )
            for key, value in display_observation.items()
        }

    def _slice_observation_value(
        self,
        key: str,
        value: ObservationValue,
        sample_count: int,
        batch_size: int,
    ) -> ObservationValue:
        """Slice one observation value along its batch axis."""
        if isinstance(value, torch.Tensor):
            return self._slice_tensor_value(
                key=key,
                value=value,
                sample_count=sample_count,
                batch_size=batch_size,
            )
        if isinstance(value, list):
            if len(value) != batch_size:
                raise RuntimeError(
                    f"Observation '{key}' has {len(value)} rows, expected "
                    f"{batch_size} from environment_indices."
                )
            return value[:sample_count]
        return value

    @staticmethod
    def _slice_tensor_value(
        key: str,
        value: torch.Tensor,
        sample_count: int,
        batch_size: int,
    ) -> torch.Tensor:
        """Slice one tensor value along its batch axis."""
        if value.shape[0] != batch_size:
            raise RuntimeError(
                f"Observation '{key}' has batch size {value.shape[0]}, expected "
                f"{batch_size} from environment_indices."
            )
        return value[:sample_count]
