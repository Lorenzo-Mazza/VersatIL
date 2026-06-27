"""Online inference explanation source."""

import torch

from versatil.explainability.constants import ExplanationSourceType
from versatil.explainability.sources.typedefs import (
    ExplanationBatch,
    ExplanationBatchConsumer,
)
from versatil.explainability.typedefs import ObservationBatch


class OnlineInferenceExplanationSource:
    """Adapts ready inference windows into explanation batches."""

    def __init__(
        self,
        consumer: ExplanationBatchConsumer,
        sample_stride: int = 1,
    ) -> None:
        """Initialize the online inference adapter.

        Args:
            consumer: Object that generates explanations for each accepted
                inference window.
            sample_stride: Explain every Nth inference timestep.

        Raises:
            ValueError: If ``sample_stride`` is not positive.
        """
        if sample_stride <= 0:
            raise ValueError(f"sample_stride must be positive. Got: {sample_stride}")
        self.consumer = consumer
        self.sample_stride = sample_stride

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
        self.consumer.explain_batch(
            batch=ExplanationBatch(
                observation=dict(observation),
                actions=None,
                display_observation={
                    key: value.detach().cpu()
                    for key, value in display_observation.items()
                },
                metadata={
                    "source": ExplanationSourceType.ONLINE_INFERENCE.value,
                    "environment_indices": environment_indices,
                    "timestep": timestep,
                },
                preprocess_observation=True,
            )
        )
