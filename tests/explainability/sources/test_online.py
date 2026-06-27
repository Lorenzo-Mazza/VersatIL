"""Tests for versatil.explainability.sources.online module."""

from unittest.mock import MagicMock

import pytest
import torch

from versatil.data.constants import Cameras
from versatil.explainability.constants import ExplanationSourceType
from versatil.explainability.sources.online import OnlineInferenceExplanationSource


class TestOnlineInferenceExplanationSource:
    def test_rejects_non_positive_sample_stride(self) -> None:
        error_message = "sample_stride must be positive. Got: 0"

        with pytest.raises(ValueError, match=error_message):
            OnlineInferenceExplanationSource(
                consumer=MagicMock(),
                sample_stride=0,
            )

    def test_forwards_ready_batches_with_online_metadata(self) -> None:
        consumer = MagicMock()
        source = OnlineInferenceExplanationSource(
            consumer=consumer,
            sample_stride=2,
        )
        observation = {
            Cameras.AGENTVIEW.value: torch.zeros(1, 1, 3, 4, 4),
        }
        display_observation = {
            Cameras.AGENTVIEW.value: torch.ones(1, 1, 3, 4, 4),
        }

        source.explain_observation_batch(
            observation=observation,
            display_observation=display_observation,
            environment_indices=[3],
            timestep=4,
        )

        consumer.explain_batch.assert_called_once()
        batch = consumer.explain_batch.call_args.kwargs["batch"]
        assert batch.metadata["source"] == ExplanationSourceType.ONLINE_INFERENCE.value
        assert batch.metadata["environment_indices"] == [3]
        assert batch.metadata["timestep"] == 4
        assert batch.preprocess_observation is True
        assert batch.actions is None

    def test_skips_timesteps_outside_frequency(self) -> None:
        consumer = MagicMock()
        source = OnlineInferenceExplanationSource(
            consumer=consumer,
            sample_stride=3,
        )

        source.explain_observation_batch(
            observation={Cameras.AGENTVIEW.value: torch.zeros(1, 1, 3, 4, 4)},
            display_observation={Cameras.AGENTVIEW.value: torch.ones(1, 1, 3, 4, 4)},
            environment_indices=[0],
            timestep=4,
        )

        consumer.explain_batch.assert_not_called()
