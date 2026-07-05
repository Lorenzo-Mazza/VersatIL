"""Tests for versatil.explainability.sources.online module."""

from unittest.mock import MagicMock

import pytest
import torch
from versatil_constants.shared import ObsKey

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

    def test_rejects_non_positive_max_samples(self) -> None:
        error_message = "max_samples must be positive when set. Got: 0"

        with pytest.raises(ValueError, match=error_message):
            OnlineInferenceExplanationSource(
                consumer=MagicMock(),
                sample_stride=1,
                max_samples=0,
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

    def test_caps_ready_samples_and_slices_final_batch(self) -> None:
        consumer = MagicMock()
        source = OnlineInferenceExplanationSource(
            consumer=consumer,
            sample_stride=1,
            max_samples=3,
        )
        first_observation = {
            Cameras.AGENTVIEW.value: torch.zeros(2, 1, 3, 4, 4),
            ObsKey.LANGUAGE.value: ["pick", "place"],
        }
        first_display_observation = {
            Cameras.AGENTVIEW.value: torch.ones(2, 1, 3, 4, 4),
        }
        second_observation = {
            Cameras.AGENTVIEW.value: torch.full((2, 1, 3, 4, 4), 2.0),
            ObsKey.LANGUAGE.value: ["lift", "drop"],
        }
        second_display_observation = {
            Cameras.AGENTVIEW.value: torch.full((2, 1, 3, 4, 4), 3.0),
        }

        source.explain_observation_batch(
            observation=first_observation,
            display_observation=first_display_observation,
            environment_indices=[0, 1],
            timestep=0,
        )
        source.explain_observation_batch(
            observation=second_observation,
            display_observation=second_display_observation,
            environment_indices=[2, 3],
            timestep=1,
        )
        source.explain_observation_batch(
            observation=second_observation,
            display_observation=second_display_observation,
            environment_indices=[4, 5],
            timestep=2,
        )

        assert consumer.explain_batch.call_count == 2
        first_batch = consumer.explain_batch.call_args_list[0].kwargs["batch"]
        second_batch = consumer.explain_batch.call_args_list[1].kwargs["batch"]
        assert first_batch.metadata["environment_indices"] == [0, 1]
        assert first_batch.observation[Cameras.AGENTVIEW.value].shape[0] == 2
        assert second_batch.metadata["environment_indices"] == [2]
        assert second_batch.observation[Cameras.AGENTVIEW.value].shape[0] == 1
        assert second_batch.observation[ObsKey.LANGUAGE.value] == ["lift"]
        assert second_batch.display_observation[Cameras.AGENTVIEW.value].shape[0] == 1
        assert source.explained_sample_count == 3

    def test_rejects_mismatched_tensor_batch_size_when_slicing(self) -> None:
        consumer = MagicMock()
        source = OnlineInferenceExplanationSource(
            consumer=consumer,
            sample_stride=1,
            max_samples=1,
        )
        error_message = (
            f"Observation '{Cameras.AGENTVIEW.value}' has batch size 1, "
            "expected 2 from environment_indices."
        )

        with pytest.raises(RuntimeError, match=error_message):
            source.explain_observation_batch(
                observation={Cameras.AGENTVIEW.value: torch.zeros(1, 1, 3, 4, 4)},
                display_observation={
                    Cameras.AGENTVIEW.value: torch.ones(1, 1, 3, 4, 4)
                },
                environment_indices=[0, 1],
                timestep=0,
            )
