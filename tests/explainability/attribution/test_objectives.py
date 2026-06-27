"""Tests for versatil.explainability.attribution.objectives module."""

import re
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
import torch

from versatil.data.constants import SampleKey
from versatil.explainability.attribution.objectives import (
    compute_policy_explanation_objective,
    repeat_action_batch,
    resolve_actions_for_explanation,
)
from versatil.models.decoding.constants import DecoderOutputKey


@pytest.fixture
def objective_policy_mock() -> MagicMock:
    policy = MagicMock()
    policy.decoder.requires_tokenized_actions = False
    policy._strip_metadata_passthrough_observations.side_effect = lambda observation: (
        observation
    )
    policy._build_algorithm_features.return_value = {"features": torch.ones(2, 3)}
    return policy


class TestComputePolicyExplanationObjective:
    def test_scores_continuous_predictions_with_action_norm(
        self,
        objective_policy_mock: MagicMock,
    ):
        objective_policy_mock.algorithm.predict.return_value = {
            "position": torch.tensor([[[3.0, 4.0]], [[0.0, 0.0]]]),
            "gripper": torch.tensor([[[12.0]], [[5.0]]]),
        }

        result = compute_policy_explanation_objective(
            policy=objective_policy_mock,
            observation={"image": torch.zeros(2, 1)},
            actions=None,
            preprocess_observation=False,
        )

        torch.testing.assert_close(result, torch.tensor([[13.0], [5.0]]))
        objective_policy_mock.algorithm.predict.assert_called_once()
        predict_kwargs = objective_policy_mock.algorithm.predict.call_args.kwargs
        torch.testing.assert_close(
            predict_kwargs["features"]["features"],
            torch.ones(2, 3),
        )
        assert predict_kwargs["network"] is objective_policy_mock.decoder

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "policy_case_name",
        ["spatial_resnet18", "smolvla", "paligemma_vlm", "prismatic_vlm"],
    )
    def test_scores_real_continuous_policy_classes(
        self,
        real_explainability_policy_case_factory: Callable,
        policy_case_name: str,
    ):
        batch_size = (
            1
            if policy_case_name.endswith("_vlm") or policy_case_name == "smolvla"
            else 2
        )
        case = real_explainability_policy_case_factory(
            case_name=policy_case_name,
            batch_size=batch_size,
        )

        result = compute_policy_explanation_objective(
            policy=case.policy,
            observation=case.observation,
            actions=None,
            preprocess_observation=False,
        )

        assert result.shape[0] == batch_size
        assert torch.isfinite(result).all()

    def test_scores_tokenized_action_predictions_with_log_likelihood(
        self,
        objective_policy_mock: MagicMock,
    ):
        objective_policy_mock.decoder.requires_tokenized_actions = True
        logits = torch.tensor(
            [
                [[0.0, 2.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 3.0], [0.0, 4.0, 0.0]],
            ]
        )
        actions = {
            SampleKey.TOKENIZED_ACTIONS.value: torch.tensor([[1, 0], [2, 1]]),
            SampleKey.IS_PAD_ACTION.value: torch.tensor(
                [[False, True], [False, False]]
            ),
        }
        objective_policy_mock.algorithm.forward.return_value = {
            DecoderOutputKey.ACTION_LOGITS.value: logits
        }

        result = compute_policy_explanation_objective(
            policy=objective_policy_mock,
            observation={"image": torch.zeros(2, 1)},
            actions=actions,
            preprocess_observation=False,
        )

        expected_log_probs = logits.log_softmax(dim=-1)
        expected = torch.tensor(
            [
                expected_log_probs[0, 0, 1],
                (expected_log_probs[1, 0, 2] + expected_log_probs[1, 1, 1]) / 2,
            ]
        )
        torch.testing.assert_close(result, expected)
        objective_policy_mock.algorithm.forward.assert_called_once()
        forward_kwargs = objective_policy_mock.algorithm.forward.call_args.kwargs
        torch.testing.assert_close(
            forward_kwargs["features"]["features"],
            torch.ones(2, 3),
        )
        assert forward_kwargs["actions"] is actions
        assert forward_kwargs["network"] is objective_policy_mock.decoder

    def test_rejects_custom_selector_for_tokenized_action_predictions(
        self,
        objective_policy_mock: MagicMock,
    ):
        objective_policy_mock.decoder.requires_tokenized_actions = True
        expected_message = (
            "output_selector is only supported when the decoder does not require "
            "tokenized actions."
        )

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            compute_policy_explanation_objective(
                policy=objective_policy_mock,
                observation={"image": torch.zeros(2, 1)},
                actions={SampleKey.TOKENIZED_ACTIONS.value: torch.zeros(2, 1)},
                preprocess_observation=False,
                output_selector=lambda predictions: next(iter(predictions.values())),
            )


class TestResolveActionsForExplanation:
    def test_generates_tokenized_pseudo_targets_for_unlabeled_batches(
        self,
        objective_policy_mock: MagicMock,
    ):
        objective_policy_mock.decoder.requires_tokenized_actions = True
        predicted_tokens = torch.tensor([[4, 5], [6, 7]])
        objective_policy_mock.algorithm.predict.return_value = {
            DecoderOutputKey.PREDICTED_ACTION_TOKENS.value: predicted_tokens
        }

        result = resolve_actions_for_explanation(
            policy=objective_policy_mock,
            observation={"image": torch.zeros(2, 1)},
            actions=None,
            preprocess_observation=False,
        )

        assert result is not None
        torch.testing.assert_close(
            result[SampleKey.TOKENIZED_ACTIONS.value],
            predicted_tokens,
        )
        objective_policy_mock.algorithm.predict.assert_called_once()
        predict_kwargs = objective_policy_mock.algorithm.predict.call_args.kwargs
        torch.testing.assert_close(
            predict_kwargs["features"]["features"],
            torch.ones(2, 3),
        )
        assert predict_kwargs["network"] is objective_policy_mock.decoder

    def test_keeps_continuous_actions_unchanged(
        self,
        objective_policy_mock: MagicMock,
    ):
        actions = {"position": torch.ones(2, 3)}

        result = resolve_actions_for_explanation(
            policy=objective_policy_mock,
            observation={"image": torch.zeros(2, 1)},
            actions=actions,
            preprocess_observation=False,
        )

        assert result is actions
        objective_policy_mock.algorithm.predict.assert_not_called()


class TestRepeatActionBatch:
    def test_repeats_action_tensors_along_batch(self):
        actions = {"tokens": torch.tensor([[1, 2], [3, 4]])}

        result = repeat_action_batch(actions=actions, repeat_count=2)

        assert result is not None
        torch.testing.assert_close(
            result["tokens"],
            torch.tensor([[1, 2], [3, 4], [1, 2], [3, 4]]),
        )

    def test_raises_when_repeat_count_is_not_positive(self):
        expected_message = "repeat_count must be positive. Got: 0"

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            repeat_action_batch(actions={"tokens": torch.ones(1, 2)}, repeat_count=0)
