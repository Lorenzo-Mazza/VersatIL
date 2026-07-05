"""Tests for versatil.explainability.online_runtime module."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from versatil.checkpoint_loading.float_policy import FloatCheckpointLoader
from versatil.explainability.online_runtime import ExplainabilityPolicyRuntime
from versatil.training.constants import PrecisionType


@pytest.fixture
def checkpoint_loader_factory() -> MagicMock:
    def factory(
        checkpoint_path: str = "/tmp/checkpoint",
        precision: str = PrecisionType.FP32.value,
    ) -> MagicMock:
        loader = MagicMock(spec=FloatCheckpointLoader)
        loader.checkpoint_path = checkpoint_path
        loader.config.experiment.precision = precision
        loader.device = torch.device("cpu")
        return loader

    return factory


class TestExplainabilityPolicyRuntime:
    def test_reuses_loader_policy_and_builds_client_identifier(
        self, checkpoint_loader_factory
    ):
        loader = checkpoint_loader_factory(checkpoint_path="/tmp/run")

        runtime = ExplainabilityPolicyRuntime(
            checkpoint_loader=loader,
            checkpoint_name="best.ckpt",
        )

        assert runtime.policy is loader.policy
        assert runtime.client_identifier == str(Path("/tmp/run") / "best")

    def test_run_inference_delegates_to_predict_action(self, checkpoint_loader_factory):
        loader = checkpoint_loader_factory(precision=PrecisionType.FP32.value)
        runtime = ExplainabilityPolicyRuntime(
            checkpoint_loader=loader,
            checkpoint_name="last.ckpt",
        )
        observation = {"left": torch.zeros(1, 1, 3, 4, 4)}

        result = runtime.run_inference(obs_dict=observation)

        loader.policy.predict_action.assert_called_once_with(obs_dict=observation)
        assert result is loader.policy.predict_action.return_value

    def test_run_inference_disables_gradients(self, checkpoint_loader_factory):
        loader = checkpoint_loader_factory(precision=PrecisionType.FP32.value)
        grad_states: list[bool] = []
        loader.policy.predict_action.side_effect = lambda obs_dict: grad_states.append(
            torch.is_grad_enabled()
        )
        runtime = ExplainabilityPolicyRuntime(
            checkpoint_loader=loader,
            checkpoint_name="last.ckpt",
        )

        runtime.run_inference(obs_dict={"left": torch.zeros(1, 1, 3, 4, 4)})

        assert grad_states == [False]
