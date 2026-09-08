"""Shared fixtures for checkpoint loading tests."""

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
import torch

from versatil.checkpoint_loading.metadata import CheckpointMetadata
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.training.constants import CheckpointKey


@pytest.fixture
def checkpoint_config_factory() -> Callable[..., MagicMock]:
    """Factory for loader configs with a policy and training config."""

    def factory(policy: MagicMock | None = None) -> MagicMock:
        selected_policy = policy or MagicMock()
        selected_policy.to.return_value = selected_policy
        selected_policy.eval.return_value = selected_policy
        selected_policy.observation_space = MagicMock(spec=ObservationSpace)
        selected_policy.action_space = MagicMock(spec=ActionSpace)
        selected_policy.action_space.actions_metadata = {"position": MagicMock()}
        selected_policy.prediction_horizon = 4
        selected_policy.decoder.observation_horizon = 2
        selected_policy.get_denoising_thresholds.return_value = {"position": 0.05}
        config = MagicMock()
        config.policy = selected_policy
        config.training = MagicMock()
        return config

    return factory


@pytest.fixture
def checkpoint_metadata_factory() -> Callable[..., CheckpointMetadata]:
    def factory(
        prediction_horizon: int = 4,
        observation_horizon: int = 2,
    ) -> CheckpointMetadata:
        observation_space = MagicMock(spec=ObservationSpace)
        observation_space.depth_cameras = {}
        action_space = MagicMock(spec=ActionSpace)
        action_space.actions_metadata = {"position": MagicMock()}
        action_space.get_total_action_dim.return_value = 3
        return CheckpointMetadata(
            observation_space=observation_space,
            action_space=action_space,
            prediction_horizon=prediction_horizon,
            observation_horizon=observation_horizon,
        )

    return factory


@pytest.fixture
def checkpoint_payload_factory() -> Callable[..., dict[str, dict[str, torch.Tensor]]]:
    """Factory for checkpoint payloads with a state dict."""

    def factory(
        state_dict: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, dict[str, torch.Tensor]]:
        return {
            CheckpointKey.STATE_DICT.value: state_dict
            or {"policy.decoder.weight": torch.tensor([1.0])}
        }

    return factory


@pytest.fixture
def lightning_module_factory() -> Callable[..., MagicMock]:
    """Factory for mocked LightningPolicy instances."""

    def factory(
        state_dict: dict[str, torch.Tensor],
        call_order: list[str] | None = None,
    ) -> MagicMock:
        lightning_module = MagicMock()
        lightning_module.state_dict.return_value = state_dict
        if call_order is not None:
            lightning_module.load_state_dict.side_effect = lambda state_dict, strict: (
                call_order.append("load_state_dict")
            )
        return lightning_module

    return factory
