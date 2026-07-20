"""Shared fixtures for checkpoint loading tests."""

from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from versatil.training.constants import CheckpointKey


@pytest.fixture
def checkpoint_config_factory() -> Callable[..., MagicMock]:
    """Factory for loader configs with a policy and training config."""

    def factory(policy: MagicMock | None = None) -> MagicMock:
        selected_policy = policy or MagicMock()
        selected_policy.to.return_value = selected_policy
        selected_policy.eval.return_value = selected_policy
        config = MagicMock()
        config.policy = selected_policy
        config.training = MagicMock()
        return config

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
        incompatible_keys = SimpleNamespace(missing_keys=[], unexpected_keys=[])
        lightning_module.load_state_dict.return_value = incompatible_keys
        if call_order is not None:
            lightning_module.load_state_dict.side_effect = lambda state_dict, strict: (
                call_order.append("load_state_dict") or incompatible_keys
            )
        return lightning_module

    return factory
