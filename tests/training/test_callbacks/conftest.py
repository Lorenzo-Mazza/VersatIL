"""Shared fixtures for the ``versatil.training.callbacks`` test package."""

from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from versatil.models.policy import Policy
from versatil.training.lightning_policy import LightningPolicy


@pytest.fixture
def staged_policy_factory(
    real_policy_factory: Callable[..., Policy],
) -> Callable[..., Policy]:
    """Real ``Policy`` with the nested module structure the staged-training
    unit tests rely on (``encoding_pipeline``, ``algorithm.{posterior_encoder,
    prior}``, ``decoder``). Heavy deps are mocked by ``real_policy_factory``."""

    def factory() -> Policy:
        algorithm = torch.nn.Module()
        algorithm.posterior_encoder = torch.nn.Linear(2, 2)
        algorithm.prior = torch.nn.Linear(2, 2)
        return real_policy_factory(
            submodules={
                "encoding_pipeline": torch.nn.Linear(2, 2),
                "algorithm": algorithm,
                "decoder": torch.nn.Linear(2, 2),
            }
        )

    return factory


@pytest.fixture
def simple_module_factory(rng: np.random.Generator) -> Callable[..., torch.nn.Module]:
    def factory(
        input_dimension: int = 4,
        output_dimension: int = 4,
    ) -> torch.nn.Module:
        module = torch.nn.Linear(input_dimension, output_dimension)
        weight_data = torch.from_numpy(
            rng.standard_normal((output_dimension, input_dimension)).astype(np.float32)
        )
        bias_data = torch.from_numpy(
            rng.standard_normal((output_dimension,)).astype(np.float32)
        )
        module.weight.data.copy_(weight_data)
        module.bias.data.copy_(bias_data)
        return module

    return factory


@pytest.fixture
def pl_module_with_policy_factory(
    simple_module_factory: Callable[..., torch.nn.Module],
) -> Callable[..., MagicMock]:
    def factory(
        policy: torch.nn.Module | None = None,
    ) -> MagicMock:
        if policy is None:
            policy = simple_module_factory()
        pl_module = MagicMock(spec=LightningPolicy)
        pl_module.policy = policy
        pl_module.parameters.return_value = policy.parameters()
        pl_module.log = MagicMock()
        pl_module.log_dict = MagicMock()
        return pl_module

    return factory


@pytest.fixture
def mock_pl_module_factory() -> Callable[..., MagicMock]:
    """Factory for a vanilla ``pl_module`` MagicMock with configurable metric
    return values for train/val hook tests and an optional ``policy.training``
    flag used by callbacks that restore train/eval mode around rollouts."""

    def factory(
        train_metrics_return: dict | np.ndarray | None = None,
        val_metrics_return: dict | np.ndarray | None = None,
        train_method_name: str = "compute_expert_usage",
        val_method_name: str = "compute_expert_usage",
        policy_training: bool | None = None,
    ) -> MagicMock:
        pl_module = MagicMock()
        getattr(
            pl_module.train_metrics, train_method_name
        ).return_value = train_metrics_return
        getattr(
            pl_module.val_metrics, val_method_name
        ).return_value = val_metrics_return
        if policy_training is not None:
            pl_module.policy.training = policy_training
        pl_module.device = torch.device("cpu")
        return pl_module

    return factory
