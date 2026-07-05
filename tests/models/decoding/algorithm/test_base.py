"""Tests for versatil.models.decoding.algorithm.base module."""

from collections.abc import Callable

import pytest
import torch
import torch.nn as nn

from versatil.models.decoding.algorithm.base import DecodingAlgorithm


class _MinimalAlgorithm(DecodingAlgorithm):
    """Smallest concrete algorithm exercising the base class default methods."""

    def forward(
        self,
        network: nn.Module,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        return network(features=features, actions=actions)

    def predict(
        self,
        network: nn.Module,
        features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return network(features=features, actions=None)


@pytest.fixture
def minimal_algorithm_factory() -> Callable[..., _MinimalAlgorithm]:
    """Factory for the minimal concrete DecodingAlgorithm."""

    def factory() -> _MinimalAlgorithm:
        return _MinimalAlgorithm()

    return factory


class TestDecodingAlgorithmInterface:
    def test_is_abstract(self):
        with pytest.raises(
            TypeError,
            match="Can't instantiate abstract class DecodingAlgorithm",
        ):
            DecodingAlgorithm()

    def test_inherits_from_nn_module(self):
        assert issubclass(DecodingAlgorithm, nn.Module)


def test_get_targets_default_returns_ground_truth_actions(
    minimal_algorithm_factory: Callable[..., _MinimalAlgorithm],
    action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
):
    algorithm = minimal_algorithm_factory()
    actions = action_dictionary_factory(
        action_keys=["position_action"],
        prediction_horizon=8,
        action_dimension=3,
    )
    targets = algorithm.get_targets(
        algorithm_output={"position_action": torch.zeros(2, 8, 3)},
        ground_truth_actions=actions,
    )
    assert targets is actions
    assert torch.equal(targets["position_action"], actions["position_action"])


def test_get_auxiliary_output_keys_default_is_empty(
    minimal_algorithm_factory: Callable[..., _MinimalAlgorithm],
):
    algorithm = minimal_algorithm_factory()
    assert algorithm.get_auxiliary_output_keys() == set()


def test_predicts_in_action_space_default_is_true(
    minimal_algorithm_factory: Callable[..., _MinimalAlgorithm],
):
    algorithm = minimal_algorithm_factory()
    assert algorithm.predicts_in_action_space is True


def test_injected_feature_keys_default_is_empty(
    minimal_algorithm_factory: Callable[..., _MinimalAlgorithm],
):
    algorithm = minimal_algorithm_factory()
    assert algorithm.injected_feature_keys() == set()
