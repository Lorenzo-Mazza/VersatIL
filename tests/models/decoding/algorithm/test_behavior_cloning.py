"""Tests for versatil.models.decoding.algorithm.behavior_cloning module."""

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
import torch

from versatil.models.decoding.algorithm.base import DecodingAlgorithm
from versatil.models.decoding.algorithm.behavior_cloning import BehavioralCloning


@pytest.fixture
def bc_factory() -> Callable[..., BehavioralCloning]:
    """Factory for BehavioralCloning instances."""

    def factory() -> BehavioralCloning:
        return BehavioralCloning()

    return factory


def test_inherits_from_decoding_algorithm(
    bc_factory: Callable[..., BehavioralCloning],
):
    bc = bc_factory()
    assert isinstance(bc, DecodingAlgorithm)


def test_predicts_in_action_space_is_true(
    bc_factory: Callable[..., BehavioralCloning],
):
    bc = bc_factory()
    assert bc.predicts_in_action_space is True


def test_auxiliary_output_keys_are_empty(
    bc_factory: Callable[..., BehavioralCloning],
):
    bc = bc_factory()
    assert bc.get_auxiliary_output_keys() == set()


def test_get_targets_returns_ground_truth_actions(
    bc_factory: Callable[..., BehavioralCloning],
    action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
):
    bc = bc_factory()
    actions = action_dictionary_factory(
        action_keys=["position_action"],
        prediction_horizon=8,
        action_dimension=3,
    )
    targets = bc.get_targets(
        algorithm_output={"position_action": torch.zeros(2, 8, 3)},
        ground_truth_actions=actions,
    )
    assert torch.equal(targets["position_action"], actions["position_action"])


class TestBehavioralCloningForward:
    def test_forwards_features_and_actions_to_network(
        self,
        bc_factory: Callable[..., BehavioralCloning],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        bc = bc_factory()
        mock_network = mock_action_decoder_factory()
        features = feature_dictionary_factory()
        actions = action_dictionary_factory(
            action_keys=["position_action"],
            prediction_horizon=8,
            action_dimension=3,
        )
        bc.forward(network=mock_network, features=features, actions=actions)
        mock_network.assert_called_once_with(features=features, actions=actions)

    def test_returns_network_output(
        self,
        bc_factory: Callable[..., BehavioralCloning],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        bc = bc_factory()
        mock_network = mock_action_decoder_factory(action_keys=["you_shall_not_pass"])
        features = feature_dictionary_factory()
        result = bc.forward(network=mock_network, features=features)
        assert set(result.keys()) == {"you_shall_not_pass"}


class TestBehavioralCloningPredict:
    def test_forwards_features_with_actions_none(
        self,
        bc_factory: Callable[..., BehavioralCloning],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        bc = bc_factory()
        mock_network = mock_action_decoder_factory()
        features = feature_dictionary_factory()
        bc.predict(network=mock_network, features=features)
        mock_network.assert_called_once_with(features=features, actions=None)

    def test_returns_network_output(
        self,
        bc_factory: Callable[..., BehavioralCloning],
        mock_action_decoder_factory: Callable[..., MagicMock],
        feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        bc = bc_factory()
        mock_network = mock_action_decoder_factory(action_keys=["where_was_Gondor?"])
        features = feature_dictionary_factory()
        result = bc.predict(network=mock_network, features=features)
        assert set(result.keys()) == {"where_was_Gondor?"}
