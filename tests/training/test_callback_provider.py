"""Tests for versatil.training.callback_provider module."""

from unittest.mock import MagicMock

import pytest
from pytorch_lightning import Callback

from versatil.configs.experiment import ExperimentConfig
from versatil.models.decoding.algorithm.variational import VariationalAlgorithm
from versatil.models.decoding.decoders.factory.free_action_transformer import (
    FreeActionTransformer,
)
from versatil.training.callback_provider import CallbackProvider


class _ImplementsProtocol:
    """Minimal class that satisfies the CallbackProvider protocol."""

    def get_callbacks(self, experiment_config: ExperimentConfig) -> list[Callback]:
        return [Callback()]


class _DoesNotImplementProtocol:
    """Class missing the get_callbacks method."""

    pass


class _WrongSignature:
    """Class with a get_callbacks that has a wrong signature (no experiment_config)."""

    def get_callbacks(self) -> list[Callback]:
        return []


@pytest.mark.unit
class TestCallbackProviderProtocol:
    def test_class_with_get_callbacks_is_recognized(self):
        instance = _ImplementsProtocol()
        assert isinstance(instance, CallbackProvider)

    def test_class_without_get_callbacks_is_not_recognized(self):
        instance = _DoesNotImplementProtocol()
        assert not isinstance(instance, CallbackProvider)

    def test_wrong_signature_still_satisfies_runtime_checkable(self):
        # runtime_checkable only checks method existence, not signature
        instance = _WrongSignature()
        assert isinstance(instance, CallbackProvider)

    def test_get_callbacks_returns_callbacks(self):
        provider = _ImplementsProtocol()
        experiment_config = MagicMock(spec=ExperimentConfig)
        callbacks = provider.get_callbacks(experiment_config=experiment_config)
        assert len(callbacks) == 1
        assert isinstance(callbacks[0], Callback)

    def test_real_components_have_get_callbacks(self):
        assert hasattr(VariationalAlgorithm, "get_callbacks")
        assert hasattr(FreeActionTransformer, "get_callbacks")
