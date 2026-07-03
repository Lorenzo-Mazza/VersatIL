"""Tests for versatil.checkpoint_loading.base module."""

import io

import pytest
import torch
from omegaconf import OmegaConf

from versatil.checkpoint_loading.base import versatil_checkpoint_safe_globals
from versatil.configs import TrainingConfig


class _ArbitraryCodeExecution:
    def __reduce__(self):
        return (print, ("arbitrary code executed",))


@pytest.mark.unit
class TestVersatilCheckpointSafeGlobals:
    def test_checkpoint_with_config_hyperparameters_loads(self):
        checkpoint = {
            "state_dict": {"layer.weight": torch.randn(2, 2)},
            "hyper_parameters": OmegaConf.structured(TrainingConfig()),
            "epoch": 3,
        }
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)

        with torch.serialization.safe_globals(versatil_checkpoint_safe_globals()):
            loaded = torch.load(buffer, weights_only=True)

        assert sorted(loaded.keys()) == ["epoch", "hyper_parameters", "state_dict"]
        torch.testing.assert_close(
            loaded["state_dict"]["layer.weight"],
            checkpoint["state_dict"]["layer.weight"],
        )

    def test_malicious_pickle_is_rejected(self):
        buffer = io.BytesIO()
        torch.save({"state_dict": _ArbitraryCodeExecution()}, buffer)
        buffer.seek(0)

        with (
            torch.serialization.safe_globals(versatil_checkpoint_safe_globals()),
            pytest.raises(Exception, match="Weights only load failed"),
        ):
            torch.load(buffer, weights_only=True)
