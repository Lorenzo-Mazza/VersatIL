"""Tests for versatil.checkpoint_loading.base module."""

import io
from collections.abc import Iterator

import pytest
import torch
from omegaconf import OmegaConf

from versatil.checkpoint_loading.base import (
    BaseCheckpointLoader,
    unregistered_checkpoint_safe_globals,
    versatil_checkpoint_safe_globals,
)
from versatil.configs import TrainingConfig


class _ArbitraryCodeExecution:
    def __reduce__(self):
        return (print, ("arbitrary code executed",))


@pytest.mark.unit
class TestVersatilCheckpointSafeGlobals:
    def test_checkpoint_with_config_hyperparameters_loads(self):
        checkpoint = {
            "state_dict": {"layer.weight": torch.ones(2, 2)},
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


@pytest.fixture
def restore_safe_globals() -> Iterator[None]:
    """Snapshot torch's global safe-globals list and restore it on teardown."""
    previous = torch.serialization.get_safe_globals()
    yield
    torch.serialization.clear_safe_globals()
    torch.serialization.add_safe_globals(previous)


@pytest.mark.unit
class TestUnregisteredCheckpointSafeGlobals:
    def test_excludes_already_registered_classes(
        self, restore_safe_globals: None
    ) -> None:
        torch.serialization.add_safe_globals([TrainingConfig])
        unregistered = unregistered_checkpoint_safe_globals()
        assert TrainingConfig not in unregistered
        assert unregistered, "expected remaining unregistered safe globals"

    def test_context_exit_preserves_prior_registrations(
        self, restore_safe_globals: None
    ) -> None:
        torch.serialization.add_safe_globals([TrainingConfig])
        with torch.serialization.safe_globals(unregistered_checkpoint_safe_globals()):
            pass
        assert TrainingConfig in torch.serialization.get_safe_globals()


@pytest.mark.unit
class TestCheckpointLoadValidation:
    def test_unexpected_critical_checkpoint_key_raises(self) -> None:
        loader = BaseCheckpointLoader(
            device=torch.device("cpu"),
            checkpoint_path="/tmp/checkpoint",
        )
        checkpoint_state_dict = {
            "policy.decoder.old_projection.weight": torch.tensor([1.0])
        }
        model_state_dict = {
            "policy.decoder.new_projection.weight": torch.tensor([1.0]),
            "policy.decoder.extra_projection.weight": torch.tensor([1.0]),
        }

        with pytest.raises(RuntimeError, match="not loaded"):
            loader._validate_checkpoint_loading(
                checkpoint_state_dict=checkpoint_state_dict,
                model_state_dict=model_state_dict,
                missing_keys=[],
                unexpected_keys=["policy.decoder.old_projection.weight"],
            )

    def test_missing_critical_model_key_raises(self) -> None:
        loader = BaseCheckpointLoader(
            device=torch.device("cpu"),
            checkpoint_path="/tmp/checkpoint",
        )
        checkpoint_state_dict = {
            "policy.encoding_pipeline.camera.weight": torch.tensor([1.0])
        }
        model_state_dict = {
            "policy.encoding_pipeline.camera.weight": torch.tensor([1.0]),
            "policy.encoding_pipeline.new_camera.weight": torch.tensor([1.0]),
        }

        with pytest.raises(RuntimeError, match="missing from the checkpoint"):
            loader._validate_checkpoint_loading(
                checkpoint_state_dict=checkpoint_state_dict,
                model_state_dict=model_state_dict,
                missing_keys=["policy.encoding_pipeline.new_camera.weight"],
                unexpected_keys=[],
            )
