"""Tests for versatil.checkpoint_loading.base module."""

import io
import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

from versatil.checkpoint_loading.base import (
    BaseCheckpointLoader,
    versatil_checkpoint_safe_globals,
)
from versatil.checkpoint_loading.metadata import CheckpointMetadata
from versatil.configs import TrainingConfig
from versatil.data.tokenization.tokenizer import Tokenizer

BASE_LOADER_MODULE = "versatil.checkpoint_loading.base"


@pytest.fixture
def metadata_loader_factory(
    checkpoint_metadata_factory: Callable[..., CheckpointMetadata],
) -> Callable[..., BaseCheckpointLoader]:
    def factory(thresholds: dict[str, float]) -> BaseCheckpointLoader:
        loader = BaseCheckpointLoader(
            device=torch.device("cpu"), checkpoint_path="/checkpoint"
        )
        loader._checkpoint_metadata = checkpoint_metadata_factory(
            observation_horizon=2, prediction_horizon=4
        )
        loader._denoising_thresholds = thresholds.copy()
        return loader

    return factory


@pytest.fixture
def tokenizer_loader_factory(
    checkpoint_config_factory: Callable[..., MagicMock],
) -> Callable[..., tuple[BaseCheckpointLoader, MagicMock]]:
    def factory(
        tokenize_observations: bool,
        has_observation_tokenizer: bool,
    ) -> tuple[BaseCheckpointLoader, MagicMock]:
        loader = BaseCheckpointLoader(
            device=torch.device("cpu"), checkpoint_path="/checkpoint"
        )
        loader._config = checkpoint_config_factory()
        loader._config.task.dataloader.tokenization.tokenize_observations = (
            tokenize_observations
        )
        tokenizer = MagicMock(spec=Tokenizer)
        tokenizer.observation_tokenizer = (
            MagicMock() if has_observation_tokenizer else None
        )
        return loader, tokenizer

    return factory


@pytest.mark.unit
class TestCheckpointMetadataAccess:
    @pytest.mark.parametrize(
        "thresholds,expected",
        [
            ({}, {}),
            ({"position": 0.05}, {"position": 0.05}),
            ({"unpredicted": 0.4}, {}),
            ({"position": 0.0, "unpredicted": 0.4}, {"position": 0.0}),
        ],
    )
    def test_thresholds_follow_the_action_space(
        self,
        metadata_loader_factory: Callable[..., BaseCheckpointLoader],
        thresholds: dict[str, float],
        expected: dict[str, float],
    ) -> None:
        loader = metadata_loader_factory(thresholds=thresholds)

        assert loader.denoising_thresholds == expected
        loader.denoising_thresholds["position"] = 10.0
        assert loader.denoising_thresholds == expected

    def test_delegates_spaces_and_horizons_to_metadata(
        self, metadata_loader_factory: Callable[..., BaseCheckpointLoader]
    ) -> None:
        loader = metadata_loader_factory(thresholds={})

        assert loader.observation_space == loader._checkpoint_metadata.observation_space
        assert loader.action_space == loader._checkpoint_metadata.action_space
        assert loader.observation_horizon == 2
        assert loader.prediction_horizon == 4


@pytest.mark.unit
@pytest.mark.parametrize("tokenize_observations", [False, True])
@pytest.mark.parametrize("has_observation_tokenizer", [False, True])
def test_saved_tokenizer_contains_required_observation_assets(
    tokenizer_loader_factory: Callable[..., tuple[BaseCheckpointLoader, MagicMock]],
    tokenize_observations: bool,
    has_observation_tokenizer: bool,
) -> None:
    loader, tokenizer = tokenizer_loader_factory(
        tokenize_observations=tokenize_observations,
        has_observation_tokenizer=has_observation_tokenizer,
    )
    expectation = (
        pytest.raises(
            ValueError,
            match=re.escape(
                "Observation tokenization requires saved observation-tokenizer "
                "assets at /checkpoint/tokenizer."
            ),
        )
        if tokenize_observations and not has_observation_tokenizer
        else does_not_raise()
    )
    with (
        patch(f"{BASE_LOADER_MODULE}.os.path.exists", return_value=True) as exists,
        patch(
            f"{BASE_LOADER_MODULE}.Tokenizer.from_pretrained", return_value=tokenizer
        ) as load_tokenizer,
        expectation,
    ):
        restored = loader._load_tokenizer(tokenizer_path="/checkpoint/tokenizer")
        assert restored == tokenizer

    exists.assert_called_once_with("/checkpoint/tokenizer")
    load_tokenizer.assert_called_once_with(
        "/checkpoint/tokenizer", device=torch.device("cpu")
    )


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
