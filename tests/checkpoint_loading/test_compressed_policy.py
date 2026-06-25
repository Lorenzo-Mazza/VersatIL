"""Tests for versatil.checkpoint_loading.compressed_policy module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from versatil.checkpoint_loading.base import BaseCheckpointLoader
from versatil.checkpoint_loading.compressed_policy import CompressedCheckpointLoader
from versatil.data.constants import Cameras
from versatil.data.normalization.normalizer import LinearNormalizer


@pytest.mark.unit
class TestCompressedCheckpointLoaderTrainingConfig:
    def test_falls_back_to_remote_config_when_local_absent(self) -> None:
        loader = CompressedCheckpointLoader.__new__(CompressedCheckpointLoader)
        loader._device = torch.device("cpu")
        loader._checkpoint_path = "/tmp/compressed"

        config = MagicMock()
        config.policy = MagicMock()

        with patch.object(
            BaseCheckpointLoader,
            "_load_config",
            return_value=config,
        ) as mock_load_config:
            loader._load_training_config(training_checkpoint_path="/tmp/train")

        mock_load_config.assert_called_once_with(config_path="/tmp/train/config.yaml")
        assert loader._policy is config.policy
        config.policy.to.assert_called_once_with(torch.device("cpu"))

    def test_prefers_local_config_when_present(self, tmp_path: Path) -> None:
        compressed_dir = tmp_path / "compressed"
        compressed_dir.mkdir()
        (compressed_dir / "config.yaml").write_text("local: true")

        loader = CompressedCheckpointLoader.__new__(CompressedCheckpointLoader)
        loader._device = torch.device("cpu")
        loader._checkpoint_path = str(compressed_dir)

        config = MagicMock()
        config.policy = MagicMock()

        with patch.object(
            BaseCheckpointLoader,
            "_load_config",
            return_value=config,
        ) as mock_load_config:
            loader._load_training_config(training_checkpoint_path="/tmp/train")

        mock_load_config.assert_called_once_with(
            config_path=str(compressed_dir / "config.yaml"),
        )


@pytest.mark.unit
class TestCompressedCheckpointLoaderMetadataProperties:
    @pytest.mark.parametrize(
        "property_name, attribute_name",
        [
            ("input_keys", "_input_keys"),
            ("output_keys", "_output_keys"),
        ],
    )
    def test_key_properties_return_copies(
        self,
        property_name: str,
        attribute_name: str,
    ) -> None:
        loader = CompressedCheckpointLoader.__new__(CompressedCheckpointLoader)
        setattr(loader, attribute_name, ["left", "depth"])

        returned = getattr(loader, property_name)
        returned.append("mutated")

        assert getattr(loader, attribute_name) == ["left", "depth"]

    def test_depth_clamp_range_uses_compressed_normalizer(self) -> None:
        normalizer = LinearNormalizer()
        normalizer.fit(
            {
                Cameras.DEPTH.value: torch.tensor([[0.1], [0.5], [0.9], [0.2]]),
            }
        )
        policy = MagicMock()
        policy.observation_space.depth_cameras = {Cameras.DEPTH.value: MagicMock()}
        loader = CompressedCheckpointLoader.__new__(CompressedCheckpointLoader)
        loader._normalizer = normalizer
        loader._policy = policy

        result = loader.depth_clamp_range

        assert result is not None
        minimum, maximum = result
        assert minimum == pytest.approx(0.1, abs=1e-5)
        assert maximum == pytest.approx(0.9, abs=1e-5)
