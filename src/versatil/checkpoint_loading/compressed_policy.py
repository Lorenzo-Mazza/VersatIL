"""Restore a policy checkpoint after post-training-compression, along with its metadata."""

import logging
import os
from typing import Any

import torch

from versatil.checkpoint_loading.base import BaseCheckpointLoader
from versatil.data.normalization.normalizer import LinearNormalizer
from versatil.post_training_compression.constants import (
    ArtifactFormat,
    CompressionFilename,
    CompressionMetadataKey,
)
from versatil.post_training_compression.serialization import load_compression_metadata
from versatil.training.constants import CheckpointFilename, CheckpointKey


class CompressedCheckpointLoader(BaseCheckpointLoader):
    """Restore compressed policy checkpoint state."""

    def __init__(
        self,
        device: torch.device,
        checkpoint_path: str,
    ) -> None:
        """Initialize and restore compressed checkpoint state."""
        super().__init__(device=device, checkpoint_path=checkpoint_path)
        self._input_keys: list[str] = []
        self._output_keys: list[str] = []
        self._metadata: dict[str, Any] = {}
        self._artifact_format = ArtifactFormat.TORCH_EXPORT_PT2.value
        self._normalizer: LinearNormalizer = LinearNormalizer()
        self._model_path = ""
        self._workflow: str | None = None
        self._load_compressed_checkpoint()

    def _load_compressed_checkpoint(self) -> None:
        """Load post-training compression metadata, config, normalizer, and tokenizer."""
        metadata_path = os.path.join(
            self._checkpoint_path,
            CompressionFilename.COMPRESSION_METADATA.value,
        )
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Compression metadata not found at {metadata_path}. "
                f"Is this a compressed checkpoint directory?"
            )
        self._metadata = load_compression_metadata(metadata_path=metadata_path)
        self._input_keys = self._metadata[CompressionMetadataKey.INPUT_KEYS.value]
        self._output_keys = self._metadata[CompressionMetadataKey.OUTPUT_KEYS.value]
        self._artifact_format = self._metadata.get(
            CompressionMetadataKey.ARTIFACT_FORMAT.value,
            ArtifactFormat.TORCH_EXPORT_PT2.value,
        )
        if (
            self._artifact_format == ArtifactFormat.EXECUTORCH_PTE.value
            and self._device.type != "cpu"
        ):
            raise ValueError(
                "ExecuTorch XNNPACK artifacts support CPU inference only, "
                f"got '{self._device}'."
            )

        model_filename = self._metadata[CompressionMetadataKey.MODEL_FILE.value]
        self._model_path = os.path.join(self._checkpoint_path, model_filename)
        if not os.path.exists(self._model_path):
            raise FileNotFoundError(
                f"Compressed model not found at {self._model_path}."
            )

        normalizer_filename = self._metadata[
            CompressionMetadataKey.NORMALIZER_FILE.value
        ]
        normalizer_path = os.path.join(self._checkpoint_path, normalizer_filename)
        if not os.path.exists(normalizer_path):
            raise FileNotFoundError(f"Normalizer not found at {normalizer_path}.")

        training_checkpoint_path = self._metadata.get(
            CompressionMetadataKey.TRAINING_CHECKPOINT_PATH.value
        )
        if training_checkpoint_path is None:
            raise ValueError(
                "Compression metadata is missing "
                f"'{CompressionMetadataKey.TRAINING_CHECKPOINT_PATH.value}'."
            )
        self._load_training_config(training_checkpoint_path=training_checkpoint_path)
        saved_thresholds = self._metadata.get(
            CompressionMetadataKey.DENOISING_THRESHOLDS.value
        )
        if saved_thresholds:
            self._policy.set_denoising_thresholds(thresholds=saved_thresholds)
        self._workflow = self._metadata.get(
            CompressionMetadataKey.QUANTIZATION_WORKFLOW.value
        )
        normalizer_state = torch.load(
            normalizer_path,
            map_location=self._device,
            weights_only=True,
        )
        self._normalizer.load_state_dict(normalizer_state)
        self._normalizer.to(self._device)

        local_tokenizer_path = os.path.join(
            self._checkpoint_path, CheckpointFilename.TOKENIZER_DIR.value
        )
        remote_tokenizer_path = os.path.join(
            training_checkpoint_path, CheckpointFilename.TOKENIZER_DIR.value
        )
        tokenizer_path = (
            local_tokenizer_path
            if os.path.exists(local_tokenizer_path)
            else remote_tokenizer_path
        )
        self._tokenizer = self._load_tokenizer(tokenizer_path=tokenizer_path)
        if self._tokenizer is not None:
            self._tokenizer.to(self._device)

        logging.info(
            "Loaded compressed checkpoint state from %s (%s, %d input keys, "
            "%d output keys)",
            self._checkpoint_path,
            self._artifact_format,
            len(self._input_keys),
            len(self._output_keys),
        )

    def _load_training_config(
        self,
        training_checkpoint_path: str,
    ) -> None:
        """Load training config for metadata access."""
        local_config_path = os.path.join(
            self._checkpoint_path, CheckpointFilename.CONFIG.value
        )
        remote_config_path = os.path.join(
            training_checkpoint_path, CheckpointFilename.CONFIG.value
        )
        config_path = (
            local_config_path
            if os.path.exists(local_config_path)
            else remote_config_path
        )
        self._config = self._load_config(config_path=config_path)
        self._policy = self._config.policy
        self._policy.to(self._device)

    @property
    def input_keys(self) -> list[str]:
        """Get the input key ordering from metadata."""
        return list(self._input_keys)

    @property
    def output_keys(self) -> list[str]:
        """Get the output key ordering from metadata."""
        return list(self._output_keys)

    @property
    def artifact_format(self) -> str:
        """Get the serialized artifact format."""
        return self._artifact_format

    @property
    def metadata(self) -> dict[str, Any]:
        """Get the loaded compression metadata."""
        return self._metadata

    @property
    def model_path(self) -> str:
        """Get the compressed model artifact path."""
        return self._model_path

    @property
    def normalizer(self) -> LinearNormalizer:
        """Get the compressed model normalizer."""
        return self._normalizer

    @property
    def workflow(self) -> str | None:
        """Get the serialized quantization workflow."""
        return self._workflow

    @property
    def depth_clamp_ranges(self) -> dict[str, tuple[float, float]]:
        """Get per-camera depth clamping ranges from the compressed normalizer."""
        clamp_ranges: dict[str, tuple[float, float]] = {}
        for depth_key in self.observation_space.depth_cameras:
            if depth_key not in self._normalizer.params_dict:
                continue
            stats = self._normalizer[depth_key].params_dict.get(
                CheckpointKey.INPUT_STATS.value
            )
            if stats is not None:
                clamp_ranges[depth_key] = (
                    float(stats["min"].item()),
                    float(stats["max"].item()),
                )
        return clamp_ranges
