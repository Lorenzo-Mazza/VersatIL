"""Loader for compressed (quantized) VersatIL policies saved as .pt2 archives."""

import logging
import os
from typing import Any

import torch
import torch.nn as nn

from versatil.common.tensor_ops import to_device
from versatil.data.constants import Cameras
from versatil.data.normalization.normalizer import LinearNormalizer
from versatil.data.transform import (
    normalize_observation,
    tokenize_observation,
    unnormalize_actions,
)
from versatil.inference.policy_loading.base import BasePolicyLoader
from versatil.post_training_compression.constants import (
    CompressionFilename,
    CompressionMetadataKey,
)
from versatil.post_training_compression.serialization import (
    load_compression_metadata,
)
from versatil.training.constants import CheckpointFilename, CheckpointKey


class CompressedPolicyLoader(BasePolicyLoader):
    """Loads a compressed policy from a .pt2 archive directory.

    Handles metadata loading, config loading from the training checkpoint,
    .pt2 model loading with backend lowering, standalone normalizer loading,
    and normalized inference.
    """

    def __init__(
        self,
        device: torch.device,
        checkpoint_path: str,
    ) -> None:
        """Initialize the compressed policy loader.

        Args:
            device: Device to load the model onto.
            checkpoint_path: Path to the compressed checkpoint directory
                containing compression_metadata.json, the .pt2 model,
                and normalizer.pt.
        """
        super().__init__(device=device, checkpoint_path=checkpoint_path)
        self._input_keys: list[str] = []
        self._output_keys: list[str] = []
        self._metadata: dict[str, Any] = {}
        self._normalizer: LinearNormalizer = LinearNormalizer()
        self._compressed_model: nn.Module | None = None
        self._load_compressed_model()

    def _load_compressed_model(self) -> None:
        """Load metadata, config, .pt2 model, normalizer, and tokenizer."""
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

        model_filename = self._metadata[CompressionMetadataKey.MODEL_FILE.value]
        model_path = os.path.join(self._checkpoint_path, model_filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Compressed model not found at {model_path}.")

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
        self._load_training_config(
            training_checkpoint_path=training_checkpoint_path,
        )
        exported_program = torch.export.load(model_path)
        self._compressed_model = exported_program.module()
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
            "Loaded compressed model from %s (%d input keys, %d output keys)",
            self._checkpoint_path,
            len(self._input_keys),
            len(self._output_keys),
        )

    def _load_training_config(
        self,
        training_checkpoint_path: str,
    ) -> None:
        """Load training config to access spaces and horizons.

        Instantiates the config and extracts the policy for metadata
        access without loading checkpoint weights.

        Args:
            training_checkpoint_path: Path to the original training
                checkpoint directory.
        """
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

    def run_inference(
        self, obs_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Run compressed policy inference with normalization.

        Normalizes observations, optionally tokenizes, converts to
        positional tensors, runs the compressed model, and converts
        output back to an unnormalized action dict.

        Args:
            obs_dict: Observation dictionary for the policy.

        Returns:
            Unnormalized action dictionary.
        """
        obs_dict = to_device(obs_dict, device=self._device)
        normalized_obs = normalize_observation(
            observation=obs_dict,
            normalizer=self._normalizer,
            observation_space=self.observation_space,
        )
        if (
            self._tokenizer is not None
            and self._tokenizer.observation_tokenizer is not None
        ):
            normalized_obs = tokenize_observation(
                observation=normalized_obs,
                obs_tokenizer=self._tokenizer.observation_tokenizer,
            )
        observation_tensors = tuple(normalized_obs[key] for key in self._input_keys)
        with torch.no_grad():
            output_tensors = self._compressed_model(*observation_tensors)

        if not isinstance(output_tensors, tuple):
            output_tensors = (output_tensors,)
        normalized_actions = {
            key: output_tensors[index] for index, key in enumerate(self._output_keys)
        }
        return unnormalize_actions(
            normalized_actions=normalized_actions,
            normalizer=self._normalizer,
            action_space=self.action_space,
        )

    @property
    def input_keys(self) -> list[str]:
        """Get the input key ordering from metadata."""
        return list(self._input_keys)

    @property
    def output_keys(self) -> list[str]:
        """Get the output key ordering from metadata."""
        return list(self._output_keys)

    @property
    def depth_clamp_range(self) -> tuple[float, float] | None:
        """Get depth image clamping range from normalizer statistics.

        Uses the standalone normalizer loaded from the compressed
        checkpoint, not the policy's normalizer.

        Returns:
            Tuple of (min, max) for clamping, or None if depth not
            in normalizer.
        """
        depth_key = Cameras.DEPTH.value
        if depth_key not in self._normalizer.params_dict:
            return None
        stats = self._normalizer[depth_key].params_dict.get(
            CheckpointKey.INPUT_STATS.value
        )
        if stats is None:
            return None
        return float(stats["min"].item()), float(stats["max"].item())
