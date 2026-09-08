"""Restore a policy checkpoint after post-training-compression, along with its metadata."""

import logging
import os
from typing import Any

import hydra
import torch
from omegaconf import OmegaConf

from versatil.checkpoint_loading.base import BaseCheckpointLoader
from versatil.checkpoint_loading.metadata import CheckpointMetadata
from versatil.data.normalization.normalizer import LinearNormalizer
from versatil.data.tokenization.action_discretizer import (
    BinnedActionDiscretizer,
    FastActionDiscretizer,
)
from versatil.models.decoding.constants import DecoderOutputKey
from versatil.models.exportable.metadata import (
    PolicyExportMetadata,
    PredictionOutput,
)
from versatil.post_training_compression.constants import (
    ArtifactFormat,
    CompressionFilename,
    CompressionMetadataKey,
)
from versatil.post_training_compression.serialization import load_compression_metadata
from versatil.training.constants import CheckpointFilename


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
        self._export_metadata = PolicyExportMetadata()
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
        self._export_metadata = PolicyExportMetadata.from_dict(
            metadata=self._metadata.get(
                CompressionMetadataKey.POLICY_EXPORT_METADATA.value,
                self._metadata.get(
                    CompressionMetadataKey.LEGACY_INFERENCE_CONTRACT.value, {}
                ),
            )
        )
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
        self._denoising_thresholds = {
            key: float(value)
            for key, value in (
                self._metadata.get(CompressionMetadataKey.DENOISING_THRESHOLDS.value)
                or {}
            ).items()
        }
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

        tokenizer_path = self._resolve_tokenizer_path(
            training_checkpoint_path=training_checkpoint_path
        )
        self._tokenizer = self._load_tokenizer(tokenizer_path=tokenizer_path)
        if self._tokenizer is not None:
            self._tokenizer.to(self._device)
        self._validate_export_metadata()

        logging.info(
            f"Loaded compressed checkpoint state from {self._checkpoint_path} "
            f"({self._artifact_format}, {len(self._input_keys)} input keys, "
            f"{len(self._output_keys)} output keys)"
        )

    def _resolve_tokenizer_path(self, training_checkpoint_path: str) -> str:
        """Select tokenizer assets for the artifact's prediction format.

        Args:
            training_checkpoint_path: Original training directory, used as the
                fallback asset location for continuous-action artifacts.

        Returns:
            Local tokenizer directory for action-token artifacts. Continuous
            artifacts prefer local assets and fall back to the training directory.

        Raises:
            FileNotFoundError: If an action-token artifact's local tokenizer
                directory is missing.
        """
        local_tokenizer_path = os.path.join(
            self._checkpoint_path, CheckpointFilename.TOKENIZER_DIR.value
        )
        if self._export_metadata.output == PredictionOutput.ACTION_TOKENS:
            if not os.path.isdir(local_tokenizer_path):
                raise FileNotFoundError(
                    "Action-token artifacts require local tokenizer assets at "
                    f"{local_tokenizer_path}."
                )
            return local_tokenizer_path
        if os.path.exists(local_tokenizer_path):
            return local_tokenizer_path
        return os.path.join(
            training_checkpoint_path, CheckpointFilename.TOKENIZER_DIR.value
        )

    def _validate_export_metadata(self) -> None:
        """Check token outputs against the tokenizer stored with the artifact.

        Raises:
            ValueError: If token output keys or tokenizer assets conflict with
                the action-reconstruction requirements.
        """
        if self._export_metadata.output != PredictionOutput.ACTION_TOKENS:
            return
        token_key = DecoderOutputKey.PREDICTED_ACTION_TOKENS.value
        if self._output_keys != [token_key]:
            raise ValueError(
                f"Action-token artifacts require output_keys=[{token_key!r}], "
                f"got {self._output_keys!r}."
            )
        if self._tokenizer is None or self._tokenizer.action_tokenizer is None:
            raise ValueError(
                "Action-token artifacts require a saved action tokenizer in the "
                "compressed checkpoint's tokenizer directory."
            )
        discretizer = self._tokenizer.action_tokenizer.action_discretizer
        if not isinstance(
            discretizer, (BinnedActionDiscretizer, FastActionDiscretizer)
        ):
            raise ValueError(
                "Action-token artifacts require a binned or FAST action discretizer; "
                f"loaded {type(discretizer).__name__}."
            )
        if not discretizer.is_fitted:
            raise ValueError(
                "Action-token artifacts require a fitted action discretizer."
            )
        expected_shape = (
            self.prediction_horizon,
            self.action_space.get_total_action_dim(),
        )
        tokenizer_shape = (discretizer.time_horizon, discretizer.action_dim)
        if tokenizer_shape != expected_shape:
            raise ValueError(
                f"Saved action tokenizer decodes shape {tokenizer_shape}, "
                f"but the policy requires {expected_shape}."
            )

    def _load_training_config(
        self,
        training_checkpoint_path: str,
    ) -> None:
        """Restore spaces and horizons from the saved training configuration.

        Args:
            training_checkpoint_path: Fallback directory for legacy artifacts
                whose training configuration remains in the original checkpoint.

        Note:
            Space configurations retain access to the complete YAML tree while
            Hydra resolves their interpolations. Construction is limited to the
            observation and action metadata used by inference.

        Raises:
            FileNotFoundError: If both configuration locations are unavailable.
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
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}.")
        self._config = OmegaConf.load(config_path)
        self._checkpoint_metadata = CheckpointMetadata(
            observation_space=hydra.utils.instantiate(
                self._config.policy.observation_space
            ),
            action_space=hydra.utils.instantiate(self._config.policy.action_space),
            prediction_horizon=self._config.policy.prediction_horizon,
            observation_horizon=self._config.policy.decoder.observation_horizon,
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
    def artifact_format(self) -> str:
        """Get the serialized artifact format."""
        return self._artifact_format

    @property
    def metadata(self) -> dict[str, Any]:
        """Get the loaded compression metadata."""
        return self._metadata

    @property
    def export_metadata(self) -> PolicyExportMetadata:
        """Graph output format and additional tensor inputs saved with the artifact."""
        return self._export_metadata

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
