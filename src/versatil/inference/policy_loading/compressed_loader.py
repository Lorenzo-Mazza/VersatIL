"""Loader for compressed (quantized) VersatIL policies saved as .pt2 archives."""

import logging
import os
from typing import Any

import hydra
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from versatil.common.tensor_ops import to_device
from versatil.data.constants import Cameras
from versatil.data.normalization.normalizer import LinearNormalizer
from versatil.data.processing.transform import (
    normalize_observation,
    tokenize_observation,
    unnormalize_actions,
)
from versatil.inference.policy_loading.base import BasePolicyLoader
from versatil.post_training_compression.compressor import PostTrainingCompressor
from versatil.post_training_compression.constants import (
    CompressionFilename,
    CompressionMetadataKey,
    QuantizationStrategy,
)
from versatil.post_training_compression.serialization import (
    load_compression_metadata,
)
from versatil.quantization.backends.base import BasePT2EBackend
from versatil.quantization.strategies import PT2EStrategy
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
        compile_model: bool = True,
    ) -> None:
        """Initialize the compressed policy loader.

        Args:
            device: Device to load the model onto.
            checkpoint_path: Path to the compressed checkpoint directory
                containing compression_metadata.json, the .pt2 model,
                and normalizer.pt.
            compile_model: Whether to compile the model with
                torch.compile before inference. For PT2E models on
                CPU this enables inductor int8 kernel fusion and is
                essential for performance. For other strategies it
                applies standard inductor compilation.
        """
        super().__init__(device=device, checkpoint_path=checkpoint_path)
        self._input_keys: list[str] = []
        self._output_keys: list[str] = []
        self._metadata: dict[str, Any] = {}
        self._normalizer: LinearNormalizer = LinearNormalizer()
        self._compressed_model: nn.Module | None = None
        self._compile_model = compile_model
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
        model = exported_program.module()
        strategy = self._metadata.get(
            CompressionMetadataKey.QUANTIZATION_STRATEGY.value
        )
        backend = self._load_backend(strategy=strategy)
        if backend is not None:
            self._validate_device(backend=backend)
        if self._compile_model and self._should_compile(
            strategy=strategy,
            device=self._device,
        ):
            model = self._compile_model_for_inference(
                model=model,
                backend=backend,
            )
        self._compressed_model = model
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

    def _load_backend(
        self,
        strategy: str | None,
    ) -> BasePT2EBackend | None:
        """Reconstruct the PT2E backend from the saved quantization config.

        Args:
            strategy: The quantization strategy from metadata.

        Returns:
            Instantiated backend, or None if not a PT2E model.
        """
        if strategy != QuantizationStrategy.PT2E.value:
            return None
        config_path = os.path.join(
            self._checkpoint_path,
            CompressionFilename.QUANTIZATION_CONFIG.value,
        )
        if not os.path.exists(config_path):
            return None
        config = OmegaConf.load(config_path)
        instance = hydra.utils.instantiate(config)
        if isinstance(instance, PostTrainingCompressor) and isinstance(
            instance.quantization, PT2EStrategy
        ):
            return instance.quantization.pt2e_backend
        else:
            return None

    @staticmethod
    def _should_compile(
        strategy: str | None,
        device: torch.device,
    ) -> bool:
        """Decide whether to compile based on strategy and device.

        CUDA + `quantize_()` models must not be compiled because
        `torch.compile` lowers int8 matmuls to torch._int_mm, which
        requires M > 16 on CUDA (cuBLAS constraint). Since inference
        batch size is unknown at load time (1 on real robot, N in
        sim), compilation is skipped entirely for this combination.

        See: https://github.com/pytorch/ao/issues/2376

        Args:
            strategy: The quantization strategy from metadata.
            device: Target inference device.

        Returns:
            True if the model should be compiled.
        """
        if (
            strategy == QuantizationStrategy.QUANTIZE_API.value
            and device.type == "cuda"
        ):
            logging.warning(
                "Skipping torch.compile for quantize_() model on CUDA. "
                "torch._int_mm requires batch > 16 on CUDA which "
                "cannot be guaranteed at inference time. "
                "See https://github.com/pytorch/ao/issues/2376"
            )
            return False
        return True

    def _validate_device(self, backend: BasePT2EBackend) -> None:
        """Validate that the device is supported by the backend.

        Args:
            backend: The PT2E backend instance.

        Raises:
            ValueError: If the device is not supported.
        """
        if self._device.type not in backend.supported_device_types:
            raise ValueError(
                f"Backend {type(backend).__name__} supports devices "
                f"{backend.supported_device_types}, got '{self._device}'."
            )

    @staticmethod
    def _compile_model_for_inference(
        model: nn.Module,
        backend: BasePT2EBackend | None,
    ) -> nn.Module:
        """Compile model with torch.compile, using backend env if available.

        For PT2E models, activates the backend environment
        permanently because torch.compile is lazy — the actual
        inductor compilation happens on the first forward pass.

        Args:
            model: The model to compile.
            backend: PT2E backend for env setup, or None.

        Returns:
            Compiled model.
        """
        if backend is not None:
            backend.activate_environment()
            compiled = torch.compile(model)
            logging.info(
                "Compiled PT2E model with %s backend",
                type(backend).__name__,
            )
        else:
            compiled = torch.compile(model)
            logging.info("Compiled model with inductor backend")
        return compiled

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
