"""Base checkpoint loader with shared config, tokenizer, and metadata access."""

import logging
import os

import hydra
import torch
from omegaconf import OmegaConf

from versatil.configs import MainConfig
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.data.tokenization.tokenizer import Tokenizer
from versatil.models.policy import Policy
from versatil.training.constants import CheckpointKey
from versatil.validation import validate_experiment


class BaseCheckpointLoader:
    """Base class for policy checkpoint loaders.

    Handles configuration loading, tokenizer setup, and provides
    shared property accessors for observation/action spaces,
    horizons, denoising thresholds, and depth clamp ranges.

    Subclasses implement concrete checkpoint restoration.
    """

    def __init__(
        self,
        device: torch.device,
        checkpoint_path: str,
    ) -> None:
        """Initialize the base checkpoint loader.

        Args:
            device: Device to load the model onto.
            checkpoint_path: Path to the checkpoint directory.
        """
        self._device = device
        self._checkpoint_path = checkpoint_path
        self._tokenizer: Tokenizer | None = None
        self._config: MainConfig | None = None
        self._policy: Policy | None = None

    def _load_config(self, config_path: str) -> MainConfig:
        """Load and validate experiment configuration from YAML.

        Args:
            config_path: Path to the config.yaml file.

        Returns:
            Instantiated MainConfig.

        Raises:
            FileNotFoundError: If config file does not exist.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}.")
        logging.info(f"Loading config from {config_path}")
        config = hydra.utils.instantiate(OmegaConf.load(config_path))
        validate_experiment(config)
        return config

    def _load_tokenizer(self, tokenizer_path: str) -> Tokenizer | None:
        """Load tokenizer from disk if the directory exists.

        Args:
            tokenizer_path: Path to the tokenizer directory.

        Returns:
            Loaded Tokenizer or None if no encoder requires tokenization.

        Raises:
            FileNotFoundError: If the config requires observation tokenization
                but the tokenizer directory does not exist.
        """
        tokenization_required = (
            self._config is not None
            and self._config.task.dataloader.tokenization.tokenize_observations
        )
        if not os.path.exists(tokenizer_path):
            if tokenization_required:
                raise FileNotFoundError(
                    f"Config requires observation tokenization but no tokenizer "
                    f"found at {tokenizer_path}. Save the tokenizer alongside "
                    f"the checkpoint."
                )
            return None
        tokenizer = Tokenizer.from_pretrained(tokenizer_path, device=self._device)
        logging.info(f"Tokenizer loaded from {tokenizer_path}")
        return tokenizer

    def _validate_checkpoint_loading(
        self,
        checkpoint_state_dict: dict[str, torch.Tensor],
        model_state_dict: dict[str, torch.Tensor],
    ) -> None:
        """Validate that critical checkpoint components were properly loaded.

        Catches issues with lazy-initialized modules where checkpoint
        weights might be silently ignored with strict=False.

        Args:
            checkpoint_state_dict: State dict from the checkpoint file.
            model_state_dict: State dict from the loaded model.

        Raises:
            RuntimeError: If critical components failed to load.
        """
        checkpoint_keys = set(checkpoint_state_dict.keys())
        model_keys = set(model_state_dict.keys())

        critical_prefixes = [
            "policy.decoder.",
            "policy.encoding_pipeline.",
            "policy.normalizer.",
        ]
        errors = []
        warnings = []

        for prefix in critical_prefixes:
            checkpoint_count = len([k for k in checkpoint_keys if k.startswith(prefix)])
            model_count = len([k for k in model_keys if k.startswith(prefix)])
            if checkpoint_count > 0 and model_count == 0:
                errors.append(
                    f"CRITICAL: Checkpoint has {checkpoint_count} keys for "
                    f"'{prefix}' but model has NONE. "
                    f"Lazy-initialized layers likely failed to load."
                )
            elif checkpoint_count > 0 and model_count < checkpoint_count:
                matched = len(
                    [
                        k
                        for k in checkpoint_keys
                        if k.startswith(prefix) and k in model_keys
                    ]
                )
                if matched < checkpoint_count:
                    warnings.append(
                        f"Checkpoint has {checkpoint_count} keys for "
                        f"'{prefix}' but model only has {model_count}. "
                        f"Matched: {matched}/{checkpoint_count}"
                    )

        lazy_module_patterns = [
            (
                ".feature_projection.linear_projections.",
                "FeatureProjection linear",
            ),
            (
                ".feature_projection.spatial_projections.",
                "FeatureProjection spatial",
            ),
            (
                ".camera_embeddings.embeddings.",
                "DynamicFeatureEmbedding",
            ),
        ]
        for lazy_pattern, module_name in lazy_module_patterns:
            checkpoint_keys_for_module = [
                k for k in checkpoint_keys if lazy_pattern in k
            ]
            model_keys_for_module = [k for k in model_keys if lazy_pattern in k]
            if len(checkpoint_keys_for_module) > 0 and len(model_keys_for_module) == 0:
                errors.append(
                    f"CRITICAL: {module_name} failed to load. "
                    f"Checkpoint has {len(checkpoint_keys_for_module)} keys "
                    f"but model has NONE. "
                    f"Example keys: {checkpoint_keys_for_module[:3]}"
                )

        sample_keys = [k for k in checkpoint_keys if k in model_keys][:5]
        for key in sample_keys:
            checkpoint_value = checkpoint_state_dict[key]
            model_value = model_state_dict[key]
            if not torch.allclose(
                checkpoint_value.to(device=model_value.device, dtype=model_value.dtype),
                model_value,
                atol=1e-6,
            ):
                errors.append(
                    f"CRITICAL: Weight mismatch for '{key}'. "
                    f"Checkpoint and model values differ after "
                    f"load_state_dict."
                )

        for warning in warnings:
            logging.warning(warning)
        if errors:
            for error in errors:
                logging.error(error)
            raise RuntimeError(
                f"Checkpoint loading validation failed with "
                f"{len(errors)} critical error(s). "
                f"The model will NOT produce correct outputs. "
                f"First error: {errors[0]}"
            )

    @property
    def device(self) -> torch.device:
        """Get the checkpoint loading device."""
        return self._device

    @property
    def checkpoint_path(self) -> str:
        """Get the checkpoint directory path."""
        return self._checkpoint_path

    @property
    def config(self) -> MainConfig:
        """Get the loaded experiment configuration."""
        return self._config

    @property
    def tokenizer(self) -> Tokenizer | None:
        """Get the loaded tokenizer, if any."""
        return self._tokenizer

    @property
    def policy(self) -> Policy:
        """Get the restored policy."""
        return self._policy

    @property
    def observation_space(self) -> ObservationSpace:
        """Get the policy's observation space."""
        return self._policy.observation_space

    @property
    def action_space(self) -> ActionSpace:
        """Get the policy's action space."""
        return self._policy.action_space

    @property
    def prediction_horizon(self) -> int:
        """Get the policy's prediction horizon."""
        return self._policy.prediction_horizon

    @property
    def observation_horizon(self) -> int:
        """Get the decoder's observation horizon."""
        return self._policy.decoder.observation_horizon

    @property
    def denoising_thresholds(self) -> dict[str, float]:
        """Get denoising thresholds filtered to predicted action keys only.

        Returns:
            Dict mapping VersatIL action key to threshold. Empty if none set.
        """
        raw = {
            key: float(param.item())
            for key, param in self._policy.denoising_thresholds.params_dict.items()
        }
        if not raw:
            return {}
        return {
            key: raw[key] for key in self.action_space.actions_metadata if key in raw
        }

    @property
    def depth_clamp_range(self) -> tuple[float, float] | None:
        """Get depth image clamping range from normalizer statistics.

        Returns:
            Tuple of (min, max) for clamping, or None if depth not in normalizer.
        """
        for depth_key in self.observation_space.depth_cameras:
            if depth_key not in self._policy.normalizer.params_dict:
                continue
            stats = self._policy.normalizer[depth_key].params_dict.get(
                CheckpointKey.INPUT_STATS.value
            )
            if stats is not None:
                return float(stats["min"].item()), float(stats["max"].item())
        return None
