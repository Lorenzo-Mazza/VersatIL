"""Standalone utility for loading a trained policy from a checkpoint directory."""

import logging
import os

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from versatil.configs import MainConfig
from versatil.data.constants import Cameras
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.data.tokenization.tokenizer import Tokenizer
from versatil.models.policy import Policy
from versatil.training.constants import MAP_PRECISION_TO_DTYPE, PrecisionType
from versatil.training.lightning_policy import LightningPolicy
from versatil.validation import validate_experiment


class PolicyLoader:
    """Loads a trained policy from a checkpoint directory.

    Handles configuration loading, checkpoint validation, tokenizer setup,
    precision conversion, and autocast inference wrapping.
    """

    def __init__(
        self,
        device: torch.device,
        checkpoint_path: str,
        checkpoint_name: str = "last.ckpt",
        precision: str = PrecisionType.BF16_MIXED.value,
        seed: int = 42,
    ):
        """Initialize the policy loader.

        Args:
            device: Device to load the model onto.
            checkpoint_path: Path to the checkpoint directory.
            checkpoint_name: Name of the checkpoint file.
            precision: Precision type for model inference.
            seed: Random seed for reproducibility.
        """
        self._device = device
        self._checkpoint_path = checkpoint_path
        self._checkpoint_name = checkpoint_name
        self._precision = precision
        self._tokenizer: Tokenizer | None = None
        self._config: MainConfig | None = None
        self._policy: Policy | None = None
        self._set_seed(seed)
        self._load_model()

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        self._rng = rng
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _load_model(self) -> None:
        """Load config, policy, and tokenizer from checkpoint directory."""
        config_path = os.path.join(self._checkpoint_path, "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Config file not found at {config_path}."
            )
        logging.info(f"Loading config from {config_path}")
        config = hydra.utils.instantiate(OmegaConf.load(config_path))
        validate_experiment(config)
        self._config = config

        checkpoint_file = os.path.join(
            self._checkpoint_path, self._checkpoint_name
        )
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(
                f"No checkpoint found at {checkpoint_file}."
            )
        logging.info(f"Loading model and tokenizer from {checkpoint_file}")

        tokenizer_path = os.path.join(self._checkpoint_path, "tokenizer")
        if os.path.exists(tokenizer_path):
            self._tokenizer = Tokenizer.from_pretrained(
                tokenizer_path, device=self._device
            )
            logging.info(f"Tokenizer loaded from {tokenizer_path}")

        self._policy = self._config.policy
        if self._tokenizer is not None:
            self._tokenizer.to(self._device)
            self._policy.set_tokenizer(self._tokenizer)

        self._policy.to(self._device).eval()

        checkpoint = torch.load(
            checkpoint_file,
            map_location=self._device,
            weights_only=False,
        )
        lightning_module = LightningPolicy(
            policy=self._policy,
            training_config=self._config.training,
        )
        lightning_module.load_state_dict(
            checkpoint["state_dict"], strict=False
        )
        self._validate_checkpoint_loading(
            checkpoint_state_dict=checkpoint["state_dict"],
            lightning_module=lightning_module,
        )

        precision_type = PrecisionType(self._precision)
        if precision_type.should_convert_model():
            self._policy = self._policy.to(precision_type.get_model_dtype())

        logging.info("Model and config successfully loaded.")

    def _validate_checkpoint_loading(
        self,
        checkpoint_state_dict: dict[str, torch.Tensor],
        lightning_module: LightningPolicy,
    ) -> None:
        """Validate that critical checkpoint components were properly loaded.

        Catches issues with lazy-initialized modules where checkpoint
        weights might be silently ignored with strict=False.

        Raises:
            RuntimeError: If critical components failed to load.
        """
        model_state = lightning_module.state_dict()
        checkpoint_keys = set(checkpoint_state_dict.keys())
        model_keys = set(model_state.keys())

        critical_prefixes = [
            "policy.decoder.",
            "policy.encoding_pipeline.",
            "policy.normalizer.",
        ]
        errors = []
        warnings = []

        for prefix in critical_prefixes:
            checkpoint_count = len(
                [k for k in checkpoint_keys if k.startswith(prefix)]
            )
            model_count = len(
                [k for k in model_keys if k.startswith(prefix)]
            )
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

        lazy_module_prefixes = [
            (
                "policy.decoder.architecture.feature_projection."
                "linear_projections.",
                "FeatureProjection linear",
            ),
            (
                "policy.decoder.architecture.feature_projection."
                "spatial_projections.",
                "FeatureProjection spatial",
            ),
            (
                "policy.decoder.architecture.camera_embeddings.embeddings.",
                "DynamicFeatureEmbedding",
            ),
        ]
        for checkpoint_prefix, module_name in lazy_module_prefixes:
            checkpoint_keys_for_module = [
                k
                for k in checkpoint_keys
                if k.startswith(checkpoint_prefix)
            ]
            model_keys_for_module = [
                k for k in model_keys if k.startswith(checkpoint_prefix)
            ]
            if (
                len(checkpoint_keys_for_module) > 0
                and len(model_keys_for_module) == 0
            ):
                errors.append(
                    f"CRITICAL: {module_name} failed to load. "
                    f"Checkpoint has {len(checkpoint_keys_for_module)} keys "
                    f"but model has NONE. "
                    f"Example keys: {checkpoint_keys_for_module[:3]}"
                )

        sample_keys = [k for k in checkpoint_keys if k in model_keys][:5]
        for key in sample_keys:
            checkpoint_value = checkpoint_state_dict[key]
            model_value = model_state[key]
            if not torch.allclose(
                checkpoint_value.to(model_value.device),
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

    def run_inference(
        self, obs_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Run policy inference with autocast and no_grad.

        Args:
            obs_dict: Observation dictionary for the policy.

        Returns:
            Action dictionary from policy.predict_action.
        """
        with torch.autocast(
            device_type=str(self._device),
            dtype=MAP_PRECISION_TO_DTYPE[self._precision],
        ):
            with torch.no_grad():
                return self._policy.predict_action(obs_dict=obs_dict)

    @property
    def device(self) -> torch.device:
        """Get the device used for inference."""
        return self._device

    @property
    def checkpoint_path(self) -> str:
        """Get the checkpoint directory path."""
        return self._checkpoint_path

    @property
    def policy(self) -> Policy:
        """Get the loaded policy."""
        return self._policy

    @property
    def config(self) -> MainConfig:
        """Get the loaded experiment configuration."""
        return self._config

    @property
    def tokenizer(self) -> Tokenizer | None:
        """Get the loaded tokenizer, if any."""
        return self._tokenizer

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
            key: raw[key]
            for key in self.action_space.actions_metadata
            if key in raw
        }

    @property
    def depth_clamp_range(self) -> tuple[float, float] | None:
        """Get depth image clamping range from normalizer statistics.

        Returns:
            Tuple of (min, max) for clamping, or None if depth not in normalizer.
        """
        depth_key = Cameras.DEPTH.value
        if depth_key not in self._policy.normalizer.params_dict:
            return None
        stats = self._policy.normalizer[depth_key].params_dict.get("input_stats")
        if stats is None:
            return None
        return float(stats["min"].item()), float(stats["max"].item())
