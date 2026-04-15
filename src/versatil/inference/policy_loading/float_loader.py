"""Floating point policy loader from training checkpoint."""

import logging
import os

import numpy as np
import torch

from versatil.inference.policy_loading.base import BasePolicyLoader
from versatil.models.policy import Policy
from versatil.training.constants import (
    CheckpointFilename,
    CheckpointKey,
    PrecisionType,
)
from versatil.training.lightning_policy import LightningPolicy


class PolicyLoader(BasePolicyLoader):
    """Loads a trained float policy from a checkpoint directory.

    Handles configuration loading, checkpoint validation, tokenizer setup,
    precision conversion, and autocast inference wrapping.
    """

    def __init__(
        self,
        device: torch.device,
        checkpoint_path: str,
        checkpoint_name: str = CheckpointFilename.DEFAULT_CHECKPOINT.value,
        seed: int = 42,
        compile_model: bool = True,
    ) -> None:
        """Initialize the policy loader.

        Args:
            device: Device to load the model onto.
            checkpoint_path: Path to the checkpoint directory.
            checkpoint_name: Name of the checkpoint file.
            seed: Random seed for reproducibility.
            compile_model: Whether to compile the policy with
                torch.compile for optimized inference.
        """
        super().__init__(device=device, checkpoint_path=checkpoint_path)
        self._checkpoint_name = checkpoint_name
        self._compile_model = compile_model
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
        config_path = os.path.join(
            self._checkpoint_path, CheckpointFilename.CONFIG.value
        )
        self._config = self._load_config(config_path=config_path)

        checkpoint_file = os.path.join(self._checkpoint_path, self._checkpoint_name)
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_file}.")
        logging.info(f"Loading model and tokenizer from {checkpoint_file}")

        self._policy = self._config.policy
        tokenizer_path = os.path.join(
            self._checkpoint_path, CheckpointFilename.TOKENIZER_DIR.value
        )
        self._tokenizer = self._load_tokenizer(tokenizer_path=tokenizer_path)
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
        state_dict_key = CheckpointKey.STATE_DICT.value
        lightning_module.load_state_dict(checkpoint[state_dict_key], strict=False)
        self._validate_checkpoint_loading(
            checkpoint_state_dict=checkpoint[state_dict_key],
            model_state_dict=lightning_module.state_dict(),
        )

        self._precision = str(self._config.experiment.precision)
        precision_type = PrecisionType(self._precision)
        if precision_type.should_convert_model():
            self._policy = self._policy.to(precision_type.get_model_dtype())

        if self._compile_model:
            self._policy = torch.compile(self._policy)
            logging.info("Compiled policy with torch.compile")

        logging.info("Model and config successfully loaded.")

    def run_inference(
        self, obs_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Run policy inference with autocast and no_grad.

        Args:
            obs_dict: Observation dictionary for the policy.

        Returns:
            Action dictionary from policy.predict_action.
        """
        with (
            torch.autocast(
                device_type=str(self._device),
                dtype=PrecisionType(self._precision).get_model_dtype(),
            ),
            torch.no_grad(),
        ):
            return self._policy.predict_action(obs_dict=obs_dict)

    @property
    def policy(self) -> Policy:
        """Get the loaded policy."""
        return self._policy
