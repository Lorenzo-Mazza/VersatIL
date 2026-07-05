"""Floating point policy checkpoint restoration."""

import logging
import os

import torch

from versatil.checkpoint_loading.base import (
    BaseCheckpointLoader,
    strip_compiled_prefixes,
    versatil_checkpoint_safe_globals,
)
from versatil.training.constants import (
    CheckpointFilename,
    CheckpointKey,
)
from versatil.training.lightning_policy import LightningPolicy


class FloatCheckpointLoader(BaseCheckpointLoader):
    """Restore a floating-point policy checkpoint, including configuration, tokenizer, normalizer (inside the policy class), and policy weights."""

    def __init__(
        self,
        device: torch.device,
        checkpoint_path: str,
        checkpoint_name: str,
    ) -> None:
        """Initialize and restore the policy checkpoint."""
        super().__init__(device=device, checkpoint_path=checkpoint_path)
        self._checkpoint_name = checkpoint_name
        self._load_model()

    def _load_model(self) -> None:
        """Load config, tokenizer, policy, and checkpoint weights."""
        config_path = os.path.join(
            self._checkpoint_path, CheckpointFilename.CONFIG.value
        )
        self._config = self._load_config(config_path=config_path)
        checkpoint_file = os.path.join(self._checkpoint_path, self._checkpoint_name)
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_file}.")
        logging.info(f"Loading policy checkpoint from {checkpoint_file}")

        self._policy = self._config.policy
        tokenizer_path = os.path.join(
            self._checkpoint_path, CheckpointFilename.TOKENIZER_DIR.value
        )
        self._tokenizer = self._load_tokenizer(tokenizer_path=tokenizer_path)
        if self._tokenizer is not None:
            self._tokenizer.to(self._device)
            self._policy.set_tokenizer(self._tokenizer)

        self._policy.to(self._device).eval()
        self._policy.device = self._device
        with torch.serialization.safe_globals(versatil_checkpoint_safe_globals()):
            checkpoint = torch.load(
                checkpoint_file,
                map_location=self._device,
                weights_only=True,
            )
        lightning_module = LightningPolicy(
            policy=self._policy,
            training_config=self._config.training,
        )
        state_dict_key = CheckpointKey.STATE_DICT.value
        checkpoint_state = strip_compiled_prefixes(checkpoint[state_dict_key])
        lightning_module.load_state_dict(checkpoint_state, strict=False)
        self._validate_checkpoint_loading(
            checkpoint_state_dict=checkpoint_state,
            model_state_dict=lightning_module.state_dict(),
        )
