"""Tokenizer class that holds both observation and action tokenizers."""

import logging
from pathlib import Path

import torch

from versatil.configs.data.tokenizer import TokenizationConfig
from versatil.data.constants import TokenizerType
from versatil.data.tokenization.observation_tokenizer import ObservationTokenizer
from versatil.data.tokenization.action_tokenizer import ActionTokenizer


class Tokenizer:
    """Tokenizer class for observations and actions.

    This class holds both observation and action tokenizers and provides
    a single interface for the Policy to interact with.

    Attributes:
        observation_tokenizer: Tokenizer for observations (language + proprio)
        action_tokenizer: Tokenizer for actions (FAST + optional language vocab mapping)
    """

    def __init__(
        self,
        observation_tokenizer: ObservationTokenizer | None = None,
        action_tokenizer: ActionTokenizer | None = None,
    ):
        """Initialize unified tokenizer.

        Args:
            observation_tokenizer: Tokenizer for observations (language + proprio)
            action_tokenizer: Tokenizer for actions (FAST + optional language vocab mapping)
        """
        self.observation_tokenizer = observation_tokenizer
        self.action_tokenizer = action_tokenizer

    @property
    def observation_vocab_size(self) -> int | None:
        """Get observation tokenizer vocab size."""
        if self.observation_tokenizer is not None:
            return self.observation_tokenizer.vocab_size
        return None

    @property
    def action_vocab_size(self) -> int | None:
        """Get action tokenizer vocab size."""
        if self.action_tokenizer is not None:
            return self.action_tokenizer.vocab_size
        return None

    def to(self, device: torch.device) -> "Tokenizer":
        """Move tokenizers to device.

        Args:
            device: Target device

        Returns:
            Self for chaining
        """
        if self.observation_tokenizer is not None:
            self.observation_tokenizer.to(device)
        if self.action_tokenizer is not None:
            self.action_tokenizer.to(device)
        return self

    def save_pretrained(self, path: str | Path) -> None:
        """Save tokenizers to disk.

        Args:
            path: Directory path to save tokenizers
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.observation_tokenizer is not None:
            obs_path = path / "observation_tokenizer"
            self.observation_tokenizer.save_pretrained(obs_path)
            logging.info(f"Saved observation tokenizer to {obs_path}")

        if self.action_tokenizer is not None:
            action_path = path / "action_tokenizer"
            self.action_tokenizer.save_pretrained(action_path)
            logging.info(f"Saved action tokenizer to {action_path}")

    @classmethod
    def from_pretrained(
        cls, path: str | Path, device: torch.device | None = None
    ) -> "Tokenizer":
        """Load tokenizers from disk.

        Args:
            path: Directory path where tokenizers were saved
            device: Target device for tensors

        Returns:
            Loaded Tokenizer instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer path not found: {path}")

        observation_tokenizer = None
        action_tokenizer = None

        obs_path = path / "observation_tokenizer"
        if obs_path.exists():
            observation_tokenizer = ObservationTokenizer.from_pretrained(
                obs_path, device=device
            )
            logging.info(f"Loaded observation tokenizer from {obs_path}")

        action_path = path / "action_tokenizer"
        if action_path.exists():
            action_tokenizer = ActionTokenizer.from_pretrained(
                action_path, device=device
            )
            logging.info(f"Loaded action tokenizer from {action_path}")

        return cls(
            observation_tokenizer=observation_tokenizer,
            action_tokenizer=action_tokenizer,
        )


def validate_tokenizer_config(config: TokenizationConfig):
    if config.tokenize_observations and config.observation_tokenizer is None:
        raise ValueError(
            "observation_tokenizer must be provided when tokenize_observations=True"
        )
    if config.tokenize_actions and config.action_tokenizer is None:
        raise ValueError("action_tokenizer must be provided when tokenize_actions=True")

    if config.action_tokenizer is not None:
        valid_tokenizers = [t.value for t in TokenizerType]
        for tokenizer in config.action_tokenizer.tokenizer_chain:
            if tokenizer not in valid_tokenizers:
                raise ValueError(
                    f"Invalid tokenizer '{tokenizer}' in chain. Must be one of {valid_tokenizers}"
                )
        # Validate language tokenizer model is provided if needed
        if (
            TokenizerType.LANGUAGE.value in config.action_tokenizer.tokenizer_chain
            and config.action_tokenizer.language_tokenizer_model is None
        ):
            raise ValueError(
                f"language_tokenizer_model must be provided when '{TokenizerType.LANGUAGE.value}' key is in tokenizer_chain"
            )
