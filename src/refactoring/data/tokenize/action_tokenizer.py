"""This module provides a wrapper around the FAST action tokenizer."""

import logging
from typing import Any

import numpy as np
import torch
from transformers import AutoProcessor


class ActionTokenizer:
    """Wrapper for FAST action tokenizer from https://arxiv.org/abs/2501.09747.

    FAST (Frequency-space Action Sequence Tokenization) provides efficient tokenization
    of multidimensional continuous action sequences into discrete tokens.

    Attributes:
        use_pretrained_weights: Whether to use the pretrained FAST tokenizer or fit on custom data.
        device: Target device for tensors.
        processor: HuggingFace AutoProcessor for FAST.
    """

    def __init__(
        self,
        use_pretrained_weights: bool = True,
        device: torch.device | None = None,
    ):
        """Initialize FAST action tokenizer.

        Args:
            use_pretrained_weights: If True, uses pretrained FAST tokenizer weights.
                If False, will fit the tokenizer on custom data during fit().
            device: Target device for tensors. If None, uses CPU.
        """
        self.use_pretrained_weights = use_pretrained_weights
        self.device = device if device is not None else torch.device("cpu")
        self.processor = AutoProcessor.from_pretrained(
            "physical-intelligence/fast", trust_remote_code=True
        )
        self._is_fitted = False
        if use_pretrained_weights:
            self._is_fitted = True

    def fit(self, action_chunks: np.ndarray) -> None:
        """Fit FAST tokenizer on action chunk data.

        Loads the FAST tokenizer code and trains it on the provided action chunks.
        Only needed if use_pretrained_weights=False.

        Args:
            action_chunks: Action chunks of shape (N, T, D) where:
                N = number of action chunks
                T = time horizon (chunk length)
                D = action dimension
                Values should be normalized to [-1, 1] range.

        Raises:
            ValueError: If trying to fit when using pretrained weights.
        """
        if self.use_pretrained_weights:
            raise ValueError(
                "Cannot fit when use_pretrained_weights=True. "
                "Initialize with use_pretrained_weights=False to fit on custom data."
            )

        logging.info(
            f"Fitting FAST tokenizer on {action_chunks.shape[0]} chunks "
            f"(time_horizon={action_chunks.shape[1]}, action_dim={action_chunks.shape[2]})"
        )
        self.processor = self.processor.fit(action_chunks)
        self._is_fitted = True


    def encode(self, action_chunks: np.ndarray | torch.Tensor) -> list[int]:
        """Encode action chunks to discrete tokens.

        Args:
            action_chunks: Action chunks of shape (B, T, D) or (T, D) where:
                B = batch size (optional)
                T = time horizon
                D = action dimension

        Returns:
            List of token integers.

        Raises:
            RuntimeError: If tokenizer has not been initialized.
        """
        if not self._is_fitted:
            raise RuntimeError("Tokenizer must be fitted or loaded before encoding")

        if isinstance(action_chunks, torch.Tensor):
            action_chunks = action_chunks.cpu().numpy()

        tokens = self.processor(action_chunks)
        return tokens

    def decode(self, tokens: list[int] | int) -> np.ndarray:
        """Decode discrete tokens back to action chunks.

        Args:
            tokens: List of token integers or single token.

        Returns:
            Reconstructed action chunks as numpy array.

        Raises:
            RuntimeError: If tokenizer has not been initialized.
        """
        if not self._is_fitted:
            raise RuntimeError("Tokenizer must be fitted or loaded before decoding")

        if isinstance(tokens, int):
            tokens = [tokens]

        decoded_actions = self.processor.decode(tokens)
        return decoded_actions

    def to(self, device: torch.device) -> "ActionTokenizer":
        """Move tokenizer to specified device.

        Args:
            device: Target device.

        Returns:
            Self for chaining.
        """
        self.device = device
        return self

    def save_pretrained(self, path: str) -> None:
        """Save fitted tokenizer to disk.

        Args:
            path: Path to save tokenizer.

        Raises:
            RuntimeError: If tokenizer has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted tokenizer")
        self.processor.save_pretrained(path)
        logging.info(f"Saved tokenizer to {path}")

    def state_dict(self) -> dict[str, Any]:
        """Get state dictionary for serialization.

        Returns:
            Dictionary containing tokenizer state.
        """
        return {
            "use_pretrained_weights": self.use_pretrained_weights,
            "device": str(self.device),
            "is_fitted": self._is_fitted,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state dictionary.

        Args:
            state_dict: State dictionary from state_dict().
        """
        self.use_pretrained_weights = state_dict["use_pretrained_weights"]
        self.device = torch.device(state_dict["device"])
        self._is_fitted = state_dict["is_fitted"]

        if self._is_fitted and self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                "physical-intelligence/fast", trust_remote_code=True
            )