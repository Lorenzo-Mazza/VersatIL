"""Discretizers for continuous action chunks."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers.processing_utils import ProcessorMixin

from versatil.data.constants import ActionDiscretizerType
from versatil.data.tokenization.binned_value_discretizer import BinnedValueDiscretizer
from versatil.data.tokenization.fast import load_fast_processor


class ActionDiscretizer(ABC):
    """Converts continuous action chunks to local discrete action IDs.

    Action chunks use shape (time_horizon, action_dim). Fitting data uses
    shape (num_chunks, time_horizon, action_dim).
    """

    @property
    @abstractmethod
    def token_count(self) -> int:
        """Number of local discrete action IDs."""

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Whether the discretizer can encode and decode actions."""

    @abstractmethod
    def fit(self, action_chunks: np.ndarray) -> None:
        """Fit on chunks with shape (num_chunks, time_horizon, action_dim)."""

    @abstractmethod
    def encode(self, action_chunk: np.ndarray) -> list[int]:
        """Encode one chunk with shape (time_horizon, action_dim)."""

    @abstractmethod
    def decode(self, token_sequences: list[list[int]]) -> np.ndarray:
        """Decode token sequences into shape (batch_size, time_horizon, action_dim)."""

    @abstractmethod
    def to(self, device: torch.device) -> None:
        """Move internal tensors to a device."""

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return serializable state."""

    @abstractmethod
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load serializable state."""

    def save_pretrained(self, path: Path) -> None:
        """Save optional external assets."""
        del path

    def load_pretrained_assets(self, path: Path) -> None:
        """Load optional external assets."""
        del path


class FastActionDiscretizer(ActionDiscretizer):
    """FAST discretizer for compressed action-token sequences."""

    def __init__(
        self,
        use_pretrained: bool = True,
        tokenizer_model: str = "physical-intelligence/fast",
    ):
        """Initialize FAST processor metadata and optional pretrained assets."""
        self.use_pretrained = use_pretrained
        self.tokenizer_model = tokenizer_model
        self.processor: ProcessorMixin | None = load_fast_processor(tokenizer_model)
        self._token_count = 2048 if use_pretrained else 1024
        self._is_fitted = use_pretrained
        self.time_horizon: int | None = None
        self.action_dim: int | None = None

    @property
    def token_count(self) -> int:
        """Number of local discrete action IDs."""
        return self._token_count

    @property
    def is_fitted(self) -> bool:
        """Whether the FAST processor can encode and decode actions."""
        return self._is_fitted

    def fit(self, action_chunks: np.ndarray) -> None:
        """Fit a local FAST processor on shape (num_chunks, time_horizon, action_dim)."""
        if self.use_pretrained:
            raise ValueError(
                "Cannot fit a pretrained FAST action discretizer. "
                "Set use_pretrained=False to fit FAST on local data."
            )
        if self.processor is None:
            raise RuntimeError("FAST processor not initialized")
        self.time_horizon = action_chunks.shape[1]
        self.action_dim = action_chunks.shape[2]
        self.processor = self.processor.fit(
            action_chunks,
            time_horizon=self.time_horizon,
            action_dim=self.action_dim,
        )
        self._is_fitted = True

    def encode(self, action_chunk: np.ndarray) -> list[int]:
        """Encode one chunk with shape (time_horizon, action_dim)."""
        if self.processor is None:
            raise RuntimeError("FAST processor not initialized")
        self.time_horizon = action_chunk.shape[-2]
        self.action_dim = action_chunk.shape[-1]
        return self.processor(action_chunk)[0]

    def decode(self, token_sequences: list[list[int]]) -> np.ndarray:
        """Decode FAST tokens into shape (batch_size, time_horizon, action_dim)."""
        if self.processor is None:
            raise RuntimeError("FAST processor not initialized")
        if self.time_horizon is None or self.action_dim is None:
            raise RuntimeError(
                "FAST action discretizer shape is unknown; encode or load a fitted "
                "discretizer before decoding"
            )
        clipped_sequences = [
            np.clip(sequence, 0, self.token_count - 1).tolist()
            for sequence in token_sequences
        ]
        decoded_actions = self.processor.decode(
            clipped_sequences,
            time_horizon=self.time_horizon,
            action_dim=self.action_dim,
        )
        if not isinstance(decoded_actions, np.ndarray):
            raise TypeError(
                f"Expected np.ndarray from FAST processor decode, got {type(decoded_actions)}"
            )
        return decoded_actions

    def to(self, device: torch.device) -> None:
        """No-op device transfer for the processor-backed discretizer."""
        del device

    def state_dict(self) -> dict[str, Any]:
        """Return serializable FAST discretizer state."""
        return {
            "type": ActionDiscretizerType.FAST.value,
            "use_pretrained": self.use_pretrained,
            "tokenizer_model": self.tokenizer_model,
            "token_count": self.token_count,
            "is_fitted": self.is_fitted,
            "time_horizon": self.time_horizon,
            "action_dim": self.action_dim,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load FAST discretizer state."""
        self.use_pretrained = state_dict["use_pretrained"]
        self.tokenizer_model = state_dict.get("tokenizer_model", self.tokenizer_model)
        self._token_count = state_dict.get("token_count", self._token_count)
        self._is_fitted = state_dict["is_fitted"]
        self.time_horizon = state_dict.get("time_horizon")
        self.action_dim = state_dict.get("action_dim")

    def save_pretrained(self, path: Path) -> None:
        """Save fitted local FAST processor assets."""
        if self.processor is not None and not self.use_pretrained:
            self.processor.save_pretrained(str(path / "fast_processor"))

    def load_pretrained_assets(self, path: Path) -> None:
        """Load saved local FAST processor assets when present."""
        fast_path = path / "fast_processor"
        if fast_path.exists():
            self.processor = load_fast_processor(str(fast_path))
            self._is_fitted = True


class BinnedActionDiscretizer(ActionDiscretizer):
    """Per-value quantile binning for chunks with shape (time_horizon, action_dim)."""

    def __init__(
        self,
        num_bins: int = 256,
        device: torch.device | None = None,
    ):
        """Initialize per-value binning for action chunks."""
        self.binner = BinnedValueDiscretizer(num_bins=num_bins, device=device)
        self.time_horizon: int | None = None
        self.action_dim: int | None = None

    @property
    def token_count(self) -> int:
        """Number of bins used for each action value."""
        return self.binner.num_bins

    @property
    def is_fitted(self) -> bool:
        """Whether bin edges have been fitted."""
        return self.binner._is_fitted

    def fit(self, action_chunks: np.ndarray) -> None:
        """Fit bin edges from shape (num_chunks, time_horizon, action_dim)."""
        self.time_horizon = action_chunks.shape[1]
        self.action_dim = action_chunks.shape[2]
        self.binner.fit(action_chunks)

    def encode(self, action_chunk: np.ndarray) -> list[int]:
        """Encode one chunk with shape (time_horizon, action_dim)."""
        tokens = self.binner.encode(action_chunk)
        return tokens.reshape(-1).detach().cpu().tolist()

    def decode(self, token_sequences: list[list[int]]) -> np.ndarray:
        """Decode bin-ID sequences into shape (batch_size, time_horizon, action_dim)."""
        if self.time_horizon is None or self.action_dim is None:
            raise RuntimeError("Binned action discretizer shape is unknown")

        expected_len = self.time_horizon * self.action_dim
        neutral_token_id = self.token_count // 2
        normalized_sequences = []
        for sequence in token_sequences:
            clipped = np.clip(sequence[:expected_len], 0, self.token_count - 1).tolist()
            if len(clipped) < expected_len:
                clipped.extend([neutral_token_id] * (expected_len - len(clipped)))
            normalized_sequences.append(clipped)

        tokens = torch.tensor(normalized_sequences, dtype=torch.long)
        tokens = tokens.reshape(
            len(token_sequences), self.time_horizon, self.action_dim
        )
        return self.binner.decode(tokens).detach().cpu().numpy()

    def to(self, device: torch.device) -> None:
        """Move bin-edge tensors to a device."""
        self.binner.to(device)

    def state_dict(self) -> dict[str, Any]:
        """Return serializable binned discretizer state."""
        return {
            "type": ActionDiscretizerType.BINNED.value,
            "num_bins": self.token_count,
            "time_horizon": self.time_horizon,
            "action_dim": self.action_dim,
            "binner": self.binner.state_dict(),
            "is_fitted": self.is_fitted,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load binned discretizer state."""
        self.time_horizon = state_dict.get("time_horizon")
        self.action_dim = state_dict.get("action_dim")
        self.binner.load_state_dict(state_dict["binner"])
