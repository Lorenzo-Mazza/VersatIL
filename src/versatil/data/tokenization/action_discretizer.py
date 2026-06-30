"""Discretizers for continuous action chunks."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.fft import idct
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
        decoded_actions = [
            self._decode_fast_token_sequence(
                token_sequence=sequence,
            )
            for sequence in token_sequences
        ]
        return np.stack(decoded_actions)

    def _decode_fast_token_sequence(self, token_sequence: list[int]) -> np.ndarray:
        """Decode one FAST BPE stream with local DCT length fix."""
        if self.processor is None:
            raise RuntimeError("FAST processor not initialized")
        if self.time_horizon is None or self.action_dim is None:
            raise RuntimeError("FAST action discretizer shape is unknown")
        bpe_tokenizer = getattr(self.processor, "bpe_tokenizer", None)
        if bpe_tokenizer is None:
            raise RuntimeError("FAST processor does not expose a BPE tokenizer")

        expected_coefficient_count = self.time_horizon * self.action_dim
        token_array = np.asarray(token_sequence, dtype=np.int64)
        token_array = np.clip(token_array, 0, self.token_count - 1)
        decoded_tokens = bpe_tokenizer.decode(token_array.tolist())
        if not isinstance(decoded_tokens, str):
            raise TypeError(
                f"Expected str from FAST BPE tokenizer decode, got {type(decoded_tokens)}"
            )
        coefficients = self._decoded_tokens_to_coefficients(
            decoded_tokens=decoded_tokens,
            expected_coefficient_count=expected_coefficient_count,
        )
        coefficient_matrix = coefficients.reshape(self.time_horizon, self.action_dim)
        return idct(
            coefficient_matrix / self._processor_scale(),
            axis=0,
            norm="ortho",
        )

    def _decoded_tokens_to_coefficients(
        self,
        decoded_tokens: str,
        expected_coefficient_count: int,
    ) -> np.ndarray:
        """Convert decoded FAST characters to a fixed-length DCT vector."""
        min_token = self._processor_min_token()
        coefficients = (
            np.asarray([ord(token) for token in decoded_tokens], dtype=np.float32)
            + min_token
        )
        if coefficients.size < expected_coefficient_count:
            return np.pad(
                coefficients,
                (0, expected_coefficient_count - coefficients.size),
                mode="constant",
                constant_values=0,
            )
        return coefficients[:expected_coefficient_count]

    def _processor_scale(self) -> float:
        """Return the FAST processor DCT scale."""
        if self.processor is None:
            raise RuntimeError("FAST processor not initialized")
        scale = getattr(self.processor, "scale", None)
        if scale is None:
            raise RuntimeError("FAST processor does not expose a scale")
        return float(scale)

    def _processor_min_token(self) -> float:
        """Return the FAST processor minimum token offset."""
        if self.processor is None:
            raise RuntimeError("FAST processor not initialized")
        min_token = getattr(self.processor, "min_token", None)
        if min_token is None:
            raise RuntimeError("FAST processor does not expose a min_token")
        return float(min_token)

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
        normalized_sequences = []
        for sequence in token_sequences:
            if len(sequence) != expected_len:
                raise ValueError(
                    "Binned action token sequence has invalid length: "
                    f"expected {expected_len} tokens for shape "
                    f"({self.time_horizon}, {self.action_dim}), got {len(sequence)}."
                )
            token_array = np.asarray(sequence)
            if np.any(token_array < 0) or np.any(token_array >= self.token_count):
                raise ValueError(
                    "Binned action token sequence contains IDs outside the valid "
                    f"range [0, {self.token_count - 1}]."
                )
            normalized_sequences.append(token_array.astype(np.int64).tolist())

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


class UniformBinnedActionDiscretizer(ActionDiscretizer):
    """Per-value uniform binning over a fixed normalized action range."""

    def __init__(
        self,
        num_bins: int = 256,
        min_value: float = -1.0,
        max_value: float = 1.0,
    ):
        """Initialize fixed-range per-value binning for action chunks."""
        if num_bins < 2:
            raise ValueError(f"num_bins must be at least 2, got {num_bins}.")
        if min_value >= max_value:
            raise ValueError(
                f"min_value must be smaller than max_value, got {min_value} >= {max_value}."
            )
        self.num_bins = num_bins
        self.min_value = min_value
        self.max_value = max_value
        self.time_horizon: int | None = None
        self.action_dim: int | None = None
        self._is_fitted = False

    @property
    def token_count(self) -> int:
        """Number of fixed bins used for each action value."""
        return self.num_bins

    @property
    def is_fitted(self) -> bool:
        """Whether action chunk shape metadata has been fitted."""
        return self._is_fitted

    def fit(self, action_chunks: np.ndarray) -> None:
        """Record shape metadata from shape (num_chunks, time_horizon, action_dim)."""
        if action_chunks.ndim != 3:
            raise ValueError(
                "Uniform binned action discretizer expects fitting data with shape "
                f"(num_chunks, time_horizon, action_dim), got {action_chunks.shape}."
            )
        self.time_horizon = action_chunks.shape[1]
        self.action_dim = action_chunks.shape[2]
        self._is_fitted = True

    def encode(self, action_chunk: np.ndarray) -> list[int]:
        """Encode one chunk with shape (time_horizon, action_dim)."""
        if not self._is_fitted:
            raise RuntimeError("Uniform binned action discretizer must be fitted first")
        clipped_actions = np.clip(action_chunk, self.min_value, self.max_value)
        bin_width = (self.max_value - self.min_value) / self.num_bins
        tokens = np.floor((clipped_actions - self.min_value) / bin_width).astype(
            np.int64
        )
        tokens = np.clip(tokens, 0, self.num_bins - 1)
        return tokens.reshape(-1).tolist()

    def decode(self, token_sequences: list[list[int]]) -> np.ndarray:
        """Decode bin-ID sequences into shape (batch_size, time_horizon, action_dim)."""
        if self.time_horizon is None or self.action_dim is None:
            raise RuntimeError("Uniform binned action discretizer shape is unknown")

        expected_len = self.time_horizon * self.action_dim
        normalized_sequences = []
        for sequence in token_sequences:
            if len(sequence) != expected_len:
                raise ValueError(
                    "Uniform binned action token sequence has invalid length: "
                    f"expected {expected_len} tokens for shape "
                    f"({self.time_horizon}, {self.action_dim}), got {len(sequence)}."
                )
            token_array = np.asarray(sequence)
            if np.any(token_array < 0) or np.any(token_array >= self.token_count):
                raise ValueError(
                    "Uniform binned action token sequence contains IDs outside the "
                    f"valid range [0, {self.token_count - 1}]."
                )
            normalized_sequences.append(token_array.astype(np.int64).tolist())

        tokens = np.asarray(normalized_sequences, dtype=np.float32)
        bin_width = (self.max_value - self.min_value) / self.num_bins
        actions = self.min_value + (tokens + 0.5) * bin_width
        return actions.reshape(len(token_sequences), self.time_horizon, self.action_dim)

    def to(self, device: torch.device) -> None:
        """No-op device transfer for fixed scalar metadata."""
        del device

    def state_dict(self) -> dict[str, Any]:
        """Return serializable uniform binned discretizer state."""
        return {
            "type": ActionDiscretizerType.UNIFORM_BINNED.value,
            "num_bins": self.num_bins,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "time_horizon": self.time_horizon,
            "action_dim": self.action_dim,
            "is_fitted": self.is_fitted,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load uniform binned discretizer state."""
        self.num_bins = state_dict["num_bins"]
        self.min_value = state_dict.get("min_value", self.min_value)
        self.max_value = state_dict.get("max_value", self.max_value)
        if self.num_bins < 2:
            raise ValueError(f"num_bins must be at least 2, got {self.num_bins}.")
        if self.min_value >= self.max_value:
            raise ValueError(
                "min_value must be smaller than max_value, got "
                f"{self.min_value} >= {self.max_value}."
            )
        self.time_horizon = state_dict.get("time_horizon")
        self.action_dim = state_dict.get("action_dim")
        self._is_fitted = state_dict["is_fitted"]
