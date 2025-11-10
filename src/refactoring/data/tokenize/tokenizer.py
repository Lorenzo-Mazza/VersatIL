"""Multi-key tokenizer for actions and observations.

This module provides a dictionary-based tokenizer that manages multiple
per-key tokenizers and operates on normalized data.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoProcessor

from refactoring.data.constants import ACTION_KEY
from refactoring.data.tokenize.action_tokenizer import ActionTokenizer
from refactoring.data.tokenize.binning_tokenizer import BinningTokenizer


class Tokenizer:
    """Multi-key tokenizer for actions and proprioceptive observations.

    This class manages multiple tokenizers (one per data key) and provides
    a unified interface for tokenization and detokenization. All tokenizers
    expect data to be pre-normalized.

    Attributes:
        device: Target device for tensors.
        tokenizers: Dictionary mapping data keys to their tokenizers.
    """

    def __init__(self, device: torch.device | None = None):
        """Initialize tokenizer.

        Args:
            device: Target device for tensors. If None, uses CPU.
        """
        self.device = device if device is not None else torch.device("cpu")
        self.tokenizers: dict[str, ActionTokenizer | BinningTokenizer] = {}

    def fit_action_tokenizer(
        self,
        action_chunks: np.ndarray,
        use_pretrained_weights: bool = True,
    ) -> None:
        """Fit or load FAST action tokenizer.

        Args:
            action_chunks: Normalized action chunks of shape (N_chunks, T, D) where:
                N_chunks = number of action chunks from dataset
                T = prediction horizon (chunk length)
                D = total action dimension
                Values should be normalized to [-1, 1] range.
            use_pretrained_weights: If True, uses pretrained FAST weights.
                If False, fits FAST tokenizer on provided action_chunks.
        """
        action_tokenizer = ActionTokenizer(
            use_pretrained_weights=use_pretrained_weights,
            device=self.device,
        )
        if not use_pretrained_weights:
            action_tokenizer.fit(action_chunks)
        self.tokenizers[ACTION_KEY] = action_tokenizer
        mode = "pretrained" if use_pretrained_weights else "fitted"
        logging.info(
            f"Initialized action tokenizer (mode={mode}, "
            f"chunks_shape={action_chunks.shape})"
        )

    def fit_proprio_tokenizer(
        self,
        normalized_proprio: dict[str, np.ndarray],
        num_bins: int,
    ) -> None:
        """Fit quantile binning tokenizer on normalized proprio data.

        Args:
            normalized_proprio: Dictionary of normalized proprioceptive arrays.
            num_bins: Number of bins for quantile-based discretization.
        """
        for key, data in normalized_proprio.items():
            binning_tokenizer = BinningTokenizer(num_bins=num_bins, device=self.device)
            binning_tokenizer.fit(data)
            self.tokenizers[key] = binning_tokenizer
            logging.info(
                f"Fitted binning tokenizer for key '{key}' with {num_bins} bins "
                f"(data shape: {data.shape})"
            )

    def tokenize(self, normalized_data: dict[str, Any]) -> dict[str, Any]:
        """Tokenize normalized data.

        Args:
            normalized_data: Dictionary of normalized data arrays/tensors.

        Returns:
            Dictionary with tokenized data where applicable, passthrough otherwise.
        """
        tokenized = {}
        for key, data in normalized_data.items():
            if key in self.tokenizers:
                tokenized[key] = self.tokenizers[key].encode(data)
            else:
                tokenized[key] = data
        return tokenized

    def detokenize(self, tokens: dict[str, Any]) -> dict[str, Any]:
        """Detokenize tokens back to normalized data.

        Args:
            tokens: Dictionary of token arrays/tensors.

        Returns:
            Dictionary with detokenized normalized data where applicable.
        """
        detokenized = {}
        for key, token_data in tokens.items():
            if key in self.tokenizers:
                detokenized[key] = self.tokenizers[key].decode(token_data)
            else:
                detokenized[key] = token_data
        return detokenized

    def to(self, device: torch.device) -> "Tokenizer":
        """Move all tokenizers to specified device.

        Args:
            device: Target device.

        Returns:
            Self for chaining.
        """
        self.device = device
        for tokenizer in self.tokenizers.values():
            tokenizer.to(device)
        return self

    def state_dict(self) -> dict[str, Any]:
        """Get state dictionary for serialization.

        Returns:
            Dictionary containing tokenizer states.
        """
        return {
            "device": str(self.device),
            "tokenizers": {
                key: tokenizer.state_dict()
                for key, tokenizer in self.tokenizers.items()
            },
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state dictionary.

        Args:
            state_dict: State dictionary from state_dict().
        """
        self.device = torch.device(state_dict["device"])
        for key, tokenizer_state in state_dict["tokenizers"].items():
            if key in self.tokenizers:
                self.tokenizers[key].load_state_dict(tokenizer_state)

    def save_pretrained(self, path: str | Path) -> None:
        """Save all tokenizers to disk using HuggingFace-style serialization.
        
        Args:
            path: Directory path to save tokenizers.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        metadata = {
            "device": str(self.device),
            "tokenizer_keys": {},
        }

        for key, tokenizer in self.tokenizers.items():
            tokenizer_dir = path / key
            tokenizer_dir.mkdir(parents=True, exist_ok=True)

            if isinstance(tokenizer, ActionTokenizer):
                tokenizer.save_pretrained(str(tokenizer_dir))
                metadata["tokenizer_keys"][key] = {
                    "type": "action",
                    "use_pretrained_weights": tokenizer.use_pretrained_weights,
                }
            elif isinstance(tokenizer, BinningTokenizer):
                torch.save(
                    {
                        "bin_edges": tokenizer.bin_edges,
                        "num_bins": tokenizer.num_bins,
                    },
                    tokenizer_dir / "binning_state.pt",
                )
                metadata["tokenizer_keys"][key] = {
                    "type": "binning",
                    "num_bins": tokenizer.num_bins,
                }

        with open(path / "config.json", "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def from_pretrained(cls, path: str | Path, device: torch.device | None = None) -> "Tokenizer":
        """Load tokenizers from disk.

        Args:
            path: Directory path where tokenizers were saved via save_pretrained().
            device: Target device for tensors. If None, uses CPU.

        Returns:
            Loaded Tokenizer instance.

        Raises:
            FileNotFoundError: If path doesn't exist or config.json is missing.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer path not found: {path}")

        config_path = path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {path}")

        with open(config_path) as f:
            metadata = json.load(f)

        if device is None:
            device = torch.device(metadata.get("device", "cpu"))
        tokenizer = cls(device=device)

        for key, info in metadata["tokenizer_keys"].items():
            tokenizer_dir = path / key

            if info["type"] == "action":
                action_tok = ActionTokenizer(
                    use_pretrained_weights=info["use_pretrained_weights"],
                    device=device,
                )
                # Load the fitted processor from disk
                if tokenizer_dir.exists():
                    action_tok.processor = AutoProcessor.from_pretrained(
                        str(tokenizer_dir),
                        trust_remote_code=True,
                    )
                    action_tok._is_fitted = True
                tokenizer.tokenizers[key] = action_tok

            elif info["type"] == "binning":
                binning_tok = BinningTokenizer(
                    num_bins=info["num_bins"],
                    device=device,
                )
                # Load binning state
                state = torch.load(tokenizer_dir / "binning_state.pt", map_location=device)
                binning_tok.bin_edges = state["bin_edges"].to(device)
                binning_tok._is_fitted = True
                tokenizer.tokenizers[key] = binning_tok

        logging.info(f"Loaded tokenizer from {path}")
        return tokenizer