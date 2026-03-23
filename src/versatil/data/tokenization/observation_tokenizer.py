"""Observation tokenizer for creating unified prompts from multiple observation keys.

This tokenizer:
1. Takes multiple observation keys (language, proprio, gripper, etc.)
2. Optionally bins continuous data into quantiles
3. Creates a unified prompt string (e.g., "TaskSpace: grasp needle, State in robot frame: 127 143 89")
4. Tokenizes the prompt using a language model tokenizer
5. Returns token IDs and padding masks
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from versatil.data.constants import (
    ObsKey,
    SampleKey,
)
from versatil.data.tokenization.binning_tokenizer import BinningTokenizer


class ObservationTokenizer:
    """Tokenizes multiple observation keys into a unified prompt.

    Creates prompts like:
    "TaskSpace: grasp needle, State in robot frame: 127 143 89, State in camera frame: 88 201 43"

    All specified observation keys are combined into a single string, then tokenized.
    """

    def __init__(
        self,
        tokenizer_model: str,
        observation_keys: list[str],
        bin_continuous_data: bool = True,
        num_bins: int = 256,
        max_token_len: int = 256,
        device: torch.device | None = None,
    ):
        """Initialize observation tokenizer.

        Args:
            tokenizer_model: HuggingFace model name (e.g., "google/gemma-2b")
            observation_keys: List of observation keys to include in prompt (order preserved)
            bin_continuous_data: Whether to bin continuous data into quantiles
            num_bins: Number of bins for quantile-based discretization
            max_token_len: Maximum token sequence length
            device: Target device for tensors
        """
        self.tokenizer_model = tokenizer_model
        self.observation_keys = observation_keys
        self.bin_continuous_data = bin_continuous_data
        self.num_bins = num_bins
        self.max_token_len = max_token_len
        self.device = device if device is not None else torch.device("cpu")
        self.language_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        if self.language_tokenizer.pad_token is None:
            self.language_tokenizer.pad_token = self.language_tokenizer.eos_token

        self.vocab_size = self.language_tokenizer.vocab_size
        self.binning_tokenizers: dict[str, BinningTokenizer] = {}
        self._is_fitted = False

    def fit(self, observation_data: dict[str, np.ndarray]) -> None:
        """Fit binning tokenizers on observation data.

        Args:
            observation_data: Dict mapping obs keys to arrays of shape (N, ..., D)
                where N is number of samples. Language keys are skipped.
        """
        if not self.bin_continuous_data:
            self._is_fitted = True
            logging.info("Binning disabled, observation tokenizer marked as fitted")
            return

        for key in self.observation_keys:
            if key == ObsKey.LANGUAGE.value:
                continue  # Language doesn't need binning

            if key not in observation_data:
                logging.warning(
                    f"Key '{key}' not found in observation data, skipping binning"
                )
                continue

            data = observation_data[key]
            binning_tok = BinningTokenizer(num_bins=self.num_bins, device=self.device)
            binning_tok.fit(data)
            self.binning_tokenizers[key] = binning_tok

        self._is_fitted = True
        logging.info(
            f"Fitted observation tokenizer on {len(self.observation_keys)} keys "
            f"({len(self.binning_tokenizers)} with binning) "
            f"(model={self.tokenizer_model}, vocab_size={self.vocab_size})"
        )

    def tokenize(self, observations: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Tokenize observations into unified prompt.

        Args:
            observations: Dict with observation data (can be batched or single sample)
                - Language keys: list[str] or list[list[str]]
                - Continuous keys: torch.Tensor or np.ndarray

        Returns:
            Dict with:
                - "tokens": Token IDs (B, max_token_len)
                - "is_pad_observation": Padding mask (B, max_token_len)
        """
        if not self._is_fitted:
            raise RuntimeError("Tokenizer must be fitted before encoding")

        # TODO: temporal dimension should be explicitly passed to this function, not inferred from inputs.
        first_tensor = next(
            (v for v in observations.values() if isinstance(v, torch.Tensor)), None
        )
        has_time_dim = first_tensor is not None and first_tensor.ndim >= 3
        batch_size, time_steps = None, None
        if has_time_dim:
            batch_size = first_tensor.shape[0]
            time_steps = first_tensor.shape[1]
            observations = {
                k: v.reshape(-1, *v.shape[2:]) if isinstance(v, torch.Tensor) else v
                for k, v in observations.items()
            }
        elif not has_time_dim:
            first_nested_list = next(
                (
                    v
                    for v in observations.values()
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], list)
                ),
                None,
            )
            if first_nested_list is not None:
                has_time_dim = True
                batch_size = len(first_nested_list)
                time_steps = len(first_nested_list[0])
                observations = {
                    k: [item for sublist in v for item in sublist]
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], list)
                    else v
                    for k, v in observations.items()
                }

        prompts = self._build_prompts(observations)
        tokenized = self.language_tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.max_token_len,
            return_tensors="pt",
        )
        tokens = tokenized["input_ids"]
        is_pad = ~tokenized["attention_mask"].to(torch.bool)
        if has_time_dim:
            # Reshape (B*T, seq) -> (B, T, seq)
            tokens = tokens.reshape(batch_size, time_steps, -1)
            is_pad = is_pad.reshape(batch_size, time_steps, -1)

        return {
            SampleKey.TOKENIZED_OBSERVATIONS.value: tokens.to(self.device),
            SampleKey.IS_PAD_OBSERVATION.value: is_pad.to(self.device),
        }

    def _build_prompts(self, observations: dict[str, Any]) -> list[str]:
        """Build prompt strings from observations.

        Prompt format:
        "Task: {language}, {key1}: {binned_value1}, {key2}: {binned_value2}, ..."

        Example:
        "Task: grasp needle, proprio robot frame: 12 34 56, gripper state: 78 90;"
        """
        first_val = next(iter(observations.values()))
        if isinstance(first_val, (torch.Tensor, np.ndarray)):
            batch_size = first_val.shape[0]
        elif isinstance(first_val, list):
            batch_size = len(first_val)
        else:
            batch_size = 1

        prompts = []
        for i in range(batch_size):
            parts = []
            for key in self.observation_keys:
                if key not in observations:
                    logging.warning(
                        f"Key '{key}' not found in observation data, skipping prompt"
                    )
                    continue

                data = observations[key]
                if key == ObsKey.LANGUAGE.value:
                    if isinstance(data, list):
                        if batch_size > 1:
                            text_list = data[i]
                            text = (
                                text_list
                                if isinstance(text_list, str)
                                else " ".join(text_list)
                            )
                        else:
                            text_list = data[0] if data else []
                            text = (
                                text_list
                                if isinstance(text_list, str)
                                else " ".join(text_list)
                            )
                    else:
                        if not isinstance(data, str):
                            raise TypeError(
                                f"Expected str for language data, got {type(data)}"
                            )
                        text = data

                    cleaned = text.lower().strip().replace("_", " ").replace("\n", " ")
                    parts.append(f"Task: {cleaned}")
                else:
                    # Handle continuous data (proprio, gripper, etc.)
                    if isinstance(data, torch.Tensor):
                        sample = data[i] if batch_size > 1 else data
                        sample = sample.cpu().float().numpy()
                    elif isinstance(data, np.ndarray):
                        sample = data[i] if batch_size > 1 else data
                    else:
                        continue

                    if self.bin_continuous_data and key in self.binning_tokenizers:
                        binned = self.binning_tokenizers[key].encode(sample)
                        sample_str = " ".join(
                            map(str, binned.cpu().float().numpy().flatten().tolist())
                        )
                    else:
                        # Use raw float values as strings
                        sample_str = " ".join(
                            f"{x:.3f}" for x in sample.flatten().tolist()
                        )

                    key_readable = key.replace("_", " ")
                    parts.append(f"{key_readable}: {sample_str}")

            prompt = ", ".join(parts) + ";\n"
            prompts.append(prompt)

        return prompts

    def to(self, device: torch.device) -> "ObservationTokenizer":
        """Move tokenizer to specified device.

        Args:
            device: Target device

        Returns:
            Self for chaining
        """
        self.device = device
        for tokenizer in self.binning_tokenizers.values():
            tokenizer.to(device)
        return self

    def state_dict(self) -> dict[str, Any]:
        """Get state dictionary for serialization.

        Returns:
            Dictionary containing tokenizer state
        """
        return {
            "tokenizer_model": self.tokenizer_model,
            "observation_keys": self.observation_keys,
            "bin_continuous_data": self.bin_continuous_data,
            "num_bins": self.num_bins,
            "max_token_len": self.max_token_len,
            "vocab_size": self.vocab_size,
            "binning_tokenizers": {
                key: tok.state_dict() for key, tok in self.binning_tokenizers.items()
            },
            "is_fitted": self._is_fitted,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state dictionary.

        Args:
            state_dict: State dictionary from state_dict()
        """
        self.tokenizer_model = state_dict["tokenizer_model"]
        self.observation_keys = state_dict["observation_keys"]
        self.bin_continuous_data = state_dict["bin_continuous_data"]
        self.num_bins = state_dict["num_bins"]
        self.max_token_len = state_dict["max_token_len"]
        self.vocab_size = state_dict["vocab_size"]
        self._is_fitted = state_dict["is_fitted"]

        for key, tok_state in state_dict["binning_tokenizers"].items():
            binning_tok = BinningTokenizer(num_bins=self.num_bins, device=self.device)
            binning_tok.load_state_dict(tok_state)
            self.binning_tokenizers[key] = binning_tok

    def save_pretrained(self, path: str | Path) -> None:
        """Save tokenizer to disk.

        Args:
            path: Directory path to save tokenizer
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "observation_tokenizer_state.pt")
        self.language_tokenizer.save_pretrained(path / "language_tokenizer")
        logging.info(f"Saved observation tokenizer to {path}")

    @classmethod
    def from_pretrained(
        cls, path: str | Path, device: torch.device | None = None
    ) -> "ObservationTokenizer":
        """Load tokenizer from disk.

        Args:
            path: Directory path where tokenizer was saved
            device: Target device for tensors

        Returns:
            Loaded ObservationTokenizer instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer path not found: {path}")
        state_dict = torch.load(
            path / "observation_tokenizer_state.pt",
            map_location=device or torch.device("cpu"),
            weights_only=False,
        )
        tokenizer = cls(
            tokenizer_model=state_dict["tokenizer_model"],
            observation_keys=state_dict["observation_keys"],
            bin_continuous_data=state_dict["bin_continuous_data"],
            num_bins=state_dict["num_bins"],
            max_token_len=state_dict["max_token_len"],
            device=device,
        )
        tokenizer.load_state_dict(state_dict)
        tokenizer.language_tokenizer = AutoTokenizer.from_pretrained(
            path / "language_tokenizer"
        )
        logging.info(f"Loaded observation tokenizer from {path}")
        return tokenizer
