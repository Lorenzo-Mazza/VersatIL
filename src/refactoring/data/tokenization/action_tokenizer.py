"""Action tokenizer with support for FAST tokenization and language vocabulary mapping."""

import logging
import os
from typing import Any
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizerFast

from refactoring.data.constants import TokenizerType, TOKENIZED_ACTIONS_KEY, IS_PAD_ACTION_KEY

# Disable tokenizers parallelism to avoid fork warnings with DataLoader workers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class ActionTokenizer:
    """Action tokenizer with FAST and language vocabulary mapping support.

    FAST (Frequency-space Action Sequence Tokenization) provides efficient tokenization
    of multidimensional continuous action sequences into discrete tokens.

    When using language tokenizer, FAST tokens are mapped to the END of the language
    vocabulary, avoiding special tokens. This allows both text and action tokens to
    coexist in the same vocabulary space.

    Attributes:
        tokenizer_chain: List of tokenizer types to apply
        use_pretrained_fast: Whether to use pretrained FAST weights
        language_tokenizer_model: HuggingFace model for language tokenizer
        fast_vocab_size: FAST tokenizer vocabulary size
        device: Target device for tensors
    """

    def __init__(
        self,
        tokenizer_chain: list[str],
        use_pretrained_fast: bool = True,
        language_tokenizer_model: str | None = None,
        fast_tokenizer_model: str  = "physical-intelligence/fast",
        num_special_tokens_to_skip: int = 128,
        max_token_len: int = 256,
        pad_token_id: int = 0,
        device: torch.device | None = None,
    ):
        """Initialize action tokenizer.

        Args:
            tokenizer_chain: List of TokenizerType values to apply in sequence
                - [TokenizerType.FAST.value]: Just FAST tokenization
                - [TokenizerType.FAST.value, TokenizerType.LANGUAGE.value]: FAST → language vocab mapping
            use_pretrained_fast: Whether to use pretrained FAST weights
            language_tokenizer_model: HuggingFace model if TokenizerType.LANGUAGE in chain
            fast_tokenizer_model : HuggingFace model for FAST tokenizer
            num_special_tokens_to_skip: Number of special tokens at end of language vocab to skip
            max_token_len: Maximum token sequence length (for padding)
            pad_token_id: Token ID to use for padding
            device: Target device for tensors
        """
        self.tokenizer_chain = tokenizer_chain
        self.use_pretrained_fast = use_pretrained_fast
        self.fast_tokenizer_model = fast_tokenizer_model
        self.language_tokenizer_model = language_tokenizer_model
        # FAST vocab sizes: 2048 for pretrained, 1024 for custom-trained
        self.fast_vocab_size = 2048 if use_pretrained_fast else 1024
        self.num_special_tokens_to_skip = num_special_tokens_to_skip
        self.max_token_len = max_token_len
        self.pad_token_id = pad_token_id
        self.device = device if device is not None else torch.device("cpu")

        self.fast_processor: PreTrainedTokenizerFast | None = None
        self.language_tokenizer: AutoTokenizer | None = None
        self.vocab_size: int | None = None
        self._is_fitted = False

        self._build_tokenizers()

    def _build_tokenizers(self) -> None:
        """Build the tokenizer chain."""
        if TokenizerType.FAST.value in self.tokenizer_chain:
            self.fast_processor = AutoProcessor.from_pretrained(
                self.fast_tokenizer_model, trust_remote_code=True
            )
            if self.use_pretrained_fast:
                self._is_fitted = True
                self.vocab_size = self.fast_vocab_size

        if TokenizerType.LANGUAGE.value in self.tokenizer_chain:
            if self.language_tokenizer_model is None:
                raise ValueError(
                    f"language_tokenizer_model must be provided when '{TokenizerType.LANGUAGE.value}' is in tokenizer_chain"
                )

            self.language_tokenizer = AutoTokenizer.from_pretrained(self.language_tokenizer_model)
            if self.language_tokenizer.pad_token is None:
                self.language_tokenizer.pad_token = self.language_tokenizer.eos_token

            # Final vocab is language tokenizer's vocab
            self.vocab_size = self.language_tokenizer.vocab_size
            required_vocab_size = self.fast_vocab_size + self.num_special_tokens_to_skip
            if self.vocab_size < required_vocab_size:
                raise ValueError(
                    f"Language tokenizer vocab size ({self.vocab_size}) is too small to hold "
                    f"FAST tokens ({self.fast_vocab_size}) + special tokens ({self.num_special_tokens_to_skip}). "
                    f"Required: {required_vocab_size}"
                )


    def fit(self, action_chunks: np.ndarray) -> None:
        """Fit FAST tokenizer on action chunk data.

        Only needed if use_pretrained_fast=False.

        Args:
            action_chunks: Action chunks of shape (N, T, D) where:
                N = number of action chunks
                T = time horizon (chunk length)
                D = action dimension
                Values should be normalized to [-1, 1] range.

        Raises:
            ValueError: If trying to fit when using pretrained weights.
        """
        if self.use_pretrained_fast:
            raise ValueError(
                "Cannot fit when use_pretrained_fast=True. "
                "Initialize with use_pretrained_fast=False to fit on custom data."
            )

        if self.fast_processor is None:
            raise RuntimeError("FAST processor not initialized")

        logging.info(
            f"Fitting FAST tokenizer on {action_chunks.shape[0]} chunks "
            f"(time_horizon={action_chunks.shape[1]}, action_dim={action_chunks.shape[2]})"
        )
        self.fast_processor = self.fast_processor.fit(action_chunks)
        self._is_fitted = True

        if self.language_tokenizer is None:
            self.vocab_size = self.fast_vocab_size

        logging.info(
            f"Fitted action tokenizer (chain={self.tokenizer_chain}, vocab_size={self.vocab_size})"
        )

    def _map_fast_to_language_vocab(self, fast_tokens: list[int] | np.ndarray) -> np.ndarray:
        """Map FAST tokens to language tokenizer vocabulary positions.

        FAST tokens are mapped to the END of the language vocabulary, avoiding special tokens:
        lang_token_id = lang_vocab_size - 1 - num_special_tokens_to_skip - fast_token_id

        Args:
            fast_tokens: FAST token IDs in range [0, fast_vocab_size)

        Returns:
            Mapped token IDs in language vocabulary space
        """
        if self.language_tokenizer is None:
            raise RuntimeError("Language tokenizer not initialized")
        fast_tokens_arr = np.array(fast_tokens) if isinstance(fast_tokens, list) else fast_tokens
        # Map FAST tokens to end of language vocab
        mapped_tokens = (
            self.language_tokenizer.vocab_size - 1
            - self.num_special_tokens_to_skip
            - fast_tokens_arr
        )
        return mapped_tokens

    def _unmap_language_to_fast_vocab(self, lang_tokens: np.ndarray | torch.Tensor) -> np.ndarray:
        """Reverse mapping from language vocab positions to FAST token IDs.

        Args:
            lang_tokens: Token IDs in language vocabulary space

        Returns:
            FAST token IDs in range [0, fast_vocab_size)
        """
        if self.language_tokenizer is None:
            raise RuntimeError("Language tokenizer not initialized")

        if isinstance(lang_tokens, torch.Tensor):
            lang_tokens = lang_tokens.cpu().numpy()
        fast_tokens = (
            self.language_tokenizer.vocab_size - 1
            - self.num_special_tokens_to_skip
            - lang_tokens
        )
        return fast_tokens

    def encode_chunk(
        self,
        action_chunk: np.ndarray | torch.Tensor,
        is_pad_mask: torch.Tensor | np.ndarray | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode a single action chunk to discrete tokens with fixed length and padding.

        Args:
            action_chunk: Action chunk of shape (T, D) where:
                T = time horizon
                D = action dimension
            is_pad_mask: Boolean mask of shape (T,) indicating which timesteps are padded.
                If provided, only non-padded actions are tokenized.

        Returns:
            Dict with:
                - TOKENIZED_ACTIONS_KEY: Token IDs (max_token_len,) padded to fixed length
                - IS_PAD_ACTION_KEY: Padding mask (max_token_len,) indicating which tokens are padding

        Raises:
            RuntimeError: If tokenizer has not been initialized.
        """
        if not self._is_fitted:
            raise RuntimeError("Tokenizer must be fitted or loaded before encoding")

        # Filter out padded actions if mask is provided
        if is_pad_mask is not None:
            if isinstance(is_pad_mask, torch.Tensor):
                is_pad_mask_np = is_pad_mask.cpu().numpy()
            else:
                is_pad_mask_np = is_pad_mask

            valid_mask = ~is_pad_mask_np
            if isinstance(action_chunk, torch.Tensor):
                action_chunk_to_tokenize = action_chunk[valid_mask].cpu().numpy()
            else:
                action_chunk_to_tokenize = action_chunk[valid_mask]
        else:
            if isinstance(action_chunk, torch.Tensor):
                action_chunk_to_tokenize = action_chunk.cpu().numpy()
            else:
                action_chunk_to_tokenize = action_chunk

        if self.fast_processor is not None:
            fast_tokens = self.fast_processor(action_chunk_to_tokenize)[0]  # Returns list[list[int]], take first
            if self.language_tokenizer is not None:
                mapped_tokens = self._map_fast_to_language_vocab(fast_tokens)
                tokens = mapped_tokens.tolist()
            else:
                tokens = fast_tokens

            tokens_len = len(tokens)
            if tokens_len < self.max_token_len:
                padding_len = self.max_token_len - tokens_len
                tokens = tokens + [self.pad_token_id] * padding_len
                is_pad = [False] * tokens_len + [True] * padding_len
            else:
                if tokens_len > self.max_token_len:
                    logging.warning(
                        f"Token length ({tokens_len}) exceeds max_token_len ({self.max_token_len}), truncating"
                    )
                tokens = tokens[: self.max_token_len]
                is_pad = [False] * self.max_token_len

            return {
                TOKENIZED_ACTIONS_KEY: torch.tensor(tokens, dtype=torch.long, device=self.device),
                IS_PAD_ACTION_KEY: torch.tensor(is_pad, dtype=torch.bool, device=self.device),
            }

        raise RuntimeError("No tokenizers in chain")

    def encode_batch(
        self,
        action_chunks: np.ndarray | torch.Tensor,
        is_pad_mask: torch.Tensor | np.ndarray | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode a batch of action chunks to discrete tokens."""
        if not self._is_fitted:
            raise RuntimeError("Tokenizer must be fitted or loaded before encoding")
        batch_size = action_chunks.shape[0]
        all_tokens = []
        all_is_pad = []
        for i in range(batch_size):
            chunk_pad_mask = is_pad_mask[i] if is_pad_mask is not None else None
            result = self.encode_chunk(action_chunks[i], is_pad_mask=chunk_pad_mask)
            all_tokens.append(result[TOKENIZED_ACTIONS_KEY])
            all_is_pad.append(result[IS_PAD_ACTION_KEY])

        return {
            TOKENIZED_ACTIONS_KEY: torch.stack(all_tokens),
            IS_PAD_ACTION_KEY: torch.stack(all_is_pad),
        }

    def encode(
        self,
        action_chunks: np.ndarray | torch.Tensor,
        is_pad_mask: torch.Tensor | np.ndarray | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode action chunk(s) to discrete tokens, automatically adapting on the input shape.
        """
        if isinstance(action_chunks, torch.Tensor):
            arr = action_chunks
        else:
            arr = np.asarray(action_chunks)

        if arr.ndim == 2:
            # Single chunk (T, D)
            return self.encode_chunk(action_chunks, is_pad_mask)
        elif arr.ndim == 3:
            # Batch (N, T, D)
            return self.encode_batch(action_chunks)
        else:
            raise ValueError(f"Expected 2D or 3D input, got shape {arr.shape}")

    def decode_chunk(self, tokens: torch.Tensor | list[int] | np.ndarray) -> np.ndarray:
        """Decode a single token sequence back to action chunk.

        Args:
            tokens: Token sequence of shape (max_token_len,) or list of token IDs

        Returns:
            Reconstructed action chunk as numpy array of shape (T, D)

        Raises:
            RuntimeError: If tokenizer has not been initialized.
        """
        if not self._is_fitted:
            raise RuntimeError("Tokenizer must be fitted or loaded before decoding")

        if isinstance(tokens, torch.Tensor):
            tokens_arr = tokens.cpu().numpy()
        elif isinstance(tokens, list):
            tokens_arr = np.array(tokens)
        else:
            tokens_arr = tokens

        valid_tokens = tokens_arr[tokens_arr != self.pad_token_id]
        if self.language_tokenizer is not None:
            fast_tokens = self._unmap_language_to_fast_vocab(valid_tokens)
        else:
            fast_tokens = valid_tokens

        if self.fast_processor is not None:
            decoded_actions = self.fast_processor.decode([fast_tokens.tolist()])
            assert isinstance(decoded_actions, np.ndarray)
            return decoded_actions[0]  # Return single chunk

        raise RuntimeError("Cannot decode without FAST processor")

    def decode_batch(self, tokens: torch.Tensor | np.ndarray) -> np.ndarray:
        """Decode a batch of discrete tokens back to action chunks.

        Args:
            tokens: Token tensor or array of shape (N, max_token_len)

        Returns:
            Reconstructed action chunks as numpy array of shape (N, T, D)

        Raises:
            RuntimeError: If tokenizer has not been initialized.
        """
        if not self._is_fitted:
            raise RuntimeError("Tokenizer must be fitted or loaded before decoding")

        if isinstance(tokens, torch.Tensor):
            tokens_arr = tokens.cpu().numpy()
        else:
            tokens_arr = tokens

        if self.language_tokenizer is not None:
            fast_tokens = self._unmap_language_to_fast_vocab(tokens_arr)
        else:
            fast_tokens = tokens_arr

        tokens_list_of_lists = []
        for i in range(fast_tokens.shape[0]):
            sample_tokens = fast_tokens[i]
            valid_tokens = sample_tokens[sample_tokens != self.pad_token_id]
            tokens_list_of_lists.append(valid_tokens.tolist())

        if self.fast_processor is not None:
            decoded_actions = self.fast_processor.decode(tokens_list_of_lists)
            assert isinstance(decoded_actions, np.ndarray)
            return decoded_actions

        raise RuntimeError("Cannot decode without FAST processor")

    def decode(self, tokens: torch.Tensor | list[int] | np.ndarray) -> np.ndarray:
        """Decode discrete tokens back to action chunks, automatically adapting on the input shape."""
        if isinstance(tokens, torch.Tensor):
            arr = tokens
        elif isinstance(tokens, list):
            arr = np.array(tokens)
        else:
            arr = tokens

        if arr.ndim == 1:
            # Single sequence (max_token_len,)
            return self.decode_chunk(tokens)
        elif arr.ndim == 2:
            # Batch (N, max_token_len)
            return self.decode_batch(tokens)
        else:
            raise ValueError(f"Expected 1D or 2D input, got shape {arr.shape}")

    def to(self, device: torch.device) -> "ActionTokenizer":
        """Move tokenizer to specified device.

        Args:
            device: Target device.

        Returns:
            Self for chaining.
        """
        self.device = device
        return self

    def save_pretrained(self, path: str | Path) -> None:
        """Save tokenizer to disk.

        Args:
            path: Directory path to save tokenizer

        Raises:
            RuntimeError: If tokenizer has not been fitted.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted tokenizer")

        # Save state dict
        torch.save(self.state_dict(), path / "action_tokenizer_state.pt")

        # Save FAST processor if fitted
        if self.fast_processor is not None and not self.use_pretrained_fast:
            self.fast_processor.save_pretrained(str(path / "fast_processor"))

        # Save language tokenizer if used
        if self.language_tokenizer is not None:
            self.language_tokenizer.save_pretrained(path / "language_tokenizer")

        logging.info(f"Saved action tokenizer to {path}")

    def state_dict(self) -> dict[str, Any]:
        """Get state dictionary for serialization.

        Returns:
            Dictionary containing tokenizer state.
        """
        return {
            "tokenizer_chain": self.tokenizer_chain,
            "use_pretrained_fast": self.use_pretrained_fast,
            "language_tokenizer_model": self.language_tokenizer_model,
            "fast_vocab_size": self.fast_vocab_size,
            "num_special_tokens_to_skip": self.num_special_tokens_to_skip,
            "vocab_size": self.vocab_size,
            "is_fitted": self._is_fitted,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state dictionary.

        Args:
            state_dict: State dictionary from state_dict().
        """
        self.tokenizer_chain = state_dict["tokenizer_chain"]
        self.use_pretrained_fast = state_dict["use_pretrained_fast"]
        self.language_tokenizer_model = state_dict["language_tokenizer_model"]
        self.fast_vocab_size = state_dict["fast_vocab_size"]
        self.num_special_tokens_to_skip = state_dict["num_special_tokens_to_skip"]
        self.vocab_size = state_dict["vocab_size"]
        self._is_fitted = state_dict["is_fitted"]

    @classmethod
    def from_pretrained(
        cls, path: str | Path, device: torch.device | None = None
    ) -> "ActionTokenizer":
        """Load tokenizer from disk.

        Args:
            path: Directory path where tokenizer was saved
            device: Target device for tensors

        Returns:
            Loaded ActionTokenizer instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer path not found: {path}")

        state_dict = torch.load(
            path / "action_tokenizer_state.pt", map_location=device or torch.device("cpu")
        )

        tokenizer = cls(
            tokenizer_chain=state_dict["tokenizer_chain"],
            use_pretrained_fast=state_dict["use_pretrained_fast"],
            language_tokenizer_model=state_dict["language_tokenizer_model"],
            num_special_tokens_to_skip=state_dict["num_special_tokens_to_skip"],
            device=device,
        )

        tokenizer.load_state_dict(state_dict)
        fast_path = path / "fast_processor"
        if fast_path.exists() and tokenizer.fast_processor is not None:
            tokenizer.fast_processor = AutoProcessor.from_pretrained(
                str(fast_path), trust_remote_code=True
            )
            tokenizer._is_fitted = True

        lang_path = path / "language_tokenizer"
        if lang_path.exists():
            tokenizer.language_tokenizer = AutoTokenizer.from_pretrained(lang_path)

        logging.info(f"Loaded action tokenizer from {path}")
        return tokenizer